//
// Created by ekin4 on 08/11/2023.
//
#include <random>
#include <tuple>
#include <array>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/phase.h>


NAMESPACE_BEGIN(mitsuba)

/**!

.. _integrator-volpath-raymarching:

Volumetric path tracer (:monosp:`volpath-raymarching`)
-------------------------------------------

.. pluginparameters::

 * - max_depth
   - |int|
   - Specifies the longest path depth in the generated output image (where -1 corresponds to
     :math:`\infty`). A value of 1 will only render directly visible light sources. 2 will lead
     to single-bounce (direct-only) illumination, and so on. (Default: -1)

 * - rr_depth
   - |int|
   - Specifies the minimum path depth, after which the implementation will start to use the
     *russian roulette* path termination criterion. (Default: 5)

 * - hide_emitters
   - |bool|
   - Hide directly visible emitters. (Default: no, i.e. |false|)

 * - absolute_tolerance
   - |float|
   - Set the tolerance threshold for the absolute difference between the RK4
     and RK5 estimates in the RK45 integrator
 * - relative_tolerance
   - |float|
   - Set the tolerance threshold for relative absolute difference between the
     RK4 and RK5 estimates in the RK45 integrator

This plugin provides a volumetric path tracer that can be used to compute approximate solutions
of the radiative transfer equation. Its implementation makes use of multiple importance sampling
to combine BSDF and phase function sampling with direct illumination sampling strategies. On
surfaces, it behaves exactly like the standard path tracer.

This integrator has special support for index-matched transmission events (i.e. surface scattering
events that do not change the direction of light). As a consequence, participating media enclosed by
a stencil shape are rendered considerably more efficiently when this shape
has a :ref:`null <bsdf-null>` or :ref:`thin dielectric <bsdf-thindielectric>` BSDF assigned
to it (as compared to, say, a :ref:`dielectric <bsdf-dielectric>` or
:ref:`roughdielectric <bsdf-roughdielectric>` BSDF).

.. note:: This integrator does not implement good sampling strategies to render
    participating media with a spectrally varying extinction coefficient. For these cases,
    it is better to use the more advanced :ref:`volumetric path tracer with
    spectral MIS <integrator-volpathmis>`, which will produce in a significantly less noisy
    rendered image.

.. warning:: This integrator does not support forward-mode differentiation.

.. tabs::
    .. code-tab::  xml

        <integrator type="volpath_raymarching">
            <integer name="max_depth" value="8"/>
        </integrator>

    .. code-tab:: python

        'type': 'volpath_raymarching',
        'max_depth': 8

*/
template <typename Float, typename Spectrum>
class VolumetricMarchedPathIntegrator : public MonteCarloIntegrator<Float, Spectrum> {

public:
    MI_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth, m_hide_emitters)
    MI_IMPORT_TYPES(Scene, Sampler, Emitter, EmitterPtr, BSDF, BSDFPtr,
                    Medium, MediumPtr, PhaseFunctionContext)


    VolumetricMarchedPathIntegrator(const Properties &props) : Base(props) {
        if (const auto sampling_mode = props.get<int>("sampling_mode", 0);
            sampling_mode == 0) {
            m_use_emitter_sampling = true;
            m_use_uni_sampling     = true;
        } else if (sampling_mode == 1) {
            m_use_emitter_sampling = false;
            m_use_uni_sampling     = true;
        } else if (sampling_mode == 2) {
            m_use_emitter_sampling = true;
            m_use_uni_sampling     = false;
        }
        if (props.has_property("tolerance") && (props.has_property("absolute_tolerance") || props.has_property("relative_tolerance"))) {
            Log(LogLevel::Error, "Cannot specify both tolerance and either of absolute_tolerance and relative_tolerance simultaneously!");
        }
        if (props.has_property("tolerance")) {
            m_absolute_tolerance = props.get<ScalarFloat>("tolerance");
            m_relative_tolerance = m_absolute_tolerance;
        }
        else {
            m_absolute_tolerance = props.get<ScalarFloat>("absolute_tolerance", dr::sqrt(dr::Epsilon<Float>));
            m_relative_tolerance = props.get<ScalarFloat>("relative_tolerance", dr::sqrt(dr::Epsilon<Float>));
        }
        m_jitter_steps = props.get<bool>("use_step_jitter", true);
        dr::make_opaque(m_absolute_tolerance, m_relative_tolerance);
    }

    MI_INLINE
    Float index_spectrum(const UnpolarizedSpectrum &spec, const UInt32 &idx) const {
        Float m = spec[0];
        if constexpr (is_rgb_v<Spectrum>) { // Handle RGB rendering
            dr::masked(m, (idx == 1u)) = spec[1];
            dr::masked(m, (idx == 2u)) = spec[2];
        } else {
            DRJIT_MARK_USED(idx);
        }
        return m;
    }

    MI_INLINE
    MediumInteraction3f sample_raymarched_interaction(MediumPtr& medium, const Ray3f &ray, Mask active) const {
        MI_MASKED_FUNCTION(ProfilerPhase::MediumSample, active);

        // initialize basic medium interaction fields
        MediumInteraction3f mei = dr::zeros<MediumInteraction3f>();
        mei.wi          = -ray.d;
        mei.sh_frame    = Frame3f(mei.wi);
        mei.time        = ray.time;
        mei.wavelengths = ray.wavelengths;

        auto [aabb_its, mint, maxt] = medium->intersect_aabb(ray);
        aabb_its &= (dr::isfinite(mint) || dr::isfinite(maxt));
        active &= aabb_its;
        dr::masked(mint, !active) = 0.f;
        dr::masked(maxt, !active) = dr::Infinity<Float>;

        mint = dr::maximum(0.f, mint);
        maxt = dr::minimum(ray.maxt, maxt);

        Float sampled_t = maxt;
        Mask valid_mi   = active && sampled_t <= maxt && dr::isfinite(mint);
        mei.t           = dr::select(valid_mi, sampled_t, dr::Infinity<Float>);
        mei.p           = ray(sampled_t);
        mei.medium      = medium;
        mei.mint        = mint;

        std::tie(mei.sigma_s, mei.sigma_n, mei.sigma_t) =
            medium->get_scattering_coefficients(mei, valid_mi);
        mei.combined_extinction = medium->get_majorant(mei, active);
        mei.radiance = medium->get_radiance(mei, active);
        return mei;
    }


    using rk_df_type = std::function<Spectrum(MediumInteraction3f&, Ray3f&, const Spectrum&, const Float&, const Mask&)>;

    std::tuple<Spectrum, Float, Float>
    integration_step_rk(rk_df_type& df, Ray3f &ray, MediumInteraction3f& mei, Spectrum& initial_value, Float& dt, Mask active, const bool no_refine_step, const bool ignore_est_every_substep) const {
        MI_MASKED_FUNCTION(ProfilerPhase::MediumSample, active)
        // --------------------- RK45CK -------------------------- //
        // In order to accelerate the numerical integration via marching, we can use an adaptive stepping algorithm
        // We utilise the embedded Runge-Kutta 4(5) method with Cash-Karp coefficients
        // With this we can find the total optical depth of the ray segment that starts at
        // current_flight_distance and ends at current_flight_distance + dt
        // Since the ODE is linear and the right hand side is independent of the dependent variable (optical_depth)
        // We can simply accumulate the results of each segment as we step through them

        struct LoopStateRK {
            MediumInteraction3f mei;
            Ray3f ray;
            Spectrum initial_value;
            Mask active;
            UInt32 niter;
            Spectrum rk_est;
            Float curr_dt;
            Float next_dt;
            Float max_initial_scale;

            DRJIT_STRUCT(LoopStateRK, mei, ray, initial_value, \
                        active, niter, rk_est, \
                        curr_dt, next_dt, max_initial_scale)
        } ls_rk = {
            mei = mei,
            ray = ray,
            initial_value = initial_value,
            active = active,
            0,
            dr::zeros<Spectrum>(),
            dt,
            dt,
            dr::max(unpolarized_spectrum(dr::abs(initial_value))),
        };

        dr::tie(ls_rk) = dr::while_loop(dr::make_tuple(ls_rk),
            [](const LoopStateRK& ls_rk) { return ls_rk.active; },
            [this, no_refine_step, ignore_est_every_substep, &df](LoopStateRK& ls_rk) {
            auto& mei = ls_rk.mei;
            auto& ray = ls_rk.ray;
            auto& initial_value = ls_rk.initial_value;
            auto& active = ls_rk.active;
            auto& niter = ls_rk.niter;
            auto& rk_est = ls_rk.rk_est;
            auto& curr_dt = ls_rk.curr_dt;
            auto& next_dt = ls_rk.next_dt;
            auto& max_initial_scale = ls_rk.max_initial_scale;

            auto rk_err_est = dr::zeros<Spectrum>();

            std::array<Spectrum, rk_stages> rk_ki;
            dr::masked(rk_est, active) = dr::zeros<Spectrum>();
            for (std::size_t idx = 0; idx < rk_stages; idx++) {
                rk_ki[idx] = initial_value;
                for (std::size_t jdx = 0; jdx < idx && !ignore_est_every_substep; jdx++) {
                    if (rk_aij[idx][jdx] != 0.0)
                        rk_ki[idx] += static_cast<ScalarFloat>(rk_aij[idx][jdx]) * rk_ki[jdx];
                }
                rk_ki[idx] = curr_dt * df(mei, ray, rk_ki[idx], static_cast<ScalarFloat>(rk_ci[idx]) * curr_dt, active);
                // nth order estimate of step
                if (rk_bi[idx] != 0.0)
                    dr::masked(rk_est, active) += rk_ki[idx] * static_cast<ScalarFloat>(rk_bi[idx]);
                // Difference estimate of step
                if (rk_ei[idx] != 0.0)
                    dr::masked(rk_err_est, active) += rk_ki[idx] * static_cast<ScalarFloat>(rk_ei[idx]);
            }

            // Error estimate from difference between 5th and 4th order estimates
            auto err_estimate = dr::max(dr::detach(unpolarized_spectrum(dr::abs(rk_err_est))));
            // Based on scipy scaling of error
            Float scale_of_err = m_absolute_tolerance;
            dr::masked(scale_of_err, active) += dr::maximum(max_initial_scale, dr::max(dr::detach(unpolarized_spectrum(dr::abs(rk_est))))) * m_relative_tolerance;
            Float error_norm = dr::select(scale_of_err > 0.0f, err_estimate / scale_of_err, 0.0f);

            Float corr = 2.0f;
            dr::masked(corr, err_estimate > 0.f && active) = 0.9f * dr::pow(error_norm, -(1.0f / rk_order));
            dr::masked(corr, (scale_of_err == 0.0f) && active) = 1.0f;

            dr::masked(next_dt, active) = dr::maximum(math::ShadowEpsilon<Float>, dr::abs(curr_dt * corr));
            next_dt = dr::copysign(next_dt, curr_dt);
            // dr::masked(next_dt, active && curr_dt < 0.f) *= -1.0f;
            //            Log(Debug, "Correction factor: %s ~ Current dt: %s | Next dt: %s ~ Minimum dt: %s ~ Error: %s", corr, curr_dt, next_dt, 5.0f * math::RayEpsilon<Float>, rk_diff_est);
            ++niter;
            active &= error_norm >= 1.0f && !no_refine_step && niter < 8;
            dr::masked(curr_dt, active) = next_dt;
        }, "VolpathMarch[rk single step]");

        //        Log(Debug, "Initial timestep: %s | Current timestep: %s | Next timestep: %s | Error estimate: %s | Estimate: %s | Scale: %s", initial_dt, curr_dt, next_dt, rk_diff_est, rk5_est, scale_of_err);
        return std::make_tuple(ls_rk.rk_est, ls_rk.curr_dt, ls_rk.next_dt);
    }

    std::pair<Spectrum, Mask> sample(const Scene *scene,
                                     Sampler *sampler,
                                     const RayDifferential3f &ray_,
                                     const Medium *initial_medium,
                                     Float * /* aovs */,
                                     Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        // If there is an environment emitter and emitters are visible: all rays will be valid
        // Otherwise, it will depend on whether a valid interaction is sampled
        Mask valid_ray = !m_hide_emitters && (scene->environment() != nullptr);

        // For now, don't use ray differentials
        Ray3f ray = ray_;

        // Tracks radiance scaling due to index of refraction changes
        Float eta(1.f);

        Spectrum throughput(1.f), result(0.f);
        MediumPtr medium = initial_medium;
        auto mei = dr::zeros<MediumInteraction3f>();
        Mask specular_chain = active && !m_hide_emitters;
        UInt32 depth = 0;

        UInt32 channel = 0;
        if (is_rgb_v<Spectrum>) {
            auto n_channels = (uint32_t) dr::size_v<Spectrum>;
            channel = (UInt32) dr::minimum(sampler->next_1d(active) * n_channels, n_channels - 1);
        }

        auto si = dr::zeros<SurfaceInteraction3f>();
        auto last_scatter_event = dr::zeros<Interaction3f>();

        Mask needs_intersection = true;
        Float last_scatter_direction_pdf = 1.f;

        /* Set up a Dr.Jit loop (optimizes away to a normal loop in scalar mode,
           generates wavefront or megakernel renderer based on configuration).
           Register everything that changes as part of the loop here */
        struct LoopState {
            Mask active;
            UInt32 depth;
            Ray3f ray;
            Spectrum throughput;
            Spectrum result;
            SurfaceInteraction3f si;
            MediumInteraction3f mei;
            MediumPtr medium;
            Float eta;
            Interaction3f last_scatter_event;
            Float last_scatter_direction_pdf;
            Mask needs_intersection;
            Mask specular_chain;
            Mask valid_ray;
            Sampler* sampler;

            DRJIT_STRUCT(LoopState, active, depth, ray, throughput, result, \
                si, mei, medium, eta, last_scatter_event, \
                last_scatter_direction_pdf, needs_intersection, \
                specular_chain, valid_ray, sampler)
        } ls = {
            active,
            depth,
            ray,
            throughput,
            result,
            si,
            mei,
            medium,
            eta,
            last_scatter_event,
            last_scatter_direction_pdf,
            needs_intersection,
            specular_chain,
            valid_ray,
            sampler
        };

        dr::tie(ls) = dr::while_loop(dr::make_tuple(ls),
            [](const LoopState& ls) { return ls.active; },
            [this, scene, channel](LoopState& ls) {

            Mask& active = ls.active;
            UInt32& depth = ls.depth;
            Ray3f& ray = ls.ray;
            Spectrum& throughput = ls.throughput;
            Spectrum& result = ls.result;
            SurfaceInteraction3f& si = ls.si;
            MediumInteraction3f& mei = ls.mei;
            MediumPtr& medium = ls.medium;
            Float& eta = ls.eta;
            Interaction3f& last_scatter_event = ls.last_scatter_event;
            Float& last_scatter_direction_pdf = ls.last_scatter_direction_pdf;
            Mask& needs_intersection = ls.needs_intersection;
            Mask& specular_chain = ls.specular_chain;
            Mask& valid_ray = ls.valid_ray;
            Sampler* sampler = ls.sampler;

            // ----------------- Handle termination of paths ------------------
            // Russian roulette: try to keep path weights equal to one, while accounting for the
            // solid angle compression at refractive index boundaries. Stop with at least some
            // probability to avoid  getting stuck (e.g. due to total internal reflection)
            active &= dr::any(unpolarized_spectrum(throughput) != 0.f);
            Float q = dr::minimum(dr::max(unpolarized_spectrum(throughput)) * dr::square(eta), .95f);
            Mask perform_rr = (depth > (uint32_t) m_rr_depth);
            active &= sampler->next_1d(active) < q || !perform_rr;
            dr::masked(throughput, perform_rr) *= dr::rcp(dr::detach(q));

            active &= depth < (uint32_t) m_max_depth;

            // ----------------------- Sampling the RTE -----------------------
            Mask active_medium  = active && (medium != nullptr);
            Mask active_surface = active && !active_medium;
            Mask act_medium_scatter = false, escaped_medium = false;

            if (dr::any_or<true>(active_medium)) {
                mei = sample_raymarched_interaction(medium, ray, active_medium);
                dr::masked(ray.maxt, active_medium) = dr::norm(scene->bbox().extents());
                Mask intersect = needs_intersection && active_medium;
                if (dr::any_or<true>(intersect))
                    dr::masked(si, intersect) = scene->ray_intersect(ray, intersect);
                needs_intersection &= !intersect;
                dr::masked(mei.t, active_medium && (si.t < mei.mint)) = dr::Infinity<Float>;

                escaped_medium = active_medium && !mei.is_valid();
                active_medium &= mei.is_valid();
            }

            if (dr::any_or<true>(active_medium)) {
                // Get maximum flight distance of ray
                auto [aabb_its, mint, maxt] = medium->intersect_aabb(ray);

                Float max_flight_distance = dr::minimum(si.t, mei.t);
                dr::masked(mei.t, active_medium) = mei.mint;
                dr::masked(mei.p, active_medium) = ray(mei.t);
                Float initial_dt = max_flight_distance - mei.t;
                initial_dt *= m_absolute_tolerance;
                dr::masked(initial_dt, initial_dt < math::ShadowEpsilon<Float>) = math::ShadowEpsilon<Float>;
                dr::masked(initial_dt, initial_dt > max_flight_distance - mei.t) = max_flight_distance - mei.t;
                Float dt = initial_dt;

                // Instantiate mask that tracks which rays are able to continue marching
                Mask iteration_mask = active_medium;

                // Instantiate tracking of optical depth, this will be used to estimate throughput
                Spectrum optical_depth(0.0f), full_path_radiance(0.0f);
                dr::masked(optical_depth, iteration_mask && medium->is_homogeneous()) = max_flight_distance * mei.sigma_t;

                // Jitter step size if needed
                if (m_jitter_steps) {
                    dr::masked(dt, iteration_mask) *= sampler->next_1d(iteration_mask);
                }
                dr::masked(dt, iteration_mask && dt > max_flight_distance - mei.t) = max_flight_distance - mei.t;

                struct LoopStateTraversal {
                    Mask iteration_mask;
                    Ray3f ray;
                    Float dt;
                    MediumInteraction3f mei;
                    MediumPtr medium;
                    Spectrum full_path_radiance;
                    Spectrum optical_depth;
                    Float max_flight_distance;

                    DRJIT_STRUCT(LoopStateTraversal, iteration_mask, ray, \
                                  dt, mei, medium, full_path_radiance, \
                                  optical_depth, max_flight_distance)
                } ls_traversal = {
                    iteration_mask = iteration_mask,
                    ray = ray,
                    dt = dt,
                    mei = mei,
                    medium = medium,
                    full_path_radiance = full_path_radiance,
                    optical_depth = optical_depth,
                    max_flight_distance = max_flight_distance
                };
                dr::tie(ls_traversal) = dr::while_loop(dr::make_tuple(ls_traversal),
                        [](const LoopStateTraversal& ls_traversal) { return dr::detach(ls_traversal.iteration_mask); },
                        [this, scene, channel](LoopStateTraversal& ls_traversal) {
                    auto& iteration_mask = ls_traversal.iteration_mask;
                    auto& ray = ls_traversal.ray;
                    auto& dt = ls_traversal.dt;
                    auto& mei = ls_traversal.mei;
                    auto& medium = ls_traversal.medium;
                    auto& full_path_radiance = ls_traversal.full_path_radiance;
                    auto& optical_depth = ls_traversal.optical_depth;
                    auto& max_flight_distance = ls_traversal.max_flight_distance;

                    rk_df_type df_opt_backward = [max_flight_distance](MediumInteraction3f& mei, Ray3f ray, const Spectrum&, const Float& dt_in, const Mask& active_dt) {
                        dr::masked(mei.p, active_dt) = ray(max_flight_distance - (mei.t + dt_in));
                        auto [sigma_s, sigma_n, sigma_t] = mei.medium->get_scattering_coefficients(mei, active_dt);
                        return sigma_t;
                    };

                    rk_df_type df_opt_forward = [](MediumInteraction3f& mei, Ray3f ray, const Spectrum&, const Float& dt_in, const Mask& active_dt) {
                        dr::masked(mei.p, active_dt) = ray(mei.t + dt_in);
                        auto [sigma_s, sigma_n, sigma_t] = mei.medium->get_scattering_coefficients(mei, active_dt);
                        return sigma_t;
                    };

                    rk_df_type df_rad = [max_flight_distance](MediumInteraction3f& mei, Ray3f ray, const Spectrum& y, const Float& dt_in, const Mask& active_dt) {
                        dr::masked(mei.p, active_dt) = ray(max_flight_distance - (mei.t + dt_in));
                        auto [sigma_s, sigma_n, sigma_t] = mei.medium->get_scattering_coefficients(mei, active_dt);
                        auto radiance = mei.medium->get_radiance(mei, active_dt);
                        return -sigma_t*y + radiance;
                    };

                    auto [next_depth, curr_dt_opt, next_dt_opt] = integration_step_rk(df_opt_backward, ray, mei, optical_depth, dt, iteration_mask && !medium->is_homogeneous(), false, true);
                    auto [next_radiance, curr_dt, next_dt] = integration_step_rk(df_rad, ray, mei, full_path_radiance, curr_dt_opt, iteration_mask, false, false);

                    if (Mask revise_optical_depth = iteration_mask && !medium->is_homogeneous() && dr::abs(curr_dt) < dr::abs(curr_dt_opt); dr::any_or<true>(revise_optical_depth)) {
                        auto [corr_depth, curr_dt_corr, next_dt_corr] = integration_step_rk(df_opt_backward, ray, mei, optical_depth, curr_dt, revise_optical_depth, true, true);
                        dr::masked(next_depth, revise_optical_depth) = corr_depth;
                    }

                    // Update accumulators and ray position
                    dr::masked(full_path_radiance, iteration_mask) += next_radiance;
                    dr::masked(optical_depth, iteration_mask && !medium->is_homogeneous()) += next_depth;

                    dr::masked(mei.t, iteration_mask) += curr_dt;
                    dr::masked(mei.p, iteration_mask) = ray(mei.t);

                    // Update step size for next iteration
                    dr::masked(dt, iteration_mask) = dr::select(max_flight_distance - mei.t < next_dt, max_flight_distance - mei.t, next_dt);

                    // Update iteration mask
                    // Marching should end when we exceed distance to the next surface
                    iteration_mask &= mei.t < max_flight_distance;
                }, "VolpathMarch[medium traversal]");

                dr::masked(optical_depth, active_medium) = ls_traversal.optical_depth;
                dr::masked(mei, active_medium) = ls_traversal.mei;
                dr::masked(mei.t, active_medium) = ls_traversal.max_flight_distance;
                // Low accuracy (i.e. the RK tolerance is large) estimates may lead to negative values even though emission is strictly positive
                full_path_radiance = dr::clip(ls_traversal.full_path_radiance, 0.0f, dr::Infinity<Spectrum>);
                dr::masked(full_path_radiance, dr::isnan(full_path_radiance)) = 0.0f;

                auto tr = dr::exp(-optical_depth);
                auto pdf_normalisation = index_spectrum(1.0f - unpolarized_spectrum(tr), channel);

                Mask sample_scattering = pdf_normalisation > 5.0f*dr::Epsilon<Float>;
                iteration_mask = active_medium && sample_scattering;
                Float desired_depth = -dr::log(1.0f - sampler->next_1d(iteration_mask) * pdf_normalisation);
                dr::masked(mei.t, iteration_mask) = mei.mint;
                dr::masked(optical_depth, iteration_mask) = 0.0f;

                dr::masked(mei.t, iteration_mask && medium->is_homogeneous()) = desired_depth / index_spectrum(mei.sigma_t, channel);
                dr::masked(optical_depth, iteration_mask && medium->is_homogeneous()) = desired_depth;
                iteration_mask &= !medium->is_homogeneous();

                dr::masked(dt, iteration_mask) = initial_dt;
                if (m_jitter_steps) {
                    dr::masked(dt, iteration_mask) *= sampler->next_1d(iteration_mask);
                }
                dr::masked(dt, iteration_mask && dt > max_flight_distance - mei.t) = max_flight_distance - mei.t;

                dr::masked(mei.p, active_medium) = ray(mei.t);
                Mask reached_depth = false;

                struct LoopStateSample {
                    Mask iteration_mask;
                    Mask reached_depth;
                    Ray3f ray;
                    Float dt;
                    MediumInteraction3f mei;
                    Spectrum optical_depth;
                    Float desired_depth;
                    Float max_flight_distance;
                    Mask active_medium;
                    UInt32 depth;

                    DRJIT_STRUCT(LoopStateSample, iteration_mask, reached_depth, \
                            ray, dt, mei, optical_depth, \
                            desired_depth, max_flight_distance, active_medium, depth)
                } ls_point_sample = {
                    iteration_mask = iteration_mask,
                    reached_depth = reached_depth,
                    ray = ray,
                    dt = dt,
                    mei = mei,
                    optical_depth = optical_depth,
                    desired_depth = desired_depth,
                    max_flight_distance = max_flight_distance,
                    active_medium = active_medium,
                    depth = depth
                };
                dr::tie(ls_point_sample) = dr::while_loop(dr::make_tuple(ls_point_sample),
                        [](const LoopStateSample& ls_point_sample) { return dr::detach(ls_point_sample.iteration_mask); },
                        [this, scene, channel](LoopStateSample& ls_point_sample) {
                    auto& iteration_mask = ls_point_sample.iteration_mask;
                    auto& reached_depth = ls_point_sample.reached_depth;
                    auto& ray = ls_point_sample.ray;
                    auto& dt = ls_point_sample.dt;
                    auto& mei = ls_point_sample.mei;
                    auto& optical_depth = ls_point_sample.optical_depth;
                    auto& desired_depth = ls_point_sample.desired_depth;
                    auto& max_flight_distance = ls_point_sample.max_flight_distance;

                    rk_df_type df_opt_backward = [max_flight_distance](MediumInteraction3f& mei, Ray3f ray, const Spectrum&, const Float& dt_in, const Mask& active_dt) {
                        dr::masked(mei.p, active_dt) = ray(max_flight_distance - (mei.t + dt_in));
                        auto [sigma_s, sigma_n, sigma_t] = mei.medium->get_scattering_coefficients(mei, active_dt);
                        return sigma_t;
                    };

                    rk_df_type df_opt_forward = [](MediumInteraction3f& mei, Ray3f ray, const Spectrum&, const Float& dt_in, const Mask& active_dt) {
                        dr::masked(mei.p, active_dt) = ray(mei.t + dt_in);
                        auto [sigma_s, sigma_n, sigma_t] = mei.medium->get_scattering_coefficients(mei, active_dt);
                        return sigma_t;
                    };

                    auto [next_depth, curr_dt, next_dt] = integration_step_rk(df_opt_forward, ray, mei, optical_depth, dt, iteration_mask, false, true);
                    if constexpr (!dr::is_jit_v<Float>)
                        Log(Trace, "Depth: %s ~ d|Depth|: %s ~ dt: %s ~ ndt: %s", optical_depth, next_depth, curr_dt, next_dt);

                    // --------------------- Correction ---------------------- //
                    // Check if we exceed the desired density by taking this step
                    // If we do, find the point between this point and the last point that reaches the desired density
                    // The desired density and the obtained density may differ since we sample the density at the point we interpolate to
                    // And if the step size is larger than 2x the mean grid spacing then this can be quite different
                    Float residual_depth = desired_depth - index_spectrum(unpolarized_spectrum(optical_depth + next_depth), channel);
                    reached_depth |= residual_depth == 0.0f;

                    if (Mask correct_depth = iteration_mask && (residual_depth < 0.0f);dr::any_or<true>(correct_depth)) {
                        Mask not_found_depth = correct_depth;
                        reached_depth |= correct_depth;
                        UInt32 niter = 0;
                        // Bisection method to find the point at which the optical thickness matches the target
                        auto a_depth = next_depth;
                        auto a_dt = curr_dt;
                        auto a_ndt = next_dt;
                        auto b_depth = dr::zeros<Spectrum>();
                        auto b_dt = dr::zeros<Float>();
                        auto b_ndt = next_dt;

                        struct LoopStateBisect {
                            Spectrum a_depth;
                            Float a_dt;
                            Float a_ndt;
                            Spectrum b_depth;
                            Float b_dt;
                            Float b_ndt;
                            UInt32 niter;
                            Ray3f ray;
                            Mask not_found_depth;
                            MediumInteraction3f mei;
                            Spectrum optical_depth;
                            Float desired_depth;

                            DRJIT_STRUCT(LoopStateBisect, a_depth, a_dt, a_ndt, b_depth, b_dt, b_ndt, \
                                                          niter, ray, not_found_depth, \
                                                          mei, optical_depth, desired_depth)
                        } ls_bisect = {
                            a_depth = a_depth,
                            a_dt = a_dt,
                            a_ndt = a_ndt,
                            b_depth = b_depth,
                            b_dt = b_dt,
                            b_ndt = b_ndt,
                            niter = niter,
                            ray = ray,
                            not_found_depth = not_found_depth,
                            mei = mei,
                            optical_depth = optical_depth,
                            desired_depth = desired_depth,
                        };
                        dr::tie(ls_bisect) = dr::while_loop(dr::make_tuple(ls_bisect),
                                [](const LoopStateBisect& ls_bisect) { return dr::detach(ls_bisect.not_found_depth); },
                                [this, scene, channel, &df_opt_forward](LoopStateBisect& ls_bisect) {
                            auto& a_depth = ls_bisect.a_depth;
                            auto& a_dt = ls_bisect.a_dt;
                            auto& a_ndt = ls_bisect.a_ndt;
                            auto& b_depth = ls_bisect.b_depth;
                            auto& b_dt = ls_bisect.b_dt;
                            auto& b_ndt = ls_bisect.b_ndt;
                            auto& niter = ls_bisect.niter;
                            auto& ray = ls_bisect.ray;
                            auto& not_found_depth = ls_bisect.not_found_depth;
                            auto& mei = ls_bisect.mei;
                            auto& optical_depth = ls_bisect.optical_depth;
                            auto& desired_depth = ls_bisect.desired_depth;

                            // Float interval_min = index_spectrum(unpolarized_spectrum(b_depth), channel);
                            // Float interval_range = index_spectrum(unpolarized_spectrum(a_depth - b_depth), channel);
                            // Float half_step = (dr::detach(a_dt) - dr::detach(b_dt)) * (residual_depth - interval_min) / interval_range + dr::detach(b_dt);
                            Float half_step = (dr::detach(a_dt) + dr::detach(b_dt)) * 0.5f;
                            auto [m_depth, m_dt, m_ndt] = integration_step_rk(df_opt_forward, ray, mei, optical_depth, half_step, not_found_depth, true, true);
                            Mask upper_half = desired_depth - index_spectrum(unpolarized_spectrum(optical_depth + m_depth), channel) <= 0.0f;

                            dr::masked(a_depth, upper_half) = m_depth;
                            dr::masked(a_dt, upper_half) = m_dt;
                            dr::masked(a_ndt, upper_half) = m_ndt;

                            dr::masked(b_depth, !upper_half) = m_depth;
                            dr::masked(b_dt, !upper_half) = m_dt;
                            dr::masked(b_ndt, !upper_half) = m_ndt;

                            ++niter;
                            not_found_depth &= index_spectrum(dr::abs(unpolarized_spectrum(a_depth - b_depth)), channel) >= dr::select(index_spectrum(mei.sigma_t, channel) > 0.0f, 5000.0f, 5.0f) * dr::Epsilon<Float> * (1.0f + desired_depth) && niter < 64;
                        }, "VolpathMarch[optical depth bisection]");

                        auto mid_depth = (ls_bisect.a_depth + ls_bisect.b_depth) * 0.5f;
                        auto mid_dt = (ls_bisect.a_dt + ls_bisect.b_dt) * 0.5f;
                        auto mid_ndt = (ls_bisect.a_ndt + ls_bisect.b_ndt) * 0.5f;

                        // Four iterations of Newton's method to polish the root found in the earlier bisection
                        // This should be quadratic convergence so by trading off some precision in the bisection we gain
                        // a big improvement here
                        for (auto iter_count = 0; iter_count < 4; iter_count++ ) {
                            dr::masked(residual_depth, correct_depth) = desired_depth - index_spectrum(unpolarized_spectrum(optical_depth + mid_depth), channel);
                            mid_dt = dr::clip(mid_dt - residual_depth/index_spectrum(-unpolarized_spectrum(df_opt_forward(mei, ray, optical_depth, mid_dt, correct_depth)), channel), b_dt, a_dt);
                            auto [n_depth, n_dt, n_ndt] = integration_step_rk(df_opt_forward, ray, mei, optical_depth, mid_dt, correct_depth, true, true);
                            auto is_improvement = !(dr::isnan(index_spectrum(unpolarized_spectrum(n_depth), channel)) || dr::isinf(index_spectrum(unpolarized_spectrum(n_depth), channel))) && dr::abs(desired_depth - index_spectrum(unpolarized_spectrum(optical_depth + n_depth), channel)) < dr::abs(residual_depth);
                            dr::masked(mid_depth, correct_depth && is_improvement) = n_depth;
                            dr::masked(mid_dt, correct_depth && is_improvement)    = n_dt;
                            dr::masked(mid_ndt, correct_depth && is_improvement)   = n_ndt;
                        }
                        // ---- //
                        dr::masked(residual_depth, correct_depth) = desired_depth - index_spectrum(unpolarized_spectrum(optical_depth + mid_depth), channel);
                        if constexpr (!dr::is_jit_v<Float>)
                            Log(Debug, "Iter: %s ~ Residual depth: %s ~ |tol|: %s ~ rel(tol) %s", niter, residual_depth, 5.0f * dr::Epsilon<Float>, desired_depth * 5.0f * dr::Epsilon<Float>);
                        dr::masked(next_depth, correct_depth) = mid_depth;
                        dr::masked(curr_dt, correct_depth) = mid_dt;
                        dr::masked(next_dt, correct_depth) = mid_ndt;
                    }

                    // ------------------------------------------------------- //

                    // Update accumulators and ray position
                    dr::masked(optical_depth, iteration_mask) += next_depth;
                    dr::masked(mei.t, iteration_mask) += curr_dt;
                    dr::masked(mei.p, iteration_mask) = ray(mei.t);

                    // Update step size for next iteration
                    dr::masked(dt, iteration_mask) = dr::select(max_flight_distance - mei.t > next_dt, max_flight_distance - mei.t, next_dt);

                    // Update iteration mask
                    // Marching should end when either throughput falls below sampled throughput or we exceed distance to the next surface
                    iteration_mask &= mei.t < max_flight_distance && !reached_depth;
                }, "VolpathMarch[scatter point sample]");

                dr::masked(optical_depth, active_medium) = ls_point_sample.optical_depth;
                dr::masked(mei, active_medium) = ls_point_sample.mei;
                dr::masked(optical_depth, active_medium && !ls_point_sample.reached_depth) = -dr::log(tr);
                std::tie(mei.sigma_s, mei.sigma_n, mei.sigma_t) = medium->get_scattering_coefficients(mei, active_medium);

                auto radiance = medium->get_radiance(mei, active_medium);
                auto [probabilities, weights] = medium->get_interaction_probabilities(radiance, mei, throughput);

                auto [prob_scatter, prob_null]     = probabilities;
                auto [weight_scatter, weight_null] = weights;

                // Default scatter probabilities only account for the balance between null density+radiance and optical density
                // But raymarching here requires that rays escape the medium occasionally so we halve the probability of scattering
                prob_scatter *= 0.5f;

                sample_scattering &= sampler->next_1d(active_medium) <= dr::detach(index_spectrum(prob_scatter, channel));
                if constexpr (!dr::is_jit_v<Float>)
                    Log(Debug, "valid: %s ~ P(s): %s ~ Sampled Scattering: %s | Optical Depth: %s ~ Flight Distance: %s ~ sigma_t: %s ~ sigma_s: %s", mei.is_valid(), prob_scatter, sample_scattering, -dr::log(tr), mei.t, mei.sigma_t, mei.sigma_s);

                dr::masked(mei.t, active_medium && !sample_scattering) = max_flight_distance;
                dr::masked(mei.p, active_medium) = ray(mei.t);
                dr::masked(tr, active_medium && sample_scattering) = dr::exp(-optical_depth);
                Spectrum path_pdf = dr::select(sample_scattering, prob_scatter * mei.sigma_t * tr / pdf_normalisation, 1.0f - prob_scatter);
                Float tr_pdf = index_spectrum(unpolarized_spectrum(path_pdf), channel);

                // ---------------- Intersection with emitters ----------------
                Mask ray_from_camera_medium = active_medium && depth == 0u;
                Mask count_direct_medium = ray_from_camera_medium || specular_chain;
                EmitterPtr emitter_medium = mei.emitter(active_medium);
                Mask active_medium_e = active_medium && (emitter_medium != nullptr) &&
                                       !(depth == 0u && m_hide_emitters);
                if (dr::any_or<true>(active_medium_e)) {
                    Spectrum weight = 1.0f;
                    Mask nonzero_density = (dr::detach(dr::mean(unpolarized_spectrum(optical_depth))) != 0.0f);
                    Mask optically_thin_medium = active_medium_e && !((medium->is_homogeneous() && nonzero_density) || !medium->is_homogeneous());
                    if (!m_use_uni_sampling && m_use_emitter_sampling) {
                        dr::masked(weight, active_medium_e && !count_direct_medium) *= 0.0f;
                    } else if (m_use_uni_sampling && m_use_emitter_sampling) {
                        DirectionSample3f ds(mei, last_scatter_event);
                        Float emitter_pdf = scene->pdf_emitter_direction(last_scatter_event, ds, active_medium_e && !optically_thin_medium);
                        Float scatter_pdf = last_scatter_direction_pdf;
                        dr::masked(weight, active_medium_e && !optically_thin_medium && !count_direct_medium) *= mis_weight(scatter_pdf, emitter_pdf);
                    }
                    auto contrib = dr::select(sample_scattering, 0.0f, full_path_radiance);
                    dr::masked(result, active_medium_e) += weight * throughput * dr::select(tr_pdf > 0.0f, contrib / tr_pdf, 0.0f);
                }

                dr::masked(throughput, active_medium) *= dr::select(tr_pdf > 0.f, tr / tr_pdf, 0.f);

                dr::masked(ray.o, active_medium && !sample_scattering) = mei.p;
                dr::masked(si.t, active_medium && !sample_scattering) = si.t - mei.t;
                dr::masked(mei.t, active_medium && !sample_scattering) = dr::Infinity<Float>;

                escaped_medium = active_medium && !mei.is_valid();
                active_medium &= mei.is_valid();
                act_medium_scatter = mei.is_valid() && active_medium;

                dr::masked(depth, act_medium_scatter) += 1;
                dr::masked(last_scatter_event, act_medium_scatter) = mei;

                // Don't estimate lighting if we exceeded number of bounces
                active &= depth < static_cast<uint32_t>(m_max_depth);
                act_medium_scatter &= active;
            }

            if (dr::any_or<true>(active_medium)) {
                if (dr::any_or<true>(act_medium_scatter)) {
                    dr::masked(throughput, act_medium_scatter) *= mei.sigma_s;

                    PhaseFunctionContext phase_ctx(sampler);
                    auto phase = mei.medium->phase_function();

                    // --------------------- Emitter sampling ---------------------
                    if (m_use_emitter_sampling) {
                        Mask sample_emitters = mei.medium->use_emitter_sampling();
                        valid_ray |= act_medium_scatter;
                        specular_chain &= !act_medium_scatter;
                        specular_chain |= act_medium_scatter && !sample_emitters;

                        Mask active_e = act_medium_scatter && sample_emitters;
                        if (dr::any_or<true>(active_e)) {
                            auto [emitted, ds] = sample_emitter(mei, scene, sampler, medium, channel, active_e);
                            auto [phase_val, phase_pdf] = phase->eval_pdf(phase_ctx, mei, ds.d, active_e);
                            auto weight = (m_use_uni_sampling ? mis_weight(ds.pdf, dr::select(ds.delta, 0.0f, phase_pdf)) : 1.0f);
                            dr::masked(result, active_e) += weight * throughput * phase_val * emitted;
                        }
                    }

                    // ------------------ Phase function sampling -----------------
                    dr::masked(phase, !act_medium_scatter) = nullptr;
                    auto [wo, phase_weight, phase_pdf]     = phase->sample(
                        phase_ctx, mei, sampler->next_1d(act_medium_scatter),
                        sampler->next_2d(act_medium_scatter),
                        act_medium_scatter);
                    act_medium_scatter &= phase_pdf > 0.f;
                    Ray3f new_ray                       = mei.spawn_ray(wo);
                    dr::masked(ray, act_medium_scatter) = new_ray;
                    needs_intersection |= act_medium_scatter;
                    dr::masked(last_scatter_direction_pdf, act_medium_scatter) = phase_pdf;
                    dr::masked(throughput, act_medium_scatter) *= phase_weight;
                }
            }

            // --------------------- Surface Interactions ---------------------
            active_surface |= escaped_medium;
            Mask intersect = active_surface && needs_intersection;
            if (dr::any_or<true>(intersect))
                dr::masked(si, intersect) = scene->ray_intersect(ray, intersect);

            if (dr::any_or<true>(active_surface)) {
                // ---------------- Intersection with emitters ----------------
                Mask ray_from_camera = active_surface && (depth == 0u);
                Mask count_direct = ray_from_camera || specular_chain;
                EmitterPtr emitter = si.emitter(scene);
                Mask active_e = active_surface && emitter != nullptr
                                && !((depth == 0u) && m_hide_emitters); // Ignore any medium emitters as this simply looks at surface emitters
                if (dr::any_or<true>(active_e)) {
                    Spectrum weight = 1.0f;
                    // Get the PDF of sampling this emitter using next event estimation
                    if (!m_use_uni_sampling && m_use_emitter_sampling) {
                        dr::masked(weight, active_e && !count_direct) *= 0.0f;
                    } else if (m_use_uni_sampling && m_use_emitter_sampling) {
                        DirectionSample3f ds(scene, si, last_scatter_event);
                        Float emitter_pdf = scene->pdf_emitter_direction(last_scatter_event, ds, active_e);
                        Float scatter_pdf = last_scatter_direction_pdf;
                        dr::masked(weight, active_e && !count_direct) *= mis_weight(scatter_pdf, emitter_pdf);
                    }
                    dr::masked(result, active_e) += weight * throughput * emitter->eval(si, active_e);
                }
            }
            active_surface &= si.is_valid();
            if (dr::any_or<true>(active_surface)) {
                BSDFContext ctx;
                BSDFPtr bsdf  = si.bsdf(ray);

                // --------------------- Emitter sampling ---------------------
                if (m_use_emitter_sampling) {
                    Mask active_e = active_surface && has_flag(bsdf->flags(), BSDFFlags::Smooth) && (depth + 1 < (uint32_t) m_max_depth);

                    if (likely(dr::any_or<true>(active_e))) {
                        auto [emitted, ds] = sample_emitter(si, scene, sampler, medium, channel, active_e);

                        // Query the BSDF for that emitter-sampled direction
                        Vector3f wo       = si.to_local(ds.d);
                        Spectrum bsdf_val = bsdf->eval(ctx, si, wo, active_e);
                        bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

                        // Determine probability of having sampled that same
                        // direction using BSDF sampling.
                        Float bsdf_pdf = bsdf->pdf(ctx, si, wo, active_e);
                        auto weight = (m_use_uni_sampling ? mis_weight(ds.pdf, dr::select(ds.delta, 0.0f, bsdf_pdf)) : 1.0f);
                        result[active_e] += weight * throughput * bsdf_val * emitted;
                    }
                }

                // ----------------------- BSDF sampling ----------------------
                auto [bs, bsdf_val] = bsdf->sample(ctx, si, sampler->next_1d(active_surface),
                                                   sampler->next_2d(active_surface), active_surface);
                bsdf_val = si.to_world_mueller(bsdf_val, -bs.wo, si.wi);

                dr::masked(throughput, active_surface) *= bsdf_val;
                dr::masked(eta, active_surface) *= bs.eta;

                Ray3f bsdf_ray                  = si.spawn_ray(si.to_world(bs.wo));
                dr::masked(ray, active_surface) = bsdf_ray;
                needs_intersection |= active_surface;

                Mask non_null_bsdf = active_surface && !has_flag(bs.sampled_type, BSDFFlags::Null);
                dr::masked(depth, non_null_bsdf) += 1;

                // update the last scatter PDF event if we encountered a non-null scatter event
                dr::masked(last_scatter_event, non_null_bsdf) = si;
                dr::masked(last_scatter_direction_pdf, non_null_bsdf) = bs.pdf;

                valid_ray |= non_null_bsdf;
                specular_chain |= non_null_bsdf && has_flag(bs.sampled_type, BSDFFlags::Delta);
                specular_chain &= !(active_surface && has_flag(bs.sampled_type, BSDFFlags::Smooth));
                Mask has_medium_trans                = active_surface && si.is_medium_transition();
                dr::masked(medium, has_medium_trans) = si.target_medium(ray.d);
            }
            active &= (active_surface | active_medium);
        }, "VolpathRaymarching integrator");

        return { ls.result, ls.valid_ray };
    }


    /// Samples an emitter in the scene and evaluates its attenuated contribution
    template <typename Interaction>
    std::tuple<Spectrum, DirectionSample3f>
    sample_emitter(const Interaction &ref_interaction, const Scene *scene,
                   Sampler *sampler, MediumPtr medium,
                   UInt32 channel, Mask active) const {
        Spectrum transmittance(1.0f);

        /// We conservatively assume that there are volume emitters in the scene and sample 3d points instead of 2d
        /// This leads to some inefficiencies due to the fact that an extra random number per is generated and unused.
        auto [ds, emitter_val] = scene->sample_emitter_direction(ref_interaction, sampler->next_3d(active), false, active);
        dr::masked(emitter_val, ds.pdf == 0.f) = 0.f;
        active &= (ds.pdf != 0.f);

        Mask is_medium_emitter = active && has_flag(ds.emitter->flags(), EmitterFlags::Medium);
        dr::masked(emitter_val, is_medium_emitter) = 0.0f;

        if (dr::none_or<false>(active)) {
            return { emitter_val, ds };
        }

        Ray3f ray = ref_interaction.spawn_ray_to(ds.p);
        Float max_dist = ray.maxt;

        // Potentially escaping the medium if this is the current medium's boundary
        if constexpr (std::is_convertible_v<Interaction, SurfaceInteraction3f>)
            dr::masked(medium, ref_interaction.is_medium_transition()) = ref_interaction.target_medium(ray.d);

        Float total_dist = 0.f;
        auto si = dr::zeros<SurfaceInteraction3f>();
        auto mei = dr::zeros<MediumInteraction3f>();
        Mask needs_intersection = true;

        struct LoopState {
            Mask active;
            Ray3f ray;
            Float total_dist;
            Spectrum emitter_val;
            Mask needs_intersection;
            Mask is_medium_emitter;
            MediumPtr medium;
            SurfaceInteraction3f si;
            MediumInteraction3f mei;
            Spectrum transmittance;
            DirectionSample3f dir_sample;
            Sampler* sampler;

            DRJIT_STRUCT(LoopState, active, ray, total_dist, emitter_val, \
                needs_intersection, is_medium_emitter, medium, si, mei, \
                transmittance, dir_sample, sampler)
        } ls = {
            active,
            ray,
            total_dist,
            emitter_val,
            needs_intersection,
            is_medium_emitter,
            medium,
            si,
            mei,
            transmittance,
            ds,
            sampler
        };

        dr::tie(ls) = dr::while_loop(dr::make_tuple(ls),
            [](const LoopState& ls) { return dr::detach(ls.active); },
            [this, scene, channel, max_dist, is_medium_emitter](LoopState& ls) {

            Mask& active = ls.active;
            Ray3f& ray = ls.ray;
            Float& total_dist = ls.total_dist;
            Spectrum& emitter_val = ls.emitter_val;
            Mask& needs_intersection = ls.needs_intersection;
            MediumPtr& medium = ls.medium;
            SurfaceInteraction3f& si = ls.si;
            MediumInteraction3f& mei = ls.mei;
            Spectrum& transmittance = ls.transmittance;
            DirectionSample3f& dir_sample = ls.dir_sample;
            Sampler* sampler = ls.sampler;

            Float remaining_dist = max_dist - total_dist;
            ray.maxt = remaining_dist;
            active &= remaining_dist > 0.f;
            if (dr::none_or<false>(active))
                return;

            Mask escaped_medium = false;
            Mask active_medium  = active && (medium != nullptr);
            Mask active_surface = active && !active_medium;

            if (dr::any_or<true>(active_medium)) {
                dr::masked(mei, active_medium) = medium->sample_interaction(ray, sampler->next_1d(active_medium), channel, active_medium);
                dr::masked(ray.maxt, active_medium && medium->is_homogeneous() && mei.is_valid()) = dr::minimum(mei.t, remaining_dist);
                Mask intersect = needs_intersection && active_medium;
                if (dr::any_or<true>(intersect))
                    dr::masked(si, intersect) = scene->ray_intersect(ray, intersect);

                dr::masked(mei.t, active_medium && (si.t < mei.t)) = dr::Infinity<Float>;
                needs_intersection &= !active_medium;

                EmitterPtr medium_em = mei.emitter(active_medium);
                Mask is_sampled_medium = active_medium && (medium_em == dir_sample.emitter) && is_medium_emitter;

                Mask is_spectral = active_medium && medium->has_spectral_extinction();
                Mask not_spectral = !is_spectral && active_medium;
                if (dr::any_or<true>(is_spectral)) {
                    Float t      = dr::minimum(remaining_dist, dr::minimum(mei.t, si.t)) - mei.mint;
                    UnpolarizedSpectrum tr  = dr::exp(-t * mei.combined_extinction);
                    UnpolarizedSpectrum free_flight_pdf = dr::select(si.t < mei.t || mei.t > remaining_dist, tr, tr * mei.combined_extinction);
                    Float tr_pdf = index_spectrum(free_flight_pdf, channel);
                    dr::masked(transmittance, is_spectral) *= dr::select(tr_pdf > 0.f, tr / tr_pdf, 0.f);
                }

                // Handle exceeding the maximum distance by medium sampling
                dr::masked(total_dist, active_medium && (mei.t > remaining_dist) && mei.is_valid()) = dir_sample.dist;
                dr::masked(mei.t, active_medium && (mei.t > remaining_dist)) = dr::Infinity<Float>;

                escaped_medium = active_medium && !mei.is_valid();
                active_medium &= mei.is_valid();

                is_sampled_medium &= active_medium;
                if (dr::any_or<true>(is_sampled_medium)) {
                    PositionSample3f ps(mei);
                    auto radiance = dr::select(is_sampled_medium, mei.radiance, 0.0);
                    dr::masked(emitter_val, is_sampled_medium) += transmittance * radiance * dr::rcp(dir_sample.pdf);
                }

                dr::masked(mei, escaped_medium) = dr::zeros<MediumInteraction3f>();
                dr::masked(total_dist, active_medium) += mei.t;

                is_spectral  &= active_medium;
                not_spectral &= active_medium;

                if (dr::any_or<true>(active_medium)) {
                    dr::masked(ray.o, active_medium) = mei.p;
                    dr::masked(si.t, active_medium)  = si.t - mei.t;

                    if (dr::any_or<true>(is_spectral))
                        dr::masked(transmittance, is_spectral) *= mei.sigma_n;
                    if (dr::any_or<true>(not_spectral))
                        dr::masked(transmittance, not_spectral) *= mei.sigma_n / mei.combined_extinction;
                }
            }

            // Handle interactions with surfaces
            Mask intersect = active_surface && needs_intersection;
            if (dr::any_or<true>(intersect))
                dr::masked(si, intersect) = scene->ray_intersect(ray, intersect);
            needs_intersection &= !intersect;
            active_surface |= escaped_medium;
            dr::masked(total_dist, active_surface) += si.t;

            active_surface &= si.is_valid() && active && !active_medium;
            if (dr::any_or<true>(active_surface)) {
                auto bsdf         = si.bsdf(ray);
                Spectrum bsdf_val = bsdf->eval_null_transmission(si, active_surface);
                bsdf_val = si.to_world_mueller(bsdf_val, si.wi, si.wi);
                dr::masked(transmittance, active_surface) *= bsdf_val;
            }

            // Update the ray with new origin & t parameter
            dr::masked(ray, active_surface) = si.spawn_ray_to(dir_sample.p);
            ray.maxt = remaining_dist;
            needs_intersection |= active_surface;

            // Continue tracing through scene if non-zero weights exist
            active &= (active_medium || active_surface) &&
                      dr::any(unpolarized_spectrum(transmittance) != 0.f);

            // If a medium transition is taking place: Update the medium pointer
            Mask has_medium_trans = active_surface && si.is_medium_transition();
            if (dr::any_or<true>(has_medium_trans)) {
                dr::masked(medium, has_medium_trans) = si.target_medium(ray.d);
            }
        }, "VolpathMarch [emitter sampling]");

        return {dr::select(is_medium_emitter, ls.emitter_val, ls.emitter_val * ls.transmittance), ls.dir_sample};
    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("VolumetricRaymarchingIntegrator[\n"
                           "  max_depth = %i,\n"
                           "  rr_depth = %i\n"
                           "]",
                           m_max_depth, m_rr_depth);
    }

    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        Float w = pdf_a / (pdf_a + pdf_b);
        return dr::select(dr::isfinite(w), w, 0.f);
    };

    MI_DECLARE_CLASS()
protected:
    bool m_use_emitter_sampling, m_use_uni_sampling, m_jitter_steps;
    ScalarFloat m_absolute_tolerance, m_relative_tolerance;

    // From Verner, doi: 10.1007/s11075-009-9290-3
    // https://www.sfu.ca/~jverner/RKV65.IIIXb.Efficient.00000144617.081204.RATOnWeb
    // static constexpr std::size_t rk_stages = 9;
    // const std::array<std::array<ScalarFloat64, rk_stages>, rk_stages> rk_aij = {
    //     std::array<ScalarFloat64, rk_stages>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    //     std::array<ScalarFloat64, rk_stages>{9.0/50, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    //     std::array<ScalarFloat64, rk_stages>{29.0/324, 25.0/324, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    //     std::array<ScalarFloat64, rk_stages>{1.0/16, 0, 3.0/16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    //     std::array<ScalarFloat64, rk_stages>{79129.0/250000, 0, -261237.0/250000, 19663.0/15625, 0.0, 0.0, 0.0, 0.0, 0.0},
    //     std::array<ScalarFloat64, rk_stages>{ 1336883.0/4909125,  0, -25476.0/30875,  194159.0/185250,  8225.0/78546, 0.0, 0.0, 0.0, 0.0},
    //     std::array<ScalarFloat64, rk_stages>{-2459386.0/14727375,  0,  19504.0/30875,  2377474.0/13615875, -6157250.0/5773131,  902.0/735, 0.0, 0.0, 0.0},
    //     std::array<ScalarFloat64, rk_stages>{ 2699.0/7410,  0, -252.0/1235, -1393253.0/3993990,  236875.0/72618, -135.0/49,  15.0/22, 0.0, 0.0},
    //     std::array<ScalarFloat64, rk_stages>{11.0/144, 0, 0, 256.0/693, 0, 125.0/504, 125.0/528, 5.0/72, 0.0}
    // };
    //
    // const std::array<ScalarFloat64, rk_stages> rk_ci = {0, 9.0/50.0, 1.0/6.0, 1.0/4.0, 53.0/100.0, 3.0/5.0, 4.0/5.0, 1.0, 1.0 },
    //                              rk_bi = { 11.0/144, 0, 0, 256.0/693, 0, 125.0/504, 125.0/528, 5.0/72, 0 },
    //                              rk_ei = { 365.0/1908.0, 0.0, 0.0, -60.0/539.0, 312500.0/366177.0, -375.0/392.0, 125.0/528.0, 9145.0/71064.0, -2995.0/17766.0 };
    //
    // const ScalarFloat rk_order = 6.0;

    // https://www.sfu.ca/~jverner/RKV98.IIa.Robust.000000351.081209.CoeffsOnlyFLOAT6040
    static constexpr std::size_t rk_stages = 17;
    const std::array<std::array<ScalarFloat64, rk_stages>, rk_stages> rk_aij = {
        std::array<ScalarFloat64, rk_stages>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        std::array<ScalarFloat64, rk_stages>{0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        std::array<ScalarFloat64, rk_stages>{-0.01988527319182291, 0.11637263332969652, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        std::array<ScalarFloat64, rk_stages>{0.0361827600517026, 0.0, 0.10854828015510781, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        std::array<ScalarFloat64, rk_stages>{2.2721142642901775, 0.0, -8.526886447976398, 6.830772183686221, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        std::array<ScalarFloat64, rk_stages>{0.050943855353893744, 0.0, 0.0, 0.1755865049809071, 0.0007022961270757468, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        std::array<ScalarFloat64, rk_stages>{0.1424783668683285, 0.0, 0.0, -0.35417994346686843, 0.07595315450295101, 0.6765157656337123, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        std::array<ScalarFloat64, rk_stages>{0.07111111111111111, 0.0, 0.0, 0.0, 0.0, 0.32799092876058983, 0.24089796012829906, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        std::array<ScalarFloat64, rk_stages>{0.07125, 0.0, 0.0, 0.0, 0.0, 0.32688424515752457, 0.11561575484247544, -0.03375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        std::array<ScalarFloat64, rk_stages>{0.048226773224658105, 0.0, 0.0, 0.0, 0.0, 0.039485599804954, 0.10588511619346581, -0.021520063204743093, -0.10453742601833482, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        std::array<ScalarFloat64, rk_stages>{-0.026091134357549235, 0.0, 0.0, 0.0, 0.0, 0.03333333333333333, -0.1652504006638105, 0.03434664118368617, 0.1595758283215209, 0.21408573218281934, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        std::array<ScalarFloat64, rk_stages>{-0.03628423396255659, 0.0, 0.0, 0.0, 0.0, -1.0961675974272087, 0.1826035504321331, 0.07082254444170684, -0.02313647018482431, 0.27112047263209327, 1.3081337494229808, 0.0, 0.0, 0.0, 0.0, 0.0},
        std::array<ScalarFloat64, rk_stages>{-0.5074635056416975, 0.0, 0.0, 0.0, 0.0, -6.631342198657237, -0.2527480100908801, -0.49526123800360955, 0.2932525545253887, 1.440108693768281, 6.237934498647056, 0.7270192054526987, 0.0, 0.0, 0.0, 0.0},
        std::array<ScalarFloat64, rk_stages>{0.6130118256955932, 0.0, 0.0, 0.0, 0.0, 9.088803891640463, -0.40737881562934486, 1.7907333894903747, 0.714927166761755, -1.438580857841723, -8.26332931206474, -1.5375705708088652, 0.34538328275648716, 0.0, 0.0, 0.0},
        std::array<ScalarFloat64, rk_stages>{-1.2116979103438739, 0.0, 0.0, 0.0, 0.0, -19.055818715595954, 1.2630606753898752, -6.913916969178458, -0.676462266509498, 3.367860445026608, 18.00675164312591, 6.83882892679428, -1.0315164519219504, 0.41291062321306227, 0.0, 0.0},
        std::array<ScalarFloat64, rk_stages>{2.1573890074940536, 0.0, 0.0, 0.0, 0.0, 23.807122198095804, 0.8862779249216556, 13.139130397598764, -2.6044157092877147, -5.193859949783873, -20.412340711541507, -12.300856252505723, 1.5215530950085394, 0.0, 0.0, 0.0}
    };

    const std::array<ScalarFloat64, rk_stages> rk_ci = { 0.0, 0.04, 0.09648736013787361, 0.1447310402068104, 0.576, 0.2272326564618766, 0.5407673435381234, 0.64, 0.48, 0.06754, 0.25, 0.6770920153543243, 0.8115, 0.906, 1.0, 1.0},
                                 rk_bi = { 0.014588852784055396, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0020241978878893325, 0.21780470845697167, 0.12748953408543898, 0.2244617745463132, 0.1787254491259903, 0.07594344758096558, 0.12948458791975614, 0.029477447612619417, 0.0 },
                                 rk_ei = { -0.0057578137681889505, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0675934530948106, 0.14099636134393978, 0.014411715396914937, -0.030796961251883054, 1.1613152578179067, -0.3222111348611858, 0.12948458791975614, 0.029477447612619417, -0.04932600711506839 };

    const ScalarFloat rk_order = 9.0;
    // ---- //

    // RK 3(2) pair (Bogacki-Shampine)
    // static constexpr std::size_t rk_stages = 4;
    // const std::array<std::array<ScalarFloat64, rk_stages>, rk_stages> rk_aij = {
    //     std::array<ScalarFloat64, rk_stages>{0.0, 0.0, 0.0, 0.0},
    //     std::array<ScalarFloat64, rk_stages>{1.0, 0.0, 0.0, 0.0},
    //     std::array<ScalarFloat64, rk_stages>{0.0, 0.75, 0.0, 0.0},
    //     std::array<ScalarFloat64, rk_stages>{2.0/9.0, 1.0/3.0, 4.0/9.0, 0.0},
    // };
    //
    // const std::array<ScalarFloat64, rk_stages> rk_ci = {0, 0.5, 0.75, 1.0},
    //                              rk_bi = { 2.0/9.0, 1.0/3.0, 4.0/9.0, 0.0 },
    //                              rk_ei = { -0.06944444444444448, 0.08333333333333331, 0.1111111111111111, -0.125 };
    //
    // const ScalarFloat rk_order = 3.0;

    // Runge-Kutta 4(5) Cash-Karp Coefficients from wikipedia
    // static constexpr std::size_t rk_stages = 7;
    // const std::array<std::array<ScalarFloat64, rk_stages>, rk_stages> rk_aij = {
    //     std::array<ScalarFloat64, rk_stages>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    //     std::array<ScalarFloat64, rk_stages>{0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    //     std::array<ScalarFloat64, rk_stages>{3.0/40.0, 9.0/40.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    //     std::array<ScalarFloat64, rk_stages>{44.0/45.0, -56.0/15.0, 32.0/9.0, 0.0, 0.0, 0.0, 0.0},
    //     std::array<ScalarFloat64, rk_stages>{19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0, 0.0, 0.0, 0.0},
    //     std::array<ScalarFloat64, rk_stages>{9017.0/3168.0, -355.0/33.0, 46372.0/5247.0, 49.0/176.0, -5103.0/18656.0, 0.0, 0.0},
    //     std::array<ScalarFloat64, rk_stages>{35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0},
    // };
    //
    // const std::array<ScalarFloat64, rk_stages> rk_ci = {0, 0.2, 0.3, 0.8, 8.0/9.0, 1.0, 1.0},
    //                              rk_bi = { 35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0 },
    //                              rk_ei = { 0.0012326388888888873, 0.0, -0.004252770290506136, 0.036979166666666674, -0.05086379716981132, 0.04190476190476192, -0.025 };
    //
    // const ScalarFloat rk_order = 5.0;
};

MI_IMPLEMENT_CLASS_VARIANT(VolumetricMarchedPathIntegrator, MonteCarloIntegrator);
MI_EXPORT_PLUGIN(VolumetricMarchedPathIntegrator, "Volumetric Raymarching Path Tracer integrator");
NAMESPACE_END(mitsuba)
