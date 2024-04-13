#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/string.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/render/srgb.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/render/volume.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class ToroidalVolume final : public Volume<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Volume, m_bbox, m_to_local)
    MI_IMPORT_TYPES(Texture)

    ToroidalVolume(const Properties &props)
        : Base(props) {

        m_cross_section_texture = props.texture<Texture>("cross_section");
        m_scale = props.get<float>("scale", 1.0f);
        m_bbox = ScalarBoundingBox3f(ScalarPoint3f(-1,-1,-1), ScalarPoint3f(1,1,1));
    }

    UnpolarizedSpectrum eval(const Interaction3f &it, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        return eval_impl(it, active);
    }

    Float eval_1(const Interaction3f &it, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        return dr::mean(eval_impl(it, active));
    }

    MI_INLINE UnpolarizedSpectrum eval_impl(const Interaction3f &it, Mask active) const {
        auto local_p = m_to_local * it.p;
        auto local_R = dr::clamp(dr::sqrt(local_p.x() * local_p.x() + local_p.y() * local_p.y()), 0.0f, 1.0f);
        auto local_Z = dr::clamp(local_p.z(), 0.0f, 1.0f);

        auto ps = PositionSample3f(
            it.p,
            it.n,
            Point2f(local_R, local_Z),
            it.time,
            1.0f,
            dr::zeros<Mask>()
        );
        auto si = SurfaceInteraction3f(ps, it.wavelengths);

        return m_scale * m_cross_section_texture->eval(si, active && m_bbox.contains(local_p));
    }

    ScalarFloat max() const override { return m_scale * m_cross_section_texture->max(); }

    void traverse(TraversalCallback *callback) override {
        callback->put_object("cross_section", m_cross_section_texture.get(), +ParamFlags::Differentiable);
        callback->put_parameter("to_local", m_to_local, +ParamFlags::Differentiable);
        Base::traverse(callback);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "ToroidalGridVolume[" << std::endl
            << "  cross_section = " << m_cross_section_texture << "," << std::endl
            << "  bbox = " << string::indent(m_bbox) << "," << std::endl
            << "  to_local = " << string::indent(m_to_local) << "," << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
protected:
    ScalarFloat m_scale;
    ref<Texture> m_cross_section_texture;
};

MI_IMPLEMENT_CLASS_VARIANT(ToroidalVolume, Volume)
MI_EXPORT_PLUGIN(ToroidalVolume, "Toroidal Volume Texture")
NAMESPACE_END(mitsuba)
