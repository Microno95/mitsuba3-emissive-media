import pytest
import drjit as dr
import mitsuba as mi

from mitsuba.scalar_rgb.test.util import fresolver_append_path


spectrum_dicts = {
    'd65': {
        "type": "d65",
    },
    'regular': {
        "type": "regular",
        "wavelength_min": 500,
        "wavelength_max": 600,
        "values": "1, 2"
    }
}


@fresolver_append_path
def create_emitter_and_spectrum(s_key='d65'):
    emitter = mi.load_dict({
        "type": "obj",
        "filename": "resources/data/tests/obj/cbox_smallbox.obj",
        "emitter" : { "type": "volumelight", "radiance" : spectrum_dicts[s_key] }
    })
    spectrum = mi.load_dict(spectrum_dicts[s_key])
    expanded = spectrum.expand()
    if len(expanded) == 1:
        spectrum = expanded[0]

    return emitter, spectrum


def test01_constructor(variant_scalar_rgb):
    # Check that the shape is properly bound to the emitter
    shape, spectrum = create_emitter_and_spectrum()
    assert shape.emitter().bbox() == shape.bbox()

    # Check that we are not allowed to specify a to_world transform directly in the emitter.
    with pytest.raises(RuntimeError):
        e = mi.load_dict({
            "type" : "volumelight",
            "to_world" : mi.ScalarTransform4f.translate([5, 0, 0])
        })


@pytest.mark.parametrize("spectrum_key", spectrum_dicts.keys())
def test02_eval(variants_vec_spectral, spectrum_key):
    # Check that eval() return the same values as the 'radiance' spectrum

    shape, spectrum = create_emitter_and_spectrum(spectrum_key)
    emitter = shape.emitter()

    it = dr.zeros(mi.SurfaceInteraction3f, 3)
    assert dr.allclose(emitter.eval(it), spectrum.eval(it))

    # Check that eval returns 0.0 when the sample point is outside the shape
    it.p = mi.ScalarPoint3f([0.0, 0.0, 0.0])
    assert dr.allclose(emitter.eval(it), 0.0)


@pytest.mark.parametrize("spectrum_key", spectrum_dicts.keys())
def test03_sample_ray(variants_vec_spectral, spectrum_key):
    # Check the correctness of the sample_ray() method

    shape, spectrum = create_emitter_and_spectrum(spectrum_key)
    emitter = shape.emitter()

    time = 0.5
    wavelength_sample = [0.5, 0.33, 0.1]
    pos_sample = [[0.2, 0.1, 0.2], [0.6, 0.9, 0.2], [0.5]*3]
    dir_sample = [[0.4, 0.5, 0.3], [0.1, 0.4, 0.9]]

    # Sample a ray (position, direction, wavelengths) on the emitter
    ray, res = emitter.sample_ray(time, wavelength_sample, pos_sample, dir_sample)

    # Sample wavelengths on the spectrum
    it = dr.zeros(mi.SurfaceInteraction3f, 3)

    # Sample a position in the shape
    ps = shape.sample_position_volume(time, pos_sample)
    pdf = mi.warp.square_to_uniform_sphere_pdf(mi.warp.square_to_uniform_sphere(dir_sample))
    it.p = ps.p
    it.n = ps.n
    it.time = time
    it.wavelengths = mi.Float(wavelength_sample) * (mi.MI_CIE_MAX - mi.MI_CIE_MIN) + mi.MI_CIE_MIN

    spec = dr.select(ps.pdf > 0.0, emitter.eval(it) * (mi.MI_CIE_MAX - mi.MI_CIE_MIN) / (ps.pdf * pdf), 0.0)
    assert dr.allclose(res, spec)
    assert dr.allclose(ray.time, time)
    assert dr.allclose(ray.o, ps.p, atol=2e-2)
    assert dr.allclose(ray.d, mi.Frame3f(ps.n).to_world(mi.warp.square_to_uniform_sphere(dir_sample)))


@pytest.mark.parametrize("spectrum_key", spectrum_dicts.keys())
def test04_sample_direction(variants_vec_spectral, spectrum_key):
    # Check the correctness of the sample_direction(), pdf_direction(), and eval_direction() methods

    shape, spectrum = create_emitter_and_spectrum(spectrum_key)
    emitter = shape.emitter()

    # Direction sampling is conditioned on a sampled position
    it = dr.zeros(mi.SurfaceInteraction3f, 3)
    it.p = [[0.2, 0.1, 0.2], [0.6, -0.9, 0.2],
            [0.4, 0.9, -0.2]]  # Some positions
    it.time = 1.0

    # Sample direction on the emitter
    samples = [[0.4, 0.5, 0.3], [0.1, 0.4, 0.9], [0.5]*3]
    ds, res = emitter.sample_direction(it, samples)

    # Sample direction on the shape
    shape_ds = dr.zeros(mi.DirectionSample3f)
    shape_ds.p = it.p
    ps = shape.sample_position_volume(it.time, samples)
    shape_ds.d = ps.p - it.p
    dist2 = dr.squared_norm(shape_ds.d)
    shape_ds.d = shape_ds.d / dr.sqrt(dist2)
    shape_ds.pdf = shape.pdf_direction_volume(it, ds)

    assert dr.allclose(ds.pdf, shape_ds.pdf)
    assert dr.allclose(ds.pdf, emitter.pdf_direction(it, ds))
    assert dr.allclose(ds.d, shape_ds.d, atol=1e-3)
    assert dr.allclose(ds.time, it.time)

    # Evaluate the spectrum (divide by the pdf)
    spec = dr.select(ds.pdf > 0.0, emitter.eval(it) / ds.pdf, 0.0)
    assert dr.allclose(res, spec)

    assert dr.allclose(emitter.eval_direction(it, ds), spec)
