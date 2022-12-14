import numpy as np
import xheadtail as xht
import xpart as xp


def test_damper_resistive():
    damper_x = xht.TransverseDamper(damping_time_x=10)
    damper_y = xht.TransverseDamper(damping_time_y=50)
    damper_xy = xht.TransverseDamper(damping_time_x=10, damping_time_y=50)

    particles0 = xp.Particles(p0c=450e9, px=[1,2,3], py=[3,4,5], x=10, y=20)
    particles1 = particles0.copy()
    particles2 = particles1.copy()
    particles3 = particles1.copy()

    damper_x.track(particles1)
    damper_y.track(particles2)
    damper_xy.track(particles3)

    assert np.isclose(damper_x.gain_x, 2/10, rtol=0, atol=1e-14)
    assert np.isclose(damper_y.gain_y, 2/50, rtol=0, atol=1e-14)
    assert damper_xy.gain_x == damper_x.gain_x
    assert damper_xy.gain_y == damper_y.gain_y

    assert np.all(particles1.py == particles0.py)
    assert np.allclose(particles1.px, particles0.px - damper_x.gain_x*particles0.px.mean(), rtol=0, atol=1e-14)

    assert np.all(particles2.px == particles0.px)
    assert np.allclose(particles2.py, particles0.py - damper_y.gain_y*particles0.py.mean(), rtol=0, atol=1e-14)

    assert np.all(particles3.px == particles1.px)
    assert np.all(particles3.py == particles2.py)

    particles4 = xp.Particles.merge([particles0, particles0])
    particles4.px[3:] = 1000
    particles4.py[3:] = 2000
    particles4.state[3:] = 0

    for damper, particles in zip([damper_x, damper_y, damper_xy],
                    [particles1, particles2, particles3]):
        pp = particles4.copy()
        damper.track(pp)
        assert np.all(pp.px[:3] == particles.px)
        assert np.all(pp.py[:3] == particles.py)
        assert np.all(pp.px[3:] == 1000)
        assert np.all(pp.py[3:] == 2000)


def test_damper_reactive():
    damper_x = xht.TransverseDamper(damping_time_x=10, beta_x=20, phi_x=30)
    damper_y = xht.TransverseDamper(damping_time_y=50, beta_y=30, phi_y=120)
    damper_xy = xht.TransverseDamper(damping_time_x=10, damping_time_y=50,
                                     beta_x=20, beta_y=30, phi_x=30, phi_y=120)

    assert np.isclose(damper_x.phi_x, 30, rtol=0, atol=1e-13)
    assert np.isclose(damper_y.phi_y, 120, rtol=0, atol=1e-13)
    assert damper_xy.phi_x == damper_x.phi_x
    assert damper_xy.phi_y == damper_y.phi_y
    assert damper_xy.beta_x == damper_x.beta_x
    assert damper_xy.beta_y == damper_y.beta_y

    particles0 = xp.Particles(p0c=450e9, px=[1,2,3], py=[3,4,5], x=10, y=20)
    particles1 = particles0.copy()
    particles2 = particles1.copy()
    particles3 = particles1.copy()

    damper_x.track(particles1)
    damper_y.track(particles2)
    damper_xy.track(particles3)

    assert np.all(particles1.py == particles0.py)
    px_ref = (particles0.px
              - damper_x.gain_x * np.sin(damper_x.phi_x /180 * np.pi) * particles0.px.mean()
              - damper_x.gain_x * np.cos(damper_x.phi_x /180 * np.pi) * particles0.x.mean() / damper_x.beta_x)
    assert np.allclose(particles1.px, px_ref, rtol=0, atol=1e-14)

    assert np.all(particles2.px == particles0.px)
    py_ref = (particles0.py
              - damper_y.gain_y * np.sin(damper_y.phi_y /180 * np.pi) * particles0.py.mean()
              - damper_y.gain_y * np.cos(damper_y.phi_y /180 * np.pi) * particles0.y.mean() / damper_y.beta_y)
    assert np.allclose(particles2.py, py_ref, rtol=0, atol=1e-14)

    assert np.all(particles3.px == particles1.px)
    assert np.all(particles3.py == particles2.py)

    particles4 = xp.Particles.merge([particles0, particles0])
    particles4.px[3:] = 1000
    particles4.py[3:] = 2000
    particles4.state[3:] = 0

    for damper, particles in zip([damper_x, damper_y, damper_xy],
                    [particles1, particles2, particles3]):
        pp = particles4.copy()
        damper.track(pp)
        assert np.all(pp.px[:3] == particles.px)
        assert np.all(pp.py[:3] == particles.py)
        assert np.all(pp.px[3:] == 1000)
        assert np.all(pp.py[3:] == 2000)

