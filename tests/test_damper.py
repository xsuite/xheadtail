import numpy as np
import xheadtail as xht
import xpart as xp

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
particles4.py[3:] = 1000

# for damper in [damper_x, damper_y, damper_xy]:
#     pp = particles4.copy()
#     damper.track(pp)
