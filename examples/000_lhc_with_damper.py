import json
import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp
import xheadtail as xht

# Load a line
fname_line = '../../xtrack/test_data/lhc_no_bb/line_and_particle.json'
with open(fname_line, 'r') as fid:
     input_data = json.load(fid)
line = xt.Line.from_dict(input_data['line'])
line.particle_ref = xp.Particles(p0c=7e12)

# Install a damper
line.append_element(
    element=xht.TransverseDamper(damping_time_x=10., damping_time_y=15.),
    name='damper'
)

# Build the tracker
tracker = line.build_tracker()

# Generate a matched bunch
particles = xp.generate_matched_gaussian_bunch(
    tracker=tracker, nemitt_x=2.5e-6, nemitt_y=2.5e-6,
    total_intensity_particles=1e11,
    num_particles=200, sigma_z=0.1)

# Kick the bunch
particles.x += 0.5e-3
particles.y += 1e-3

# Track!
tracker.track(particles, num_turns=100, turn_by_turn_monitor=True)

# Plot
res = tracker.record_last_track

import matplotlib.pyplot as plt
plt.close('all')
fig1 = plt.figure(1)
sp1 = fig1.add_subplot(2,1,1)
sp2 = fig1.add_subplot(2,1,2)
sp1.plot(np.mean(res.x, axis=0))
sp2.plot(np.mean(res.y, axis=0))
plt.show()
