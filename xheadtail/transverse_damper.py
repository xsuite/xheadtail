import numpy as np
from scipy.special import k0
from scipy.constants import c, e

import xpart as xp
import xobjects as xo

def _compute_mean(particles, coord):
    if isinstance(particles._context, xo.ContextPyopencl):
        raise NotImplementedError
    mask = particles.state > 0
    res = getattr(particles, coord)[mask].mean()
    return res

class TransverseDamper:

    def __init__(self, dampingrate_x, dampingrate_y, phase=90,
                 local_beta_function=None, *args, **kwargs):
        '''Ideal transverse damper with an in-place "measurement"
        (transverse "pick-up") of the transverse dipole moment.
        Note: a single bunch in the beam is assumed, i.e. this works on
        the entire beam's moments.

        Arguments:
            - dampingrate_x, dampingrate_y: horizontal and vertical
                damping rates in turns (e.g. 50 turns for a typical 2018
                LHC ADT set-up)
            - phase: phase of the damper kick in degrees with respect to
                the transverse position "pick-up". The default value of
                90 degrees corresponds to a typical resistive damper.
            - local_beta_function: the optics beta function at the
                transverse position "pick-up" (e.g. in the local place
                of this Element). This is required if the damper is not
                a purely resistive damper (or exciter), i.e. if the
                phase is not 90 (or 270) degrees. The beta function is
                assumed to be the same for both transverse planes,
                otherwise use two instances of the TransverseDamper.
        '''

        if dampingrate_x and not dampingrate_y:
            self.gain_x = 2/dampingrate_x
            self.track = self.track_horizontal
            self.prints('Damper in horizontal plane active')
        elif not dampingrate_x and dampingrate_y:
            self.gain_y = 2/dampingrate_y
            self.track = self.track_vertical
            self.prints('Damper in vertical plane active')
        elif not dampingrate_x and not dampingrate_y:
            self.prints('Dampers not active')
        else:
            self.gain_x = 2/dampingrate_x
            self.gain_y = 2/dampingrate_y
            self.track = self.track_all
            self.prints('Dampers active')
        if phase != 90 and phase != 270 and not local_beta_function:
            raise TypeError(
                'TransverseDamper: numeric local_beta_function value at '
                'position of damper missing! (Required because of non-zero '
                'reactive damper component.)')
        self.phase_in_2pi = phase / 360. * 2*np.pi
        self.local_beta_function = local_beta_function

    # will be overwritten at initialisation
    def track(self, particles: xp.Particles):
        pass

    def track_horizontal(self, particles: xp.Particles):
        particles.px -= self.gain_x * np.sin(self.phase_in_2pi) * _compute_mean(particles, 'px')
        if self.local_beta_function:
            particles.px -= (self.gain_x * np.cos(self.phase_in_2pi) *
                        _compute_mean(particles, 'x') / self.local_beta_function)

    def track_vertical(self, particles: xp.Particles):
        particles.py -= self.gain_y * np.sin(self.phase_in_2pi) * _compute_mean(particles, 'py')
        if self.local_beta_function:
            particles.py -= (self.gain_y * np.cos(self.phase_in_2pi) *
                        _compute_mean(particles, 'y') / self.local_beta_function)

    def track_all(self, particles: xp.Particles):
        particles.px -= self.gain_x * np.sin(self.phase_in_2pi) * _compute_mean(particles, 'px')
        particles.py -= self.gain_y * np.sin(self.phase_in_2pi) * _compute_mean(particles, 'py')
        if self.local_beta_function:
            particles.px -= (self.gain_x * np.cos(self.phase_in_2pi) *
                        _compute_mean(particles, 'x') / self.local_beta_function)
            particles.py -= (self.gain_y * np.cos(self.phase_in_2pi) *
                        _compute_mean(particles, 'y') / self.local_beta_function)

    @classmethod
    def horizontal(cls, dampingrate_x, *args, **kwargs):
        return cls(dampingrate_x, 0, *args, **kwargs)

    @classmethod
    def vertical(cls, dampingrate_y, *args, **kwargs):
        return cls(0, dampingrate_y, *args, **kwargs)
