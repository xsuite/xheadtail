import numpy as np
from scipy.special import k0
from scipy.constants import c, e

import xobjects as xo
import xpart as xp
import xtrack as xt


def with_hidden_lost_particles(track):
    def track_without_lost_particles(self, particles):
        if particles.lost_particles_are_hidden:
            track(self, particles)
        else:
            particles.hide_lost_particles()
            track(self, particles)
            particles.unhide_lost_particles()
    return track_without_lost_particles

class TransverseDamper(xt.BeamElement):

    _xofields = {
        'gain_x': xo.Float64,
        'gain_y': xo.Float64,
        '_sin_phi_x': xo.Float64,
        '_sin_phi_y': xo.Float64,
        '_cos_phi_x': xo.Float64,
        '_cos_phi_y': xo.Float64,
        'beta_x': xo.Float64,
        'beta_y': xo.Float64,
    }

    iscollective = True

    def __init__(self, damping_time_x=None, damping_time_y=None,
                 gain_x=None, gain_y=None, phi_x=90., phi_y=90., **kwargs):

        xt.BeamElement.__init__(self, **kwargs)

        if damping_time_x:
            assert gain_x is None, (
                'Only one of `damping_time_x` and `gain_x` can be passed.')
            self.damping_time_x = damping_time_x

        if damping_time_y:
            assert gain_y is None, (
                'Only one of `damping_time_y` and `gain_y` can be passed.')
            self.damping_time_y = damping_time_y

        self.phi_x = phi_x
        self.phi_y = phi_y

    @with_hidden_lost_particles
    def track(self, particles):
        if self.gain_x != 0:
            particles.px -= (self.gain_x * self._sin_phi_x
                             * particles.px.mean())
            if self.beta_x != 0:
                particles.px -= (self.gain_x * self._cos_phi_x
                                 * particles.x.mean() / self.beta_x)
        if self.gain_y != 0:
            particles.py -= (self.gain_y * self._sin_phi_y
                             * particles.py.mean())
            if self.beta_y != 0:
                particles.py -= (self.gain_y * self._cos_phi_y
                                 * particles.y.mean() / self.beta_y)
    @property
    def damping_time_x(self):
        if self.gain_x == 0:
            return None
        return 2/self.gain_x

    @damping_time_x.setter
    def damping_time_x(self, value):
        if value is None:
            self.gain_x = 0
        elif value == 0:
            raise ValueError('Only non-zero values allowed for `damping_time_x`.')
        else:
            self.gain_x = 2/value

    @property
    def damping_time_y(self):
        if self.gain_y == 0:
            return None
        return 2/self.gain_y

    @damping_time_y.setter
    def damping_time_y(self, value):
        if value is None:
            self.gain_y = 0
        elif value == 0:
            raise ValueError('Only non-zero values allowed for `damping_time_y`.')
        else:
            self.gain_y = 2/value

    @property
    def phi_x(self):
        return np.arctan2(self._sin_phi_x, self._cos_phi_x) * 180 / np.pi

    @phi_x.setter
    def phi_x(self, value):
        self._sin_phi_x = np.sin(value / 180 * np.pi)
        self._cos_phi_x = np.cos(value / 180 * np.pi)

    @property
    def phi_y(self):
        return np.arctan2(self._sin_phi_y, self._cos_phi_y) * 180 / np.pi

    @phi_y.setter
    def phi_y(self, value):
        self._sin_phi_y = np.sin(value / 180 * np.pi)
        self._cos_phi_y = np.cos(value / 180 * np.pi)

    @property
    def dampingrate_x(self):
        raise NameError(
            '`dampingrate_x` is deprecated. Please use damping_time_x instead.')

    @dampingrate_x.setter
    def dampingrate_x(self, value):
        raise NameError(
            '`dampingrate_x` is deprecated. Please use damping_time_x instead.')

    @property
    def dampingrate_y(self):
        raise NameError(
            '`dampingrate_y` is deprecated. Please use damping_time_y instead.')

    @dampingrate_y.setter
    def dampingrate_y(self, value):
        raise NameError(
            '`dampingrate_y` is deprecated. Please use damping_time_y instead.')
