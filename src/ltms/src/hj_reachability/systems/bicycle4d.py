import jax.numpy as jnp

from .. import dynamics
from .. import sets

__all__ = ['Bicycle4D']


class Bicycle4D(dynamics.Dynamics):

    def __init__(self,
                 min_steer, max_steer,
                 min_accel, max_accel,
                 min_disturbances=None, 
                 max_disturbances=None,
                 wheelbase=0.32,
                 control_mode="min",
                 disturbance_mode="max",
                 control_space=None,
                 disturbance_space=None):

        self.wheelbase = wheelbase

        if min_disturbances is None:
            min_disturbances = [0] * 4
        if max_disturbances is None:
            max_disturbances = [0] * 4

        if control_space is None:
            control_space = sets.Box(jnp.array([min_steer, min_accel]),
                                     jnp.array([max_steer, max_accel]))
        
        if disturbance_space is None:
            disturbance_space = sets.Box(jnp.array(min_disturbances),
                                         jnp.array(max_disturbances))
        super().__init__(control_mode, 
                         disturbance_mode, 
                         control_space, 
                         disturbance_space)

    def with_mode(self, mode: str):
        assert mode in ["reach", "avoid"]
        if mode == "reach":
            self.control_mode = "min"
            self.disturbance_mode = "max"
        elif mode == "avoid":
            self.control_mode = "max"
            self.disturbance_mode = "min"
        return self

    ## Dynamics
    #   x_dot = v * cos(yaw) + d_1
    #   y_dot = v * sin(yaw) + d_2
    #   yaw_dot = (v * tan(u1))/L + d3
    #   delta_dot = u1 + d4
    #   v_dot = u2 + d5
    
    def __call__(self, state, control, disturbance, time):
        x, y, yaw, vel = state
        steer, accel = control
        return jnp.array([
            vel * jnp.cos(yaw),
            vel * jnp.sin(yaw),
            (vel * jnp.tan(steer))/self.wheelbase,
            accel,
        ])
    
    def optimal_control_and_disturbance(self, state, time, grad_value):
        # OBS: only when vel >= 0

        if self.control_mode == 'max':
            opt_ctrl = jnp.where(0 <= grad_value[2:], self.control_space.hi, self.control_space.lo)
        else:
            opt_ctrl = jnp.where(0 <= grad_value[2:], self.control_space.lo, self.control_space.hi)

        if self.disturbance_mode == 'max':
            opt_dstb = jnp.where(0 <= grad_value, self.disturbance_space.hi, self.disturbance_space.lo)
        else:
            opt_dstb = jnp.where(0 <= grad_value, self.disturbance_space.lo, self.disturbance_space.hi)

        return opt_ctrl, opt_dstb
    
    def partial_max_magnitudes(self, state, time, value, grad_value_box):
        del value, grad_value_box # unused
        return jnp.abs(self(state, self.control_space.max_magnitudes, self.disturbance_space.max_magnitudes, time))