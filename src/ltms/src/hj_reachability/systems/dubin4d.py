import jax.numpy as jnp

from .. import dynamics
from .. import sets

__all__ = ['DubinsCar4D']


class DubinsCar4D(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 min_turn_rate, max_turn_rate,
                 min_accel, max_accel,
                 min_disturbances=None, 
                 max_disturbances=None,
                 control_mode="min",
                 disturbance_mode="max",
                 control_space=None,
                 disturbance_space=None):

        if min_disturbances is None:
            min_disturbances = [0] * 4
        if max_disturbances is None:
            max_disturbances = [0] * 4

        if control_space is None:
            control_space = sets.Box(jnp.array([min_turn_rate, min_accel]),
                                     jnp.array([max_turn_rate, max_accel]))
        
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
    # Implements the affine dynamics 
    #   `dx_dt = f(x, t) + G_u(x, t) @ u + G_d(x, t) @ d`.
    # 
    #   x_dot = v * cos(yaw) + d_1
    #   y_dot = v * sin(yaw) + d_2
    #   yaw_dot = u1 + d3
    #   v_dot = u2 + d4
        

    def open_loop_dynamics(self, state, time):
        x, y, yaw, vel = state
        return jnp.array([
            vel * jnp.cos(yaw),
            vel * jnp.sin(yaw),
            0,
            0.,
        ])

    def control_jacobian(self, state, time):
        return jnp.array([
            [0., 0.],
            [0., 0.],
            [1., 0.],
            [0., 1.],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.identity(4)
