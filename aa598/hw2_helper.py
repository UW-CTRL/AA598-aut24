import jax.numpy as jnp
# import jax
from dynamaxsys import *


# @jax.jit
def simulate_dynamics(dynamics, state, controls, dt):
    T = controls.shape[0]
    states = [state]
    for c in controls:
        state = dynamics.discrete_step(state, c, 0., dt)
        states.append(state)
    return jnp.stack(states)
