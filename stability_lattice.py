import numpy as np
import jax
from jax import jit
import jax.numpy as jnp
from functools import partial
from typing import Union

import VorticesMotionPeriodic as vmp
import utils

Array = Union[np.ndarray, jnp.ndarray]

from jax import config
config.update("jax_enable_x64", True)

def velocities_for_jacobian(state: Array, gammas: Array, indices: Array, n: int, uv: Array, L: float, m: int) -> Array:
  """
    Returns the induced velocity in the translating frame for every vortex in the system
  """

  velocities = vmp._every_induced_velocity(state, gammas, indices, L, m=4)
  
  # transform velocities to co rotating frame
  velocities = velocities.at[:n].set(velocities[:n] - uv[0])
  velocities = velocities.at[n:].set(velocities[n:] - uv[1])
  
  return velocities

  
def stability_eigs(state: Array, gammas: Array, L: float, uv: Array) -> [Array, Array]:
  """ Computes the eigenvalues and eigenvectors of an equilibrium's stability matrix """
  
  n = len(gammas)
  ind = utils.indices(n)
  
  jacobian = jax.jacfwd(velocities_for_jacobian)(state, gammas, ind, n, uv, L, m=4)
  
  eigenvalues, eigenvectors = jnp.linalg.eig(jacobian)
  
  return eigenvalues, eigenvectors

def stability_directions(state: Array, gammas: Array, L: float, uv: Array) -> [Array, Array, Array]:
  """ Computes the stable and unstable directions of an equilibrium, tolerance is set to 10^{-5} """
  
  eigenvalues, eigenvectors = stability_eigs(state, gammas, L, uv)
  n = len(gammas)
  
  bool_if_stable = jnp.where(jnp.real(eigenvalues) < -1e-5, np.full(2*n, True), np.full(2*n, False))
  bool_if_unstable = jnp.where(jnp.real(eigenvalues) > 1e-5, np.full(2*n, True), np.full(2*n, False))
  bool_if_zero = jnp.where(jnp.abs(eigenvalues) < 1e-5, np.full(2*n, True), np.full(2*n, False))
  
  stable_eigenvectors = jnp.transpose(eigenvectors[:, bool_if_stable])
  unstable_eigenvectors = jnp.transpose(eigenvectors[:, bool_if_unstable])
  zero_eigenvectors = jnp.transpose(eigenvectors[:, bool_if_zero])
  
  return eigenvalues, stable_eigenvectors, unstable_eigenvectors, zero_eigenvectors

