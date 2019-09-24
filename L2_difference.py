import numpy as np


def L2_diff(analytical_phi, approximate_phi, gcenter):
    phi_difference = analytical_phi - approximate_phi
    N = gcenter.size
    pot_diff_norm = np.sqrt(np.sum(np.square(phi_difference)/N))
    # pot_diff_norm = np.sqrt(np.sum(np.square(phi_difference)))
    return pot_diff_norm


# finite volume L2 norm error
def L2_diff(analytical_phi, approximate_phi, grid):
    total_vol = np.sum(grid.vol)
    phi_difference = analytical_phi - approximate_phi
    L2_norm = np.sqrt(np.sum(phi_difference**2)/total_vol)
    return L2_norm
