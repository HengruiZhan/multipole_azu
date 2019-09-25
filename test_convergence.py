import analytical_potential as ap
import comparitive_difference as comdf
import numpy as np
import matplotlib.pyplot as plt
import grid
import L2_difference as L2
import multipole
np.set_printoptions(threshold=np.inf)

nr = 64
nz = 128

"""
nr = 128
nz = 256
# nz = 128
"""

rlim = (0.0, 1.0)
zlim = (-1.0, 1.0)
# zlim = (-0.5, 0.5)

g = grid.Grid(nr, nz, rlim, zlim)

dens = g.scratch_array()
"""
# density of a perfect sphere
sph_center = (0.0, 0.0)
radius = np.sqrt((g.r2d - sph_center[0])**2 + (g.z2d - sph_center[1])**2)
a = 0.2
dens[radius <= a] = 1.0
# analytical potential of a perfect sphere
density = 1.0
phi_anal = g.scratch_array()
for i in range(g.nr):
    for j in range(g.nz):
        sph_phi = ap.Ana_Sph_pot(g.r[i], g.z[j], a, density)
        phi_anal[i, j] = sph_phi.potential
"""

"""
# density of double sphere
a = 0.1
sph1_center = (0.0, 0.5)
sph2_center = (0.0, -0.5)
mask1 = (g.r2d-sph1_center[0])**2 + (g.z2d-sph1_center[1])**2 <= a**2
dens[mask1] = 1.0
mask2 = (g.r2d-sph2_center[0])**2 + (g.z2d-sph2_center[1])**2 <= a**2
dens[mask2] = 1.0
density = 1.0

# analytical potential of double sphere
phi_anal = g.scratch_array()
for i in range(g.nr):
    for j in range(g.nz):
        double_sph = ap.Ana_Double_Sph_pot(g.r[i], g.z[j], sph1_center,
                                           sph2_center, a, density)
        phi_anal[i, j] = double_sph.potential
        """

# density of a MacLaurin spheroid
sph_center = (0.0, 0.0)
a_1 = 0.23
a_3 = 0.10
mask = g.r2d**2/a_1**2 + g.z2d**2/a_3**2 <= 1
dens[mask] = 1.0
density = 1.0

# analytical potential of the MacLaurin spheroid
phi_anal = g.scratch_array()
for i in range(g.nr):
    for j in range(g.nz):
        mac_phi = ap.Ana_Mac_pot(a_1, a_3, g.r[i], g.z[j], density)
        phi_anal[i, j] = mac_phi.potential

"""
# density of double MacLaurin spheroid
sph1_center = (0.0, 0.5)
sph2_center = (0.0, -0.5)
a_1 = 0.23
a_3 = 0.10
mask1 = (g.r2d-sph1_center[0])**2/a_1**2 + (g.z2d -
                                            sph1_center[1])**2/a_3**2 <= 1
dens[mask1] = 1.0
mask2 = (g.r2d-sph2_center[0])**2/a_1**2 + (g.z2d -
                                            sph2_center[1])**2/a_3**2 <= 1
dens[mask2] = 1.0
density = 1.0

phi_anal = g.scratch_array()
for i in range(g.nr):
    for j in range(g.nz):
        dou_mac_phi = ap.Ana_Double_Mac(a_1, a_3, g.r[i], g.z[j],
                                        sph1_center, sph2_center, density)
        phi_anal[i, j] = dou_mac_phi.potential
"""
"""
center = (0, 0)
n_moments = 0
m = multipole.Multipole(g, n_moments, 2*g.dr, center=center)

m.compute_expansion(dens)

phi = g.scratch_array()
for i in range(g.nr):
    for j in range(g.nz):
        phi[i, j] = m.phi_point(g.r[i], g.z[j])
"""

# test brute force version Multipole
center = (0, 0)
n_moments = 0
m = multipole.Multipole(g, n_moments, center=center)

m.compute_expansion(dens)
phi = m.phi()

phi_norm = np.sqrt(np.sum(phi**2/phi.size))
L2normerr = L2.L2_diff(phi_anal, phi, g.scratch_array())
# L2normerr = L2.L2_diff(phi_anal, phi, g)
# normalized L2 norm error
L2normerr = L2normerr/phi_norm

print("lmax=", n_moments)
print(L2normerr)
