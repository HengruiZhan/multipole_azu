import analytical_potential as ap
import comparitive_difference as comdf
import numpy as np
import matplotlib.pyplot as plt
import grid
import L2_difference as L2
import multipole

"""
nr = 101
nz = 201
"""
"""
nr = 257
nz = 513
"""
"""
nr = 513
nz = 1025
"""

nr = 129
nz = 257

rlim = (0.0, 1.0)
zlim = (-1.0, 1.0)

"""
rlim = (0.0, 200)
zlim = (-100, 100)
"""
g = grid.Grid(nr, nz, rlim, zlim)

dens = g.scratch_array()
"""
print(g.r.shape)
print(g.z.shape)
print(np.amax(g.r))
print(np.amax(g.z))
print(np.amin(g.r))
print(np.amin(g.z))
"""

# density of a MacLaurin spheroid
sph_center = (0.0, 0.0)
a_1 = 0.23
a_3 = 0.10
mask = g.r2d**2/a_1**2 + g.z2d**2/a_3**2 <= 1
dens[mask] = 1.0
density = 1.0

"""
plt.imshow(np.transpose(dens), origin="lower",
           interpolation="nearest",
           extent=[g.rlim[0], g.rlim[1],
                   g.zlim[0], g.zlim[1]])
ax = plt.gca()
ax.set_aspect("equal")
plt.savefig("dens.png")
e = np.sqrt(1 - (a_3/a_1)**2)
print(e)
"""

# analytical potential of the MacLaurin spheroid
phi_mac = g.scratch_array()
for i in range(g.nr):
    for j in range(g.nz):
        mac_phi = ap.Ana_Mac_pot(a_1, a_3, g.r[i], g.z[j], density)
        phi_mac[i, j] = mac_phi.potential

"""
plt.imshow(np.log10(np.abs(np.transpose(phi_mac))), origin="lower",
           interpolation="nearest",
           extent=[g.rlim[0], g.rlim[1],
                   g.zlim[0], g.zlim[1]])

plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.savefig("anal_mac.png")
"""

center = (0.0, 0.0)
n_moments = 10
m = multipole.Multipole(g, n_moments, 2*g.dr, center=center)

m.compute_expansion(dens)

phi = g.scratch_array()
for i in range(g.nr):
    for j in range(g.nz):
        phi[i, j] = m.phi_point(g.r[i], g.z[j])

# phi = run.run_multipole(m, g, dens)

plt.imshow(np.log10(np.abs(np.transpose(phi))), origin="lower",
           interpolation="nearest",
           extent=[g.rlim[0], g.rlim[1],
                   g.zlim[0], g.zlim[1]])
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.savefig("lmax=10.png")

L2normerr = L2.L2_diff(phi_mac, phi, g.scratch_array())
print(L2normerr)
print("lmax=", n_moments)
