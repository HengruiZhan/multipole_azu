from numba import jit
import numpy as np
import scipy.constants as sc
from scipy.special import sph_harm


class Multipole():
    # def __init__(self, grid, n_moments, dr, center=(0.0, 0.0)):
    def __init__(self, grid, n_moments, dr, center=(0.0, 0.0)):

        self.g = grid
        self.n_moments = n_moments
        self.dr_mp = dr
        self.center = center

        # compute the bins
        # this computation method is not correct
        r_max = max(abs(self.g.rlim[0] - center[0]), abs(self.g.rlim[1] -
                                                         center[0]))
        z_max = max(abs(self.g.zlim[0] - center[1]), abs(self.g.zlim[1] -
                                                         center[1]))

        dmin = 0.0

        dmax = np.sqrt(r_max**2 + z_max**2)

        self.n_bins = int(dmax/dr)

        # bin boundaries
        self.r_bin = np.linspace(dmin, dmax, self.n_bins)

        # storage for the inner and outer multipole moment functions
        # we'll index the list by multipole moment l

        self.m_r = []
        self.m_i = []
        for l in range(self.n_moments+1):
            self.m_r.append(np.zeros((self.n_bins), dtype=np.complex128))
            self.m_i.append(np.zeros((self.n_bins), dtype=np.complex128))

    def compute_harmonics(self, l, r, z):
        radius = np.sqrt((r - self.center[0])**2 +
                         (z - self.center[1])**2)
        theta = np.arctan2(r, z)

        Y_lm = sph_harm(0, l, 0.0, theta)
        R_lm = np.sqrt(4*np.pi/(2*l + 1)) * radius**l * Y_lm
        I_lm = np.nan_to_num(np.sqrt(4*np.pi/(2*l + 1)) *
                             Y_lm / radius**(l+1))

        return R_lm, I_lm

    @jit
    def compute_expansion(self, rho):
        # rho is density that lives on a grid self.g

        # loop over cells
        for i in range(self.g.nr):
            for j in range(self.g.nz):

                # for each cell, i,j, compute r and theta (polar angle from z)
                # and determine which shell we are in

                radius = np.sqrt((self.g.r[i] - self.center[0])**2 +
                                 (self.g.z[j] - self.center[1])**2)

                # tan(theta) = r/z
                theta = np.arctan2(self.g.r[i], self.g.z[j])

                # loop over the multipole moments, l (m = 0 here)
                m_zone = rho[i, j] * self.g.vol[i, j]
                for l in range(self.n_moments+1):

                    # compute Y_l^m (note: we use theta as the polar
                    # angle, scipy is opposite)
                    Y_lm = sph_harm(0, l, 0.0, theta)

                    R_lm = np.sqrt(4*np.pi/(2*l + 1)) * radius**l * Y_lm
                    I_lm = np.sqrt(4*np.pi/(2*l + 1)) * Y_lm / radius**(l+1)

                    # add to the all of the appropriate inner or outer
                    # moment functions
                    imask = radius <= self.r_bin
                    omask = radius > self.r_bin
                    self.m_r[l][imask] += R_lm * m_zone
                    self.m_i[l][omask] += I_lm * m_zone
        # check whether m_r[l] is not 0
        # print("m_r is", self.m_r)
        # print("m_i is", self.m_i)

    def sample_mtilde(self, l, r):
        # this returns the result of Eq. 19

        # we need to find which be we are in
        # print("r is", r)
        mu_m = np.argwhere(self.r_bin <= r)[-1][0]
        mu_p = np.argwhere(self.r_bin > r)[0][0]

        assert mu_p == mu_m + 1

        mtilde_r = (r - self.r_bin[mu_m])/(self.r_bin[mu_p] - self.r_bin[mu_m]
                                           ) * self.m_r[l][mu_p] + \
            (r - self.r_bin[mu_p])/(self.r_bin[mu_m] -
                                    self.r_bin[mu_p]) * self.m_r[l][mu_m]

        mtilde_i = (r - self.r_bin[mu_m])/(self.r_bin[mu_p] - self.r_bin[mu_m]
                                           ) * self.m_i[l][mu_p] + \
            (r - self.r_bin[mu_p])/(self.r_bin[mu_m] -
                                    self.r_bin[mu_p]) * self.m_i[l][mu_m]

        return mtilde_r, mtilde_i

    @jit
    def compute_phi(self, r, z, dr, dz):
        # return Phi(r), using Eq. 20

        r += dr
        z += dz

        radius = np.sqrt((r - self.center[0])**2 +
                         (z - self.center[1])**2)

        # tan(theta) = r/z
        theta = np.arctan2(r, z)

        phi_zone = 0.0
        for l in range(self.n_moments+1):
            mtilde_r, mtilde_i = self.sample_mtilde(l, radius)

            Y_lm = sph_harm(0, l, 0.0, theta)
            R_lm = np.sqrt(4*np.pi/(2*l + 1)) * radius**l * Y_lm
            I_lm = np.nan_to_num(np.sqrt(4*np.pi/(2*l + 1)) *
                                 Y_lm / radius**(l+1))

            phi_zone += sc.G * (mtilde_r * np.conj(I_lm) +
                                np.conj(mtilde_i) * R_lm)

        return -np.real(phi_zone)

    def phi_point(self, r, z):
        """calculate the potential of a specific point"""
        dr = self.g.dr/2
        dz = self.g.dz/2

        # phi_0 = self.compute_phi(r, z, 0, 0)
        phi_m_r = self.compute_phi(r, z, -dr, 0)
        phi_m_z = self.compute_phi(r, z, 0, -dz)
        phi_p_r = self.compute_phi(r, z, dr, 0)
        phi_p_z = self.compute_phi(r, z, 0, dz)
        # phi = 1/total_area(sum(phi(x_face)*area))

        area_m_r = 2*np.pi*(r-dr)*dz
        area_m_z = np.pi*((r+dr)**2-(r-dr)**2)
        area_p_r = 2*np.pi*(r+dr)*dz
        area_p_z = np.pi*((r+dr)**2-(r-dr)**2)
        total_area = area_m_r+area_m_z+area_p_r+area_p_z

        phi_m_r_area = phi_m_r*area_m_r
        phi_m_z_area = phi_m_z*area_m_z
        phi_p_r_area = phi_p_r*area_p_r
        phi_p_z_area = phi_p_z*area_p_z
        phi = (phi_m_r_area+phi_m_z_area+phi_p_r_area+phi_p_z_area)/total_area

        return phi

    """
    @jit
    def phi(self):
        phi = self.g.scratch_array()
        for i in range(self.g.nr):
            for j in range(self.g.nz):
                phi[i, j] = self.phi_point(self.g.r[i], self.g.z[j])
        return phi
        """
    
    
    #brute force version of multipole expansion
    class Multipole():
    # def __init__(self, grid, n_moments, dr, center=(0.0, 0.0)):
    def __init__(self, grid, n_moments, center=(0.0, 0.0)):

        self.g = grid
        self.n_moments = n_moments
        self.center = center

        self.radius = np.sqrt((self.g.r2d - self.center[0])**2 +
                              (self.g.z2d - self.center[1])**2)

        self.m_r = []
        self.m_i = []
        for l in range(self.n_moments+1):
            self.m_r.append(self.g.complex_array())
            self.m_i.append(self.g.complex_array())

    @jit
    def compute_expansion(self, rho):
        # rho is density that lives on a grid self.g

        # loop over cells
        for i in range(self.g.nr):
            for j in range(self.g.nz):
                radius = np.sqrt((self.g.r[i] - self.center[0])**2 +
                                 (self.g.z[j] - self.center[1])**2)
                # tan(theta) = r/z
                theta = np.arctan2(self.g.r[i], self.g.z[j])
                m_zone = rho[i, j] * self.g.vol[i, j]
                # loop over the multipole moments, l (m = 0 here)
                for l in range(self.n_moments+1):
                    # compute Y_l^m (note: we use theta as the polar
                    # angle, scipy is opposite)
                    Y_lm = sph_harm(0, l, 0.0, theta)

                    R_lm = np.sqrt(4*np.pi/(2*l + 1)) * radius**l * Y_lm
                    I_lm = np.sqrt(4*np.pi/(2*l + 1)) * Y_lm / radius**(l+1)

                    # assign to the the  inner or outer moment functions
                    self.m_r[l][i, j] = R_lm * m_zone
                    self.m_i[l][i, j] = I_lm * m_zone

    @jit
    def compute_phi_face(self, dr, dz):
        # return Phi(r), using Eq. 20

        r_face = self.g.r2d + dr
        z_face = self.g.z2d + dz

        radius = np.sqrt((r_face - self.center[0])**2 +
                         (z_face - self.center[1])**2)

        # tan(theta) = r/z
        theta = np.arctan2(r_face, z_face)

        phi_zone = self.g.complex_array()
        for l in range(self.n_moments+1):

            Y_lm = sph_harm(0, l, 0.0, theta)
            R_lm = np.sqrt(4*np.pi/(2*l + 1)) * radius**l * Y_lm
            I_lm = np.nan_to_num(np.sqrt(4*np.pi/(2*l + 1)) *
                                 Y_lm / radius**(l+1))

            # loop over each cell to assign the face potential

            for i in range(self.g.nr):
                for j in range(self.g.nz):
                    inmask = self.radius <= radius[i, j]
                    outmask = self.radius > radius[i, j]
                    # equation 20
                    # buggy
                    phi_zone[i, j] += sc.G * (np.sum(self.m_r[l][inmask]) *
                                              np.conj(I_lm[i, j]) +
                                              np.conj(np.sum(
                                                  self.m_i[l][outmask])) *
                                              R_lm[i, j])

        return -np.real(phi_zone)

    @jit
    def phi(self):
        # calculate the potential of a specific point
        dr = self.g.dr/2
        dz = self.g.dz/2

        phi = self.g.scratch_array()

        # phi_0 = self.compute_phi(r, z, 0, 0)

        phi_m_r = self.compute_phi_face(-dr, 0)
        phi_m_z = self.compute_phi_face(0, -dz)
        phi_p_r = self.compute_phi_face(dr, 0)
        phi_p_z = self.compute_phi_face(0, dz)

        # phi = (phi_m_r+phi_m_z+phi_p_r+phi_p_z)/4

        # phi = 1/total_area(sum(phi(x_face)*area))
        # compute the areas in different direction

        area_m_r = self.g.scratch_array()
        area_m_z = self.g.scratch_array()
        area_p_r = self.g.scratch_array()
        area_p_z = self.g.scratch_array()
        total_area = self.g.scratch_array()

        for i in range(self.g.nr):
            for j in range(self.g.nz):
                area_m_r[i, j] = 2*np.pi*(self.g.r2d[i, j]-dr)*dz
                area_m_z[i, j] = np.pi*((self.g.r2d[i, j]+dr)**2 -
                                        (self.g.r2d[i, j]-dr)**2)
                area_p_r[i, j] = 2*np.pi*(self.g.r2d[i, j]+dr)*dz
                area_p_z[i, j] = np.pi*((self.g.r2d[i, j]+dr)**2 -
                                        (self.g.r2d[i, j]-dr)**2)
                total_area[i, j] = area_m_r[i, j]+area_m_z[i, j] +\
                    area_p_r[i, j]+area_p_z[i, j]

        phi_m_r_area = phi_m_r*area_m_r
        phi_m_z_area = phi_m_z*area_m_z
        phi_p_r_area = phi_p_r*area_p_r
        phi_p_z_area = phi_p_z*area_p_z
        phi = (phi_m_r_area+phi_m_z_area+phi_p_r_area+phi_p_z_area)/total_area

        return phi
