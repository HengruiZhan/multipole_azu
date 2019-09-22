import numpy as np


class Grid():
    def __init__(self, nr, nz, rlim=(0.0, 1.0), zlim=(-1.0, 1.0)):
        """an axisymmetric grid"""
        """
        self.nr = nr
        self.nz = nz

        self.rlim = rlim
        self.zlim = zlim

        self.dr = (self.rlim[1] - self.rlim[0])/self.nr
        self.dz = (self.zlim[1] - self.zlim[0])/self.nz

        # coordinates of the grid cell center
        self.r = (np.arange(self.nr) + 0.5)*self.dr + self.rlim[0]
        # self.r2d = np.repeat(self.r, self.nz).reshape(self.nr, self.nz)

        self.z = (np.arange(self.nz) + 0.5)*self.dz + self.zlim[0]
        # self.z2d = np.transpose(np.repeat(self.z, self.nr).reshape(self.nz,
        #                                                            self.nr))
        """

        self.nr = nr - 1
        self.nz = nz - 1

        """
        self.nr = nr
        self.nz = nz
        """

        self.rlim = rlim
        self.zlim = zlim

        self.dr = (self.rlim[1] - self.rlim[0])/nr
        self.dz = (self.zlim[1] - self.zlim[0])/nz

        # self.r = (np.arange(self.nr) + 0.5)*self.dr + self.rlim[0]
        self.r = np.linspace(self.rlim[0]+self.dr/2, self.rlim[1]-self.dr/2,
                             self.nr)

        # self.z = (np.arange(self.nz) + 0.5)*self.dz + self.zlim[0]
        self.z = np.linspace(self.zlim[0]+self.dz/2, self.zlim[1]-self.dz/2,
                             self.nz)

        self.r2d, self.z2d = np.meshgrid(self.r, self.z, indexing='ij')

        self.vol = np.pi*2.0*self.r2d*self.dr*self.dz

    """
    def set_bin(self, inner, inner_max, s, t):
        # inner is used to dtermine the bin size
        # in terms of atomic radial measure
        # s and t are used to determined the outer radial m_zone
        # see section 8.10.2.2 of the Flash User's Guide
        # Multipole Poisson solver (improved version)
        self.bin = []
        self.bin[0] = inner*dr
        while
        """

    def scratch_array(self):
        return np.zeros((self.nr, self.nz), dtype=np.float64)
