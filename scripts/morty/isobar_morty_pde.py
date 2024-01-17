import numpy as np
import matplotlib.pyplot as plt
import time
from morty_pde_ode_compare import FormatAssist


class IsobarSolve(FormatAssist):
    def __init__(self, nodes, z1, z2, nu1, nu2, dt, tf, lama, lamb, lamc,
                 lamd, lamd_m1, FYd, br_c_d, br_dm1_d, vol1, vol2, r_1a,
                 r_2a, r_1b, r_2b, r_1c, r_2c, r_1d_m1, r_2d, r_1d, FYb,
                 FYc, FYd_m1):
        """
        This class allows for the solve of a system of PDEs by 
        solving each individually in a Jacobi-like manner.
        This approach will provide a more accurate result,
        as contributions from other nuclides in the isobar
        will be included.
        The next step up from this approach is to incorporate
        the spatial solve within the depletion solver itself.

        """
        self.nodes = nodes
        self.z1 = z1
        self.zs = np.linspace(0, z1+z2, nodes)
        self.dt = dt
        self.ts = np.arange(0, tf+dt, dt)
        self.nu_vec = self._format_spatial(nu1, nu2)

        self.mu = {}
        self.mu['a'] = self._format_spatial((lama + r_1a), (lama + r_2a))
        self.mu['b'] = self._format_spatial((lamb + r_1b), (lamb + r_2b))
        self.mu['c'] = self._format_spatial((lamc + r_1c), (lamb + r_2c))
        self.mu['d_m1'] = self._format_spatial((lamd_m1 + r_1d_m1), (lamd_m1 + r_2d))
        self.mu['d'] = self._format_spatial((lamd + r_1d), (lamd + r_2d))
        self.S = {}
        self.S['a'] = self._format_spatial((FYa/vol1), (0/vol2))

        self.lama = lama
        self.lamb = lamb
        self.lamc = lamc
        self.lamd = lamd
        self.lamd_m1 = lamd_m1

        self.FYb = FYb
        self.FYc = FYc
        self.FYd = FYd
        self.FYd_m1 = FYd_m1

        self.br_c_d = br_c_d
        self.br_dm1_d = br_dm1_d
        
        self.vol1 = vol1
        self.vol2 = vol2

        return

    def _external_PDE_no_step(self, conc, isotope):
        S_vec = self.S[isotope]
        mu_vec = self.mu[isotope]
        J = np.arange(0, self.nodes)
        Jm1 = np.roll(J,  1)
        dz = np.diff(self.zs)[0]

        conc_mult = 1 - mu_vec * self.dt
        add_source = S_vec * self.dt
        lmbda = (self.nu_vec * self.dt / dz)
        conc = add_source + conc_mult * conc + lmbda * (conc[Jm1] - conc)

        return conc

    def _initialize_result_mat(self):
        """
        Set up the 3D result matrix of the form
            time, space, nuclide
            with 5 nucldies available

        """
        result_mat = np.zeros((len(self.ts), self.nodes, 5))
        self.conc_a = np.array([0] * self.nodes)
        result_mat[0, :, 0] = self.conc_a
        self.conc_b = np.array([0] * self.nodes)
        result_mat[0, :, 1] = self.conc_b
        self.conc_c = np.array([0] * self.nodes)
        result_mat[0, :, 2] = self.conc_c
        self.conc_d_m1 = np.array([0] * self.nodes)
        result_mat[0, :, 3] = self.conc_d_m1
        self.conc_d = np.array([0] * self.nodes)
        result_mat[0, :, 4] = self.conc_d
        return result_mat
    
    def _update_sources(self):
        """
        Update source terms based on concentrations

        """
        self.S['b'] = self._format_spatial((self.conc_a*self.lama + self.FYb/self.vol1), (self.conc_a*self.lama), vector_form=True)
        self.S['c'] = self._format_spatial((self.conc_b*self.lamb + self.FYc/self.vol1), (self.conc_b*self.lama), vector_form=True) 
        self.S['d_m1'] = self._format_spatial((self.FYd_m1/self.vol1), (0/self.vol2)) 
        self.S['d'] = self._format_spatial((self.br_c_d*self.conc_c*self.lamc + self.FYd/self.vol1 + self.br_dm1_d*self.conc_d_m1*self.lamd_m1),
                                        (self.br_c_d*self.conc_c*self.lamc + self.br_dm1_d*self.conc_d_m1*self.lamd_m1), vector_form=True) 
        return

    def _update_result_mat(self, result_mat, ti):
        """
        Updates the result matrix with new concentrations
        """
        result_mat[ti, :, 0] = self.conc_a
        result_mat[ti, :, 1] = self.conc_b
        result_mat[ti, :, 2] = self.conc_c
        result_mat[ti, :, 3] = self.conc_d_m1
        result_mat[ti, :, 4] = self.conc_d
        return result_mat


    def serial_MORTY_solve(self):
        """
        Run MORTY, calculating the isobar concentrations in-core and ex-core
            for the given problem.
        
        """
        result_mat = self._initialize_result_mat()
        for ti, t in enumerate(ts[:-1]):
    
            self._update_sources()

            self.conc_a = self._external_PDE_no_step(self.conc_a, 'a')
            self.conc_b = self._external_PDE_no_step(self.conc_b, 'b')
            self.conc_c = self._external_PDE_no_step(self.conc_c, 'c')
            self.conc_d_m1 = self._external_PDE_no_step(self.conc_d_m1, 'd_m1')
            self.conc_d = self._external_PDE_no_step(self.conc_d, 'd')
    
            result_mat = self._update_result_mat(result_mat, ti)
    
        return result_mat


    def parallel_MORTY_solve(self):
        """
        Run MORTY, calculating the isobar concentrations in-core and ex-core
            for the given problem.
        
        """
        import multiprocessing
        result_mat = self._initialize_result_mat()
        for ti, t in enumerate(ts[:-1]):
            # Parallelization is easy due to Jacobi appraoch
            #   Each isobar is independent from the others at each iteration

            self._update_sources()

            with multiprocessing.Pool() as pool:
                res_list = pool.starmap(self._external_PDE_no_step, [(self.conc_a, 'a'),
                                                    (self.conc_b, 'b'),
                                                    (self.conc_c, 'c'),
                                                    (self.conc_d_m1, 'd_m1'),
                                                    (self.conc_d, 'd')])
            self.conc_a = res_list[0]
            self.conc_b = res_list[1]
            self.conc_c = res_list[2]
            self.conc_d_m1 = res_list[3]
            self.cond_d = res_list[4]

            result_mat = self._update_result_mat(result_mat, ti)

        return result_mat



if __name__ == '__main__':
    # Test this module using MSRE 135 isobar
    parallel = False
    L = 200.66 #824.24
    V = 2e6
    frac_in = 0.33
    frac_out = 0.67
    z1 = frac_in * L
    z2 = frac_out * L
    vol1 = frac_in * V
    vol2 = frac_out * V
    core_diameter = 140.335
    fuel_frac = 0.225
    xsarea = fuel_frac * (np.pi * (core_diameter/2)**2)
    nu = 75708 / xsarea
    nu1 = nu
    nu2 = nu
    loss_core = 6e12 * 2666886.8E-24
    #isotope = 'test'
    #mu = {'test': [2.1065742176025568e-05 + loss_core, 2.1065742176025568e-05]}
    #S = {'test': [24568909090.909092, 0]}
    #initial_guess = [0, 0]
    spacenodes = 100
    dz = np.diff(np.linspace(0, z1+z2, spacenodes))[0]
    lmbda = 0.9
    dt = lmbda * dz / nu1
    tf = 10
    ts = np.arange(0, tf+dt, dt)
    print(f'Number of iterations: {len(ts)}')
    isotopea = 'Sb135'
    isotopeb = 'Te135'
    isotopec = 'I135'
    isotoped_m1 = 'Xe135_m1'
    isotoped = 'Xe135'
    br_c_d = 0.8349109
    br_dm1_d = 0.997
    lama = np.log(2) / 1.68
    lamb = np.log(2) / (19)
    lamc = np.log(2) / (6.57 * 3600)
    lamd_m1 = np.log(2) / (15.29 * 60)
    lamd = np.log(2) / (9.14 * 3600)
    zs = np.linspace(0, z1+z2, spacenodes)

    PC = 1 / (3.2e-11)
    P = 8e6 # 8MW
    # Yields from ENDF OpenMC thermal data
    Ya = 0.00145764
    Yb = 0.0321618
    Yc = 0.0292737
    Yd_m1 = 0.0110156
    Yd = 0.000785125
    FYa = PC * P * Ya   # atoms/s = fissions/J * J/s * yield_fraction
    FYb = PC * P * Yb
    FYc = PC * P * Yc
    FYd_m1 = PC * P * Yd_m1
    FYd = PC * P * Yd
    phi_th = 6E12
    r_1a = 0
    r_2a = 0
    r_1b = 0
    r_2b = 0
    r_2c = 0
    r_2d = 0
    ng_I135 = 80.53724E-24
    ng_Xe135 = 2_666_886.8E-24
    ng_Xe135_m1 = 0 #10_187_238E-24
    r_1c = phi_th * ng_I135
    r_1d = phi_th * ng_Xe135
    r_1d_m1 = phi_th * ng_Xe135_m1


    start = time.time()
    solver = IsobarSolve(spacenodes, z1, z2, nu1, nu2, dt, tf, lama, lamb,
                         lamc, lamd, lamd_m1, FYd, br_c_d, br_dm1_d, vol1,
                         vol2, r_1a, r_2a, r_1b, r_2b, r_1c, r_2c, r_1d_m1,
                         r_2d, r_1d, FYb, FYc, FYd_m1)
    if parallel:
        result_mat = solver.parallel_MORTY_solve()
    else:
        result_mat = solver.serial_MORTY_solve()
    end = time.time()
    print(f'Time taken : {round(end-start)}s')
    
    
    # Plotting
    if tf > 3600 * 24:
        ts = ts / (3600*24)
        units = 'd'
    else:
        units = 's'
    labels = [isotopea, isotopeb, isotopec, isotoped_m1, isotoped]
    for i, iso in enumerate(labels):
        plt.plot(ts[:-2], result_mat[0:-2, 0, i], label=f'{iso} Exiting Core')
        plt.plot(ts[:-2], result_mat[0:-2, 1, i], label=f'{iso} Entering Core')
    plt.xlabel(f'Time [{units}]')
    plt.ylabel('Concentration [at/cc]')
    plt.yscale('log')
    plt.legend()
    plt.savefig('conc_time.png')
    plt.close()