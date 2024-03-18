import numpy as np
import matplotlib.pyplot as plt
from time import time
from morty_pde_ode_compare import FormatAssist
import os


class IsobarSolve(FormatAssist):
    def __init__(self, nodes, z1, z2, nu1, nu2, lmbda, tf, lams,
                 FYs, br_c_d, br_dm1_d, vol1, vol2, losses):
        """
        This class allows for the solve of a system of PDEs by
        solving each individually in a Jacobi-like manner.
        This approach will provide a more accurate result,
        as contributions from other nuclides in the isobar
        will be included.
        The next step up from this approach is to incorporate
        the spatial solve within the depletion solver itself.

        Parameters
        ----------
        nodes : int
            Number of spatial nodes
        z1 : float
            Position of in-core to ex-core transition
        z2 : float
            Position of ex-core to in-core transition
        nu1 : float
            Velocity in-core
        nu2 : float
            Velocity ex-core
        lmbda : float
            Time step times velocity divized by spatial mesh size
        tf : float
            Final time
        lams : dict
            key : str
                Isobar nuclide indicator (a, b, c, d, or d_m1)
            value : float
                Decay constant for given nuclide
        FYs : dict
            key : string
                Isobar nuclide indicator (a, b, c, d, or d_m1)
            value : float
                Fission yield for nuclice
        br_c_d : float
            Branching ratio for nuclide "c" to "d"
        br_dm1_d : float
            Branching ratio for nuclide "dm1" to "d"
        vol1 : float
            Volume of in-core region
        vol2 : float
            Volume of ex-core region
        losses : dict
            key : string
                Isobar nuclide indicator (a, b, c, d, or d_m1)
            value : float
                Loss term due to parasitic absorption or other similar terms

        """
        self.nodes = nodes
        self.z1 = z1
        self.zs = np.linspace(0, z1+z2, nodes)
        self.dt = lmbda * dz / nu1
        self.ts = np.arange(0, tf+self.dt, self.dt)
        self.nu_vec = self._format_spatial(nu1, nu2)

        self.mu = {}
        self.mu['a'] = self._format_spatial((lams['a'] + losses['1a']),
                                            (lams['a'] + losses['2a']))
        self.mu['b'] = self._format_spatial((lams['b'] + losses['1b']),
                                            (lams['b'] + losses['2b']))
        self.mu['c'] = self._format_spatial((lams['c'] + losses['1c']),
                                            (lams['c'] + losses['2c']))
        self.mu['d_m1'] = self._format_spatial((lams['d_m1']
                                                + losses['1d_m1']),
                                               (lams['d_m1'] + losses['2d']))
        self.mu['d'] = self._format_spatial((lams['d'] + losses['1d']),
                                            (lams['d'] + losses['2d']))
        self.S = {}
        self.S['a'] = self._format_spatial((FYs['a']/vol1), (0/vol2))

        self.lama = lams['a']
        self.lamb = lams['b']
        self.lamc = lams['c']
        self.lamd = lams['d']
        self.lamd_m1 = lams['d_m1']

        self.FYb = FYs['b']
        self.FYc = FYs['c']
        self.FYd = FYs['d']
        self.FYd_m1 = FYs['d_m1']

        self.br_c_d = br_c_d
        self.br_dm1_d = br_dm1_d

        self.vol1 = vol1
        self.vol2 = vol2

        return

    def _external_PDE_no_step(self, conc, isotope):
        """
        This function applies a single time step iteration of the PDE

        Parameters
        ----------
        conc : 1D vector
            Concentration over spatial nodes at previous time
        isotope : string
            Nuclide isobar indicator (a, b, c, d, or d_m1)

        Returns
        -------
        conc : 1D vector
            Concentration over spatial nodes at current time
        """
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

    def _initialize_result_mat(self, PDE=True):
        """
        Set up the 3D result matrix with the form
            time, space, nuclide
            with 5 nucldies in the isobar available

        Parameters
        ----------
        PDE : bool
            If using PDE True, False for ODE

        Returns
        -------
        result_mat : 3D matrix (2D if PDE False)
            Holds values over time, space, and nuclide (in that order)

        """
        if PDE:
            nodes = self.nodes
        else:
            nodes = 1
        result_mat = np.zeros((len(self.ts), nodes, 5))
        self.conc_a = np.array([0] * nodes)
        result_mat[0, :, 0] = self.conc_a
        self.conc_b = np.array([0] * nodes)
        result_mat[0, :, 1] = self.conc_b
        self.conc_c = np.array([0] * nodes)
        result_mat[0, :, 2] = self.conc_c
        self.conc_d_m1 = np.array([0] * nodes)
        result_mat[0, :, 3] = self.conc_d_m1
        self.conc_d = np.array([0] * nodes)
        result_mat[0, :, 4] = self.conc_d
        return result_mat

    def _update_sources(self, PDE=True):
        """
        Update source terms based on concentrations

        Parameters
        ----------
        PDE : bool
            True if PDE, False if ODE

        """
        if PDE:
            vector_form = True
        else:
            vector_form = False
        self.S['b'] = self._format_spatial((self.conc_a*self.lama
                                            + self.FYb/self.vol1),
                                           (self.conc_a*self.lama),
                                           vector_form=vector_form)
        self.S['c'] = self._format_spatial((self.conc_b*self.lamb
                                            + self.FYc/self.vol1),
                                           (self.conc_b*self.lama),
                                           vector_form=vector_form)
        self.S['d_m1'] = self._format_spatial((self.FYd_m1/self.vol1),
                                              (0/self.vol2))
        self.S['d'] = self._format_spatial((self.br_c_d*self.conc_c*self.lamc
                                            + self.FYd/self.vol1
                                            + self.br_dm1_d
                                            * self.conc_d_m1
                                            * self.lamd_m1),
                                           (self.br_c_d*self.conc_c*self.lamc
                                            + self.br_dm1_d
                                            * self.conc_d_m1
                                            * self.lamd_m1),
                                           vector_form=vector_form)
        return

    def _update_result_mat(self, result_mat, ti):
        """
        Updates the result matrix with new concentrations

        Parameters
        ----------
        result_mat : 3D matrix
            Holds values over time, space, and nuclide (in that order)
        ti : int
            Current time index

        Returns
        -------
        result_mat : 3D matrix
            Holds values over time, space, and nuclide (in that order)
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

        Returns
        -------
        result_mat : 3D matrix
            Holds values over time, space, and nuclide (in that order)

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

    def _external_ODE_no_step(self, conc, isotope, t):
        """
        This function applies a single time step iteration of the ODE

        Parameters
        ----------
        conc : float
            Initial concentration
        isotope : string
            Nuclide isobar indicator (a, b, c, d, or d_m1)

        Returns
        -------
        conc : float
            Concentration at current time
        """

        conc = (conc * np.exp(-self.mu[isotope][0] * t) +
                self.S[isotope][0] / self.mu[isotope][0] *
                (1 - np.exp(-self.mu[isotope][0] * t)))

        return conc

    def ode_solve(self):
        """
        Solve the time dependent ODE

        """
        ODE_result_mat = self._initialize_result_mat(False)
        conc0_a = self.conc_a
        conc0_b = self.conc_b
        conc0_c = self.conc_c
        conc0_d = self.conc_d
        conc0_d_m1 = self.conc_d_m1

        for ti, t in enumerate(ts[:-1]):
            self._update_sources(False)

            self.conc_a = self._external_ODE_no_step(conc0_a, 'a', t)
            self.conc_b = self._external_ODE_no_step(conc0_b, 'b', t)
            self.conc_c = self._external_ODE_no_step(conc0_c, 'c', t)
            self.conc_d_m1 = self._external_ODE_no_step(conc0_d_m1, 'd_m1', t)
            self.conc_d = self._external_ODE_no_step(conc0_d, 'd', t)

            ODE_result_mat = self._update_result_mat(ODE_result_mat, ti)

        return ODE_result_mat

    def parallel_MORTY_solve(self):
        """
        Run MORTY, calculating the isobar concentrations in-core and ex-core
            for the given problem.

        Returns
        -------
        result_mat : 3D matrix
            Holds values over time, space, and nuclide (in that order)

        """
        import multiprocessing
        result_mat = self._initialize_result_mat()
        for ti, t in enumerate(ts[:-1]):
            # Parallelization is easy due to Jacobi appraoch
            #   Each isobar is independent from the others at each iteration

            self._update_sources()

            with multiprocessing.Pool() as pool:
                res_list = pool.starmap(self._external_PDE_no_step,
                                        [(self.conc_a, 'a'),
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
    gif = False
    ode = True
    scaled_flux = True
    savedir = './images'
    tf = 1.25 * 24 * 3600
    spacenodes = 100

    L = 608.06  # 824.24
    V = 2116111
    frac_in = 0.33
    frac_out = 0.67
    core_outlet_node = int(spacenodes * frac_in)
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
    dz = np.diff(np.linspace(0, z1+z2, spacenodes))[0]
    lmbda = 0.9
    dt = lmbda * dz / nu1
    ts = np.arange(0, tf+dt, dt)
    print(f'Number of iterations: {len(ts)}')
    isotopea = 'Sb135'
    isotopeb = 'Te135'
    isotopec = 'I135'
    isotoped_m1 = 'Xe135_m1'
    isotoped = 'Xe135'
    br_c_d = 0.8349109
    br_dm1_d = 0.997
    lams = {}
    lams['a'] = np.log(2) / 1.68
    lams['b'] = np.log(2) / 19
    lams['c'] = np.log(2) / (6.57*3600)
    lams['d'] = np.log(2) / (15.29*3600)
    lams['d_m1'] = np.log(2) / (9.14*3600)

    zs = np.linspace(0, z1+z2, spacenodes)

    PC = 1 / (3.2e-11)
    P = 8e6  # 8MW
    # Yields from ENDF OpenMC thermal data
    Ya = 0.00145764
    Yb = 0.0321618
    Yc = 0.0292737
    Yd_m1 = 0.0110156
    Yd = 0.000785125
    FYs = {}  # atoms/s = fissions/J * J/s * yield_fraction
    FYs['a'] = PC * P * Ya
    FYs['b'] = PC * P * Yb
    FYs['c'] = PC * P * Yc
    FYs['d'] = PC * P * Yd
    FYs['d_m1'] = PC * P * Yd_m1
    phi_th = 6E12
    losses = {}
    losses['1a'] = 0
    losses['2a'] = 0
    losses['1b'] = 0
    losses['2b'] = 0
    losses['2c'] = 0
    losses['2d'] = 0
    ng_I135 = 80.53724E-24
    ng_Xe135 = 2_666_886.8E-24
    ng_Xe135_m1 = 0  # 10_187_238E-24
    losses['1c'] = phi_th * ng_I135
    losses['1d'] = phi_th * ng_Xe135
    losses['1d_m1'] = phi_th * ng_Xe135_m1

    start = time()
    solver = IsobarSolve(spacenodes, z1, z2, nu1, nu2, lmbda, tf,
                         lams, FYs, br_c_d, br_dm1_d, vol1,
                         vol2, losses)
    if parallel:
        result_mat = solver.parallel_MORTY_solve()
    else:
        result_mat = solver.serial_MORTY_solve()
    if ode:
        if scaled_flux:
            P = 8e6 * frac_in
            FYs['a'] = PC * P * Ya
            FYs['b'] = PC * P * Yb
            FYs['c'] = PC * P * Yc
            FYs['d'] = PC * P * Yd
            FYs['d_m1'] = PC * P * Yd_m1
            phi_th = 6E12 * frac_in
            losses['1c'] = phi_th * ng_I135
            losses['1d'] = phi_th * ng_Xe135
            losses['1d_m1'] = phi_th * ng_Xe135_m1
            solver = IsobarSolve(spacenodes, z1, z2, nu1, nu2, lmbda, tf,
                                 lams, FYs, br_c_d, br_dm1_d, vol1,
                                 vol2, losses)

        ode_result_mat = solver.ode_solve()
    end = time()
    print(f'Time taken : {round(end-start)}s')

    # Plotting
    savedir = './images'
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    if tf > 3600 * 24:
        ts = ts / (3600*24)
        units = 'd'
    else:
        units = 's'
    labels = [isotopea, isotopeb, isotopec, isotoped_m1, isotoped]
    for i, iso in enumerate(labels):
        plt.plot(ts[:-2], result_mat[0:-2, core_outlet_node, i],
                 label=f'{iso} Exiting Core')
        plt.plot(ts[:-2], result_mat[0:-2, -1, i],
                 label=f'{iso} Entering Core')
        if ode:
            plt.plot(ts[:-2], ode_result_mat[0:-2, 0, i],
                     label=f'{iso} ODE', linestyle='--')

        plt.xlabel(f'Time [{units}]')
        plt.ylabel('Concentration [at/cc]')
        plt.yscale('log')
        plt.legend()
        plt.savefig(f'{savedir}/nuc_{iso}_conc_time.png')
        plt.close()

    if ode:
        for i, iso in enumerate(labels):
            print('-' * 50)
            print(f'{iso} atom densities')
            PDE_val_core_inlet = result_mat[-2, 0, i]
            print(f'PDE core inlet: {PDE_val_core_inlet}')
            ODE_val = ode_result_mat[-2, 0, i]
            print(f'ODE {ODE_val}')
            pcnt_diff = ((PDE_val_core_inlet - ODE_val) /
                         (PDE_val_core_inlet) * 100)
            print(f'{iso} PDE/ODE diff: {round(pcnt_diff, 3)}%')

    # Gif
    if gif:
        print(f'Estimated time to gif completion: {round(0.08 * len(ts))} s')
        start = time()
        from matplotlib.animation import FuncAnimation
        fig, ax = plt.subplots()
        max_conc = np.max(result_mat[0:-2, :, :])

        def update(frame):
            ax.clear()
            plt.xlabel('Space [cm]')
            plt.vlines(z1, 0, 1e1 * max_conc, color='black')
            plt.ylabel('Concentration [at/cc]')
            plt.ylim((1e-5 * max_conc, 1e1 * max_conc))
            plt.yscale('log')

            for i, iso in enumerate(labels):
                ax.plot(zs, result_mat[frame, :, i],
                        label=f'{iso}', marker='.')
            ax.set_title(f'Time: {round(frame*dt, 4)} s')
            plt.legend()
        animation = FuncAnimation(fig, update, frames=len(ts), interval=1)
        animation.save(f'{savedir}/isobar_evolution.gif', writer='pillow')
        plt.close()
        print(f'Gif took {time() - start} s')
