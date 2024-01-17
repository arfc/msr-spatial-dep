import numpy as np
import matplotlib.pyplot as plt
import time

def format_spatial(zs, z1, term1, term2, vector_form=False):
    """
    Distribute term 1 in < z1, and term 2 in > z1.
    Returns a list of terms corresponding to each z. 
    If term1 and term2 are vectors, use `vector_form=True`
    """
    return_list = []
    for zi, z in enumerate(zs):
        if z < z1:
            if not vector_form:
               return_list.append(term1)
            else:
               return_list.append(term1[zi])
        elif z > z1:
            if not vector_form:
               return_list.append(term2)
            else:
               return_list.append(term2[zi])
        else:
            if not vector_form:
               return_list.append(0.5 * (term1 + term2))
            else:
               return_list.append(0.5 * (term1[zi] + term2[zi]))
    return return_list
          

def external_PDE_no_step(conc, isotope, dt, mu, S, nu_vec, z1, z2, nodes=100):
    zs = np.linspace(0, z1+z2, nodes)
    S_vec = S[isotope]
    nu_vec = nu_vec
    mu_vec = mu[isotope]
    J = np.arange(0, nodes)
    Jm1 = np.roll(J,  1)
    dz = np.diff(zs)[0]

    S_vec = np.asarray(S_vec)
    nu_vec = np.asarray(nu_vec)
    mu_vec = np.asarray(mu_vec)
    conc = np.asarray(conc)

    conc_mult = 1 - mu_vec * dt
    add_source = S_vec * dt
    lmbda = (nu_vec * dt / dz)
    conc = add_source + conc_mult * conc + lmbda * (conc[Jm1] - conc)

    return conc



def serial_MORTY_solve(tf, dt, spacenodes, mu, S, nu_vec, z1, z2, lama, lamb,
                       lamc, lamd_m1, FYb, FYc, FYd_m1, FYd, br_c_d, br_dm1_d,
                       vol1, vol2, format_spatial, external_PDE_no_step):
    """
    Run MORTY, calculating the isobar concentrations in-core and ex-core
         for the given problem.
    
    """
    ts = np.arange(0, tf+dt, dt)
    zs = np.linspace(0, z1+z2, spacenodes)
    result_mat = np.zeros((len(ts), spacenodes, 5))
    conc_a = np.array([0] * spacenodes)
    result_mat[0, :, 0] = conc_a
    conc_b = np.array([0] * spacenodes)
    result_mat[0, :, 1] = conc_b
    conc_c = np.array([0] * spacenodes)
    result_mat[0, :, 2] = conc_c
    conc_d_m1 = np.array([0] * spacenodes)
    result_mat[0, :, 3] = conc_d_m1
    conc_d = np.array([0] * spacenodes)
    result_mat[0, :, 4] = conc_d
    for ti, t in enumerate(ts[:-1]):
 
         S['b'] = format_spatial(zs, z1, (conc_a*lama + FYb/vol1), (conc_a*lama), vector_form=True)
         S['c'] = format_spatial(zs, z1, (conc_b*lamb + FYc/vol1), (conc_b*lama), vector_form=True) 
         S['d_m1'] = format_spatial(zs, z1, (FYd_m1/vol1), (0/vol2)) 
         S['d'] = format_spatial(zs, z1, (br_c_d*conc_c*lamc + FYd/vol1 + br_dm1_d*conc_d_m1*lamd_m1),
                                         (br_c_d*conc_c*lamc + br_dm1_d*conc_d_m1*lamd_m1), vector_form=True) 

         conc_a = external_PDE_no_step(conc_a, 'a', dt, mu, S, nu_vec, z1, z2, nodes=spacenodes)
         conc_b = external_PDE_no_step(conc_b, 'b', dt, mu, S, nu_vec, z1, z2, nodes=spacenodes)
         conc_c = external_PDE_no_step(conc_c, 'c', dt, mu, S, nu_vec, z1, z2, nodes=spacenodes)
         conc_d_m1 = external_PDE_no_step(conc_d_m1, 'd_m1', dt, mu, S, nu_vec, z1, z2, nodes=spacenodes)
         conc_d = external_PDE_no_step(conc_d, 'd', dt, mu, S, nu_vec, z1, z2, nodes=spacenodes)
 
 
         result_mat[ti, :, 0] = conc_a
         result_mat[ti, :, 1] = conc_b
         result_mat[ti, :, 2] = conc_c
         result_mat[ti, :, 3] = conc_d_m1
         result_mat[ti, :, 4] = conc_d
    return result_mat


def parallel_MORTY_solve(tf, dt, spacenodes, mu, S, nu_vec, z1, z2, lama, lamb,
                       lamc, lamd_m1, FYb, FYc, FYd_m1, FYd, br_c_d, br_dm1_d,
                       vol1, vol2, format_spatial, external_PDE_no_step):
    """
    Run MORTY, calculating the isobar concentrations in-core and ex-core
         for the given problem.
    
    """
    import multiprocessing
    ts = np.arange(0, tf+dt, dt)
    result_mat = np.zeros((len(ts), spacenodes, 5))
    conc_a = np.array([0] * spacenodes)
    result_mat[0, :, 0] = conc_a
    conc_b = np.array([0] * spacenodes)
    result_mat[0, :, 1] = conc_b
    conc_c = np.array([0] * spacenodes)
    result_mat[0, :, 2] = conc_c
    conc_d_m1 = np.array([0] * spacenodes)
    result_mat[0, :, 3] = conc_d_m1
    conc_d = np.array([0] * spacenodes)
    result_mat[0, :, 4] = conc_d
    zs = np.linspace(0, z1+z2, spacenodes)
    for ti, t in enumerate(ts[:-1]):
         # Parallelization is easy due to Jacobi appraoch
         #   Each isobar is independent from the others at each iteration
 
         S['b'] = format_spatial(zs, z1, (conc_a*lama + FYb/vol1), (conc_a*lama), vector_form=True)
         S['c'] = format_spatial(zs, z1, (conc_b*lamb + FYc/vol1), (conc_b*lama), vector_form=True) 
         S['d_m1'] = format_spatial(zs, z1, (FYd_m1/vol1), (0/vol2)) 
         S['d'] = format_spatial(zs, z1, (br_c_d*conc_c*lamc + FYd/vol1 + br_dm1_d*conc_d_m1*lamd_m1),
                                         (br_c_d*conc_c*lamc + br_dm1_d*conc_d_m1*lamd_m1), vector_form=True) 


         with multiprocessing.Pool() as pool:
             res_list = pool.starmap(external_PDE_no_step, [(conc_a, 'a', dt, mu, S, nu_vec, z1, z2, spacenodes),
                                                 (conc_b, 'b', dt, mu, S, nu_vec, z1, z2, spacenodes),
                                                 (conc_c, 'c', dt, mu, S, nu_vec, z1, z2, spacenodes),
                                                 (conc_d_m1, 'd_m1', dt, mu, S, nu_vec, z1, z2, spacenodes),
                                                 (conc_d, 'd', dt, mu, S, nu_vec, z1, z2, spacenodes)])
         conc_a = res_list[0]
         conc_b = res_list[1]
         conc_c = res_list[2]
         conc_d_m1 = res_list[3]
         cond_d = res_list[4]

         #conc_a = external_PDE_no_step(conc_a, 'a', dt, mu, S, nu_vec, z1, z2, nodes=spacenodes)
         #conc_b = external_PDE_no_step(conc_b, 'b', dt, mu, S, nu_vec, z1, z2, nodes=spacenodes)
         #conc_c = external_PDE_no_step(conc_c, 'c', dt, mu, S, nu_vec, z1, z2, nodes=spacenodes)
         #conc_d_m1 = external_PDE_no_step(conc_d_m1, 'd_m1', dt, mu, S, nu_vec, z1, z2, nodes=spacenodes)
         #conc_d = external_PDE_no_step(conc_d, 'd', dt, mu, S, nu_vec, z1, z2, nodes=spacenodes)
 
 
         result_mat[ti, :, 0] = conc_a
         result_mat[ti, :, 1] = conc_b
         result_mat[ti, :, 2] = conc_c
         result_mat[ti, :, 3] = conc_d_m1
         result_mat[ti, :, 4] = conc_d
    return result_mat



if __name__ == '__main__':
    # Test this module using MSRE 135 isobar
    parallel = False
    L = 824.24
    V = 2e6
    frac_in = 0.33
    frac_out = 0.67
    z1 = frac_in * L
    z2 = frac_out * L
    vol1 = frac_in * V
    vol2 = frac_out * V
    nu1 = 66666.66
    nu2 = 66666.66
    loss_core = 6e12 * 2666886.8E-24
    #isotope = 'test'
    #mu = {'test': [2.1065742176025568e-05 + loss_core, 2.1065742176025568e-05]}
    #S = {'test': [24568909090.909092, 0]}
    #initial_guess = [0, 0]
    spacenodes = 100
    dz = np.diff(np.linspace(0, z1+z2, spacenodes))[0]
    lmbda = 0.9
    dt = lmbda * dz / nu1
    tf = 0.1 * 0.1
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

   # S = {'a': [FYa/vol1, 0/vol2]}

#    mu = {
#      'a': [lama + r_1a, lama + r_2a],
#      'b': [lamb + r_1b, lamb + r_2b],
#      'c': [lamc + r_1c, lamc + r_2c],
#      'd_m1': [lamd_m1 + r_1d, lamd_m1 + r_2d],
#      'd': [lamd + r_1d, lamd + r_2d],
#    }
    nu_vec = format_spatial(zs, z1, nu1, nu2)
    mu = {}
    mu['a'] = format_spatial(zs, z1, (lama + r_1a), (lama + r_2a))
    mu['b'] = format_spatial(zs, z1, (lamb + r_1b), (lamb + r_2b))
    mu['c'] = format_spatial(zs, z1, (lamc + r_1c), (lamb + r_2c))
    mu['d_m1'] = format_spatial(zs, z1, (lamd_m1 + r_1d_m1), (lamd_m1 + r_2d))
    mu['d'] = format_spatial(zs, z1, (lamd + r_1d), (lamd + r_2d))
    S = {}
    S['a'] = format_spatial(zs, z1, (FYa/vol1), (0/vol2))

    start = time.time()
    if parallel:
        result_mat = parallel_MORTY_solve(tf, dt, spacenodes, mu, S, nu_vec, z1, z2,
                                     lama, lamb, lamc, lamd_m1, FYb, FYc, FYd_m1,
                                     FYd, br_c_d, br_dm1_d, vol1, vol2, format_spatial,
                                     external_PDE_no_step)
    else:
        result_mat = serial_MORTY_solve(tf, dt, spacenodes, mu, S, nu_vec, z1, z2,
                                     lama, lamb, lamc, lamd_m1, FYb, FYc, FYd_m1,
                                     FYd, br_c_d, br_dm1_d, vol1, vol2, format_spatial,
                                     external_PDE_no_step)
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