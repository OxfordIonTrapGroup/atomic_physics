import numpy as np
from .wigner import wigner3j

class Rates:
    def __init__(self, ion):
        self.ion = ion

        if ion.Gamma is None:
            ion.calc_Scattering()

    def get_spont(self):
        """ Returns the spontaneous emission matrix. """
        spont = np.zeros((self.ion.num_states, self.ion.num_states))
        ion = self.ion
        I_dim = np.rint(2.0*ion.I+1).astype(int)
        for _, transition in ion.transitions.items():
            A = transition.A
            upper = transition.upper
            lower = transition.lower
            Ju = upper.J
            Jl = lower.J
            Mu = np.arange(-Ju, Ju+1)
            Ml = np.arange(-Jl, Jl+1)
            jdim_u = int(np.rint(2*Ju+1))
            jdim_l = int(np.rint(2*Jl+1))
            jdim = jdim_u + jdim_l

            # calculate scattering rates in the high-field basis so we can
            # forget about nuclear spin
            Gamma_hf = np.zeros((jdim, jdim))
            for i_u in range(jdim_u):
                for q in [-1, 0, 1]:
                    if abs(Mu[i_u] + q) > Jl:
                        continue
                    i_l = np.argwhere(Ml == Mu[i_u]+q)
                    # Gamma_hf[i_l, i_u+jdim_l] = wigner3j(
                    #     Jl, 1, Ju, -(Mu[i_u]+q), q, Mu[i_u])**2
                    Gamma_hf[i_l, i_u+jdim_l] = wigner3j(
                        Jl, 1, Ju, -(Mu[i_u]+q), q, Mu[i_u])
            Gamma_hf *= np.sqrt(A*(2*Ju+1))

            # introduce the (still decoupled) nuclear spin
            Gamma_hf = np.kron(Gamma_hf, np.identity(I_dim))

            # now couple...
            M = np.concatenate((ion.M[ion.slice(lower)],
                                ion.M[ion.slice(upper)]))
            subspace = np.r_[ion.slice(lower), ion.slice(upper)]
            num = len(M)
            Gamma = np.zeros(Gamma_hf.shape)
            VV = np.zeros((num, num))
            for ii in range(num):
                for jj in range(num):
                    VV[ii, jj] = (ion.V[subspace[ii], subspace[jj]])

            # conclusion: there seems to be a sign missing here that
            # would lead to some interference between terms...need to look
            # at the physics next...
            Gamma = np.power(np.abs((VV.T)@Gamma_hf@(VV)),2)
            # print(VV[1+16,17]**2*8)
            # print(np.power(VV[16:,17],2)*8)
            print(np.min(np.min(Gamma_hf)))
            # this would make sense if there were a sign in the scattering
            # amplitude...
            # print(Gamma_hf[1, 16:])
            # print("ho")
            # print((VV.T@Gamma_hf)[1,16:])
            # print(Gamma[1,16:])

            # # low-field test...
            I = ion.I
            from sympy.physics.wigner import wigner_6j
            M = np.concatenate((ion.M[ion.slice(lower)],
                                ion.M[ion.slice(upper)]))
            F = np.concatenate((ion.F[ion.slice(lower)],
                                ion.F[ion.slice(upper)]))
            test = np.zeros((len(M), len(M)))
            l_dim = len(ion.M[ion.slice(lower)])
            u_dim = len(ion.M[ion.slice(upper)])
            for i in range(l_dim):
                for j in range(u_dim):
                    ind_i = i
                    ind_j = j + l_dim
                    M_i = M[ind_i]
                    M_j = M[ind_j]
                    F_i = F[ind_i]
                    F_j = F[ind_j]
                    q = M_i - M_j
                    test[ind_i, ind_j] = A*(
                        (2*Ju+1)*(2*F_i+1)*(2*F_j+1)
                        *(wigner3j(F_i, 1, F_j, -M_i, q, M_j))**2
                        *(wigner_6j(Jl, I, F_i, F_j, 1, Ju)**2))
            from pprint import pprint
            # print("test")
            # pprint(test[1, 16:])
            # # pprint((test-Gamma)[1, :])
            # pprint("F: {}".format(F[16:]))
            # pprint("MF: {}".format(M[16:]))
            pprint("mi ax {}".format(ion.MIax[subspace][:16]))
            pprint("mj ax {}".format(ion.MJax[subspace][:16]))
            # pprint("gamma")
            # pprint(Gamma_hf[0, 16:])
            # pprint(test[1, 17])
            # pprint(Gamma[1, 17])
            # pprint("error {}:".format(np.max(np.max(np.abs(test-Gamma)))))

            # high-field test...
            # M = np.concatenate((ion.M[ion.slice(lower)],
            #                     ion.M[ion.slice(upper)]))
            # MI = np.concatenate((ion.MI[ion.slice(lower)],
            #                      ion.MI[ion.slice(upper)]))
            # MJ = np.concatenate((ion.MJ[ion.slice(lower)],
            #                      ion.MJ[ion.slice(upper)]))
            # test = np.zeros((len(M), len(M)))
            # l_dim = len(ion.M[ion.slice(lower)])
            # u_dim = len(ion.M[ion.slice(upper)])
            # print(l_dim, u_dim)
            # for i in range(l_dim):
            #     for j in range(u_dim):
            #         ind_i = i
            #         ind_j = j + l_dim
            #         if MI[ind_i] != MI[ind_j]:
            #             continue
            #         M_i = MJ[ind_i]
            #         M_j = MJ[ind_j]
            #         q = M_i - M_j
            #         if q not in [-1, 1, 0]:
            #             continue
            #         test[ind_i, ind_j] = A*(2*Ju+1)*(wigner3j(Jl, 1, Ju, -M_i, q, M_j))**2
            # from pprint import pprint
            # pprint(test)
            # pprint(Gamma_hf)

            # pprint("err: {}".format(np.max(np.max(np.abs(test-Gamma_hf)))))

            return Gamma_hf


    def get_stim(self, lasers):
        """ Returns the stimulated emission matrix for a list of lasers. """
        for laser in lasers:
            pass
        return np.zeros((self.ion.num_states, self.ion.num_states))

    def get_tranitions(self, lasers):
        """
        Returns the complete transitions matrix for a given set of lasers.
        """
        return self.get_spont() + self.get_stim(lasers)
