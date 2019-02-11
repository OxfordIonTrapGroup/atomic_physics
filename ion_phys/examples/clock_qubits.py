# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 11:38:00 2018

@author: hanley
"""

from ion_phys.atoms.ca_43_p import atom
import ion_phys.common as ip
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
import scipy.optimize as so

#Physical constants
muB = sc.physical_constants['Bohr magneton'][0]
muN = sc.physical_constants['nuclear magneton'][0]
amu = sc.physical_constants['unified atomic mass unit'][0]

level = atom["levels"][ip.Level(n=4, L=0, S=1/2, J=1/2)]

def local_gradient(B, M1, M2):
    ip.init(atom, B+1E-7)
    E = level["E"]/(1E9*sc.h)
    M = level["M"]
    F = level["F"]
    trans_freq_plus = E[np.logical_and(M==M1,F==F1)]- E[np.logical_and(M==M2,F==F2)]

    ip.init(atom, B-1E-7)
    E = level["E"]/(1E9*sc.h)
    M = level["M"]
    F = level["F"]
    trans_freq_minus = E[np.logical_and(M==M1,F==F1)]- E[np.logical_and(M==M2,F==F2)]

    return abs((trans_freq_plus- trans_freq_minus)/2E-9)


F1 = 4
F2 = 3

M1 = np.arange(-F1, F1+1)
M2 = np.arange(-F2, F2+1)

for m_init in M1:
    for m_final in M2:
        if abs(m_final-m_init)>1:
            continue
        else:
            res = so.minimize_scalar(local_gradient, bounds=(0,1),
                                     args = (m_init, m_final), method='bounded',
                                     options = {'xatol':1E-10, 'maxiter':500})
            magic_field =  res.x*1e4
            if magic_field < 1:
                print(m_init, ' --> ', m_final, ': No magic field')
            else:
                print(m_init, ' --> ', m_final, ':  Magic field: ', res.x*1e4, ' Gauss')

