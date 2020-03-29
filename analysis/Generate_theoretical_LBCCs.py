"""
Generate, as a function of mu, a number of theoretical LBCC L(mu, T)
"""

import pandas as pd
import numpy as np

from modules.lbcc_funs import get_analytic_limb_brightening_curve

Temps = [1.e6, 1.5e6, 2.e6]
R0 = 1.01
n_mu = 50

out_file = "/Users/turtle/Dropbox/MyNACD/analysis/theoretic_lbcc/theoretic_lbcc.csv"

mu_limb = np.sqrt(1-1/(R0**2))
mu_vec = np.linspace(mu_limb, 1, num=n_mu)

# generate DataFrame for writing to csv
output = pd.DataFrame(data={'mu': mu_vec})

for index, temp in enumerate(Temps):
    temp_F, temp_mu = get_analytic_limb_brightening_curve(t=temp, g_type=1, R0=R0, mu_vec=mu_vec)
    temp_L = np.log10(temp_F/temp_F[-1])
    col_label = "T" + '{:.1f}'.format(temp/1e6)

    output[col_label] = temp_L


# write to csv
output.to_csv(out_file)


# --- Generate approximations for Beta and y ------------------
# epsilon for determining line approx in L(T)
eps = .05e6
# if logT, find line approx in L(log10(T))
logT = True
log_eps = .01

out_file2 = "/Users/turtle/Dropbox/MyNACD/analysis/theoretic_lbcc/theoretic_lbcc_slope_int.csv"

output2 = pd.DataFrame(data={'mu': mu_vec})

for index, temp in enumerate(Temps):
    if logT:
        log_temp = np.log10(temp)
        log_lo  = log_temp-log_eps
        log_hi  = log_temp+log_eps
        temp_lo = 10**(log_lo)
        temp_hi = 10**(log_hi)
    else:
        temp_lo = temp - eps
        temp_hi = temp + eps
    # recover F values for perturbed temperatures
    F_lo, temp_mu = get_analytic_limb_brightening_curve(t=temp_lo, g_type=1, R0=R0, mu_vec=mu_vec)
    F_hi, temp_mu = get_analytic_limb_brightening_curve(t=temp_hi, g_type=1, R0=R0, mu_vec=mu_vec)
    F_actual, temp_mu = get_analytic_limb_brightening_curve(t=temp, g_type=1, R0=R0, mu_vec=mu_vec)
    # calculate L
    L_lo = np.log10(F_lo/F_lo[-1])
    L_hi = np.log10(F_hi/F_hi[-1])
    L_actual = np.log10(F_actual/F_actual[-1])
    # calculate line approximation at L_actual
    if logT:
        L_slope = (L_hi - L_lo)/(2*log_eps)
        L_intercept = -L_slope*log_temp + L_actual
        # convert to beta and y
        beta = 1/(L_slope + 1)
        y = -L_intercept*beta
    else:
        L_slope = (L_hi - L_lo)/(2*eps)
        L_intercept = L_actual - beta*temp
        # convert to beta and y (1e6 is experimental. It seems to work, but has no basis
        beta = 1/(L_slope*1e6 + 1)
        y = -L_intercept*beta

    slope_label = "L_slope" + '{:.1f}'.format(temp/1e6)
    int_label = "L_int" + '{:.1f}'.format(temp/1e6)

    output2[slope_label] = beta
    output2[int_label] = y

# write to csv
output2.to_csv(out_file2)
