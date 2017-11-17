import numpy as np

def etminan(C, Cpi, F2x=3.74):
  """Calculate the radiative forcing from CO2, CH4 and N2O.

  This function uses the updated formulas of Etminan et al. (2016),
  including the overlaps between CO2, methane and nitrous oxide.

  Reference: Etminan et al, 2016, JGR, doi: 10.1002/2016GL071930

  Inputs:
    C: [CO2, CH4, N2O] concentrations, [ppm, ppb, ppb]
    Cpi: pre-industrial [CO2, CH4, N2O] concentrations

  Keywords:
    F2x: radiative forcing from a doubling of CO2.

  Returns:
    3-element array of radiative forcing: [F_CO2, F_CH4, F_N2O]
  """

  Cbar = 0.5 * (C[0] + Cpi[0])
  Mbar = 0.5 * (C[1] + Cpi[1])
  Nbar = 0.5 * (C[2] + Cpi[2])

  # Note the original formula uses 5.36 in place of F2x/log(2).
  F = np.zeros(3)
  F[0] = (-2.4e-7*(C[0] - Cpi[0])**2 + 7.2e-4*np.fabs(C[0]-Cpi[0]) - \
    2.1e-4 * Nbar + F2x/np.log(2)) * np.log((C[0]+Cpi[0])/Cpi[0])
  F[1] = (-1.3e-6*Mbar - 8.2e-6*Nbar + 0.043) * (np.sqrt(C[1]) - \
    np.sqrt(Cpi[1]))
  F[2] = (-8.0e-6*Cbar + 4.2e-6*Nbar - 4.9e-6*Mbar + 0.117) * \
    (np.sqrt(C[2]) - np.sqrt(Cpi[2]))

  return F