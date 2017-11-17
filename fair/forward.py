import inspect
import numpy as np
from scipy.optimize import root
from constants import molwt, lifetime, radeff
from constants.general import M_ATMOS
from forcing.ghg import etminan
from forcing import ozone_tr

def iirf_interp_funct(alp_b,a,tau,targ_iirf):
	# ref eq. (7) of Millar et al ACP (2017)
    iirf_arr = alp_b*(np.sum(a*tau*(1.0 - np.exp(-100.0/(tau*alp_b)))))
    return iirf_arr   -  targ_iirf

def fair_scm(emissions,
             other_rf=0.0,
             q=np.array([0.33,0.41]),
             tcrecs=np.array([1.6,2.75]),
             d=np.array([239.0,4.1]),
             a=np.array([0.2173,0.2240,0.2824,0.2763]),
             tau=np.array([1000000,394.4,36.54,4.304]),
             r0=32.40,
             rc=0.019,
             rt=4.165,
             F2x=3.74,
             C_0=np.array([278., 722., 275.] + [0.]*28),
             natural=np.array([202., 10.]),
             iirf_max=97.0,
             restart_in=False,
             restart_out=False,
             tcr_dbl=70.0):

  # Conversion between ppm CO2 and GtC emissions
  ppm_gtc   = M_ATMOS/1e18*molwt.C/molwt.AIR

  # Conversion between ppb/ppt concentrations and Mt/kt emissions
  # in the RCP databases ppb = Mt and ppt = kt so factor always 1e18
  emis2conc = M_ATMOS/1e18*np.asarray(molwt.aslist)/molwt.AIR

  # Funny units for nitrogen emissions - N2O is expressed in N2 equivalent
  n2o_sf = molwt.N2O/molwt.N2
  emis2conc[2] = emis2conc[2] / n2o_sf

  # Number of individual gases and radiative forcing agents to consider
  # just test with WMGHGs + trop ozone
  ngas = 31
  nF   = 5

  # If TCR and ECS are supplied, calculate the q1 and q2 model coefficients 
  # (overwriting any other q array that might have been supplied)
  # ref eq. (4) and (5) of Millar et al ACP (2017)
  k = 1.0 - (d/tcr_dbl)*(1.0 - np.exp(-tcr_dbl/d))  # Allow TCR to vary
  if type(tcrecs) in [np.ndarray,list]:
    q =  (1.0 / F2x) * (1.0/(k[0]-k[1])) * np.array([
      tcrecs[0]-tcrecs[1]*k[1],tcrecs[1]*k[0]-tcrecs[0]])

  # Convert any list to a numpy array for (a) speed and (b) consistency.
  # Goes through all variables in scope and converts them.
  frame = inspect.currentframe()
  args, _, _, values = inspect.getargvalues(frame)
  for i in args:
    if type(values[i]) is list:
      exec(i + '= np.array(' + i + ')')

  # Set up the output timeseries variables
  if type(emissions) is not np.ndarray or emissions.shape[1] != 40:
    raise ValueError("emissions timeseries should be a nt x 40 numpy array")
  carbon_boxes_shape = (emissions.shape[0], a.shape[0])
  thermal_boxes_shape = (emissions.shape[0], d.shape[0])
  nt = emissions.shape[0]

  # Check natural emissions and convert to 2D array if necessary
#  print natural
  if natural.ndim==1:
    if natural.shape[0]!=2:
      raise ValueError(
        "natural emissions should be a 2-element or nt x 2 array")
    natural = np.tile(natural, nt).reshape((nt,2))
  elif natural.ndim==2:
    if natural.shape[1]!=2 or natural.shape[0]!=nt:
      raise ValueError(
        "natural emissions should be a 2-element or nt x 2 array")
  else:
    raise ValueError(
      "natural emissions should be a 2-element or nt x 2 array")

  F = np.zeros((nt, nF))
  C_acc = np.zeros(nt)
  iirf = np.zeros(nt)
  R_i = np.zeros(carbon_boxes_shape)
  T_j = np.zeros(thermal_boxes_shape)

  C = np.zeros((nt, ngas))
  T = np.zeros(nt)

  if restart_in:
    R_i[0]=restart_in[0]
    T_j[0]=restart_in[1]
    C_acc[0] = restart_in[2]
  else:
    # Initialise the carbon pools to be correct for first timestep in
    # numerical method
    R_i[0,:] = a * (np.sum(emissions[0,1:3])) / ppm_gtc

  # CO2 is a delta from pre-industrial. Other gases are absolute concentration
  C[0,0] = np.sum(R_i[0,:],axis=-1)
  C[0,1:3] = C_0[1:3] - C_0[1:3]*(1.0 - np.exp(-1.0/np.array(
    lifetime.aslist[1:3]))) + (natural[0,:] + 0.5 * (
    emissions[0,2:4])) / emis2conc[1:3]
  C[0,3:] = C_0[3:] - C_0[3:]*(1.0 - np.exp(-1.0/np.array(
    lifetime.aslist[3:]))) + (0.5 * (
    emissions[0,12:])) / emis2conc[3:]

  # CO2, CH4 and methane are co-dependent and from Etminan relationship
  F[0,0:3] = etminan(C[0,0:3], C_0[0:3], F2x=F2x)

  # Minor (F- and H-gases) are linear in concentration
  # the factor of 0.001 here is because radiative efficiencies are given
  # in W/m2/ppb and concentrations of minor gases are in ppt.
  F[0,3] = np.sum((C[0,3:] - C_0[3:]) * radeff.aslist[3:] * 0.001)

  # Tropospheric ozone. Assuming no temperature/chemistry feedback it can live
  # outside the forward model.
  F[:,4] = ozone_tr.regress(emissions)

  if restart_in == False:
    # Update the thermal response boxes
    T_j[0,:] = (q/d)*(np.sum(F[0,:]))

  # Sum the thermal response boxes to get the total temperature anomaly
  T[0]=np.sum(T_j[0,:],axis=-1)

  for t in range(1,nt):
      
    # Calculate the parametrised iIRF and check if it is over the maximum 
    # allowed value
    iirf[t] = rc * C_acc[t-1]  + rt*T[t-1]  + r0
    if iirf[t] >= iirf_max:
      iirf[t] = iirf_max
      
    # Linearly interpolate a solution for alpha
    if t == 1:
      time_scale_sf = (root(iirf_interp_funct,0.16,args=(a,tau,iirf[t])))['x']
    else:
      time_scale_sf = (root(iirf_interp_funct,time_scale_sf,args=(
        a,tau,iirf[t])))['x']

    # Multiply default timescales by scale factor
    tau_new = tau * time_scale_sf

    # CARBON DIOXIDE
    # Compute the updated concentrations box anomalies from the decay of the
    # previous year and the additional emissions
    R_i[t,:] = R_i[t-1,:]*np.exp(-1.0/tau_new) + a*(np.sum(
      emissions[t,1:3])) / ppm_gtc
    # Sum the boxes to get the total concentration anomaly
    C[t,0] = np.sum(R_i[...,t,:],axis=-1)
    # Calculate the additional carbon uptake
    C_acc[t] =  C_acc[t-1] + 0.5*(np.sum(emissions[t-1:t+1,1:3])) - (
      C[t,0] - C[t-1,0])*ppm_gtc

    # METHANE
    C[t,1] = C[t-1,1] - C[t-1,1]*(1.0 - np.exp(-1.0/lifetime.CH4)) + (
      natural[t,0] + 0.5 * (emissions[t,3] + emissions[t-1,3])) / emis2conc[1]

    # NITROUS OXIDE
    C[t,2] = C[t-1,2] - C[t-1,2]*(1.0 - np.exp(-1.0/lifetime.N2O)) + (
      natural[t,1] + 0.5 * (emissions[t,4] + emissions[t-1,4])) / emis2conc[2]

    # OTHER WMGHGs
    C[t,3:] = C[t-1,3:] - C[t-1,3:]*(1.0 - np.exp(-1.0/np.array(
      lifetime.aslist[3:]))) + (0.5 * (
      emissions[t,12:] + emissions[t-1,12:])) / emis2conc[3:]

    # Calculate the total radiative forcing
    F[t,0:3] = etminan(C[t,0:3], C_0[0:3], F2x=F2x)
    F[t,3] = np.sum((C[t,3:] - C_0[3:]) * radeff.aslist[3:] * 0.001)

    # Update the thermal response boxes
    T_j[t,:] = T_j[t-1,:]*np.exp(-1.0/d) + q*(1-np.exp((-1.0)/d))*np.sum(
      F[t,:])
    # Sum the thermal response boxes to get the total temperature anomaly
    T[t]=np.sum(T_j[t,:],axis=-1)

  # add delta CO2 concentrations to initial value
  C[:,0] = C[:,0] + C_0[0]

  if restart_out:
    restart_out_val=(R_i[-1],T_j[-1],C_acc[-1])
    return C, F, T, restart_out_val
  else:
    return C, F, T
