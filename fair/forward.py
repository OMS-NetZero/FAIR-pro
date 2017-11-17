import inspect
import numpy as np
from scipy.optimize import root
from constants import molwt, lifetime, radeff
from constants.general import M_ATMOS
from forcing.ghg import etminan

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
             natural=np.array([0, 202., 10.] + [0.]*28),
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
  # just test with WMGHGs for now
  ngas = 31
  nF   = 4

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
  integ_len = emissions.shape[0]
#  elif type(other_rf) is np.ndarray:
#    carbon_boxes_shape = (other_rf.shape[0], a.shape[0])
#    thermal_boxes_shape = (other_rf.shape[0], d.shape[0])
#    integ_len = len(other_rf)
#    emissions = np.zeros(integ_len)
#  else:
#    raise ValueError(
#      "Neither emissions or other_rf is defined as a timeseries")

  RF = np.zeros((integ_len, nF))
  C_acc = np.zeros(integ_len)
  iirf = np.zeros(integ_len)
  R_i = np.zeros(carbon_boxes_shape)
  T_j = np.zeros(thermal_boxes_shape)

  C = np.zeros((integ_len, ngas))
  T = np.zeros(integ_len)

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
  C[0,1:3] = C_0[1:3] - C_0[1:3]*(1.0 - np.exp(-1.0/np.array(lifetime.aslist[1:3]))) + (natural[1:3] + 0.5 * (emissions[0,2:4])) / emis2conc[1:3]
  C[0,3:] = C_0[3:] - C_0[3:]*(1.0 - np.exp(-1.0/np.array(lifetime.aslist[3:]))) + (natural[3:] + 0.5 * (emissions[0,12:])) / emis2conc[3:]

  # CO2, CH4 and methane are co-dependent and from Etminan relationship
  RF[0,0:3] = etminan(C[0,0:3], C_0[0:3], F2x=F2x)

  # Minor (F- and H-gases) are linear in concentration
  # the factor of 0.001 here is because radiative efficiencies are given
  # in W/m2/ppb and concentrations of minor gases are in ppt.
  RF[0,3] = np.sum((C[0,3:] - C_0[3:]) * radeff.aslist[3:] * 0.001)

  if restart_in == False:
    # Update the thermal response boxes
    T_j[0,:] = (q/d)*(np.sum(RF[0,:]))

  # Sum the thermal response boxes to get the total temperature anomaly
  T[0]=np.sum(T_j[0,:],axis=-1)

  for x in range(1,integ_len):
      
    # Calculate the parametrised iIRF and check if it is over the maximum 
    # allowed value
    iirf[x] = rc * C_acc[x-1]  + rt*T[x-1]  + r0
    if iirf[x] >= iirf_max:
      iirf[x] = iirf_max
      
    # Linearly interpolate a solution for alpha
    if x == 1:
      time_scale_sf = (root(iirf_interp_funct,0.16,args=(a,tau,iirf[x])))['x']
    else:
      time_scale_sf = (root(iirf_interp_funct,time_scale_sf,args=(a,tau,iirf[x])))['x']

    # Multiply default timescales by scale factor
    tau_new = tau * time_scale_sf

    # CARBON DIOXIDE
    # Compute the updated concentrations box anomalies from the decay of the
    # previous year and the additional emissions
    R_i[x,:] = R_i[x-1,:]*np.exp(-1.0/tau_new) + a*(np.sum(emissions[x,1:3])) / ppm_gtc
    # Sum the boxes to get the total concentration anomaly
    C[x,0] = np.sum(R_i[...,x,:],axis=-1)
    # Calculate the additional carbon uptake
    C_acc[x] =  C_acc[x-1] + 0.5*(np.sum(emissions[x-1:x+1,1:3])) - (
      C[x,0] - C[x-1,0])*ppm_gtc

    # METHANE
    C[x,1] = C[x-1,1] - C[x-1,1]*(1.0 - np.exp(-1.0/lifetime.CH4)) + (
      natural[1] + 0.5 * (emissions[x,3] + emissions[x-1,3])) / emis2conc[1]

    # NITROUS OXIDE
    C[x,2] = C[x-1,2] - C[x-1,2]*(1.0 - np.exp(-1.0/lifetime.N2O)) + (
      natural[2] + 0.5 * (emissions[x,4] + emissions[x-1,4])) / emis2conc[2]

    # OTHER WMGHGs
    C[x,3:] = C[x-1,3:] - C[x-1,3:]*(1.0 - np.exp(-1.0/np.array(
      lifetime.aslist[3:]))) + (natural[3:] + 0.5 * (
      emissions[x,12:] + emissions[x-1,12:])) / emis2conc[3:]

    # Calculate the total radiative forcing
    RF[x,0:3] = etminan(C[x,0:3], C_0[0:3], F2x=F2x)
    RF[x,3] = np.sum((C[x,3:] - C_0[3:]) * radeff.aslist[3:] * 0.001)

    # Update the thermal response boxes
    T_j[x,:] = T_j[x-1,:]*np.exp(-1.0/d) + q*(1-np.exp((-1.0)/d))*np.sum(RF[x,:])
    # Sum the thermal response boxes to get the total temperature anomaly
    T[x]=np.sum(T_j[x,:],axis=-1)

  # add delta CO2 concentrations to initial value
  C[:,0] = C[:,0] + C_0[0]

  if restart_out:
    restart_out_val=(R_i[-1],T_j[-1],C_acc[-1])
    return C, RF, T, restart_out_val
  else:
    return C, RF, T
