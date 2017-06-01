import numpy as np
from scipy.optimize import root

def iirf100_interp_funct(alp_b,a,tau,targ_iirf100):
    iirf100_arr = alp_b*(np.sum(a*tau*(1.0 - np.exp(-100.0/(tau*alp_b)))))
    return iirf100_arr   -  targ_iirf100

def fair_scm(tstep=1.0,
             emissions=False,
             other_rf=0.0,
             co2_concs=False,
             q=np.array([0.33,0.41]),
             tcrecs=np.array([1.6,2.75]),
             d=np.array([239.0,4.1]),
             a=np.array([0.2173,0.2240,0.2824,0.2763]),
             tau=np.array([1000000,394.4,36.54,4.304]),
             r0=32.40,
             rC=0.019,
             rT=4.165,
             F_2x=3.74,
             C_0=278.0,
             ppm_gtc=2.123,
             iirf100_max=97.0,
             in_state=[[0.0,0.0,0.0,0.0],[0.0,0.0],0.0],
             restart_out=False):

  # # # ------------ CALCULATE Q ARRAY ------------ # # #
  # If TCR and ECS are supplied, overwrite the q array
  k = 1.0 - (d/70.0)*(1.0 - np.exp(-70.0/d))
  if type(tcrecs) in [np.ndarray,list]:
    q =  (1.0 / F_2x) * (1.0/(k[0]-k[1])) \
        * np.array([tcrecs[0]-k[1]*tcrecs[1],k[0]*tcrecs[1]-tcrecs[0]])

  # # # ------------ SET UP OUTPUT TIMESERIES VARIABLES ------------ # # #
  # by default FAIR is not concentration driven
  conc_driven=False
  if type(emissions) in [np.ndarray,list]:
    integ_len = len(emissions)
    if (type(other_rf) in [np.ndarray,list]) \
       and (len(other_rf)!=integ_len):
        raise ValueError("The emissions and other_rf timeseries don't have the same length")
    elif type(other_rf) in [int,float]:
        other_rf = np.linspace(other_rf,other_rf,num=integ_len)

    carbon_boxes_shape = (integ_len,4)
    thermal_boxes_shape = (integ_len,2)
  
  elif type(co2_concs) in [np.ndarray,list]:
    integ_len = len(co2_concs)
    conc_driven = True
    if (type(other_rf) in [np.ndarray,list]) \
       and (len(other_rf)!=integ_len):
        raise ValueError("The concentrations and other_rf timeseries don't have the same length")
    elif type(other_rf) in [int,float]:
        other_rf = np.linspace(other_rf,other_rf,num=integ_len)

    carbon_boxes_shape = (integ_len,4)
    thermal_boxes_shape = (integ_len,2)

  elif type(other_rf) in [np.ndarray,list]:
    integ_len = len(other_rf)
    if type(emissions) in [int,float]:
        emissions = np.linspace(emissions,emissions,num=integ_len)
    else:
        emissions = np.zeros(integ_len)

    carbon_boxes_shape = (integ_len,4)
    thermal_boxes_shape = (integ_len,2)

  else:
    raise ValueError("Neither emissions, co2_concs or other_rf is defined as a timeseries")

  RF = np.zeros(integ_len)
  C_acc = np.zeros(integ_len)
  iirf100 = np.zeros(integ_len)
  R_i = np.zeros(carbon_boxes_shape)
  C = np.zeros(integ_len)
  T_j = np.zeros(thermal_boxes_shape)
  T = np.zeros(integ_len)

  # # # ------------ FIRST TIMESTEP ------------ # # #
  R_i_pre = in_state[0]
  T_j_pre = in_state[1]
  C_acc_pre = in_state[2]

  if conc_driven:
    C[0] = co2_concs[0]
  
  else:
    # Calculate the parametrised iIRF and check if it is over the maximum 
    # allowed value
    iirf100[0] = r0 + rC*C_acc_pre + rT*np.sum(T_j_pre)

    if iirf100[0] >= iirf100_max:
      iirf100[0] = iirf100_max
      
    # Determine a solution for alpha
    time_scale_sf = (root(iirf100_interp_funct,0.16,args=(a,tau,iirf100[0])))['x']

    # Multiply default timescales by scale factor
    tau_new = time_scale_sf * tau

    # Compute the updated concentrations box anomalies from the decay of the 
    # previous year and the emisisons
    R_i[0] = R_i_pre*np.exp(-tstep/tau_new) \
              + (emissions[0,np.newaxis])*a*tau_new*(1-np.exp(-tstep/tau_new)) / ppm_gtc

    C[0] = np.sum(R_i[0]) + C_0

    # Calculate the additional carbon uptake
    C_acc[0] =  C_acc_pre + emissions[0] - (C[0]-(np.sum(R_i_pre) + C_0)) * ppm_gtc

  # Calculate the radiative forcing using the previous timestep's CO2 concentration
  RF[0] = (F_2x/np.log(2.)) * np.log(C[0] /C_0) \
            + other_rf[0]

  # Update the thermal response boxes
  T_j[0] = RF[0,np.newaxis]*q*(1-np.exp((-tstep)/d)) + T_j_pre*np.exp(-tstep/d)

  # Sum the thermal response boxes to get the total temperature anomlay
  T[0]=np.sum(T_j[0])

  # # # ------------ REST OF RUN ------------ # # #
  for x in range(1,integ_len):
    if conc_driven:
      C[x] = co2_concs[x]
    
    else:
      # Calculate the parametrised iIRF and check if it is over the maximum 
      # allowed value
      iirf100[x] = r0 + rC*C_acc[x-1] + rT*T[x-1]
      if iirf100[x] >= iirf100_max:
        iirf100[x] = iirf100_max
        
      # Determine a solution for alpha
      if x == 1:
        time_scale_sf = (root(iirf100_interp_funct,0.16,args=(a,tau,iirf100[x])))['x']
      else:
        time_scale_sf = (root(iirf100_interp_funct,time_scale_sf,args=(a,tau,iirf100[x])))['x']

      # Multiply default timescales by scale factor
      tau_new = time_scale_sf * tau

      # Compute the updated concentrations box anomalies from the decay of the previous year and the emisisons
      R_i[x] = R_i[x-1]*np.exp(-tstep/tau_new) \
              + (emissions[x,np.newaxis])*a*tau_new*(1-np.exp(-tstep/tau_new)) / ppm_gtc

      # Sum the boxes to get the total concentration anomaly
      C[x] = np.sum(R_i[x]) + C_0

      # Calculate the additional carbon uptake
      C_acc[x] =  C_acc[x-1] + emissions[x] * tstep - (C[x]-C[x-1]) * ppm_gtc

    # Calculate the radiative forcing using the previous timestep's CO2 concentration
    RF[x] = (F_2x/np.log(2.)) * np.log((C[x-1]) /C_0) + other_rf[x]

    # Update the thermal response boxes
    T_j[x] = T_j[x-1]*np.exp(-tstep/d) + RF[x,np.newaxis]*q*(1-np.exp(-tstep/d))
    
    # Sum the thermal response boxes to get the total temperature anomaly
    T[x]=np.sum(T_j[x])

  if restart_out:
    return C, T, (R_i[-1],T_j[-1],C_acc[-1])
  else:
    return C, T