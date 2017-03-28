import numpy as np
from scipy.optimize import root
from scipy.ndimage.filters import gaussian_filter1d

def iirf_interp_funct(alp_b,a,tau,iirf,x):

    iirf_arr = alp_b*(np.sum(a*tau*(1.0 - np.exp(-100.0/(tau*alp_b)))))
    return iirf_arr   -  iirf[x]

def fair_scm(emissions,other_rf=0.0,tcrecs=np.array([1.75,2.5]),d=np.array([4.1,239.0]),a=np.array([0.2173,0.2240,0.2824,0.2763]),tau=np.array([1000000,394.4,36.54,4.304]),r0=32.40,rc=0.019,rt=4.165,F_2x=3.74,C_0=278.0,ppm_gtc=2.123,iirf_max=97.0):

  #Calculate the q1 and q2 model coefficients from the TCR, ECS and thermal response timescales.
  k = 1.0 - (d/70.0)*(1.0 - np.exp(-70.0/d))
  q = np.transpose((1.0 / F_2x) * (1.0/(k[0]-k[1])) * np.array([tcrecs[0]-tcrecs[1]*k[1],tcrecs[1]*k[0]-tcrecs[0]]))

  #Set up the output timeseries variables
  carbon_boxes_shape = tuple(list(emissions.shape) + [4])
  thermal_boxes_shape = tuple(list(emissions.shape) + [2])
  
  RF = np.zeros_like(emissions)
  C_acc = np.zeros_like(emissions)
  iirf = np.zeros_like(emissions)
  R_i = np.zeros(carbon_boxes_shape)
  T_j = np.zeros(thermal_boxes_shape)

  C = np.zeros_like(emissions)
  T = np.zeros_like(emissions)

  #Initialise the carbon pools to be correct for first timestep in numerical method
  R_i[0,:] = a * emissions[0,np.newaxis] / ppm_gtc
  C[0] = np.sum(R_i[0,:],axis=-1)
  if type(other_rf) == float:
    RF[0] = (F_2x/np.log(2.)) * np.log((C[0] + C_0) /C_0) + other_rf
  else:
    RF[0] = (F_2x/np.log(2.)) * np.log((C[0] + C_0) /C_0) + other_rf[x]
  #Update the thermal response boxes
  T_j[0,:] = (q/d)*(RF[0,np.newaxis])
  #Sum the thermal response boxes to get the total temperature anomlay
  T[0]=np.sum(T_j[0,:],axis=-1)

  for x in range(1,emissions.shape[-1]):
      
    #Calculate the parametrised iIRF and check if it is over the maximum allowed value
    iirf[x] = rc * C_acc[x-1]  + rt*T[x-1]  + r0
    if iirf[x] >= iirf_max:
      iirf[x] = iirf_max
      
    #Linearly interpolate a solution for alpha
    if x == 1:
      time_scale_sf = (root(iirf_interp_funct,0.16,args=(a,tau,iirf,x)))['x']
    else:
      time_scale_sf = (root(iirf_interp_funct,time_scale_sf,args=(a,tau,iirf,x)))['x']

    #Multiply default timescales by scale factor
    tau_new = tau * time_scale_sf

    #Compute the updated concentrations box anomalies from the decay of the pervious year and the additional emisisons
    R_i[x,:] = R_i[x-1,:]*np.exp(-1.0/tau_new) + a*(emissions[x,np.newaxis]) / ppm_gtc
    #Summ the boxes to get the total concentration anomaly
    C[x] = np.sum(R_i[...,x,:],axis=-1)
    #Calculate the additional carbon uptake
    C_acc[x] =  C_acc[x-1] + 0.5*(emissions[x] +emissions[x-1]) - (C[x] - C[x-1])*ppm_gtc

    #Calculate the total radiative forcing
    if type(other_rf) == float:
      RF[x] = (F_2x/np.log(2.)) * np.log((C[x] + C_0) /C_0) + other_rf
    else:
      RF[x] = (F_2x/np.log(2.)) * np.log((C[x] + C_0) /C_0) + other_rf[x]

    #Update the thermal response boxes
    T_j[x,:] = T_j[x-1,:]*np.exp(-1.0/d) + (q/d)*(RF[x,np.newaxis])
    #Sum the thermal response boxes to get the total temperature anomlay
    T[x]=np.sum(T_j[x,:],axis=-1)


  return C + C_0, T





