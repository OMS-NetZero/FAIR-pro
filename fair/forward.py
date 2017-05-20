import numpy as np
from scipy.optimize import root

def iirf_interp_funct(alp_b,a,tau,targ_iirf):
  # ref eq. (7) of Millar et al ACP (2017)
    iirf_arr = alp_b*(np.sum(a*tau*(1.0 - np.exp(-100.0/(tau*alp_b)))))
    return iirf_arr   -  targ_iirf

def fair_scm(emissions=False,
             other_rf=0.0,
             q=np.array([0.33,0.41]),
             tcrecs=np.array([1.6,2.75]),
             d=np.array([239.0,4.1]),
             a=np.array([0.2173,0.2240,0.2824,0.2763]),
             tau=np.array([1000000,394.4,36.54,4.304]),
             r0=32.40,
             rc=0.019,
             rt=4.165,
             F_2x=3.74,
             C_0=278.0,
             ppm_gtc=2.123,
             iirf_max=97.0,
             restart_in=False,
             restart_out=False):

  # So I can't get the timestepping to work for single year steps. At the 
  # moment if you want to do a single timestep and you give an emissions array 
  # of length 1 then all you do is initialise our variables. You don't enter 
  # our loop as integ_len will be equal to 1 and range(1,1) returns nothing.

  # Ultimately you should be able to get identical results whether you 
  # timestep FAIR year by year or run it continuously.

  # My solution is to properly define what each of the values in each array 
  # represent. We can change this but in the interest of having something here 
  # is attempt 1, based off what the code already does
    # emissions in timestep x
        # average of all global emissions in timestep x. The model treats them 
        # as if they've all been released instantaneously on Jan 1st of 
        # timestep x 
        # (see line R_i[0,:] = a * emissions[0,np.newaxis] / ppm_gtc)
    # concentrations in timestep x
        # concentrations at the start of timestep x i.e. on Jan 1st. It is 
        # directly affected by emissions in timestep x. This value is used to 
        # calculate radiative forcing
    # radiative forcing in timestep x
        # radiative forcing at the start of timestep x i.e. on Jan 1st. 
    # temperature in timestep x
        # global mean temperatures at the end of the timestep i.e. on Dec 
        # 31st. Note that we're implicitly assuming that the radiative forcing 
        # applies throughout the entire year
        # (see line T_j[x,:] = T_j[x-1,:]*np.exp(-1.0/d) + q*(1-np.exp((-1.0)/d))*RF[x,np.newaxis]) 
        # we also had the conflicting line 
        # T_j[0,:] = (q/d)*(RF[0,np.newaxis]
        # previously (I've changed it now). The two lines are conflicting as 
        # the first one assumes that the radiative forcing acts over the 
        # length of the year whilst the second assumes that the radiative 
        # forcing acts in an infinitesimally small period of time).

  # With these definitions it is now clear that initialisation and restart 
  # values must be values from some previous point in time (even if that's 
  # December 31st of the previous year). Hence they should not be returned.

  # Of course I don't think any of this matters for overall results but I 
  # think it does matter if we want people to pick this up and use it as that 
  # requires a coherent story. Something which the 'master' branch code does 
  # not currently tell.

  # The ultimate question we have to answer is, if you run FAIR for a single 
  # year, what are you actually doing?

  # If TCR and ECS are supplied, calculate the q1 and q2 model coefficients 
  # (overwriting any other q array that might have been supplied)
  # ref eq. (4) and (5) of Millar et al ACP (2017)
  k = 1.0 - (d/70.0)*(1.0 - np.exp(-70.0/d))
  if type(tcrecs) in [np.ndarray,list]:
    q =  (1.0 / F_2x) * (1.0/(k[0]-k[1])) * np.array([tcrecs[0]-tcrecs[1]*k[1],tcrecs[1]*k[0]-tcrecs[0]])

  # Set up the output timeseries variables
  # emissions must be a numpy array for this to work
  if type(emissions) in [np.ndarray,list]:
    carbon_boxes_shape = tuple(list(emissions.shape) + [4])
    print '\nChanging carbon_boxes_shape = tuple(list(emissions.shape) + [4]) to'
    print '(len(emissions),4) makes no difference?'
    print carbon_boxes_shape == (len(emissions),4)
    thermal_boxes_shape = tuple(list(emissions.shape) + [2])
    print '\nChanging thermal_boxes_shape = tuple(list(emissions.shape) + [2]) to'
    print '(len(emissions),2) makes no difference?'
    print thermal_boxes_shape == (len(emissions),2)
    integ_len = emissions.shape[-1]
    print '\nChanging integ_len = emissions.shape[-1] to'
    print 'len(emissions) makes no difference?'
    print integ_len == len(emissions)
  elif type(other_rf) in [np.ndarray,list]:
    carbon_boxes_shape = tuple(list(other_rf.shape) + [4])
    print '\nChanging carbon_boxes_shape = tuple(list(other_rf.shape) + [4]) to'
    print '(len(other_rf),4) makes no difference?'
    print carbon_boxes_shape == (len(other_rf),4)
    thermal_boxes_shape = tuple(list(other_rf.shape) + [2])
    print '\nChanging thermal_boxes_shape = tuple(list(other_rf.shape) + [2]) to'
    print '(len(other_rf),2) makes no difference?'
    print thermal_boxes_shape == (len(other_rf),2)
    integ_len = other_rf.shape[-1]
    print '\nChanging integ_len = other_rf.shape[-1] to'
    print 'len(other_rf) makes no difference?'
    print integ_len == len(other_rf)
    emissions = np.zeros(integ_len)
  else:
    raise ValueError("Neither emissions or other_rf is defined as a timeseries")

  RF = np.zeros(integ_len)
  C_acc = np.zeros(integ_len)
  iirf = np.zeros(integ_len)
  R_i = np.zeros(carbon_boxes_shape)
  T_j = np.zeros(thermal_boxes_shape)

  C = np.zeros(integ_len)
  T = np.zeros(integ_len)

  if restart_in:
    R_i_pre=restart_in[0]
    T_j_pre=restart_in[1]
    C_acc_pre = restart_in[2]

    T_pre=np.sum(T_j_pre)

    # Calculate the parametrised iIRF and check if it is over the maximum allowed value
    iirf[0] = rc * C_acc_pre  + rt * T_pre  + r0
    if iirf[0] >= iirf_max:
      iirf[0] = iirf_max
      
    # Linearly interpolate a solution for alpha
    time_scale_sf = (root(iirf_interp_funct,0.16,args=(a,tau,iirf[0])))['x']

    # Multiply default timescales by scale factor
    tau_new = tau * time_scale_sf

    # Compute the updated concentrations box anomalies from the decay of the previous year and the emisisons
    R_i[0,:] = R_i_pre[0]*np.exp(-1.0/tau_new) + a*(emissions[0,np.newaxis]) / ppm_gtc
    print '\nChanging R_i[0,:] to'
    print 'R_i[0] makes no difference?'
    print R_i[0,:] == R_i[0]


  else:
    # Initialise the carbon pools to be correct for first timestep in numerical method
    R_i[0,:] = a * emissions[0,np.newaxis] / ppm_gtc
    print '\nChanging R_i[0,:] to'
    print 'R_i[0] makes no difference?'
    print R_i[0,:] == R_i[0]

  C[0] = np.sum(R_i[0,:],axis=-1)
  print '\nChanging np.sum(R_i[0,:],axis=-1) to'
  print 'np.sum(R_i[0]) makes no difference?'
  print np.sum(R_i[0,:],axis=-1) == np.sum(R_i[0])

  if type(other_rf) == float:
    RF[0] = (F_2x/np.log(2.)) * np.log((C[0] + C_0) /C_0) + other_rf
  else:
    RF[0] = (F_2x/np.log(2.)) * np.log((C[0] + C_0) /C_0) + other_rf[0]

  # Update the thermal response boxes
  if restart_in:
    T_j[0,:] = T_j_pre*np.exp(-1.0/d) \
               + q*(1-np.exp((-1.0)/d))*(RF[0,np.newaxis])
  else:
    T_j[0,:] = q*(1-np.exp((-1.0)/d))*(RF[0,np.newaxis])

  print '\nChanging T_j[0,:] to'
  print 'T_j[0] makes no difference?'
  print T_j[0,:] == T_j[0]

  # Sum the thermal response boxes to get the total temperature anomlay
  T[0]=np.sum(T_j[0,:],axis=-1)
  print '\nChanging np.sum(T_j[0,:],axis=-1) to'
  print 'np.sum(T_j[0]) makes no difference?'
  print np.sum(T_j[0,:],axis=-1) == np.sum(T_j[0])

  for x in range(1,integ_len):
      
    # Calculate the parametrised iIRF and check if it is over the maximum allowed value
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

    # Compute the updated concentrations box anomalies from the decay of the previous year and the emisisons
    R_i[x,:] = R_i[x-1,:]*np.exp(-1.0/tau_new) + a*(emissions[x-1,np.newaxis]) / ppm_gtc
    print '\nChanging R_i[x,:] to'
    print 'R_i[x] makes no difference?'
    print R_i[x,:] == R_i[x]

    # Sum the boxes to get the total concentration anomaly
    C[x] = np.sum(R_i[...,x,:],axis=-1)
    print '\nChanging np.sum(R_i[...,x,:],axis=-1) to'
    print 'np.sum(R_i[x]) makes no difference?'
    print np.sum(R_i[...,x,:],axis=-1) == np.sum(R_i[x])

    # Calculate the additional carbon uptake
    C_acc[x] =  C_acc[x-1] + emissions[x] - (C[x] - C[x-1])*ppm_gtc

    # Calculate the total radiative forcing
    if type(other_rf) == float:
      RF[x] = (F_2x/np.log(2.)) * np.log((C[x] + C_0) /C_0) + other_rf
    else:
      RF[x] = (F_2x/np.log(2.)) * np.log((C[x] + C_0) /C_0) + other_rf[x]

    # Update the thermal response boxes
    T_j[x,:] = T_j[x-1,:]*np.exp(-1.0/d) + q*(1-np.exp((-1.0)/d))*RF[x,np.newaxis]
    print '\nChanging T_j[x,:] to'
    print 'T_j[x] makes no difference?'
    print T_j[x,:] == T_j[x]
    
    # Sum the thermal response boxes to get the total temperature anomaly
    T[x]=np.sum(T_j[x,:],axis=-1)
    print '\nChanging np.sum(T_j[x,:],axis=-1) to'
    print 'np.sum(T_j[x]) makes no difference?'
    print np.sum(T_j[x,:],axis=-1) == np.sum(T_j[x])

  if restart_out:
    restart_out_val=(R_i[-1],T_j[-1],C_acc[-1])
    return C + C_0, T, restart_out_val
  else:
    return C + C_0, T