"""
Python module for forward calculations with the FAIR climate model

List of classes, exceptions and functions exported by the module. 

# # ------------ CLASSES ------------ # #
sublime snippet for module description in header is 'hclsdesc'

# # ------------ EXCEPTIONS ------------ # #
sublime snippet for exception description in header is 'hexcdesc'

# # ------------ FUNCTIONS ------------ # #
iirf100_interp_funct: calculate difference between iIRF100 and target iIRF100 for given alpha

iirf100_interp_funct_multi_rf: calcualtes difference between calculated and target iirf100 function for multi RFs

fair_scm: run fair forward calculation

plot_fair: plot fair output variables

temp_to_forcing: convert temperature profile into RF profile

forcing_to_conc: converts an inputted RF profile into compatible CO2 concentration profile

conc_to_emissions: converts temperature and co2 concentration profile into compatible CO2 emissions profile

# # ------------ ANY OTHER OBJECTS EXPORTED ------------ # #
describe them here
"""

# # # ------------ IMPORT REQUIRED MODULES ------------ # # #
# # ------------ STANDARD LIBRARY ------------ # #

# # ------------ THIRD PARTY ------------ # #
import numpy as np
from scipy.optimize import root
import scipy.ndimage.filters as filters1
#-------------------------------------------

# # ------------ LOCAL APPLICATION/LIBRARY SPECIFIC ------------ # #

def iirf100_interp_funct(alpha,a,tau,targ_iirf100):
    """
    Calculate difference between iIRF100 and target iIRF100 for given alpha

    # # ------------ ARGUMENTS ------------ # #
    alpha: (float/int)
      carbon pool response time scaling factor (dimensionless)

    a: (np.array)
      fraction of emitted carbon which goes into each carbon pool (dimensionless)

    tau: (np.array)
      response time of each carbon pool when alpha = 1 (yrs)

    targ_iirf100: (float/int)
      target value of the 100-year integrated impulse response (iIRF100)

    # # ------------ RETURN VALUE ------------ # #
    iirf100_arr - targ_iirf100: (float)
      difference between target and calculated iIRF100
    """
    # # # ------------ IMPORT REQUIRED MODULES ------------ # # #
    # # ------------ STANDARD LIBRARY ------------ # #

    # # ------------ THIRD PARTY ------------ # #
    import numpy as np
    # # ------------ LOCAL APPLICATION/LIBRARY SPECIFIC ------------ # #

    # # # ------------ CODE ------------ # # #
    iirf100_arr = alpha*(np.sum(a*tau*(1.0 - np.exp(-100.0/(tau*alpha)))))
    return iirf100_arr   -  targ_iirf100

# Define a varient of the above function that is necessary for quick calculations of multiple radiative forcing profiles.
def iirf100_interp_funct_multi_rf(alpha,a,tau,targ_iirf100):
    """
    Calculate difference between iIRF100 and target iIRF100 for given alpha

    # # ------------ ARGUMENTS ------------ # #
    alpha: (float/int)
      carbon pool response time scaling factor (dimensionless)

    a: (np.array)
      fraction of emitted carbon which goes into each carbon pool (dimensionless)

    tau: (np.array)
      response time of each carbon pool when alpha = 1 (yrs)

    targ_iirf100: (float/int)
      target value of the 100-year integrated impulse response (iIRF100)

    # # ------------ RETURN VALUE ------------ # #
    iirf100_arr - targ_iirf100: (float)
      difference between target and calculated iIRF100
    """
    # # # ------------ IMPORT REQUIRED MODULES ------------ # # #
    # # ------------ STANDARD LIBRARY ------------ # #

    # # ------------ THIRD PARTY ------------ # #
    import numpy as np
    # # ------------ LOCAL APPLICATION/LIBRARY SPECIFIC ------------ # #

    # # # ------------ CODE ------------ # # #
    iirf100_arr = alpha*(np.sum((a*tau*(1.0 - np.exp(-100.0/(tau*alpha[:,np.newaxis])))),axis=1))
    return iirf100_arr   -  targ_iirf100

# Define the FAIR simple climate model function. input_params is the default set of parameters, later we will define each parameter by calling different values (or columns for the case of multiple values for each parameter).
def fair_scm(tstep=1.0,
             emissions=False,
             other_rf=0.0,
             co2_concs=False,
                                        input_params=np.array([0.33,0.41,1.6,2.75,239.0,4.1,0.2173,0.2240,0.2824,0.2763,1000000,394.4,36.54,4.304,32.40,0.019,4.165,3.74,278.0,2.123,97.0]),
             in_state=[[0.0,0.0,0.0,0.0],[0.0,0.0],0.0],
             restart_out=False):
    """
    Run fair forward calculation

    Takes input emissions/concentrations and forcing arrays and returns global 
    mean atmospheric CO_2 concentrations and temperatures.

    # # ------------ ARGUMENTS ------------ # #
    sublime snippet for variable description in header is 'hvardesc'

    tstep:^ (float)
      Length of a timestep (yrs)

    emissions:^ (np.array/list/float/int/bool)
      CO_2 emissions timeseries (GtC/yr). If a scalar then emissions are 
      assumed to be constant throughout the run. If false then emissions 
      aren't used.

    other_rf:^ (np.array/list/float/int)
      Non-CO_2 radiative forcing timeseries (W/m^2). If a scalar then other_rf 
      is assumed to be constant throughout the run.

    co2_concs:^ (np.array/list/bool)
      Atmospheric CO_2 concentrations timeseries (ppmv). If emissions are 
      supplied then co2_concs is not used.

    input_params:^ (np.array)
      The array which contains all the parameter values. The variables each 
      column corresponds to are listed below:
    
      q:^ (np.array)
      position = 0,1
        response of each thermal box to radiative forcing (K/(W/m^2)). 
        Over-written if tcrecs is supplied (be careful as this is default 
        behaviour).

      tcrecs:^ (np.array)
      position = 2,3
        Transient climate response (TCR) and equilibrium climate sensitivity 
        (ECS) array (K). tcrecs[0] is TCR and tcrecs[1] is ECS.

      d:^ (np.array)
      position = 4,5
        response time of each thermal box (yrs)

      a:^ (np.array)
      position = 6,7,8,9
        fraction of emitted carbon which goes into each carbon pool 
        (dimensionless)

      tau:^ (np.array)
      position = 10,11,12,13
        unscaled response time of each carbon pool (yrs)

      r0:^ (float)
      position = 14
        pre-industrial 100-year integrated impulse response (iIRF100) (yrs)

      rC:^ (float)
      position = 15
        sensitivity of iIRF100 to CO_2 uptake by the land and oceans (yrs/GtC)

      rT:^ (float)
      position = 16
        sensitivity of iIRF100 to increases in global mean temperature (yrs/K)

      F_2x:^ (float)
      position = 17
        radiative forcing due to a doubling of atmospheric CO_2 concentrations 
        (W/m^2)

      C_0:^ (float)
      position = 18
        pre-industrial atmospheric CO_2 concentrations (ppmv)

      ppm_gtc:^ (float)
      position = 19
        ppmv to GtC conversion factor (GtC/ppmv)

      iirf100_max:^ (float)
      position = 20
        maximum allowed value of iIRF100 (keeps the model stable) (yrs)

    in_state:^ (list/np.array)
      initial state of the climate system with elements:
        [0]: (np.array/list)
          co_2 concentration of each carbon pool (ppmv)
        [1]: (np.array/list)
          temp of each temperature response box (K)
        [2]: (float)
          cumulative carbon uptake (GtC)

    restart_out:^ (bool)
      whether to return the final state of the climate system or not

    ^ => Keyword argument

    # # ------------ RETURN VALUE ------------ # #
    sublime snippet for variable description in header is 'hvardesc'
    # ------------ DEFAULT ------------ #
    C: (np.array)
      timeseries of atmospheric CO_2 concentrations (ppmv)

    T: (np.array)
      timeseries of global mean temperatures (K)

    # ------------ IF RESTART_OUT == TRUE ------------ #
    As above with the addition of a tuple with elements
    [0]: (np.array/list)
      co_2 concentration of each carbon pool at the end of the run (ppmv)
    [1]: (np.array/list)
      temp of each temperature response box at the end of the run (K)
    [2]: (float)
      cumulative carbon uptake at the end of the run (GtC)

    # # ------------ SIDE EFFECTS ------------ # #
    document side effects here

    # # ------------ EXCEPTIONS ------------ # #
    sublime snippet for exception description in header is 'hexcdesc'

    # # ------------ RESTRICTIONS ------------ # #
    Document any restrictions on when the function can be called
    """

    # One line break before anything else
    # # # ------------ IMPORT REQUIRED MODULES ------------ # # #
    # # ------------ STANDARD LIBRARY ------------ # #

    # # ------------ THIRD PARTY ------------ # #
    import numpy as np
    from scipy.optimize import root
    import time

    # # ------------ LOCAL APPLICATION/LIBRARY SPECIFIC ------------ # #

    # # # ------------ CODE ------------ # # #

    # # # ------------ SET UP OUTPUT TIMESERIES VARIABLES ------------ # # #
    # Register current time such that the total run time can be displayed if required
    begin_time = time.time()
    # the p_dims variable is used to store the number of different values for each parameter
    if type(input_params) in [np.ndarray,list]:
      if input_params.ndim==1:
        p_dims = 1
        input_params = input_params.reshape((1,input_params.shape[0]))
      else:
        p_dims = input_params.shape[0]
      
    # the rf_dims variable is used to store the number of different radiative forcing profiles
    if type(other_rf) in [np.ndarray,list]:
      if other_rf.ndim==1:
        rf_dims = 1
        other_rf = other_rf[...,np.newaxis].transpose()
      else:
        rf_dims = other_rf.shape[0]
    
    elif type(other_rf) in [np.ndarray,float]:
        rf_dims = 1
        if type(emissions) in [np.ndarray,float]:
            other_rf = np.full([1,emissions.size],other_rf)
        else:
            other_rf = np.full([1,co2_concs.size],other_rf)
        
    # the integ_len variable is used to store the length of our timeseries along with rf_dims and p_dims such that empty parameters of the right dimensions, size and shape can be easily made
    # by default FAIR is not concentration driven
    conc_driven=False
    # here we check if FAIR is emissions driven
    if type(emissions) in [np.ndarray,list]:
        integ_len = tuple([rf_dims] + [p_dims] + [len(emissions)])
        if (type(other_rf) in [np.ndarray,list]) and (other_rf.shape[-1]!=integ_len[-1]):
            raise ValueError("The emissions and other_rf timeseries don't have the same length")
        elif type(other_rf) in [int,float]:
            other_rf = np.full(integ_len,other_rf)
  
    # here we check if FAIR is concentration driven
    elif type(co2_concs) in [np.ndarray,list]:
        integ_len = tuple([rf_dims] + [p_dims] + [len(co2_concs)])
        conc_driven = True
        if (type(other_rf) in [np.ndarray,list]) and (other_rf.shape[-1]!=integ_len[2]):
            raise ValueError("The concentrations and other_rf timeseries don't have the same length")
        elif type(other_rf) in [int,float]:
            other_rf = np.full(integ_len,other_rf)

    # finally we check if only a non-CO2 radiative forcing timeseries has been supplied
    elif type(other_rf) in [np.ndarray,list]:
        integ_len = tuple([rf_dims] + [p_dims] + [other_rf.shape[-1]])
        if type(emissions) in [int,float]:
            emissions = np.full(integ_len[-1],emissions)
        else:
            emissions = np.zeros(integ_len[-1])

    else:
        raise ValueError("Neither emissions, co2_concs or other_rf is defined as a timeseries")
    
    RF = np.zeros(integ_len)
    C_acc = np.zeros(integ_len)
    iirf100 = np.zeros(integ_len)

    carbon_boxes_shape = tuple([integ_len[-1]] + [rf_dims] + [p_dims] + [4])
    R_i = np.zeros(carbon_boxes_shape)
    C = np.zeros(integ_len)

    thermal_boxes_shape = tuple([integ_len[-1]] + [rf_dims] + [p_dims] + [2])
    T_j = np.zeros(thermal_boxes_shape)
    T = np.zeros(integ_len)
    
    # Define correct size and shape arrays for each parameter
    q = np.zeros(thermal_boxes_shape)
    tcrecs = np.zeros(thermal_boxes_shape)
    d = np.zeros(thermal_boxes_shape)
    a = np.zeros(carbon_boxes_shape)
    tau = np.zeros(carbon_boxes_shape)
    r0 = np.zeros(integ_len)
    rC = np.zeros(integ_len)
    rT = np.zeros(integ_len)
    F_2x = np.zeros(integ_len)
    C_0 = np.zeros(integ_len)
    ppm_gtc = np.zeros(integ_len)
    iirf100_max = np.zeros(integ_len)
    
    # Pull out parameter values from input_params
    q = input_params[...,0:2]
    tcrecs = input_params[...,2:4]
    d = input_params[...,4:6]
    a = input_params[...,6:10]
    tau = input_params[...,10:14]
    r0 = input_params[...,14]
    rC = input_params[...,15]
    rT = input_params[...,16]
    F_2x = input_params[...,17]
    C_0 = input_params[...,18]
    ppm_gtc = input_params[...,19]
    iirf100_max = input_params[...,20]
    
    time_scale_sf = np.zeros(integ_len)
    tau_new = np.zeros(tau.shape)
    C_out = np.zeros(integ_len)
    iirf_arr = np.zeros(integ_len)
    
        # # # ------------ CALCULATE Q ARRAY ------------ # # #
    # If TCR and ECS are supplied, overwrite the q array
    k = np.zeros(d.shape)
    k = 1.0 - (d/70.0)*(1.0 - np.exp(-70.0/d))
    if type(tcrecs) in [np.ndarray,list]:
        q =  ((1.0 / F_2x) * (1.0/(k[...,0]-k[...,1])) \
            * np.array([tcrecs[...,0]-k[...,1]*tcrecs[...,1],k[...,0]*tcrecs[...,1]-tcrecs[...,0]])).transpose()

    # # # ------------ FIRST TIMESTEP ------------ # # #
    R_i_pre = in_state[0]
    C_pre = np.sum(R_i_pre) + C_0
    T_j_pre = in_state[1]
    C_acc_pre = in_state[2]

    if conc_driven:
        C[0] = co2_concs[0]
  
    else:
        # Calculate the parametrised iIRF and check if it is over the maximum 
        # allowed value
        iirf100[...,0] = r0 + rC*C_acc_pre + rT*np.sum(T_j_pre)
        for i in range(0,rf_dims):
            for j in range(0,p_dims):
                if iirf100[i,j,0] >= iirf100_max[j]:
                  iirf100[i,j,0] = iirf100_max[j]
          
        # Determine a solution for alpha using scipy's root finder
        # First check which iirf100_interp_funct needs to be called, i.e. do we have multiple radiative forcing profiles and hence do we need to call that specific function?
        if rf_dims == 1:
            for j in range(0,p_dims):
                time_scale_sf[0,j,0] = (root(iirf100_interp_funct,0.16,args=(a[j],tau[j],iirf100[0,j,0])))['x']
        else:
            for i in range(0,rf_dims):
                time_scale_sf[i,:,0] = (root(iirf100_interp_funct_multi_rf,0.16,args=(a,tau,iirf100[i,:,0])))['x']

        # Multiply default timescales by scale factor
        tau_new = tau * time_scale_sf[...,0,np.newaxis]

        # Compute the updated concentrations box anomalies from the decay of the 
        # previous year and the emisisons
        R_i[0,...,:] = R_i_pre*np.exp(-tstep/tau_new) \
                  + (emissions[0,np.newaxis])*a*tau_new*(1-np.exp(-tstep/tau_new)) / ppm_gtc[:,np.newaxis]

        C[...,0] = np.sum(R_i[0,...,:],axis=-1) + C_0

        # Calculate the additional carbon uptake
        C_acc[...,0] =  C_acc_pre + emissions[0] - (C[...,0]-(np.sum(R_i_pre) + C_0)) * ppm_gtc

    # Calculate the radiative forcing using the previous timestep's CO2 concentration
    RF[...,0] = (F_2x/np.log(2.)) * np.log(C_pre/C_0) + other_rf[:,0,np.newaxis]
        
    # Update the thermal response boxes
    T_j[0,...,:] = RF[...,0,np.newaxis]*q*(1-np.exp((-tstep)/d)) + T_j_pre*np.exp(-tstep/d)

    # Sum the thermal response boxes to get the total temperature anomlay
    T[...,0] = np.sum(T_j[0,...,:],axis=-1)
    
    # # # ------------ REST OF RUN ------------ # # #
    start_time = time.time()
    
    for x in range(1,integ_len[-1]):
        if conc_driven:
          C[...,x] = co2_concs[x]
        
        else:
          # Calculate the parametrised iIRF and check if it is over the maximum 
          # allowed value 
          iirf100[...,x] = r0 + rC*C_acc[...,x-1] + rT*T[...,x-1]
          for i in range(0,rf_dims):
            for j in range(0,p_dims):
                if iirf100[i,j,x] >= iirf100_max[j]:
                  iirf100[i,j,x] = iirf100_max[j]
            
          # Determine a solution for alpha using scipy's root finder
          if rf_dims == 1:
              for j in range(0,p_dims):
                  time_scale_sf[0,j,x] = (root(iirf100_interp_funct,time_scale_sf[0,j,x-1],args=(a[j],tau[j],iirf100[0,j,x])))['x']
          else:
              for i in range(0,rf_dims):
                  time_scale_sf[i,:,x] = (root(iirf100_interp_funct_multi_rf,time_scale_sf[i,:,x-1],args=(a,tau,iirf100[i,:,x])))['x'] 

                
          # Multiply default timescales by scale factor
          tau_new = time_scale_sf[...,x,np.newaxis] * tau

          # Compute the updated concentrations box anomalies from the decay of the previous year and the emisisons
          R_i[x,...,:] = R_i[x-1,...,:]*np.exp(-tstep/tau_new) \
                  + (emissions[x,np.newaxis])*a*tau_new*(1-np.exp(-tstep/tau_new)) / ppm_gtc[:,np.newaxis]

          # Sum the boxes to get the total concentration anomaly
          C[...,x] = np.sum(R_i[x,...,:],axis=-1) + C_0

          # Calculate the additional carbon uptake
          C_acc[...,x] =  C_acc[...,x-1] + emissions[x] * tstep - (C[...,x]-C[...,x-1]) * ppm_gtc

        # Calculate the radiative forcing using the previous timestep's CO2 concentration

        RF[...,x] = (F_2x/np.log(2.)) * np.log((C[...,x-1]) /C_0) + other_rf[:,x,np.newaxis]    

        # Update the thermal response boxes
        T_j[x,...,:] = T_j[x-1,...,:]*np.exp(-tstep/d) + RF[...,x,np.newaxis]*q*(1-np.exp(-tstep/d))
        
        # Sum the thermal response boxes to get the total temperature anomaly
        T[...,x] = np.sum(T_j[x,...,:],axis=-1)
        
        # Allows live progress of percentage of completion as well as an estimate of the time remaining for large computations
        if p_dims + rf_dims > 100:
          p = round(((x / (integ_len[-1] - 1.))*100.),0)
          tr = ((((time.time() - start_time) / x) * (integ_len[-1] - 1)) - (time.time() - start_time))
          tr_mins = int(tr/60.)
          tr_secs = round((tr/60. - int(tr/60.))*60.,0)
        
          if tr_mins == 1.0:
              print '{0}\r'.format('Completed: %.0f%%. Time remaining: %.0f minute and %.0f seconds.' % (p, tr_mins, tr_secs)),
          else:
              print '{0}\r'.format('Completed: %.0f%%. Time remaining: %.0f minutes and %.0f seconds.' % (p, tr_mins, tr_secs)),
   
    # # # ------------ OUTPUT ------------ # # #
    # Prints the total runtime
    if p_dims + rf_dims > 100:
        mins = int((time.time() - begin_time)/60)
        secs = round(((time.time() - begin_time)/60 - int((time.time() - begin_time)/60))*60,0)
        if mins == 1.0:
          print("---------- Run time was %.0f minute and %.0f seconds ----------" % (mins, secs))
        else:
          print("---------- Run time was %.0f minutes and %.0f seconds ----------" % (mins, secs))
        
    # Ensure that output form matches input
    if rf_dims == 1:
        if p_dims == 1:
            if restart_out:
                return C[0,0], T[0,0], (R_i[-1],T_j[-1],C_acc[-1])
            else:
                return C[0,0], T[0,0]
        else:
            if restart_out:
                return C[0], T[0], (R_i[-1],T_j[-1],C_acc[-1])
            else:
                return C[0], T[0]
    else:
        if p_dims == 1:
            if restart_out:
                return C[:,0,:], T[:,0,:], (R_i[-1],T_j[-1],C_acc[-1])
            else:
                return C[:,0,:], T[:,0,:]
        else:    
            if restart_out:
                return C, T, (R_i[-1],T_j[-1],C_acc[-1])
            else:
                return C, T
        

def plot_fair(emms,
              conc,
              forc,
              temp,
              y_0=0,
              tuts=False,
              infig=False,
              inemmsax=None,
              inconcax=None,
              inforcax=None,
              intempax=None,
              colour={'emms':'black',
                     'conc':'blue',
                     'forc':'orange',
                     'temp':'red'},
              label=None,
              linestyle='-',
             ):
    """
    Function to plot FAIR variables

    Takes some of the work out of making a panel plot and ensures that the 
    variables appear as they are interpreted by fair e.g. fluxes are constant 
    over the timestep rather than the default linear interpolation between 
    values as done by most plotting routines.

    # # ------------ ARGUMENTS ------------ # #
    sublime snippet for variable description in header is 'hvardesc'

    emms: (np.array/list)
      CO_2 emissions timeseries (GtC)

    conc: (np.array/list)
      CO_2 concentrations timeseries (ppmv)

    forc: (np.array/list)
      Non-CO_2 forcing timeseries (W/m^2)

    temp: (np.array/list)
      Global mean temperature timeseries (K)

    y_0:^ (float/int)
      starting value of your timeseries, used to set min of time axis (same as time units)

    tuts:^ (string/bool)
      time units. If not supplied then 'units unknown' is printed

    infig:^ (matplotlib.figure.Figure)
      pre-existing figure we should plot onto

    inemmsax:^ (subplots.AxesSubplot)
      pre-existing axis the emissions should be plotted onto

    inconcax:^ (subplots.AxesSubplot)
      pre-existing axis the CO_2 concentrations should be plotted onto

    inforcax:^ (subplots.AxesSubplot)
      pre-existing axis the non-CO_2 forcing should be plotted onto

    intempax:^ (subplots.AxesSubplot)
      pre-existing axis the global mean temperature should be plotted onto

    colour:^ (dict)
      dictionary of colours to use for each timeseries

    label:^ (str/bool)
      name of the emissions timeseries

    linestyle:^ (str)
      linestyle to use for the timeseries. Default is '-' i.e. solid

    ^ => Keyword argument

    # # ------------ RETURN VALUE ------------ # #
    fig: (matplotlib.figure.Figure)
      the figure object

    emmsax: (subplots.AxesSubplot)
      emissions subplot

    concax: (subplots.AxesSubplot)
      CO_2 concentrations subplot

    forcax: (subplots.AxesSubplot)
      non-CO_2 forcing subplot

    tempax: (subplots.AxesSubplot)
      global mean temperature subplot

    # # ------------ SIDE EFFECTS ------------ # #
    document side effects here

    # # ------------ EXCEPTIONS ------------ # #
    sublime snippet for exception description in header is 'hexcdesc'

    # # ------------ RESTRICTIONS ------------ # #
    Document any restrictions on when the function can be called

    """

    # One line break before anything else
    # # # ------------ IMPORT REQUIRED MODULES ------------ # # #
    # # ------------ STANDARD LIBRARY ------------ # #
    from math import ceil

    # # ------------ THIRD PARTY ------------ # #
    import numpy as np

    from matplotlib import pyplot as plt
    plt.style.use('seaborn-darkgrid')
    plt.rcParams['figure.figsize'] = 16, 9
    plt.rcParams['lines.linewidth'] = 1.5

    font = {'weight' : 'normal',
          'size'   : 16}

    plt.rc('font', **font)

    # # ------------ LOCAL APPLICATION/LIBRARY SPECIFIC ------------ # #

    # # # ------------ CODE ------------ # # #
    # # ------------ SORT OUT INPUT VARIABLES ------------ # #
    pts = {'emms':emms,
         'forc':forc,
         'conc':conc,
         'temp':temp}

    integ_len = 0

    for j,var in enumerate(pts):
        if type(pts[var]) == list:
            pts[var] = np.array(pts[var])
            integ_len = len(pts[var])
        elif type(pts[var]) == np.ndarray:
            integ_len = len(pts[var])

    if integ_len == 0:
        for name in pts:
            print "{0}: {1}".format(name,type(pts[name]))
        raise ValueError("Error: I can't work out which one of your input variables is a timeseries")

    for j,var in enumerate(pts):
        if type(pts[var]) == np.ndarray and len(pts[var]) == integ_len:
            pass
        elif type(pts[var]) == np.ndarray and len(pts[var]) != integ_len:
            for name in pts:
                print "{0}: {1}\nlength: {2}".format(name,
                                                     type(pts[name]),
                                                     len(pts[name]))
            raise ValueError("Error: Your timeseries are not all the same length, I don't know what to do")
        else:
            if type(pts[var]) in [float,int]:
                pts[var] = np.full(integ_len,pts[var])
            else:
                pts[var] = np.zeros(integ_len)

    # # ------------ SORT OUT TIME VARIABLE ------------ # #
    # state variables are valid at the end of the timestep so we
    # go from 1 - integ_len + 1 rather than 0 - integ_len
    time = np.arange(0.99,integ_len+0.99) + y_0

    if not tuts:
        tuts = 'units unknown'

    # # ------------ PREPARE FLUX VARIABLES FOR PLOTTING  ------------ # #
    # Flux variables are assumed constant throughout the timestep to make this appear 
    # on the plot we have to do the following if there's fewer than 1000 timesteps
    fmintsp = 1000
    if integ_len < fmintsp:
        # work out small you need to divide each timestep to get 1000 timesteps
        div = ceil(fmintsp/integ_len)
        ftime = np.arange(0,integ_len,1.0/div) + y_0
        fluxes = ['emms','forc']
        for f in fluxes:
            tmp = []
            for j,v in enumerate(pts[f]):
                for i in range(0,int(div)):
                    tmp.append(v)
            pts[f] = tmp
            
    else:
        ftime = time - 0.5
    
    if not infig:
        fig = plt.figure()
        emmsax = fig.add_subplot(221)
        concax = fig.add_subplot(222)
        forcax = fig.add_subplot(223)
        tempax = fig.add_subplot(224)
    else:
        fig = infig
        emmsax = inemmsax
        concax = inconcax
        forcax = inforcax
        tempax = intempax

    emmsax.plot(ftime,pts['emms'],color=colour['emms'],label=label,ls=linestyle)
    emmsax.set_ylabel(r'Emissions (GtC.yr$^{-1}$)')
    if label is not None:
        emmsax.legend(loc='best')
    concax.plot(time,pts['conc'],color=colour['conc'],ls=linestyle)
    concax.set_ylabel('CO$_2$ concentrations (ppm)')
    concax.set_xlim(emmsax.get_xlim())
    forcax.plot(ftime,pts['forc'],color=colour['forc'],ls=linestyle)
    forcax.set_ylabel('Non-CO$_2$ radiative forcing (W.m$^{-2}$)')
    forcax.set_xlabel('Time ({0})'.format(tuts))
    tempax.plot(time,pts['temp'],color=colour['temp'],ls=linestyle)
    tempax.set_ylabel('Temperature anomaly (K)')
    tempax.set_xlabel(forcax.get_xlabel())
    tempax.set_xlim(forcax.get_xlim())
    fig.tight_layout()

    return fig,emmsax,concax,forcax,tempax






### 2) --- Functions in this section define the FAIR inverse model ---
        #  FAIR MODEL INVERSE. CONTAINS FUNCTIONS WHICH:
        #        1) CALCULATE THE RADIATIVE FORCING FROM A GIVEN TEMPERATURE PROFILE
        #        2) CALCULATE THE CO2 CONCENTRATION FROM A GIVEN RADIATIVE FORCING
        #        3) CALCULATE THE ANNUAL EMISSIONS PROFILE FROM A GIVEN CO2 CONCENTRATION

def temp_to_forcing(T_in, 
                  other_rf = None, 
                  other_rf_in = False, 
                  tcrecs = np.array([1.6,2.75]), 
                  tstep = 1.0,
                  F_2x = 3.74,
                  d=np.array([239.0,4.1]),
                  q=np.array([0.33,0.41]),
                  in_state = np.array([[0.0,0.0],[0.0,0.0]])):
    """
    Convert a temperature profile into a compatible RF profile.

    # # ------------ ARGUMENTS ------------ # #
    T_in: (np.array)
      inputted temperature profile we wish to convert to RF profile

    other_rf:^ (np.array or None)
      other_rf component which contributes to the temperature response but we wish to separate out of this rf calculation. Eg. if we wish to isolate the contribution to temperature from CO2 emissions we plug in other_rf as non-CO2 forcing.

    other_rf_in:^ (boolean)
      Boolean value switch to determine if we have are inputting other_rf.

    tcrecs:^ (np.array)
      values of TCR and ECS.
     
    tstep:^ (int/float)
      time step interval - have not tested for anything other than 1 year.
      
    F_2x:^ (int/float)
      value of radative forcing upon doubling of CO2 concentrations.
      
    d:^ (np.array)
      position = 0,1
        response time of each thermal box (yrs)
      
    q:^ (np.array)
      position = 2,3
        response of each thermal box to radiative forcing (K/(W/m^2)). 
        Over-written if tcrecs is supplied (be careful as this is default 
        behaviour).

    in_state:^ (np.array)
      inputted state of vectors, ie. [T_comp], [co2_rf_pre, other_rf_pre]. Default = zeros.

    ^ => Keyword argument
    
    # # ------------ RETURN VALUE ------------ # #
    RF: (np.array)
      timeseries of radiative forcing compatible with the inputted tempeature profile, split into 2 temeprature bins (Wm^{-2}).

    # # ------------ SIDE EFFECTS ------------ # #
    document side effects here

    # # ------------ EXCEPTIONS ------------ # #
    sublime snippet for exception description in header is 'hexcdesc'

    # # ------------ RESTRICTIONS ------------ # #
    Document any restrictions on when the function can be called
    """
    # # # ------------ IMPORT REQUIRED MODULES ------------ # # #
    # # ------------ STANDARD LIBRARY ------------ # #
    # # ------------ THIRD PARTY ------------ # #
    import numpy as np
    # # ------------ LOCAL APPLICATION/LIBRARY SPECIFIC ------------ # #
    # # # ------------ CODE ------------ # # #
    
    # --- INITIALIZE ARRAYS ---
    co2_rf = np.zeros(T_in.size)
    T_comp = np.zeros((T_in.size,2))
    
    if isinstance(other_rf,np.ndarray) != True:
        if other_rf_in == False:
            other_rf = np.zeros(T_in.size)
        else:
            print 'Warning: you haven\'t inputted any other_rf profile but you have turned other_rf_in to True. Other_rf might be of wrong data type.'
    elif isinstance(other_rf,np.ndarray) & other_rf_in == True:
        if other_rf.size != T_in.size:
            print 'Warning: other_rf needs to be the same size as T_in.'
    elif isinstance(other_rf,np.ndarray) & other_rf_in == False:
        print 'Warning: you have inputted an other_RF profile but haven\'t turned the switch other_rf_in to True.'
    elif other_rf_in == True & isinstance(other_rf,np.ndarray) != True:
        print 'Warning: other_rf must be of type numpy.ndarray'
    
    # --- CALCULATE Q ARRAY ---
    # If TCR and ECS are supplied, overwrite the q array
    k = 1.0 - (d/70.0)*(1.0 - np.exp(-70.0/d))
    if type(tcrecs) in [np.ndarray,list]:
        q =  (1.0 / F_2x) * (1.0/(k[0]-k[1])) \
            * np.array([tcrecs[0]-k[1]*tcrecs[1],k[0]*tcrecs[1]-tcrecs[0]])
    
    ### --- FIRST TIMESTEP ---
    # Set the initial values of key parameters as the in_state values, default = zeros.
    T_comp_pre = in_state[0]
    co2_rf_pre = in_state[1,0]
    other_rf_pre = in_state[1,1]
    
    # Calculate the temperature contribution to the total from each box, after it has decayed from the previous timestep
    T_d = T_comp_pre[:]*(np.exp((-tstep)/d[:]))

    # Estimate the co2 radiative forcing required to give that temperature change between last timestep and this one
    co2_rf[0] = (T_in[0] - np.sum(T_d) - other_rf[0]*np.sum(q*(1-np.exp((-tstep)/d)))) / (np.sum(q*(1-np.exp((-tstep)/d)))) 
    
    # Calcualate the contribution to the temperature boxes from the changed radiative forcing in this timestep, compared to the previous one
    T_f = q[:]*(co2_rf[0]+other_rf[0]) * (1-np.exp((-tstep)/d[:]))
        
    # Total temperature of each box is the sum of the decayed temperature from previous step and the forced temp from this step
    T_comp[0,:] = T_d + T_f
   # ---------------------
    
    ### --- REST OF RUN ---
    for x in range(1,T_in.size):
        # Calcualate temperature contribution from each box, assuming it decays since last timestep
        T_d = T_comp[x-1,:]*(np.exp((-tstep)/d[:]))
        
        # Calculate the required radiative forcing to produce observved temperature at this timestep, 
            #given the sum of the decayed temperatures in each box, from previous timestep
        co2_rf[x] = (T_in[x] - np.sum(T_d) - other_rf[x]*np.sum(q*(1-np.exp((-tstep)/d)))) / (np.sum(q*(1-np.exp((-tstep)/d)))) 
        
        # Calculate the contribution to the temperature of each box from the RF change from the previous step
        T_f = q[:]*(co2_rf[x]+other_rf[x]) * (1-np.exp((-tstep)/d[:]))
        
        # Total temperature contribution of each box is sum of decayed temperature from previous step 
            #and forcing temperature from this step
        T_comp[x,:] = T_d + T_f
    # -------------------
        
    return co2_rf

def forcing_to_conc(RF_in, 
                   RF_ext = None, 
                   RF_ext_in = False, 
                   a = 5.396, 
                   Cpreind = 278.):
    """
    Convert a radiative forcing profile into a compatible CO2 concentration profile.

    # # ------------ ARGUMENTS ------------ # #
    RF_in: (np.array)
      inputted RF profile we wish to convert to a CO2 concentrations time series.

    RF_ext:^ (np.array or None)
      other_rf component which contributes to the temperature response but we wish to separate out of this rf calculation. Eg. if we wish to isolate the contribution to RF from CO2 concentrations we plug in RF_ext as non-CO2 forcing.

    RF_ext_in:^ (boolean)
      Boolean value switch to determine if we have are inputting other_rf or not.

    a:^ (float)
      F_2x/np.log(2) -> where F_2x is the forcing contribution from doubling CO2 concentrations (F_2x = 3.74 Wm^{-2}).
     
    C_preind:^ (int/float)
      Preindustrial CO2 concentration.

    ^ => Keyword argument
    
    # # ------------ RETURN VALUE ------------ # #
    C: (np.array)
      timeseries of CO2 concentrations compatible with the inputted RF profile.

    # # ------------ SIDE EFFECTS ------------ # #
    document side effects here

    # # ------------ EXCEPTIONS ------------ # #
    sublime snippet for exception description in header is 'hexcdesc'

    # # ------------ RESTRICTIONS ------------ # #
    Document any restrictions on when the function can be called
    """
    # # # ------------ IMPORT REQUIRED MODULES ------------ # # #
    # # ------------ STANDARD LIBRARY ------------ # #
    # # ------------ THIRD PARTY ------------ # #
    import numpy as np
    # # ------------ LOCAL APPLICATION/LIBRARY SPECIFIC ------------ # #
    # # # ------------ CODE ------------ # # #
    
    # Initializing arrays
    # Create array to hold concentration values
    C = np.zeros(RF_in.size)
    
    if isinstance(RF_ext,np.ndarray) != True:
        if RF_ext_in == False:
            RF_ext = np.zeros(RF_in.size)
        else:
            print 'Warning: you haven\'t inputted any RF_ext profile but you have turned RF_ext_in to True. RF_ext might be of wrong data type.'
    elif isinstance(RF_ext,np.ndarray) & RF_ext_in == True:
        if RF_ext.size != RF_in.size:
            print 'Warning: RF_ext needs to be the same size as RF_in.'
    elif isinstance(RF_ext,np.ndarray) & RF_ext_in == False:
        print 'Warning: you have inputted an RF_ext profile but haven\'t turned the switch RF_ext_in to True.'
    elif RF_ext_in == True & isinstance(RF_ext,np.ndarray) != True:
        print 'Warning: RF_ext must be of type numpy.ndarray'
    
    ## Check inputs are of the same size if external forcing data is also provided
    #if RF_ext_in == True:
    #    if RF_in.size != RF_ext.size:
    #        print 'The inputs RF_in and RF_ext are not of the same size!'
    #elif RF_ext_in == False:
    #    RF_ext = np.zeros(RF_in.size)
    
    # Compute required co2 concentration to produce inputted logarithmic forcing
    for i in range(0, RF_in.size):
        C[i]=np.exp((RF_in[i]-RF_ext[i])/a)*Cpreind
    
    return C

def conc_to_emissions(co2_conc, T_input, 
                           t_const = np.array([1000000,394.4,36.54,4.303]), 
                           pool_splits = np.array([0.2173,0.2240,0.2824,0.2763]),
                           GtC_2_ppmv = 0.471,
                           r0=32.40,
                           rC=0.019,
                           rT=4.165,
                           iirf100_max = 97.,
                           Cpreind = 278,
                           tstep = 1.0,
                           year_smoothing = 5,
                           in_state=[[0.0,0.0,0.0,0.0],[0.0,0.0],0.0]):
    """
    Converts carbon dioxide concentration timeseries to carbon emissions profile using a 4 pool carbon cycle model.

    # # ------------ ARGUMENTS ------------ # #
    co2_conc: (np.array)
      inputted CO2 concentration timeseries.

    T_input: (np.array)
      inputted temperature timeseries.
    
    t_const:^ (np.array)
      time constants for the carbon pools to decay.

    pool_splits:^ (np.array)
       fraction of emitted carbon which goes into each carbon pool (dimensionless).

    GtC_2_ppmv:^ (float)
      Conversion factor to get from GtC to ppmv (GtC/ppmv).

    r0:^ (float)
      pre-industrial 100-year integrated impulse response (iIRF100) (yrs).

    rC:^ (float)
      sensitivity of iIRF100 to CO2 uptake by the land and oceans (yrs/GtC).

    rT:^ (float)
      sensitivity of iIRF100 to increases in global mean temperature (yrs/K).

    iirf100_max:^ (float)
      max allowed value of iirf100 value. Since eventually the value saturates and hte approximate form implied becomes inaccurate, we fix it after iirf100 goes too large (yrs).
     
    Cpreind:^ (int/float)
      preindustrial CO2 concentration (ppmv).

    tstep:^ (int/float)
      time step between calculated points -> havent tested for anything but 1 year interval (yrs).

    year_smoothing:^ (int)
      sigma interval over which to smooth the resulting emissions scenario (yrs).

    in_state:^ (np.array)
      input state of system. in_state[0,:] = C_comp_pre, in_state[1,:] = T_j_pre, in_state[2,:] = C_acc_pre. Standard = all set to zero.

    ^ => Keyword argument
    
    # # ------------ RETURN VALUE ------------ # #
    E: (np.array)
      timeseries of CO2 emissions time series compatible with the inputted concentration and temperature profiles. Unsmoothed (i.e. can have 2 timestep noise introduced due to nature of finite difference integral used).
      
    E_smooth: (np.array)
      E output but smoothed with a gaussian function set to a sigma size equal to year_smoothing parameter.

    # # ------------ SIDE EFFECTS ------------ # #
    document side effects here

    # # ------------ EXCEPTIONS ------------ # #
    sublime snippet for exception description in header is 'hexcdesc'

    # # ------------ RESTRICTIONS ------------ # #
    Document any restrictions on when the function can be called
    """
    # # # ------------ IMPORT REQUIRED MODULES ------------ # # #
    # # ------------ STANDARD LIBRARY ------------ # #
    # # ------------ THIRD PARTY ------------ # #
    import numpy as np
    from scipy.optimize import root
    import scipy.ndimage.filters as filters1
    # # ------------ LOCAL APPLICATION/LIBRARY SPECIFIC ------------ # #
    # # # ------------ CODE ------------ # # #
     
    #initialize the carbon pools, emissions, accumulated carbon and iIRF100 vectors. 
    #Give an intial guess of alpha for root function to work from.
    C_comp = np.zeros((co2_conc.size, 4))
    E = np.zeros(co2_conc.size)
    C_acc = np.zeros(co2_conc.size)
    iirf100 = np.zeros(co2_conc.size)
    alph_t = 0.16
    
    ###---------FIRST TIMESTEP----------
    #set the initial values of key parameters as the in_state values, default = zeros.
    C_comp_pre = in_state[0]
    C_pre = np.sum(C_comp_pre) + Cpreind
    T_j_pre = in_state[1]
    C_acc_pre = in_state[2]

    # Calculate the parametrised iIRF100 and check if it is over the maximum allowed value
    iirf100[0] = r0 + rC*C_acc[0] + rT*T_input[0]
    if iirf100[0] >= iirf100_max:
        iirf100[0] = iirf100_max
    
    #find the value of alpha
    alph_t = (root(iirf100_interp_funct,alph_t,args=(pool_splits,t_const,iirf100[0])))['x']
    
    #compute the carbon in each pool
    C_comp[0,:] = C_comp_pre[:]*np.exp((-tstep)/(alph_t*t_const[:]))
    
    #compute the emissions required to give change in CO2 concentration
    E[0] = (co2_conc[0] - np.sum(C_comp[0,:],-1) - C_pre) / (alph_t*np.sum(pool_splits*t_const*(1-np.exp((-tstep)/(alph_t*t_const))))*GtC_2_ppmv)
    
    #recompute the distribution of carbon in each pool for better estimation of emissions in next timestep
    C_comp[0,:] = C_comp[0,:] + alph_t*pool_splits[:]*t_const[:]*E[0]*GtC_2_ppmv*(1-np.exp((-tstep)/(alph_t*t_const[:])))
    
    #calculate the accumulated carbon in the land and oceans
    C_acc[0] =  C_acc_pre + E[0]*tstep - ((co2_conc[0]-C_pre)/GtC_2_ppmv)
    
    
    ###----------REST OF RUN-------------
    for i in range(1, co2_conc.size):
        #estimate the value of iIRF100, given the temperature and accumulated carbon in previous timestep
        iirf100[i] = r0 + rC*C_acc[i-1] + rT*T_input[i-1]
        if iirf100[i] > iirf100_max:
            iirf100[i] = iirf100_max
        
        #calculate the value of alpha using scipys root finder
        alph_t = (root(iirf100_interp_funct,alph_t,args=(pool_splits,t_const,iirf100[i])))['x']
        
        #compute the distribution of carbon between the pools
        C_comp[i,:] = C_comp[i-1,:]*np.exp((-tstep)/(alph_t*t_const[:]))
        
        #calculate the emissions required in this year to cause change in CO2 concentration
        E[i] = (co2_conc[i] - np.sum(C_comp[i,:],-1) - Cpreind) / (alph_t*np.sum(pool_splits*t_const*(1-np.exp((-tstep)/(alph_t*t_const))))*GtC_2_ppmv)
        
        #recalculate the distribution of carbon in each pool for better estimation in next timestep
        C_comp[i,:] = C_comp[i,:] + \
            alph_t*pool_splits[:]*t_const[:]*E[i]*GtC_2_ppmv*(1-np.exp((-tstep)/(alph_t*t_const[:])))
        
        #calculate the accumulated carbon in the land and sea
        C_acc[i] =  C_acc[i-1] + E[i]*tstep - ((co2_conc[i]-co2_conc[i-1])/GtC_2_ppmv)
        
    ###----------------------------------
   
    # We apply a gaussian filter to smooth the resulting curve 
    E_smooth = filters1.gaussian_filter1d(E, year_smoothing)
    
    return E, E_smooth




