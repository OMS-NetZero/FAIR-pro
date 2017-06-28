"""
Python module for forward calculations with the FAIR climate model

List of classes, exceptions and functions exported by the module. 

# # ------------ CLASSES ------------ # #
sublime snippet for module description in header is 'hclsdesc'

# # ------------ EXCEPTIONS ------------ # #
sublime snippet for exception description in header is 'hexcdesc'

# # ------------ FUNCTIONS ------------ # #
iirf100_interp_funct: calculate difference between iIRF100 and target iIRF100 for given alpha

fair_scm: run fair forward calculation

plot_fair: plot fair output variables

# # ------------ ANY OTHER OBJECTS EXPORTED ------------ # #
describe them here
"""

# # # ------------ IMPORT REQUIRED MODULES ------------ # # #
# # ------------ STANDARD LIBRARY ------------ # #

# # ------------ THIRD PARTY ------------ # #
import numpy as np
from scipy.optimize import root

# # ------------ LOCAL APPLICATION/LIBRARY SPECIFIC ------------ # #

# Define a function which gives the relationship between iIRF_100 and scaling factor, alpha
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

# define a function that: for 4 inputs and a length L:
# leaves arrays as they are
# converts scalar types into a array of length L, all elements the same scalar value (constant emissions or concentrations)
# converts False types (ie. no argument given) into an array of length L of zeros
# checks the all the final arrays have the same length and raises an error if not
def emissions_concentrations_sort(A,B,C,D,length):
    Arr = [A,B,C,D]
    for x in range(len(Arr)):
        if type(Arr[x]) in [int,float]:
            Arr[x] = np.full(length,Arr[x])
        elif type(Arr[x]) == bool:
            Arr[x] = np.zeros(length)
    if any(len(x) != length for x in Arr):
	    raise ValueError("One or more of the emissions/concentrations given or other_rf timeseries doesn't have the same length")
    else:
	    return Arr

# Define three functions that give the Radiative forcing due to CH4 and N2O as per equations given in IPCC AR5 8.SM
def f(M, N):
	return 0.47 * np.log(1 + 2.01 * 10**(-5) * M * N * 0.75 + 5.31 * 10**(-15) * M * (M * N)**(1.52)) # see IPCC AR5 Table 8.SM.1

def RF_M(M, N, M_0, N_0, alp_m=0.036):
	return alp_m * (np.sqrt(M+M_0) - np.sqrt(M_0)) - (f(M+M_0, N_0) - f(M_0, N_0))

def RF_N(M, N, M_0, N_0, alp_n=0.12):
	return alp_n * (np.sqrt(N+N_0) - np.sqrt(N_0)) - (f(M_0, N+N_0) - f(M_0, N_0))

# Define the FAIR simple climate model function
def fair_scm(tstep=1.0,
             emissions=False,
			 M_emissions=False,
			 N_emissions=False,
             other_rf=0.0,
             co2_concs=False,
			 M_concs=False,
			 N_concs=False,
             q=np.array([0.33,0.41]),
             tcrecs=np.array([1.6,2.75]),
             d=np.array([239.0,4.1]),
             a=np.array([0.2173,0.2240,0.2824,0.2763]),
             tau=np.array([1000000,394.4,36.54,4.304]),
			 tau_M=12.4,
			 tau_N=121,
             r0=32.40,
             rC=0.019,
             rT=4.165,
             F_2x=3.74,
             C_0=278.0,
			 M_0=722,
			 N_0=270,
             ppm_gtc=2.123,
			 ppb_TgM=2.838,
			 ppb_TgN=7.787,
             iirf100_max=97.0,
             in_state=[[0.0,0.0,0.0,0.0],[0.0,0.0],0.0,0.0,0.0],
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
	
    M_emissions:^ (np.array/list/float/int/bool)
      CH4 emissions timeseries (Tg/yr). If a scalar then emissions are 
      assumed to be constant throughout the run. If false then CH4 emissions 
      aren't used.
	  
    N_emissions:^ (np.array/list/float/int/bool)
      N2O emissions timeseries (Tg/yr). If a scalar then emissions are 
      assumed to be constant throughout the run. If false then N2O emissions 
      aren't used.

    other_rf:^ (np.array/list/float/int)
      Non-CO_2 radiative forcing timeseries (W/m^2). If a scalar then other_rf 
      is assumed to be constant throughout the run.

    co2_concs:^ (np.array/list/bool)
      Atmospheric CO_2 concentrations timeseries (ppmv). If emissions are 
      supplied then co2_concs is not used.
	  
    M_concs:^ (np.array/list/bool)
      Atmospheric CH4 concentrations timeseries (ppbv). If M_emissions are 
      supplied then M_concs is not used.
	  
    N_concs:^ (np.array/list/bool)
      Atmospheric N2O concentrations timeseries (ppbv). If N_emissions are 
      supplied then N_concs is not used.

    q:^ (np.array)
      response of each thermal box to radiative forcing (K/(W/m^2)). 
      Over-written if tcrecs is supplied (be careful as this is default 
      behaviour).

    tcrecs:^ (np.array)
      Transient climate response (TCR) and equilibrium climate sensitivity 
      (ECS) array (K). tcrecs[0] is TCR and tcrecs[1] is ECS.

    d:^ (np.array)
      response time of each thermal box (yrs)

    a:^ (np.array)
      fraction of emitted carbon which goes into each carbon pool 
      (dimensionless)

    tau:^ (np.array)
      unscaled response time of each carbon pool (yrs)
	  
    tau_M:^ (float/int)
      lifetime of atmospheric CH4 (yrs) (value used for IPCC AR5 metrics)
	  
    tau_N:^ (float/int)
      lifetime of atmospheric N2O (yrs) (value used for IPCC AR5 metrics)

    r0:^ (float)
      pre-industrial 100-year integrated impulse response (iIRF100) (yrs)

    rC:^ (float)
      sensitivity of iIRF100 to CO_2 uptake by the land and oceans (yrs/GtC)

    rT:^ (float)
      sensitivity of iIRF100 to increases in global mean temperature (yrs/K)

    F_2x:^ (float)
      radiative forcing due to a doubling of atmospheric CO_2 concentrations 
      (W/m^2)

    C_0:^ (float)
      pre-industrial atmospheric CO_2 concentrations (ppmv)
	  
    M_0:^ (float)
      pre-industrial atmospheric CH4 concentrations (ppbv)
	  
    N_0:^ (float)
      pre-industrial atmospheric N2O concentrations (ppbv)

    ppm_gtc:^ (float)
      ppmv to GtC conversion factor (GtC/ppmv)
	  
    ppb_TgM:^ (float)
      ppbv to Tg of CH4 conversion factor (TgCH4/ppbv)
	  
    ppb_TgN:^ (float)
      ppbv to Tg of N2O conversion factor (TgN2O/ppbv)

    iirf100_max:^ (float)
      maximum allowed value of iIRF100 (keeps the model stable) (yrs)

    in_state:^ (list/np.array)
      initial state of the climate system with elements:
        [0]: (np.array/list)
          co_2 concentration of each carbon pool (ppmv)
        [1]: (np.array/list)
          temp of each temperature response box (K)
        [2]: (float)
          cumulative carbon uptake (GtC)
        [3]: (float)
          CH4 concentration (ppbv)
        [4]: (float)
          N2O concentration (ppbv)

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

    # # ------------ LOCAL APPLICATION/LIBRARY SPECIFIC ------------ # #

    # # # ------------ CODE ------------ # # #

    # # # ------------ CALCULATE Q ARRAY ------------ # # #
    # If TCR and ECS are supplied, overwrite the q array
    k = 1.0 - (d/70.0)*(1.0 - np.exp(-70.0/d))
    if type(tcrecs) in [np.ndarray,list]:
        q =  (1.0 / F_2x) * (1.0/(k[0]-k[1])) \
            * np.array([tcrecs[0]-k[1]*tcrecs[1],k[0]*tcrecs[1]-tcrecs[0]])

    # # # ------------ SET UP OUTPUT TIMESERIES VARIABLES ------------ # # #
    # the integ_len variable is used to store the length of our timeseries
    # by default FAIR is not concentration driven
    conc_driven=False
    
	# here we check if we want to include CH4 or N2O ie. if they are given as arguments in fair_scm
    if all(type(x) == bool for x in [M_emissions,N_emissions,M_concs,N_concs]):
        include_M_N = [False,False]
    elif any(type(x) != bool for x in [M_emissions,M_concs]):
	    include_M_N = [True,False]
    elif any(type(x) != bool for x in [N_emissions,N_concs]):
	    include_M_N = [False,True]
    else:
	    include_M_N = [True,True]
	
	# here we check if FAIR is emissions driven, for now assuming if CO_2 is emissions driven, the other GHGs are as well
    if type(emissions) in [np.ndarray,list]:
        integ_len = len(emissions)
        [emissions,M_emissions,N_emissions,other_rf] = emissions_concentrations_sort(emissions,M_emissions,N_emissions,other_rf,integ_len)
        # if (type(other_rf) in [np.ndarray,list]) and (len(other_rf)!=integ_len):
            # raise ValueError("The emissions and other_rf timeseries don't have the same length")
        # elif type(other_rf) in [int,float]:
            # other_rf = np.full(integ_len,other_rf)
  
    # here we check if FAIR is concentration driven
    elif type(co2_concs) in [np.ndarray,list]:
        integ_len = len(co2_concs)
        conc_driven = True
        [co2_concs,M_concs,N_concs,other_rf,integ_len] = emissions_concentrations_sort(co2_concs,M_concs,N_concs,other_rf,integ_len)
        # if (type(other_rf) in [np.ndarray,list]) and (len(other_rf)!=integ_len):
            # raise ValueError("The concentrations and other_rf timeseries don't have the same length")
        # elif type(other_rf) in [int,float]:
            # other_rf = np.full(integ_len,other_rf)

    # finally we check if only a non-CO2 radiative forcing timeseries has been supplied
    elif type(other_rf) in [np.ndarray,list]:
        integ_len = len(other_rf)
        [emissions,M_emissions,N_emissions,other_rf] = emissions_concentrations_sort(emissions,M_emissions,N_emissions,other_rf,integ_len)
        # if type(emissions) in [int,float]:
            # emissions = np.full(integ_len,emissions)
        # else:
            # emissions = np.zeros(integ_len)

    else:
        raise ValueError("Neither emissions, co2_concs or other_rf is defined as a timeseries")

    RF = np.zeros(integ_len)
    C_acc = np.zeros(integ_len)
    iirf100 = np.zeros(integ_len)

    carbon_boxes_shape = (integ_len,4)
    R_i = np.zeros(carbon_boxes_shape)
    C = np.zeros(integ_len)
    M = np.zeros(integ_len)
    N = np.zeros(integ_len)
    
    thermal_boxes_shape = (integ_len,2)
    T_j = np.zeros(thermal_boxes_shape)
    T = np.zeros(integ_len)

    # # # ------------ FIRST TIMESTEP ------------ # # #
    R_i_pre = in_state[0]
    C_pre = np.sum(R_i_pre) + C_0
    T_j_pre = in_state[1]
    C_acc_pre = in_state[2]
    M_pre = in_state[3]
    N_pre = in_state[4]

    if conc_driven:
        C[0] = co2_concs[0]
        M[0] = M_concs[0]
        N[0] = N_concs[0]
  
    else:
        # Calculate the parametrised iIRF and check if it is over the maximum 
        # allowed value
        iirf100[0] = r0 + rC*C_acc_pre + rT*np.sum(T_j_pre)
        if iirf100[0] >= iirf100_max:
          iirf100[0] = iirf100_max
          
        # Determine a solution for alpha using scipy's root finder
        time_scale_sf = (root(iirf100_interp_funct,0.16,args=(a,tau,iirf100[0])))['x']

        # Multiply default timescales by scale factor
        tau_new = time_scale_sf * tau

        # Compute the updated concentrations box anomalies from the decay of the 
        # previous year and the emisisons
        R_i[0] = R_i_pre*np.exp(-tstep/tau_new) \
                  + (emissions[0,np.newaxis])*a*tau_new*(1-np.exp(-tstep/tau_new)) / ppm_gtc

        C[0] = np.sum(R_i[0]) + C_0
        
        # Compute the concentrations of the other GHGs from the decay of the previous year and yearly emissions
        M[0] = M_pre*np.exp(-tstep/tau_M) + M_emissions[0]*tau_M*(1-np.exp(-tstep/tau_M)) / ppb_TgM
        
        N[0] = N_pre*np.exp(-tstep/tau_N) + N_emissions[0]*tau_N*(1-np.exp(-tstep/tau_N)) / ppb_TgN

        # Calculate the additional carbon uptake
        C_acc[0] =  C_acc_pre + emissions[0] - (C[0]-(np.sum(R_i_pre) + C_0)) * ppm_gtc

    # Calculate the radiative forcing using the previous timestep's CO2 concentration

    RF[0] = (F_2x/np.log(2.)) * np.log(C_pre/C_0) + other_rf[0] + RF_M(M_pre,N_pre,M_0,N_0) + RF_N(M_pre,N_pre,M_0,N_0)

    # Update the thermal response boxes
    T_j[0] = RF[0,np.newaxis]*q*(1-np.exp((-tstep)/d)) + T_j_pre*np.exp(-tstep/d)

    # Sum the thermal response boxes to get the total temperature anomlay
    T[0] = np.sum(T_j[0])

    # # # ------------ REST OF RUN ------------ # # #
    for x in range(1,integ_len):
        if conc_driven:
          C[x] = co2_concs[x]
          M[x] = M_concs[x]
          N[x] = N_concs[x]
        
        else:
          # Calculate the parametrised iIRF and check if it is over the maximum 
          # allowed value
          iirf100[x] = r0 + rC*C_acc[x-1] + rT*T[x-1]
          if iirf100[x] >= iirf100_max:
            iirf100[x] = iirf100_max
            
          # Determine a solution for alpha using scipy's root finder
          time_scale_sf = (root(iirf100_interp_funct,time_scale_sf,args=(a,tau,iirf100[x])))['x']

          # Multiply default timescales by scale factor
          tau_new = time_scale_sf * tau

          # Compute the updated concentrations box anomalies from the decay of the previous year and the emisisons
          R_i[x] = R_i[x-1]*np.exp(-tstep/tau_new) \
                  + (emissions[x,np.newaxis])*a*tau_new*(1-np.exp(-tstep/tau_new)) / ppm_gtc

          # Sum the boxes to get the total concentration anomaly
          C[x] = np.sum(R_i[x]) + C_0
          
          # Compute the concentrations for the other GHGs from the decay of previous year and yearly emissions
          M[x] = M[x-1]*np.exp(-tstep/tau_M) + M_emissions[x]*tau_M*(1-np.exp(-tstep/tau_M)) / ppb_TgM
          
          N[x] = N[x-1]*np.exp(-tstep/tau_N) + N_emissions[x]*tau_N*(1-np.exp(-tstep/tau_N)) / ppb_TgN

          # Calculate the additional carbon uptake
          C_acc[x] =  C_acc[x-1] + emissions[x] * tstep - (C[x]-C[x-1]) * ppm_gtc

        # Calculate the radiative forcing using the previous timestep's CO2 concentration
        RF[x] = (F_2x/np.log(2.)) * np.log((C[x-1]) /C_0) + other_rf[x] + RF_M(M[x-1],N[x-1],M_0,N_0) + RF_N(M[x-1],N[x-1],M_0,N_0)

        # Update the thermal response boxes
        T_j[x] = T_j[x-1]*np.exp(-tstep/d) + RF[x,np.newaxis]*q*(1-np.exp(-tstep/d))
        
        # Sum the thermal response boxes to get the total temperature anomaly
        T[x] = np.sum(T_j[x])

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
    emmsax.set_ylabel('Emissions (GtC)')
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