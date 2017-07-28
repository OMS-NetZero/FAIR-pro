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

# Define a function that edits our emission or concentration arrays as appropriate based on arguments given
def emissions_concentrations_sort(Arr,
                                  length):
    """
    Takes input unperturbed/ perturbed concentrations of CH4 and N2O 
    and returns radiative forcing caused.
    # # ------------ ARGUMENTS ------------ # #
    sublime snippet for variable description in header is 'hvardesc'
    
    Arr:^ (np.array/list/float/int/bool)
      Array containing all gas emissions or concentrations 
      and other radiative forcing
      
    length:^ (float)
      length of the timeseries (Yr)
    ^ => Keyword argument
    # # ------------ RETURN VALUE ------------ # #
    sublime snippet for variable description in header is 'hvardesc'
    Returns an array containing the edited emissions or concentrations
    and other radiative forcings as follows:
    -leaves lists/arrays alone
    -replaces int/float type with array at that constant value for length
    -replaces False type with array of zeros of length
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

    # # ------------ LOCAL APPLICATION/LIBRARY SPECIFIC ------------ # #

    # # # ------------ CODE ------------ # # #
    for i,x in enumerate(Arr):
        if type(x) in [int,float]:
            Arr[i] = np.full(length,x)
        elif type(x) == bool:
            Arr[i] = np.zeros(length)
        elif len(x) != length:
            raise ValueError("One or more of the emissions/concentrations given or other_rf timeseries doesn't have the same length")
    return Arr

# Define a function that gives the Radiative forcing due to CH4 as per Etminan et al. 2016, table 1
def RF_M(M,
         M_0=722.0,
         N_0=270.0,
         alp_m=0.036):
    """
    Takes input unperturbed/ perturbed concentrations of CH4 and N2O 
    and returns radiative forcing caused by CH4
    # # ------------ ARGUMENTS ------------ # #
    sublime snippet for variable description in header is 'hvardesc'
    
    M:^ (float)
      Current CH4 concentration (ppbv)
      
    N:^ (float)
      Current N2O concentration (ppbv)
      
    M_0:^ (float)
      Unperturbed CH4 concentration (ppbv)
      
    M:^ (float)
      Unperturbed N2O concentration (ppbv)
      
    a3:^ (float)
      constant given in Etminan et al. 2016, notation as 
      used there
      
    b3:^ (float)
      constant given in Etminan et al. 2016, notation as 
      used there
      
    K:^ (float)
      constant term given in Etminan et al. 2016.
    ^ => Keyword argument
    # # ------------ RETURN VALUE ------------ # #
    sublime snippet for variable description in header is 'hvardesc'
    Returns the radiative forcing due to CH4
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

    # # ------------ LOCAL APPLICATION/LIBRARY SPECIFIC ------------ # #

    # # # ------------ CODE ------------ # # #
        
    def f(M, N):
        return 0.47 * np.log(1 + 2.01 * 10**(-5) * (M * N)**0.75 + 5.31 * 10**(-15) * M * (M * N)**(1.52))
        
    return alp_m * (np.sqrt(M) - np.sqrt(M_0)) - (f(M, N_0) - f(M_0, N_0))
    
# Define a function that gives the Radiative forcing due to N2O as per Etminan et al. 2016, table 1
def RF_N(N,
         M_0=722.0,
         N_0=270.0,
         alp_n = 0.12):
    """
    Takes input unperturbed/ perturbed concentrations of CO2, CH4, N2O 
    and returns radiative forcing caused by N2O
    # # ------------ ARGUMENTS ------------ # #
    sublime snippet for variable description in header is 'hvardesc'
    
    C:^ (float)
      Current CO2 concentration (ppbv)
    
    M:^ (float)
      Current CH4 concentration (ppbv)
      
    N:^ (float)
      Current N2O concentration (ppbv)
    
    C_0:^ (float)
      Unperturbed CO2 concentration (ppbv)
    
    M_0:^ (float)
      Unperturbed CH4 concentration (ppbv)
      
    M:^ (float)
      Unperturbed N2O concentration (ppbv)
      
    a2:^ (float)
      constant given in Etminan et al. 2016, notation as 
      used there
      
    b2:^ (float)
      constant given in Etminan et al. 2016, notation as 
      used there
      
    c2:^ (float)
      constant given given in Etminan et al. 2016, notation as
      used there
      
    K:^ (float)
      Constant term as in Etminan et al. 2016.
    ^ => Keyword argument
    # # ------------ RETURN VALUE ------------ # #
    sublime snippet for variable description in header is 'hvardesc'
    Returns the radiative forcing due to N2O
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

    # # ------------ LOCAL APPLICATION/LIBRARY SPECIFIC ------------ # #

    # # # ------------ CODE ------------ # # #
    
    def f(M, N):
        return 0.47 * np.log(1 + 2.01 * 10**(-5) * (M * N)**0.75 + 5.31 * 10**(-15) * M * (M * N)**(1.52))
        
    return alp_n * (np.sqrt(N) - np.sqrt(N_0)) - (f(M_0, N) - f(M_0, N_0))
    
# Define a function that gives the Radiative forcing due to CO2 as per Etminan et al. 2016, table 1
def RF_C(C,
         C_0=278.0,
         F_2x=3.74):
    """
    Takes input unperturbed/ perturbed concentrations of CO2, N2O, CH4 
    and returns radiative forcing caused by CO2
    # # ------------ ARGUMENTS ------------ # #
    sublime snippet for variable description in header is 'hvardesc'
    
    C:^ (float)
      Current CO2 concentration (ppmv)
    
    M:^ (float)
      Current CH4 concentration (ppmv)
      
    N:^ (float)
      Current N2O concentration (ppbv)
    
    C_0:^ (float)
      Unperturbed CO2 concentration (ppbv)
    
    M_0:^ (float)
      Unperturbed CH4 concentration (ppbv)
      
    M:^ (float)
      Unperturbed N2O concentration (ppbv)
      
    a2:^ (float)
      constant given in Etminan et al. 2016, notation as 
      used there
      
    b2:^ (float)
      constant given in Etminan et al. 2016, notation as 
      used there
      
    c2:^ (float)
      constant given given in Etminan et al. 2016, notation as
      used there
      
    K:^ (float)
      Constant term as in Etminan et al. 2016.
    ^ => Keyword argument
    # # ------------ RETURN VALUE ------------ # #
    sublime snippet for variable description in header is 'hvardesc'
    Returns the radiative forcing due to CO2
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

    # # ------------ LOCAL APPLICATION/LIBRARY SPECIFIC ------------ # #

    # # # ------------ CODE ------------ # # #
    
    return (F_2x / np.log(2.0)) * np.log(C/C_0)

# Define a function that returns the radiative forcing due to any other trace gases not considered explicitly (eg. CFC-11, CFC12, HFC134a etc.)
def RF_other_gases(conc,
                   conc_0,
                   RE):
    """
    Takes input unperturbed/ perturbed concentrations of any other trace gas 
    and returns radiative forcing caused.
    # # ------------ ARGUMENTS ------------ # #
    sublime snippet for variable description in header is 'hvardesc'
    
    conc:^ (float/np.array)
      Current gas concentration (ppbv)
    
    conc_0:^ (float/np.array)
      Unperturbed gas concentration (ppbv)
      
    RE:^ (float/np.array)
      radiative efficiency of the gas species (W/m^2(ppbv)^-1)
    ^ => Keyword argument
    # # ------------ RETURN VALUE ------------ # #
    sublime snippet for variable description in header is 'hvardesc'
    Returns the radiative forcing due the gas
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

    # # ------------ LOCAL APPLICATION/LIBRARY SPECIFIC ------------ # #

    # # # ------------ CODE ------------ # # #
    
    return RE*(conc-conc_0)

# Define the FAIR simple climate model function
def fair_scm(tstep=1.0,
             emissions=False,
             M_emissions=False,
             N_emissions=False,
             MK_gas_emissions_def=[False,False,False,False,False,False],
             other_rf=0.0,
             co2_concs=False,
             M_concs=False,
             N_concs=False,
             MK_gas_concs_def=[False,False,False,False,False,False],
             q=np.array([0.33,0.41]),
             tcrecs=np.array([1.6,2.75]),
             d=np.array([239.0,4.1]),
             a=np.array([0.2173,0.2240,0.2824,0.2763]),
             tau=np.array([1000000,394.4,36.54,4.304]),
             tau_MK_gas=np.array([57.0,143.0,118.0,12.2,13.9,31.3]),
             r0=32.40,
             rC=0.019,
             rT=4.165,
             F_2x=3.74,
             MK_gas_RE=np.array([0.26,0.32,0.30,0.21,0.16,0.17]),
             C_0=278.0,
             M_0=722.0,
             N_0=270.0,
             MK_gas_0=np.zeros(6),
             ppm_gtc=2.123,
             ppb_MtCH4=2.78,
             ppb_MtN2O=7.559,
             ppb_KtX=np.array([137.37,120.91,187.38,86.47,102.03,153.82])*10**(3)/5.6523,
             iirf100_max=97.0,
             in_state=[[0.0,0.0,0.0,0.0],[0.0,0.0],0.0,0.0,0.0,[0.0,0.0,0.0,0.0,0.0,0.0]],
             restart_out=False,
             MAGICC_model = False,
             S_OH_CH4 = -0.29,
             S_T_CH4 = 0.0316,
             tau_M_0 = 9.6,
             tau_N_0=121.0,
             S_N2O = -0.05):
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
      CH4 emissions timeseries (MtCH4/yr). If a scalar then emissions are 
      assumed to be constant throughout the run. If false then CH4 emissions 
      aren't used.
	  
    N_emissions:^ (np.array/list/float/int/bool)
      N2O emissions timeseries (MtN2O/yr). If a scalar then emissions are 
      assumed to be constant throughout the run. If false then N2O emissions 
      aren't used.
      
    MK_gas_emissions:^ (np.array/list)
      2D array containing selected Montreal/Kyoto gas emissions 
      timeseries (KtX/yr). If any element within scalar then emissions are 
      assumed to be constant throughout the run for that species. 
      If false then these emissions aren't used. The gases (currently) used are: 
      [CFC-11,CFC-12,CFC-113,HCFC-22,HCFC-134,CCl4] in that order. The array
      must be in the format: [[CFC-11(0),CFC-11(1),...],[CFC-12(0),...],...]
      ie. gas species 0th dimension, timeseries 1st dimension.
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
      
    MK_gas_concs:^ (np.array/list)
      2D array containing selected Montreal/Kyoto gas concentrations 
      timeseries (ppbv). If MK_gas_emissions are supplied then NK_gas_concs 
      is not used. The gases (currently) used are: 
      [CFC-11,CFC-12,CFC-113,HCFC-22,HCFC-134,CCl4] in that order. The array
      must be in the format: [[CFC-11(0),CFC-11(1),...],[CFC-12(0),...],...]
      ie. gas species 0th dimension, timeseries 1st dimension.
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
      
    tau_MK_gas:^ (list/np.array)
      lifetimes of selected gases. Order as per MK_gas_emissions (yrs)
    r0:^ (float)
      pre-industrial 100-year integrated impulse response (iIRF100) (yrs)
    rC:^ (float)
      sensitivity of iIRF100 to CO_2 uptake by the land and oceans (yrs/GtC)
    rT:^ (float)
      sensitivity of iIRF100 to increases in global mean temperature (yrs/K)
    F_2x:^ (float)
      radiative forcing due to a doubling of atmospheric CO_2 concentrations 
      (W/m^2)
      
    MK_gas_RE:^ (np.array/list)
      array containing radiative efficiency data for selected Montreal/Kyoto 
      gases, in same order as MK_gas_emissions (W/m^2 ppb^-1)
    C_0:^ (float)
      pre-industrial atmospheric CO_2 concentrations (ppmv)
	  
    M_0:^ (float)
      pre-industrial atmospheric CH4 concentrations (ppbv)
	  
    N_0:^ (float)
      pre-industrial atmospheric N2O concentrations (ppbv)
      
    MK_gas_0:^ (np.array/list)
      pre-industrial atmospheric concentrations of selected Montreal/Kyoto
      gases (ppbv), order as MK_gas_emissions
    ppm_gtc:^ (float)
      ppmv to GtC conversion factor (GtC/ppmv)
	  
    ppb_MtCH4:^ (float)
      ppbv to Mt of CH4 conversion factor. Taken from MAGICC (MtCH4/ppbv)
	  
    ppb_MtN2O:^ (float)
      ppbv to Mt of N2O conversion factor. Taken from MAGICC (MtN2O/ppbv)
      
    ppb_KtX:^ (np.array,list)
      ppbv to Kt of selected Montreal/Kyoto gas conversion factor (KtX/ppbv),
      order same as MK_gas_emissions.
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
          
        [5]: (array)
          Montreal/Kyoto gas concentrations (ppbv)
    restart_out:^ (bool)
      whether to return the final state of the climate system or not
      
    MAGICC_model:^ (bool)
      whether to use approximate MAGICC CH4 and N2O lifetime models
      
    S_OH_CH4:^ (float)
      Sensitivity coefficient to use for MAGICC CH4 lifetime 
      OH response
      
    S_T_CH4:^ (float)
      Sensitivity coefficient to use for MAGICC CH4 lifetime
      temperature response (/K)
      
    tau_M_0:^ (float)
      pre-industrial tropospheric methane lifetime (Yrs)
      
    tau_N_0:^ (float)
      pre-industrial N2O lifetime (Yrs)
      
    S_N2O:^ (float)
      sensitivity coefficient of N2O on itself ()
    ^ => Keyword argument
    # # ------------ RETURN VALUE ------------ # #
    sublime snippet for variable description in header is 'hvardesc'
    # ------------ DEFAULT ------------ #
    C: (np.array)
      timeseries of atmospheric CO_2 concentrations (ppmv)
    T: (np.array)
      timeseries of global mean temperatures (K)
      
    RF:(np.array)
      timeseries of total radiative forcing (W/m^2)
      
    M:(np.array)
      timeseries of atmospheric CH4 concentrations (ppbv)
      
    N: (np.array)
      timeseries of atmospheric N2O concentrations (ppbv)
      
    MK_gas: (np.ndarray)
      2D array containing concentration timeseries of the Montreal/Kyoto
      gases included (ppbv) in the order:
      [CFC-11,CFC-12,CFC-113,HCFC-22,HCFC-134,CCl4]
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
    import pandas as pd

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
    MK_gas_emissions = MK_gas_emissions_def[:]
    MK_gas_concs = MK_gas_concs_def[:]
    
	# here we check if FAIR is emissions driven, for now assuming if CO_2 is emissions driven, the other GHGs are as well
    # swapaxes is needed for MK_gas_emissions since the following code needs it to be a format with time down, whereas the input has time across
    if type(emissions) in [np.ndarray,list]:
        integ_len = len(emissions)
        [emissions,M_emissions,N_emissions,other_rf] = emissions_concentrations_sort([emissions,M_emissions,N_emissions,other_rf],integ_len)
        MK_gas_emissions = np.swapaxes(emissions_concentrations_sort(MK_gas_emissions,integ_len),0,1)   

    # here we check if FAIR is concentration driven
    elif type(co2_concs) in [np.ndarray,list]:
        integ_len = len(co2_concs)
        conc_driven = True
        [co2_concs,M_concs,N_concs,other_rf] = emissions_concentrations_sort([co2_concs,M_concs,N_concs,other_rf],integ_len)
        MK_gas_concs = np.swapaxes(emissions_concentrations_sort(MK_gas_concs,integ_len),0,1)

    # finally we check if only a non-CO2 radiative forcing timeseries has been supplied
    elif type(other_rf) in [np.ndarray,list]:
        integ_len = len(other_rf)
        [emissions,M_emissions,N_emissions,other_rf] = emissions_concentrations_sort([emissions,M_emissions,N_emissions,other_rf],integ_len)
        MK_gas_emissions = np.swapaxes(emissions_concentrations_sort(MK_gas_emissions,integ_len),0,1)

    else:
        raise ValueError("Neither emissions, co2_concs or other_rf is defined as a timeseries")

    RF = np.zeros(integ_len)
    C_acc = np.zeros(integ_len)
    iirf100 = np.zeros(integ_len)
    M_iirf100 = np.zeros(integ_len)
    M_lifetime = np.zeros(integ_len)
    N_lifetime = np.zeros(integ_len)

    carbon_boxes_shape = (integ_len,4)
    R_i = np.zeros(carbon_boxes_shape)
    C = np.zeros(integ_len)
    M = np.zeros(integ_len)
    N = np.zeros(integ_len)
    MK_gas = np.zeros((integ_len,6))
    
    thermal_boxes_shape = (integ_len,2)
    T_j = np.zeros(thermal_boxes_shape)
    T = np.zeros(integ_len)
    
    co2_RF = np.zeros(integ_len)
    M_RF = np.zeros(integ_len)
    N_RF = np.zeros(integ_len)
    MK_gas_RF = np.zeros((integ_len,6))

    # # # ------------ FIRST TIMESTEP ------------ # # #
    R_i_pre = in_state[0]
    C_pre = np.sum(R_i_pre) + C_0
    T_j_pre = in_state[1]
    C_acc_pre = in_state[2]
    M_pre = in_state[3] + M_0
    N_pre = in_state[4] + N_0
    MK_gas_pre = in_state[5] + MK_gas_0

    if conc_driven:
        C[0] = co2_concs[0]
        M[0] = M_concs[0]
        N[0] = N_concs[0]
        MK_gas[0] = MK_gas_concs[0]

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
        
        # Now do the same thing for Methane OH lifetime:
        if MAGICC_model:
            tropOH = S_OH_CH4 * (np.log(M_pre) - np.log(M_0))
            tau_1 = tau_M_0 * np.exp(-1*tropOH)
            tau_CH4_trop = tau_M_0 / ((tau_M_0 / tau_1) + S_T_CH4 * np.sum(T_j_pre))
            tau_M_new = (1/tau_CH4_trop + 1/120.0 + 1/150.0 + 1/200.0)**(-1)
        else:
            tau_M_new = (1/tau_M_0 + 1/120.0 + 1/150.0 + 1/200.0)**(-1) # add all the Methane lifetime factors to the OH lifetime impact
            
        M_lifetime[0] = tau_M_new
        
        # Finally the same for N2O
        if MAGICC_model:
            tau_N_new = tau_N_0 * (N_pre / N_0) ** (S_N2O)
        else:
            tau_N_new = tau_N_0
            
        N_lifetime[0] = tau_N_new
        
        # Compute the updated concentrations box anomalies from the decay of the 
        # previous year and the emisisons
        R_i[0] = R_i_pre*np.exp(-tstep/tau_new) \
                  + (emissions[0,np.newaxis])*a*tau_new*(1-np.exp(-tstep/tau_new)) / ppm_gtc

        C[0] = np.sum(R_i[0]) + C_0
        
        # Compute the concentrations of the other GHGs from the decay of the previous year and yearly emissions (NB. M_pre - M_0 is the concentration anomaly)
        M[0] = (M_pre)*np.exp(-tstep/tau_M_new) \
                + M_emissions[0]*tau_M_new*(1-np.exp(-tstep/tau_M_new)) / ppb_MtCH4
            
        N[0] = (N_pre)*np.exp(-tstep/tau_N_new) \
                + N_emissions[0]*tau_N_new*(1-np.exp(-tstep/tau_N_new)) / ppb_MtN2O
        
        MK_gas[0] = (MK_gas_pre-MK_gas_0)*np.exp(-tstep/tau_MK_gas) \
                     + MK_gas_emissions[0]*tau_MK_gas*(1-np.exp(-tstep/tau_MK_gas)) / ppb_KtX \
                     + MK_gas_0

        # Calculate the additional carbon uptake
        C_acc[0] =  C_acc_pre + emissions[0] - (C[0]-(np.sum(R_i_pre) + C_0)) * ppm_gtc

    # Calculate the radiative forcing due to each gas, storing each in a separate array, then sum them to obtain the total RF
    co2_RF[0] = RF_C(C=C_pre,C_0=C_0)
    M_RF[0] = RF_M(M=M_pre,M_0=M_0,N_0=N_0)
    N_RF[0] = RF_N(N=N_pre,N_0=N_0)
    MK_gas_RF[0] = RF_other_gases(MK_gas_pre,MK_gas_0,MK_gas_RE)
    
    RF[0] = co2_RF[0] + M_RF[0] + N_RF[0] + np.sum(MK_gas_RF[0]) + other_rf[0]

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
          MK_gas[x] = MK_gas_concs[x]
        
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
          
          # Now same calculation for Methane using MAGICC lifetime model
          
          if MAGICC_model:
            tropOH = S_OH_CH4 * (np.log(M[x-1]) - np.log(M_0))
            tau_1 = tau_M_0 * np.exp(-1*tropOH)
            tau_CH4_trop = tau_M_0 / ((tau_M_0 / tau_1) + S_T_CH4 * T[x-1])
            tau_M_new = (1/tau_CH4_trop + 1/120.0 + 1/150.0 + 1/200.0)**(-1)
          else:
            tau_M_new = (1/tau_M_0 + 1/120.0 + 1/150.0 + 1/200.0)**(-1)
          
          M_lifetime[x] = tau_M_new

          # Finally the same for N2O (MAGICC/TAR lifetime formula)
          if MAGICC_model:
            tau_N_new = tau_N_0 * (N[x-1] / N_0) ** (S_N2O)
          else:
            tau_N_new = tau_N_0
            
          N_lifetime[x] = tau_N_new
        
        # Compute the updated concentrations box anomalies from the decay of the previous year and the emisisons
          R_i[x] = R_i[x-1]*np.exp(-tstep/tau_new) \
                  + (emissions[x,np.newaxis])*a*tau_new*(1-np.exp(-tstep/tau_new)) / ppm_gtc

          # Sum the boxes to get the total concentration anomaly
          C[x] = np.sum(R_i[x]) + C_0
          
          # Compute the concentrations for the other GHGs from the decay of previous year and yearly emissions (NB. M[x-1] - M_0 is the concentration anomaly)
          M[x] = (M[x-1])*np.exp(-tstep/tau_M_new) \
                  + M_emissions[x]*tau_M_new*(1-np.exp(-tstep/tau_M_new)) / ppb_MtCH4 \
          
          N[x] = (N[x-1])*np.exp(-tstep/tau_N_new) \
                  + N_emissions[x]*tau_N_new*(1-np.exp(-tstep/tau_N_new)) / ppb_MtN2O \
                  
          MK_gas[x] = (MK_gas[x-1]-MK_gas_0)*np.exp(-tstep/tau_MK_gas) \
                     + MK_gas_emissions[x]*tau_MK_gas*(1-np.exp(-tstep/tau_MK_gas)) / ppb_KtX \
                     + MK_gas_0

          # Calculate the additional carbon uptake
          C_acc[x] =  C_acc[x-1] + emissions[x] * tstep - (C[x]-C[x-1]) * ppm_gtc

        # Calculate the individual and total radiative forcing using the previous timestep's gas concentrations
        co2_RF[x] = RF_C(C=C[x-1],C_0=C_0)
        M_RF[x] = RF_M(M=M[x-1],M_0=M_0,N_0=N_0)
        N_RF[x] = RF_N(N[x-1],N_0=N_0)
        MK_gas_RF[x] = RF_other_gases(MK_gas[x-1],MK_gas_0,MK_gas_RE)
    
        RF[x] = co2_RF[x] + M_RF[x] + N_RF[x] + np.sum(MK_gas_RF[x]) + other_rf[x]
        
        # Update the thermal response boxes
        T_j[x] = T_j[x-1]*np.exp(-tstep/d) + RF[x,np.newaxis]*q*(1-np.exp(-tstep/d))
        
        # Sum the thermal response boxes to get the total temperature anomaly
        T[x] = np.sum(T_j[x])

    # Now we gather together all the relevant data such that we can output it all in a nested dict. We have to swapaxes the MK gases to get them in the right format.
    MK_gas = MK_gas.swapaxes(0,1)
    MK_gas_emissions = MK_gas_emissions.swapaxes(0,1)
    MK_gas_RF = MK_gas_RF.swapaxes(0,1)
    
    # Creating the dictionaries to then be nested within one output
    emissions_out = {'CO2' : emissions, 'CH4' : M_emissions, 'N2O' : N_emissions,
                     'CFC11' : MK_gas_emissions[0], 'CFC12' : MK_gas_emissions[1], 'CFC113' : MK_gas_emissions[2], 
                     'HCFC22' : MK_gas_emissions[3], 'HFC134a' : MK_gas_emissions[4], 'CCl4' : MK_gas_emissions[5]}
    concentration_out = {'CO2' : C, 'CH4' : M, 'N2O' : N ,
                         'CFC11' : MK_gas[0], 'CFC12' : MK_gas[1], 'CFC113' : MK_gas[2], 'HCFC22' : MK_gas[3], 'HFC134a' : MK_gas[4], 'CCl4' : MK_gas[5]}
    forcing_out = {'total' : RF, 'other' : other_rf, 'CO2' : co2_RF, 'CH4' : M_RF, 'N2O' : N_RF,
                   'CFC11' : MK_gas_RF[0], 'CFC12' : MK_gas_RF[1], 'CFC113' : MK_gas_RF[2], 
                   'HCFC22' : MK_gas_RF[3], 'HFC134a' : MK_gas_RF[4], 'CCl4' : MK_gas_RF[5]}
    lifetime_out = {'CH4' : M_lifetime, 'N2O': N_lifetime}
    
    # Create the output dictionary
    out = {'emissions' : emissions_out , 'concentration' : concentration_out , 'forcing' : forcing_out , 'lifetime' : lifetime_out , 'temperature' : T} 
    
    if restart_out:
        return C, T, (R_i[-1],T_j[-1],C_acc[-1])
    else:
        return out


def plot_fair(emms,
              M_emms,
              N_emms,
              conc,
              M_conc,
              N_conc,
              forc,
              temp,
              y_0=0,
              tuts=False,
              infig=False,
              inemmsax=None,
              inM_emmsax=None,
              inN_emmsax=None,
              inconcax=None,
              inM_concax=None,
              inN_concax=None,
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
    
    M_emms: (np.array/list)
      CH_4 emissions timeseries (MtCH4)
    N_emms: (np.array/list)
      N_2O emissions timeseries (MtN2O)
      
    conc: (np.array/list)
      CO_2 concentrations timeseries (ppmv)
      
    M_conc: (np.array/list)
      CH_4 concentrations timeseries (ppbv)
      
    N_conc: (np.array/list)
      N_2O concentrations timeseries (ppbv)
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
      pre-existing axis the CO_2 emissions should be plotted onto
      
    inM_emmsax:^ (subplots.AxesSubplot)
      pre-existing axis the CH_4 emissions should be plotted onto
      
    inN_emmsax:^ (subplots.AxesSubplot)
      pre-existing axis the N_2O emissions should be plotted onto
    inconcax:^ (subplots.AxesSubplot)
      pre-existing axis the CO_2 concentrations should be plotted onto
      
    inM_concax:^ (subplots.AxesSubplot)
      pre-existing axis the CH_4 concentrations should be plotted onto
      
    inN_concax:^ (subplots.AxesSubplot)
      pre-existing axis the N_2O concentrations should be plotted onto
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
      CO_2 emissions subplot
    concax: (subplots.AxesSubplot)
      CO_2 concentrations subplot
      
    M_emmsax: (subplots.AxesSubplot)
      CH_4 emissions subplot
    M_concax: (subplots.AxesSubplot)
      CH_4 concentrations subplot
      
    N_emmsax: (subplots.AxesSubplot)
      N_2O emissions subplot
    N_concax: (subplots.AxesSubplot)
      N_2O concentrations subplot
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
         'M_emms':M_emms,
         'N_emms':N_emms,
         'forc':forc,
         'conc':conc,
         'M_conc':M_conc,
         'N_conc':N_conc,
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
        fluxes = ['emms','M_emms','N_emms','forc']
        for f in fluxes:
            tmp = []
            for j,v in enumerate(pts[f]):
                for i in range(0,int(div)):
                    tmp.append(v)
            pts[f] = tmp
            
    else:
        ftime = time - 0.5
    
    if not infig:
        fig = plt.figure(figsize=(16,18))
        emmsax = fig.add_subplot(421)
        concax = fig.add_subplot(422)
        M_emmsax = fig.add_subplot(423)
        M_concax = fig.add_subplot(424)
        N_emmsax = fig.add_subplot(425)
        N_concax = fig.add_subplot(426)
        forcax = fig.add_subplot(427)
        tempax = fig.add_subplot(428)
    else:
        fig = infig
        emmsax = inemmsax
        concax = inconcax
        M_emmsax = inM_emmsax
        M_concax = inM_concax
        N_emmsax = inN_emmsax
        N_concax = inN_concax
        forcax = inforcax
        tempax = intempax

    emmsax.plot(ftime,pts['emms'],color=colour['emms'],label=label,ls=linestyle)
    emmsax.set_ylabel('CO$_2$ emissions (GtC)')
    if label is not None:
        emmsax.legend(loc='best')
    concax.plot(time,pts['conc'],color=colour['conc'],ls=linestyle)
    concax.set_ylabel('CO$_2$ concentrations (ppm)')
    concax.set_xlim(emmsax.get_xlim())
    M_emmsax.plot(ftime,pts['M_emms'],color=colour['emms'],ls=linestyle)
    M_emmsax.set_ylabel('CH$_4$ emissions (MtCH$_4$)')
    M_concax.plot(time,pts['M_conc'],color=colour['conc'],ls=linestyle)
    M_concax.set_ylabel('CH$_4$ concentrations (ppb)')
    M_concax.set_xlim(M_emmsax.get_xlim())
    N_emmsax.plot(ftime,pts['N_emms'],color=colour['emms'],ls=linestyle)
    N_emmsax.set_ylabel('N$_2$O emissions (MtN$_2$O)')
    N_concax.plot(time,pts['N_conc'],color=colour['conc'],ls=linestyle)
    N_concax.set_ylabel('N$_2$O concentrations (ppb)')
    N_concax.set_xlim(N_emmsax.get_xlim())
    forcax.plot(ftime,pts['forc'],color=colour['forc'],ls=linestyle)
    forcax.set_ylabel('Non-CO$_2$/CH$_4$/N$_2$O radiative forcing (W.m$^{-2}$)')
    forcax.set_xlabel('Time ({0})'.format(tuts))
    tempax.plot(time,pts['temp'],color=colour['temp'],ls=linestyle)
    tempax.set_ylabel('Temperature anomaly (K)')
    tempax.set_xlabel(forcax.get_xlabel())
    tempax.set_xlim(forcax.get_xlim())
    fig.tight_layout()

    return fig,emmsax,concax,M_emmsax,M_concax,N_emmsax,N_concax,forcax,tempax