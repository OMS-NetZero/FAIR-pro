"""
Module for doing calculations with the FAIR simple climate model

# # ------------ CLASSES ------------ # #
FAIR: Class for computing with the FAIR simple climate model
"""

# # # ------------ IMPORT REQUIRED MODULES ------------ # # #
# # ------------ STANDARD LIBRARY ------------ # #

# # ------------ THIRD PARTY ------------ # #
import numpy as np

# # ------------ LOCAL APPLICATION/LIBRARY SPECIFIC ------------ # #



class FAIR(object):
    """
    FAIR: Class for computing with the FAIR simple climate model

    # # ------------ METHODS ------------ # #
    run: performs a full run of the model for a given emissions timeseries

    time_step: performs a single timestep

    calc_k_q: calculates k and q arrays from FAIR parameters

    iirf_interp_function: calculate the integrated impulse response target function

    get_tau_sf: calculate the carbon pool decay timeconstant scaling factor for a given target value of the integrated impulse response

    print_para: print the current value of a FAIR instance's parameters

    tune_carbon_cycle: tune FAIR's carbon cycle to another model's emissions to forcing response

    sort_inputs: check FAIR inputs to make sure they're compatible and determine how FAIR is to be driven

    plot: plot FAIR's key variables

    # # ------------ INSTANCE VARIABLES ------------ # #
    All instance variables are optional. See .__init__.__doc__ for options.

    # # ------------ SUBCLASS INTERFACES ------------ # #
    will work out what this means once I start using them
    """

    # One line break before anything else
    # # # ------------ CODE ------------ # # #    
    # sublime snippet for a method is 'met'
    # your first method will probably be '__init__'

    def __init__(self,
                 tstep=1.0,
                 emissions=False,
                 other_rf=0.0,
                 co2_concs_in=False,
                 R_i_pre=[0.0,0.0,0.0,0.0],
                 T_j_pre=[0.0,0.0],
                 C_acc_pre=0.0,
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
                 ):
        """
        Initialises an instance of the FAIR class. 

        Sets attributes and properties based off the input arguments. Performs 
        checks for ...
    
        # # ------------ ARGUMENTS ------------ # #
        tstep:^ (float)
          Length of a timestep (yrs)

        emissions:^ (np.array/list/float/int/bool)
          CO_2 emissions timeseries (GtC/yr). If a scalar then emissions are 
          assumed to be constant throughout the run. If false then emissions 
          aren't used.

        other_rf:^ (np.array/list/float/int)
          Non-CO_2 radiative forcing timeseries (W/m^2). If a scalar then other_rf 
          is assumed to be constant throughout the run.

        co2_concs_in:^ (np.array/list/bool)
          Atmospheric CO_2 concentrations timeseries (ppmv). If emissions are 
          supplied then co2_concs_in is not used.
        
        R_i_pre: (np.array/list/bool)
          Atmospheric CO_2 concentrations in each box just before the run begins

        T_j_pre: (np.array/list/bool)
          Surface temperatures in each pool just before the run begins

        C_acc_pre: (float/int)
          Total CO_2 uptake by land and oceans up until the start of the run

        q:^ (np.array)
          response of each thermal box to radiative forcing (K/(W/m^2)). 
          Over-written if tcrecs is supplied.

        tcrecs:^ (np.array)
          array with elements:
            [0]: (float/int)
              Transient climate response i.e. TCR (K)
            [1]: (float/int)
              equilibrium climate sensitivity i.e. ECS (K)

        d:^ (np.array)
          response time of each thermal box (yrs)

        a:^ (np.array)
          fraction of emitted carbon which goes into each carbon pool 
          (dimensionless)

        tau:^ (np.array)
          unscaled response time of each carbon pool (yrs)

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

        ppm_gtc:^ (float)
          ppmv to GtC conversion factor (GtC/ppmv)

        iirf100_max:^ (float)
          maximum allowed value of iIRF100 (keeps the model stable) (yrs)

        ^ => Keyword argument
    
        # # ------------ RETURN VALUE ------------ # #
        sublime snippet for variable description in header is 'hvardesc'
    
        # # ------------ SIDE EFFECTS ------------ # #
        All the arguments are set as attributes of the FAIR instance. If 
        any of the emissions, concentrations or radiative forcing inputs are 
        floats then they are extended to be the same length as the supplied 
        timeseries.
        
        On top of these, the following are also set to be attributes of the 
        FAIR instance:
        
        conc_driven: (bool)
          whether the run is concentration driven or not
          
        n: (int)
          number of timesteps in our input arrays

        x: (int)
          timestep of the run we're up to (set to -1)

        sf_x: (float)
          carbon pool decay time scaling factor in timestep x (set to 0.16)

        R_i_x: (4 x 1 np.array)
          Atmospheric CO_2 anomaly in each box in the timestep x (set to 
          R_i_pre) (ppmv)

        C_pre: (float/int)
          Atmospheric CO_2 concentrations at the start of the run (ppmv)

        C_x: (float/int)
          Atmospheric CO_2 concentrations in the timestep x (set to 
          C_pre) (ppmv)

        T_j_x: (2 x 1 np.array)
          Surface temperature anomaly in each box in the timestep x (set to 
          T_j_pre) (K)

        T_pre: (float/int)
          Surface temperatures at the start of the run (K)

        T_x: (float/int)
          Surface temperatures in the timestep x (set to T_pre) (K)

        C_acc_x: (float/int)
          Accumulated CO_2 up to timestep x (set to C_acc_pre) (GtC)

        k: (2D np.array)
          realised warming in each temperature response after 70 
          years of 1%/yr CO2 increase as a fraction of equilibrium warming 
          (dimensionless)

        setting_inputs: (bool)
          If true, FAIR won't check to make sure that input timeseries are all the same. If false, every time FAIR's inputs are updated it will check to make sure that you have a coherent set. 
        
        If tcrecs is supplied as a valid 2D array then the q values are 
        re-calculated and may be over-written. If tcrecs is not supplied as a 
        valid 2D array its value is set to False.
        
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

        # # # ------------ SET ATTRIBUTES AND PROPERTIES ------------ # # #
        # have to set 'hidden' properties the first time otherwise the setter 
        # functions will try to calculate q and k when not all their 
        # properties are set
        # # ------------ INPUTS ------------ # #
        self.tstep = tstep
        self._emissions = emissions # property
        self._other_rf = other_rf # property
        self._co2_concs_in = co2_concs_in # property
        self.R_i_pre = R_i_pre
        self.T_j_pre = T_j_pre
        self.C_acc_pre = C_acc_pre
        self.q = q
        self._tcrecs = tcrecs # property
        self._d = d # property
        self.a = a
        self.tau = tau
        self.r0 = r0
        self.rC = rC
        self.rT = rT
        self._F_2x = F_2x # property
        self.C_0 = C_0
        self.ppm_gtc = ppm_gtc
        self.iirf100_max = iirf100_max

        # # ------------ DERIVED ------------ # #
        self.x = -1
        self.sf_x = 0.16
        self.R_i_x = self.R_i_pre
        self.C_pre = np.sum(self.R_i_pre) + self.C_0
        self.C_x = self.C_pre
        self.T_j_x = self.T_j_pre
        self.T_pre = np.sum(self.T_j_pre)
        self.T_x = self.T_pre
        self.C_acc_x = self.C_acc_pre
        
        # check input timeseries
        self.setting_inputs = False
        self.co2_concs_in = co2_concs_in
        
        # calculate k and q
        self.calc_k_q()

    def sort_inputs(self):
      """
      Sort out FAIR's input timeseries

      If self.setting_inputs == True then no checks are done. Warnings are printed when FAIR comes across something unexpected.
    
      # # ------------ ARGUMENTS ------------ # #
      sublime snippet for variable description in header is 'hvardesc'
    
      ^ => Keyword argument
    
      # # ------------ RETURN VALUE ------------ # #
      sublime snippet for variable description in header is 'hvardesc'
    
      # # ------------ SIDE EFFECTS ------------ # #
      May over-write some input timeseries but warnings are issued if so.
      
      # # ------------ EXCEPTIONS ------------ # #
      sublime snippet for exception description in header is 'hexcdesc'
    
      # # ------------ RESTRICTIONS ------------ # #
      Document any restrictions on when the function can be called
    
      """
    
      # One line break before anything else
      if not self.setting_inputs:
          print "Checking inputs"
          # # # ----------- SET UP OUTPUT TIMESERIES VARIABLES ----------- # # #
          # by default FAIR is not concentration driven
          self.conc_driven=False
          # here we check if FAIR is emissions driven
          if type(self.emissions) in [np.ndarray,list]:
              self.n = len(self.emissions)
              if type(self.emissions) in [list]:
                  self._emissions = np.array(self.emissions)
              if (type(self.other_rf) in [np.ndarray,list]) and (len(self.other_rf)!=self.n):
                  raise ValueError("The emissions and other_rf timeseries don't have the same length\n"
                                   + "If you're setting inputs then make sure FAIR.setting_inputs == True")
              elif type(self.other_rf) in [int,float]:
                  self._other_rf = np.full(self.n,self.other_rf)
                  print "other_rf is a scalar so will be constant throughout the timeseries"
              print "Ready for emissions driven work"
        
          # here we check if FAIR is concentration driven
          elif type(self.co2_concs_in) in [np.ndarray,list]:
              self.n = len(self.co2_concs_in)
              self.conc_driven = True
              if type(self.co2_concs_in) in [list]:
                  self._co2_concs_in = np.array(self.co2_concs_in)
              if (type(self.other_rf) in [np.ndarray,list]) and (len(self.other_rf)!=self.n):
                  raise ValueError("The concentrations and other_rf timeseries don't have the same length\n"
                                   + "If you're setting inputs then make sure FAIR.setting_inputs == True")
              elif type(self.other_rf) in [int,float]:
                  self._other_rf = np.full(self.n,self.other_rf)
                  print "other_rf is a scalar so will be constant throughout the timeseries"
              print "Ready for CO_2 concentration driven work"

          # finally we check if only a non-CO2 radiative forcing timeseries has been supplied
          elif type(self.other_rf) in [np.ndarray,list]:
              self.n = len(self.other_rf)
              if type(self.other_rf) in [list]:
                  self._other_rf = np.array(self.other_rf)
              if type(self.emissions) in [int,float]:
                  self._emissions = np.full(self.n,self.emissions)
                  print "emissions is a scalar so will be constant throughout the timeseries"
              else:
                  self._emissions = np.zeros(self.n)
                  print "couldn't find emissions, setting to zero"
              print "Ready for Non-CO_2 forcing driven work"

          else:
              raise ValueError("None of emissions, co2_concs_in or other_rf are defined as a timeseries")

    def calc_k_q(self):
        """
        Calculates k and q arrays from FAIR parameters
    
        # # ------------ ARGUMENTS ------------ # #
        sublime snippet for variable description in header is 'hvardesc'

        self: (FAIR object instance)
          FAIR object instance
    
        # # ------------ RETURN VALUE ------------ # #
        sublime snippet for variable description in header is 'hvardesc'
    
        # # ------------ SIDE EFFECTS ------------ # #
        Calculates and sets self.k and self.q of a FAIR object instance
        
        # # ------------ EXCEPTIONS ------------ # #
        sublime snippet for exception description in header is 'hexcdesc'
    
        # # ------------ RESTRICTIONS ------------ # #
        Document any restrictions on when the function can be called
    
        """
    
        # One line break before anything else
        # # # ------------ CALCULATE Q ARRAY ------------ # # #
        # If TCR and ECS are supplied, overwrite the q array
        self.k = 1.0 - (self.d/70.0)*(1.0 - np.exp(-70.0/self.d))
        if type(self.tcrecs) in [np.ndarray,list]:
            self.q =  (1.0 / self.F_2x) * (1.0/(self.k[0]-self.k[1])) \
                * np.array([self.tcrecs[0]-self.k[1]*self.tcrecs[1],self.k[0]*self.tcrecs[1]-self.tcrecs[0]])
        else:
            self.tcrecs = False

    def iirf_interp_funct(self,cp_scale,iirf_targ):
        """
        Calculate the integrated impulse response target function

        The iIRF calculation is done for a given scaling of the carbon pool 
        decay timeconstants. The difference between the calculated value and 
        the target 100 year integrated impulse response function is returned.
    
        # # ------------ ARGUMENTS ------------ # #
        sublime snippet for variable description in header is 'hvardesc'

        cp_scale: (int)
          scaling of the carbon pool decay timeconstants (dimensionless)

        iirf_targ: (float/int)
          target value of the 100yr integrated impulse response function (years)
    
        ^ => Keyword argument
    
        # # ------------ RETURN VALUE ------------ # #
        sublime snippet for variable description in header is 'hvardesc'

        iirf_arr - iirf_targ: (float/int)
          difference between 100yr iIRF and the target iIRF (years)
    
        # # ------------ SIDE EFFECTS ------------ # #
        document side effects here
        
        # # ------------ EXCEPTIONS ------------ # #
        sublime snippet for exception description in header is 'hexcdesc'
    
        # # ------------ RESTRICTIONS ------------ # #
        Document any restrictions on when the function can be called
    
        """
    
        # One line break before anything else
        iirf_arr = cp_scale*(np.sum(self.a*self.tau*(1.0 - np.exp(-100.0/(self.tau*cp_scale)))))
        return iirf_arr - iirf_targ

    def get_tau_sf(self,iirf_targ):
        """
        Returns the solution for the scaling of the carbon decay timeconstants.

        The solution is found when the calculated iIRF matches the target iIRF to within whatever tolerance is the default of the scipy root function
    
        # # ------------ ARGUMENTS ------------ # #
        sublime snippet for variable description in header is 'hvardesc'
    
        iirf_targ: (float/int)
          target value of the 100yr integrated impulse response function (years)

        ^ => Keyword argument
    
        # # ------------ RETURN VALUE ------------ # #
        sublime snippet for variable description in header is 'hvardesc'
    
        # # ------------ SIDE EFFECTS ------------ # #
        document side effects here
        
        # # ------------ EXCEPTIONS ------------ # #
        sublime snippet for exception description in header is 'hexcdesc'
    
        # # ------------ RESTRICTIONS ------------ # #
        Document any restrictions on when the function can be called
    
        """
    
        # One line break before anything else
        from scipy.optimize import root
        return (root(self.iirf_interp_funct,self.sf_x,args=(iirf_targ)))['x']
        
        
    def time_step(self):
        """
        Run FAIR for a single timestep

        Calculates and updates the FAIR instance's timestep variables.
    
        # # ------------ ARGUMENTS ------------ # #
        sublime snippet for variable description in header is 'hvardesc'
    
        ^ => Keyword argument
    
        # # ------------ RETURN VALUE ------------ # #
        sublime snippet for variable description in header is 'hvardesc'
    
        # # ------------ SIDE EFFECTS ------------ # #
        Creates/updates the following attributes:

        R_i_x_pre: (4 x 1 np.array)
          Atmospheric CO_2 concentrations in each pool in the timestep before 
          x (ppmv)

        C_x_pre: (float/int)
          Atmospheric CO_2 concentrations in the timestep before x (ppmv)

        T_j_x_pre: (2 x 1 np.array)
          Surface temperature anomaly in each box in the timestep before x (K)
    
        T_x_pre: (float/int)
          Surface temperatures in the timestep before x (K)

        C_acc_x_pre: (float/int)
          Accumulated CO_2 up to the timestep before x (GtC)

        x: (int)
          timestep of the run we're up to

        iirf100_x: (float/int)
          100yr integrated impulse response function in timestep x (yrs)

        sf_x: (float/int)
          carbon pool response time scaling factor in timestep x

        tau_x: (4 x 1 np.array)
          response time of each carbon pool in timestep x (yrs)

        R_i_x: (4 x 1 np.array)
          Atmospheric CO_2 anomaly in each box in the timestep x (ppmv)

        C_x: (float/int)
          Atmospheric CO_2 concentrations in the timestep x (set to 
          C_pre) (ppmv)

        T_x: (float/int)
          Surface temperatures in the timestep x (set to T_pre) (K)

        # # ------------ EXCEPTIONS ------------ # #
        sublime snippet for exception description in header is 'hexcdesc'
    
        # # ------------ RESTRICTIONS ------------ # #
        Document any restrictions on when the function can be called
        """
    
        # One line break before anything else
        # # # ------------ UPDATE PREVIOUS VALUES ------------ # # #
        self.R_i_x_pre = self.R_i_x
        self.C_x_pre = self.C_x
        self.T_j_x_pre = self.T_j_x
        self.T_x_pre = self.T_x
        self.C_acc_x_pre = self.C_acc_x
        
        self.x += 1

        if self.conc_driven:
            self.C_x = self.co2_concs_in[self.x]
        else:
            self.iirf100_x = self.r0 + self.rC*self.C_acc_x_pre + self.rT*self.T_x_pre
            
            if self.iirf100_x > self.iirf100_max:
                self.iirf100_x = self.iirf100_max

            self.sf_x = self.get_tau_sf(self.iirf100_x)
            self.tau_x = self.tau*self.sf_x

            self.R_i_x = (self.R_i_x_pre*np.exp(-self.tstep/self.tau_x) 
                          + self.emissions[self.x,np.newaxis]
                            * self.a*self.tau_x
                            * (1-np.exp(-self.tstep/self.tau_x)) 
                            / self.ppm_gtc)

            self.C_x = np.sum(self.R_i_x) + self.C_0
            self.C_acc_x = (self.C_acc_x_pre + self.emissions[self.x] 
                            - (self.C_x - self.C_x_pre)*self.ppm_gtc)

        self.RF_CO2_x = (self.F_2x/np.log(2.))*np.log((self.C_x_pre)/self.C_0)
        self.RF_x = self.RF_CO2_x + self.other_rf[self.x]

        self.T_j_x = (self.T_j_x_pre*np.exp(-self.tstep/self.d) 
                      + self.RF_x[np.newaxis]*self.q*(1-np.exp(-self.tstep/self.d)))
        self.T_x = np.sum(self.T_j_x)

    def run(self):
        """
        Run the model for the entire timeseries
    
        # # ------------ ARGUMENTS ------------ # #
        sublime snippet for variable description in header is 'hvardesc'
    
        ^ => Keyword argument
    
        # # ------------ RETURN VALUE ------------ # #
        sublime snippet for variable description in header is 'hvardesc'
    
        # # ------------ SIDE EFFECTS ------------ # #
        Sets/updates the following attributes:

        R_i: (self.n x 4 np.array)
          Atmospheric CO_2 concentrations in each of the four carbon pools at the end of each timestep (ppmv)

        C: (self.n x 1 np.array)
          Total atmospheric CO_2 anomaly at the end of each timestep (ppmv)

        T_j: (self.n x 2 np.array)
          Surface temperature anomaly in each response box at the end of each timestep (K)

        T: (self.n x 1 np.array)
          Total surface temperature anomaly at the end of each timestep (K)

        C_acc: (self.n x 1 np.array)
          Cumulative uptake of carbon at the end of each timestep (GtC)
        
        iirf100: (self.n x 1 np.array)
          100yr integrated impulse response function throughout each timestep (yrs)

        RF: (self.n x 1 np.array)
          Global mean radiative forcing throughout each timestep (W/m^2)

        # # ------------ EXCEPTIONS ------------ # #
        sublime snippet for exception description in header is 'hexcdesc'
    
        # # ------------ RESTRICTIONS ------------ # #
        Document any restrictions on when the function can be called
    
        """
    
        # One line break before anything else
        # # # ------------ SET UP OUTPUT ARRAYS ------------ # # #
        self.R_i = np.zeros((self.n,4))
        self.C = np.zeros((self.n))
        self.T_j = np.zeros((self.n,2))
        self.T = np.zeros((self.n))
        self.C_acc =np.zeros((self.n))
        self.iirf100 = np.zeros((self.n))
        self.RF_CO2 = np.zeros((self.n))
        self.RF = np.zeros((self.n))

        # ------------ INITIALISE VARIABLES ------------ #
        self.x = -1
        self.sf_x = 0.16
        self.R_i_x = self.R_i_pre
        self.C_pre = np.sum(self.R_i_pre) + self.C_0
        self.C_x = self.C_pre
        self.T_j_x = self.T_j_pre
        self.T_pre = np.sum(self.T_j_pre)
        self.T_x = self.T_pre
        self.C_acc_x = self.C_acc_pre
        
        for x in range(0,self.n):
            # run the timestep
            self.time_step()
            # save the output
            if not self.conc_driven:
                self.R_i[x] = self.R_i_x
                self.iirf100[x] = self.iirf100_x
            self.T_j[x] = self.T_j_x
            self.C[x] = self.C_x
            self.T[x] = self.T_x
            self.C_acc[x] = self.C_acc_x
            self.RF_CO2[x] = self.RF_CO2_x
            self.RF[x] = self.RF_x

    def print_para(self):
        """
        Print out FAIR input parameters in human readable form    
        """
    
        # One line break before anything else
        print 'TCR: {0} K'.format(self.tcrecs[0])
        print 'ECS: {0} K'.format(self.tcrecs[1])
        
        print 'd1: {0} yrs'.format(self.d[0])
        print 'd2: {0} yrs'.format(self.d[1])
        
        print 'a1: {0}'.format(self.a[0])
        print 'a2: {0}'.format(self.a[1])
        print 'a3: {0}'.format(self.a[2])
        print 'a4: {0}'.format(self.a[3])

        print 'tau1: {0} yrs'.format(self.tau[0])
        print 'tau2: {0} yrs'.format(self.tau[1])
        print 'tau3: {0} yrs'.format(self.tau[2])
        print 'tau4: {0} yrs'.format(self.tau[3])
        
        print 'r0: {0} yrs'.format(self.r0)
        print 'rC: {0} yrs/GtC'.format(self.rC)
        print 'rT: {0} yrs/K'.format(self.rT)

        print 'F_2x: {0} W/m^2'.format(self.F_2x)

        print 'C_0: {0} ppmv'.format(self.C_0)

        print 'ppmv --> GtC: {0} GtC/ppmv'.format(self.ppm_gtc)

        print 'Max iirf: {0} yrs'.format(self.iirf100_max) 
            
    def plot(self,
             y_0=0,
             tuts=False,
             colour={'emms':'black',
                 'conc':'blue',
                 'forc':'orange',
                 'temp':'red'},
             showplt=False
             ):
        """
        Function to plot FAIR variables

        Takes some of the work out of making a panel plot and ensures that the 
        variables appear as they are interpreted by fair e.g. fluxes are
        constant over the timestep rather than the default linear 
        interpolation between values as done by most plotting routines.
    
        # # ------------ ARGUMENTS ------------ # #
        y_0:^ (float/int)
          starting value of your timeseries, used to set min of time axis (
          same as time units)

        tuts:^ (string/bool)
          time units. If not supplied then 'units unknown' is printed
        
        colour:^ (dict)
          dictionary of colours to use for each timeseries

        showplt:^ (bool)
          whether to show the plot before returning the figure or not

        ^ => Keyword argument
    
        # # ------------ RETURN VALUE ------------ # #
        sublime snippet for variable description in header is 'hvardesc'
    
        # # ------------ SIDE EFFECTS ------------ # #
        Shows a plot of the instance's timeseries
        
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
        # read our input arrays into a dictionary which will hold all our plotting points
        pts = {'emms':self.emissions,
               'forc':self.RF,
               'forc-co2':self.RF_CO2,
               'forc-non-co2':self.other_rf,
               'conc':self.C,
               'temp':self.T}

        # define our flux variables, all others assumed to be state variables
        fluxes = ['emms','forc','forc-co2','forc-non-co2']
        integ_len = self.n

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

        if not tuts:
            tuts = 'units unknown'

        # # ------------ PREPARE STATE VARIABLES FOR PLOTTING ------------ # #
        # state variables are valid at the end of the timestep so we
        # go from 1 - integ_len + 1 rather than 0 - integ_len
        pts['stime'] = np.arange(0.99,integ_len+0.99) + y_0
        # pre-pend the pre-run value 
        pts['stime'] = np.insert(pts['stime'],0,0.0)
        pts['conc'] = np.insert(pts['conc'],0,self.C_pre)
        pts['temp'] = np.insert(pts['temp'],0,self.T_pre)

        # # ------------ PREPARE FLUX VARIABLES FOR PLOTTING  ------------ # #
        # Flux variables are assumed constant throughout the timestep. To make 
        # this appear on the plot we have to do the following if there's fewer 
        # than 1000 timesteps
        fmintsp = 1000
        if integ_len < fmintsp:
            # work out how many pieces you need to divide each timestep into 
            # in order to get 1000 timesteps
            div = ceil(fmintsp/integ_len)
            # work out how long each piece is (in the units of a timestep)
            plen = 1.0/div
            # create a flux timeseries
            pts['ftime'] = np.arange(0,integ_len,plen) + y_0
            # update our flux timeseries to show the step changes properly
            for f in fluxes:
                pts[f] = [v for v in pts[f] for i in range(0,int(div))]
                pts[f] = np.array(pts[f])
        # if there's enough timesteps we just plot the value in the middle of the timestep
        else:
            pts['ftime'] = np.arange(0.5,integ_len+0.5) + y_0

        # # ------------ PLOT FIGURE ------------ # #
        fig = plt.figure()
        emmsax = fig.add_subplot(221)
        concax = fig.add_subplot(222)
        forcax = fig.add_subplot(223)
        tempax = fig.add_subplot(224)

        emmsax.plot(pts['ftime'],pts['emms'],color=colour['emms'])
        emmsax.set_ylabel(r'CO$_2$ emissions (GtC.yr$^{-1}$)')
        concax.plot(pts['stime'],pts['conc'],color=colour['conc'])
        concax.set_ylabel('CO$_2$ concentrations (ppm)')
        concax.set_xlim(emmsax.get_xlim())
        forcax.plot(pts['ftime'],pts['forc'],color=colour['forc'],label='Total')
        forcax.plot(pts['ftime'],pts['forc-co2'],label=r'CO$_2$')
        forcax.plot(pts['ftime'],pts['forc-non-co2'],label=r'Non-CO$_2$')
        forcax.set_ylabel('Radiative forcing (W.m$^{-2}$)')
        forcax.set_xlabel('Time ({0})'.format(tuts))
        forcax.legend()
        tempax.plot(pts['stime'],pts['temp'],color=colour['temp'])
        tempax.set_ylabel('Temperature anomaly (K)')
        tempax.set_xlabel(forcax.get_xlabel())
        tempax.set_xlim(forcax.get_xlim())
        fig.tight_layout()

        if showplt:
            fig.show()

        return fig,emmsax,concax,forcax,tempax
            
    def tune_carbon_cycle(self,
                          emissions=False,
                          targ_co2_forc=False,
                          non_co2_forc=False,
                          tstep=1.0,
                          F_2x=3.74,
                          C_0=278,
                          a0_bnds=[0.0,1.0],
                          a1_bnds=[0.0,1.0],
                          a2_bnds=[0.0,1.0],
                          r0_bnds=[0.0,100.0],
                          beta_bnds=[0.0,100.0]):
        """
        Tune FAIR's carbon cycle to another model's emissions to forcing response

        Adjust the fraction of CO2 which goes into each carbon response pool 
        as well as the feedback parameters to tune FAIR's response to some 
        other emissions to forcing relationship. You must supply emissions and 
        non-CO2 radiative forcing timeseries as well as a target CO2 forcing 
        timeseries. 

        The temperature and carbon feedbacks are scaled such that the ratio 
        between their two controlling parameters, rT/rC, remains constant.

        You should also double check that you are using the correct CO2 
        forcing parameters, F_2x and C_0, whenever you run this tuning method.
      
        # # ------------ ARGUMENTS ------------ # #
        emissions: (np.array/list)
          emissions with which to drive FAIR

        targ_co2_forc: (np.array/list)
          target CO_2 forcing (W/m^2)

        non_co2_forc: (np.array/list)
          non CO_2 forcing (W/m^2) with which to drive FAIR

        tstep: (float)
          length of a timestep in our timeseries (yrs)

        F_2x: (float)
          forcing due to a doubling of CO_2 to use in our tuning

        C_0: (float)
          pre-industrial CO_2 concentration to use in our tuning

        a0_bnds: (list)
          minimum and maximum allowed values for a0
      
        a1_bnds: (list)
          minimum and maximum allowed values for a1

        a2_bnds: (list)
          minimum and maximum allowed values for a2

        r0_bnds: (list)
          minimum and maximum allowed values for r0

        beta_bnds: (list)
          minimum and maximum allowed values for beta, the scaling parameter 
          of rC and rT

        ^ => Keyword argument
      
        # # ------------ SIDE EFFECTS ------------ # #
        Updates the FAIR class instance's timestep, CO2 forcing, carbon pool 
        distribution and feedback parameters to their values at the end of the 
        tuning.

        Plots the fit between FAIR with the tuned parameters and the target 
        timeseries. 
        """
      
        # One line break before anything else
        # # # ------------ IMPORT REQUIRED MODULES ------------ # # #
        # # ------------ THIRD PARTY ------------ # #
        import scipy.optimize

        print "=============================================="
        print "Tuning FAIR's carbon cycle"
        print "=============================================="

        # define our values of rC and rT when beta = 1.0
        rC_base = 0.019
        rT_base = 4.165

        # Check we have input timeseries
        self.setting_inputs = True
        if type(emissions) not in [np.ndarray,list]:
            print "emissions type: {0}".format(type(emissions))
            raise ValueError("Error: Emissions must be a timeseries")
        else:
            self.emissions = np.array(emissions)

        if type(targ_co2_forc) not in [np.ndarray,list]:
            print "targ_co2_forc type: {0}".format(type(targ_co2_forc))
            raise ValueError("Error: Target forcing must be a timeseries")
        elif len(targ_co2_forc) != len(emissions):
            print "len(emissions) = {0}".format(len(emissions))
            print "len(targ_co2_forc) = {0}".format(len(targ_co2_forc))
            raise ValueError("Error: Target forcing and emissions timeseries must be the same length")
        elif type(targ_co2_forc) in [list]:
            targ_co2_forc = np.array(targ_co2_forc)

        if type(non_co2_forc) not in [np.ndarray,list]:
            raise ValueError("Error: tuning without a non-CO_2 forcing timeseries is not yet implemented")
        elif len(non_co2_forc) != len(emissions):
            print "len(emissions) = {0}".format(len(emissions))
            print "len(non_co2_forc) = {0}".format(len(non_co2_forc))
            raise ValueError("Error: Non-CO_2 forcing and emissions timeseries must be the same length")
        else:
            self.setting_inputs = False
            self.other_rf = non_co2_forc

        # Print out tuning parameters to confirm for the user
        print "Tuning carbon cycle with"
        print "tsep = {0} years".format(tstep)
        self.tstep = tstep
        print "F_2X = {0} W/m^2".format(F_2x)
        self.F_2x = F_2x
        print "C_0 = {0} ppmv\n".format(C_0)
        self.C_0 = C_0

        # define our emissions to forcing function
        def emms_to_forc(emissions,a0,a1,a2,r0,beta):
            """
            Function to determine FAIR CO2 forcings from input CO2 emissions
          
            More details of behaviour if required.
          
            # # ------------ ARGUMENTS ------------ # #
            emissions: (np.array/list)
              CO_2 emissions timeseries

            a0: (float)
              fraction of emissions which go into the slowest uptake carbon pool

            a1: (float)
              fraction of emissions which go into the second slowest uptake 
              carbon pool

            a2: (float)
              fraction of emissions which go into the third slowest uptake 
              carbon pool

            r0: (float)
              pre-industrial 100yr integrated impulse response function (yrs)

            beta: (flaot)
              scaling factor of rC and rT parameters. When beta = 1.0, rC = 
              0.019 yr/GtC and rT = 4.165 yr/K
          
            ^ => Keyword argument
          
            # # ------------ RETURN VALUE ------------ # #
            sublime snippet for variable description in header is 'hvardesc'
          
            # # ------------ SIDE EFFECTS ------------ # #
            a3 = 1.0 - a0 - a1 - a2
          
            # # ------------ EXCEPTIONS ------------ # #
            sublime snippet for exception description in header is 'hexcdesc'
          
            # # ------------ RESTRICTIONS ------------ # #
            Document any restrictions on when the function can be called
          
            """
          
            # One line break before anything else
            # # # ------------ IMPORT REQUIRED MODULES ------------ # # #
            # # ------------ STANDARD LIBRARY ------------ # #
            import sys

            # # ------------ THIRD PARTY ------------ # #
        
            # # ------------ LOCAL APPLICATION/LIBRARY SPECIFIC ------------ # #
        
            # # # ------------ CODE ------------ # # #
            
            # print which tuning run we're up to
            print "\r{0}".format(self.tuning_run),
            sys.stdout.flush()
            
            # turn on setting inputs to avoid many messages
            self.setting_inputs = True
            self.emissions = emissions
            self.setting_inputs = False
            self.a = [a0,a1,a2,1 - a0 - a1 - a2]
            self.r0 = r0
            self.rC = beta * rC_base
            self.rT = beta * rT_base
            self.run()
            self.tuning_run += 1

            return self.RF_CO2

        self.tuning_run = 0
        print "Tuning runs required"

        mins = [a0_bnds[0],a1_bnds[0],a2_bnds[0],r0_bnds[0],beta_bnds[0]]
        maxs = [a0_bnds[1],a1_bnds[1],a2_bnds[1],r0_bnds[1],beta_bnds[1]]

        para,cov = scipy.optimize.curve_fit(emms_to_forc,
                                            emissions,
                                            targ_co2_forc,
                                            p0 = [self.a[0],
                                                  self.a[1],
                                                  self.a[2],
                                                  self.r0,
                                                  self.rC/rC_base,],
                                            bounds = (mins,maxs)
                                            )

        # read our parameters back into self
        self.a = [para[0],para[1],para[2],1-para[0]-para[1]-para[2]]
        self.r0 = para[3]
        self.rC = para[4] * rC_base
        self.rT = para[4] * rT_base

        # print summary
        print "Tuning finished"
        print ""
        print "Parameters used"
        print "-----------------"
        print "F_2x = {0} W/m^2".format(self.F_2x)
        print "C_0 = {0} ppmv".format(self.C_0)
        print ""
        print "Output parameters"
        print "-----------------"
        print "a0 = {0}".format(self.a[0])
        print "a1 = {0}".format(self.a[1])
        print "a2 = {0}".format(self.a[2])
        print "a3 = {0}".format(self.a[3])
        print "sum of a = {0}".format(np.sum(self.a))
        print ""
        print "r0 = {0} yrs".format(self.r0)
        print "rC = {0} yrs/GtC".format(self.rC)
        print "rT = {0} yrs/K".format(self.rT)

        self.run()
        fig,emmsax,concax,forcax,tempax = self.plot(showplt=False)
        forcax.plot(np.arange(0,len(targ_co2_forc)),targ_co2_forc,label='Target',color='green')
        forcax.legend()
        fig.show()

    # # ------------ PROPERTIES ------------ # #
    @property
    def emissions(self):
        return self._emissions

    @emissions.setter
    def emissions(self,val):
        self._emissions = val
        self.sort_inputs()

    @property
    def other_rf(self):
        return self._other_rf

    @other_rf.setter
    def other_rf(self,val):
        self._other_rf = val
        self.sort_inputs()

    @property
    def co2_concs_in(self):
        return self._co2_concs_in

    @co2_concs_in.setter
    def co2_concs_in(self,val):
        self._co2_concs_in = val
        self.sort_inputs()

    @property
    def tcrecs(self):
        return self._tcrecs

    @tcrecs.setter
    def tcrecs(self, val):     
        if type(val) not in [np.ndarray,list] or len(val) != 2:
            raise ValueError("d must be an array or list with 2 elements")
        self._tcrecs = np.array(val,dtype=float)
        self.calc_k_q()
    
    @property
    def F_2x(self):
        return self._F_2x

    @F_2x.setter
    def F_2x(self, val):        
        if type(val) not in [int,float]:
            raise ValueError("Non-numeric F_2x won't work")
        self._F_2x = float(val)
        self.calc_k_q()

    @property
    def d(self):
        return self._d

    @d.setter
    def d(self, val):       
        if type(val) not in [list, np.ndarray] or len(val) != 2:
            raise ValueError("d must be an array or list with 2 elements")
        self._d = np.array(val,dtype=float)
        self.calc_k_q() 

if __name__ == '__main__':
    print "hi"