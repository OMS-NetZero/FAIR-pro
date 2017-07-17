"""
FAIR simple climate model

# # ------------ CLASSES ------------ # #
sublime snippet for module description in header is 'hclsdesc'

FAIR: Class for computing with the FAIR simple climate model

# # ------------ EXCEPTIONS ------------ # #
sublime snippet for exception description in header is 'hexcdesc'

# # ------------ FUNCTIONS ------------ # #
sublime snippet for module description in header is 'hfundesc'

# # ------------ ANY OTHER OBJECTS EXPORTED ------------ # #
describe them here
"""

def FAIR(object):
    """
    FAIR: Class for computing with the FAIR simple climate model

    More detailed summary of class' behaviour if required. If the class subclasses another class and its behaviour is mostly inherited from that class, mention this and summarize the differences here. 
    "override" indicates that a subclass method replaces a superclass method and does not call the superclass method. 
    "extend" indicates that a subclass method calls the superclass method (in addition to its own behavior). 

    # # ------------ METHODS ------------ # #
    sublime snippet for method description in header is 'hmetdesc'

    run: performs a full run of the model for a given emissions timeseries

    time_step: performs a single timestep

    calc_k_q: calculates k and q arrays from FAIR parameters

    iirf_interp_function: calculate the integrated impulse response target function

    get_tau_sf: calculate the carbon pool decay timeconstant scaling factor for a given target value of the integrated impulse response

    print_para: print the current value of a FAIR instance's parameters

    # # ------------ INSTANCE VARIABLES ------------ # #
    sublime snippet for variable description in header is 'hvardesc'

    All instance variables are optional. See __init__ method docstring for options.

    # # ------------ SUBCLASS INTERFACES ------------ # #
    will work out what this means once I start using them
    """

    # One line break before anything else
    # sublime snippet for a method is 'met'
    # your first method will probably be '__init__'

    def __init__(self,
                 tstep=1.0,
                 emissions=False,
                 other_rf=0.0,
                 co2_concs=False,
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

        Sets attributes and properties based off the input arguments. Performs checks for ....[[]]
    
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

        co2_concs:^ (np.array/list/bool)
          Atmospheric CO_2 concentrations timeseries (ppmv). If emissions are 
          supplied then co2_concs is not used.
        
        R_i_pre: (np.array/list/bool)
          Atmospheric CO_2 concentrations in each box just before the run begins

        T_j_pre: (np.array/list/bool)
          Surface temperatures in each pool just before the run begins

        C_acc_pre: (float/int)
          Total CO_2 uptake by land and oceans up until the start of the run

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
        All the arguments are set as attributes of the FAIR instance. If either the emissions, concentrations or radiative forcing inputs are floats then they are extended to be the same length as the supplied timeseries.
        
        On top of these, the following are also set to be attributes of the FAIR instance:
        
        conc_driven: (bool)
          whether the run is concentration driven or not
          
        n: (int)
          number of timesteps in our input arrays

        k: (2D np.array)
          realised warming in each temperature response after 70 
          years of 1%/yr CO2 increase as a fraction of equilibrium warming 
          (dimensionless)
        
        q: (2D np.array)
          sensitivity of each temperature response to radiative forcing 
          (K W^-1 m^2)
        
        
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

        # # # ----------- SET UP OUTPUT TIMESERIES VARIABLES ----------- # # #
        # by default FAIR is not concentration driven
        conc_driven=False
        # here we check if FAIR is emissions driven
        if type(emissions) in [np.ndarray,list]:
            n = len(emissions)
            if (type(other_rf) in [np.ndarray,list]) and (len(other_rf)!=n):
                raise ValueError("The emissions and other_rf timeseries don't have the same length")
            elif type(other_rf) in [int,float]:
                other_rf = np.full(n,other_rf)
      
        # here we check if FAIR is concentration driven
        elif type(co2_concs) in [np.ndarray,list]:
            n = len(co2_concs)
            conc_driven = True
            if (type(other_rf) in [np.ndarray,list]) and (len(other_rf)!=n):
                raise ValueError("The concentrations and other_rf timeseries don't have the same length")
            elif type(other_rf) in [int,float]:
                other_rf = np.full(n,other_rf)

        # finally we check if only a non-CO2 radiative forcing timeseries has been supplied
        elif type(other_rf) in [np.ndarray,list]:
            n = len(other_rf)
            if type(emissions) in [int,float]:
                emissions = np.full(n,emissions)
            else:
                emissions = np.zeros(n)

        else:
            raise ValueError("Neither emissions, co2_concs or other_rf is defined as a timeseries")

        # # # ------------ SET ATTRIBUTES ------------ # # #
        self.tstep = tstep
        self.emissions = emissions
        self.other_rf = other_rf
        self.co2_concs = co2_concs
        self.R_i_pre = R_i_pre
        self.T_j_pre = T_j_pre
        self.C_acc_pre = C_acc_pre
        self.a = a
        self.tau = tau
        self.r0 = r0
        self.rC = rC
        self.rT = rT
        self.F_2x = F_2x
        self.C_0 = C_0
        self.ppm_gtc = ppm_gtc
        self.iirf100_max = iirf100_max

        # # # ------------ SET PROPERTIES ------------ # # #
        # have to set 'hidden' properties the first time otherwise the setter 
        # functions will try to calculate q and k when not all their 
        # properties are set
        self._tcrecs = tcrecs
        self._d = d

        #Get the length of the emissions array
        self.n = len(emissions)

        # calculate k and q
        self.calc_k_q()

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
        self.k = 1.0 - (self.d/70.0)*(1.0 - np.exp(-70.0/self.d))
        self.q = np.transpose((1.0 / self.F_2x) * (1.0/(self.k[0]-self.k[1])) * np.array([self.TCR-self.ECS*self.k[1],self.ECS*self.k[0]-self.TCR]))
            

        # # # ------------ CALCULATE Q ARRAY ------------ # # #
        # If TCR and ECS are supplied, overwrite the q array
        self.k = 1.0 - (d/70.0)*(1.0 - np.exp(-70.0/d))
        if type(tcrecs) in [np.ndarray,list]:
            self.q =  (1.0 / F_2x) * (1.0/(k[0]-k[1])) \
                * np.array([tcrecs[0]-k[1]*tcrecs[1],k[0]*tcrecs[1]-tcrecs[0]])

    def iirf_interp_funct(self,cp_scal,iirf_targ):
        """
        Calculate the integrated impulse response target function

        The iIRF calculation is done for a given scaling of the carbon pool 
        decay timeconstants. The difference between the calculated value and 
        the target 100 year integrated impulse response function is returned.
    
        # # ------------ ARGUMENTS ------------ # #
        sublime snippet for variable description in header is 'hvardesc'

        cp_scal: (int)
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
        iirf_arr = alp_b*(np.sum(self.a*self.tau*(1.0 - np.exp(-100.0/(self.tau*alp_b)))))
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

        if self.x == 1:
            return (root(self.iirf_interp_funct,0.16,args=(iirf_targ)))['x']
        else:
            return (root(self.iirf_interp_funct,self.sf,args=(iirf_targ)))['x']
        
        
    def time_step(self):
        """
        Calculate FAIR output variables for the next timestep
    
        # # ------------ ARGUMENTS ------------ # #
        sublime snippet for variable description in header is 'hvardesc'
    
        ^ => Keyword argument
    
        # # ------------ RETURN VALUE ------------ # #
        sublime snippet for variable description in header is 'hvardesc'
    
        # # ------------ SIDE EFFECTS ------------ # #
        Creates/updates the following attributes:

        iirf100_x: (float/int)
          100yr iIRF to be used in the timestep's CO2 uptake 
          calculations (years)
        
        # # ------------ EXCEPTIONS ------------ # #
        sublime snippet for exception description in header is 'hexcdesc'
    
        # # ------------ RESTRICTIONS ------------ # #
        Document any restrictions on when the function can be called
    
        """
    
        # One line break before anything else
        try:
            self.x
        except AttributeError:
            self.x = 0
            self.C_pre = np.sum(self.R_i_pre) + self.C_0
            self.T_pre = np.sum(self.T_j_pre)

            if self.conc_driven:
                self.C_x = self.co2_concs[0]
            else:
                self.iirf100_x = self.r0 + self.rC*self.C_acc_pre + self.rT*self.T_pre
                if self.iirf100_x > self.iirf100_max:
                    self.iirf100_x = self.iirf100_max

                self.sf_x = self.get_tau_sf(self.iirf100_x)

                self.tau_x = self.tau*self.sf_x

                self.R_i_x = (self.R_i_pre*np.exp(-self.tstep/self.tau_x) 
                              + (self.emissions[0],np.newaxis)
                                * self.a*self.tau_x
                                * (1-np.exp(-self.tstep/self.tau_x)) 
                                / self.ppm_gtc
                             )

                self.C_x = np.sum(self.R_i_x) + self.C_0

                self.C_acc_x = self.C_acc_pre + self.emissions[0] - (self.C_x - self.C_pre)*self.ppm_gtc
