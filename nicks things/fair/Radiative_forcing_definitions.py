# Define a function that gives the Radiative forcing due to both CH4 and N2O as per equations given in IPCC AR5 8.SM (CH4/N2O before)
def RF_M_N(M,
           N,
           M_0,
           N_0,
           alp_m=0.036,
           alp_n=0.12):
    """
    Takes input unperturbed/ perturbed concentrations of CH4 and N2O 
    and returns radiative forcing caused.

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
      
    alp_m:^ (float)
      CH4 RF constant given in Myhre et al. (1998)
      
    alp_n:^ (float)
      N2O RF constant given in Myhre et al. (1998)

    ^ => Keyword argument

    # # ------------ RETURN VALUE ------------ # #
    sublime snippet for variable description in header is 'hvardesc'
    Returns the radiative forcing due to CH4 and N2O

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
    
    # first define a function used for both CH4 and N2O RF (see Myhre et al. 1998 as IPCC AR5 has typo) - purely to make code more readable
    def f(M, N):
        return 0.47 * np.log(1 + 2.01 * 10**(-5) * (M * N)**0.75 + 5.31 * 10**(-15) * M * (M * N)**(1.52)) # see IPCC AR5 Table 8.SM.1

    return alp_m * (np.sqrt(M) - np.sqrt(M_0)) - (f(M, N_0) - f(M_0, N_0)) + alp_n * (np.sqrt(N) - np.sqrt(N_0)) - (f(M_0, N) - f(M_0, N_0))
    
    
    
    
# Define a function that gives the Radiative forcing due to CH4 as per Etminan et al. 2016, table 1
def RF_M(M,
         N,
         M_0=722.0,
         N_0=270.0,
         a3=-1.3*10**(-6),
         b3=-8.2*10**(-6),
         K=0.043):
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
        
    return (a3*np.mean([M,M_0]) + b3*np.mean([N,N_0]) + K) * (np.sqrt(M) - np.sqrt(M_0))
    
# Define a function that gives the Radiative forcing due to N2O as per Etminan et al. 2016, table 1
def RF_N(C,
         M,
         N,
         C_0=278.0,
         M_0=722.0,
         N_0=270.0,
         a2=-8.0*10**(-6),
         b2=4.2*10**(-6),
         c2=-4.9*10**(-6),
         K=0.117):
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
    
    return (a2*np.mean([C,C_0]) + b2*np.mean([N,N_0]) + c2*np.mean([M,M_0]) + K) * (np.sqrt(N) - np.sqrt(N_0))
    
# Define a function that gives the Radiative forcing due to CO2 as per Etminan et al. 2016, table 1
def RF_C(C,
         N,
         C_0=278.0,
         N_0=270.0,
         a1=-2.4*10**(-7),
         b1=7.2*10**(-4),
         c1=-2.1*10**(-4),
         K=5.36):
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
    
    return (a1*(C-C_0)**(2) + b1*abs(C-C_0) + c1*np.mean([N,N_0]) + K) * np.log(C/C_0)

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
    
