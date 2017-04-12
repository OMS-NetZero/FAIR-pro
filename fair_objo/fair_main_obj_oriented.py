"""
# -*- coding: utf-8 -*-
# @Author: Zebedee Nicholls
# @Date:   2017-04-03 20:28:08
# @Last Modified by:   Zebedee Nicholls
# @Last Modified time: 2017-04-10 10:52:47

Module used to run the FAIR simple climate model. 

# # ------------ CLASSES ------------ # #
FAIR: Class for computing with the FAIR simple climate model.
 This 
class allows the user to timestep their model, run it in full for a given 
emissions series, set, update and print all parameters. 

Each function is described by its docstring. To access these use 
[function-name].__doc__ or read below.

Bibtex reference: @article{millar_modified_2016, title = {A modified 
impulse-response representation of the global response to carbon dioxide 
emissions}, volume = {2016}, 
url = {http://www.atmos-chem-phys-discuss.net/acp-2016-405/}, 
doi = {10.5194/acp-2016-405}, 
journal = {Atmospheric Chemistry and Physics Discussions}, 
author = {Millar, R. J. and Nicholls, Z. R. and Friedlingstein, P. and 
Allen, M. R.}, year = {2016}, keywords = {Carbon cycle, 
FAIR, Simple climate model, Temperature response}, pages = {1--20} }
"""

# # # ------------ IMPORT REQUIRED MODULES ------------ # # #
import numpy as np
from scipy.optimize import root

# # # ------------ CODE ------------ # # #
class FAIR(object):
	"""
	Class for computing with the FAIR simple climate-carbon-cycle model.

	# # ------------ METHODS ------------ # #
	:run(): performs a full run of the model for a given emissions timeseries
	:time_step(): performs a single timestep
	:cacl_k_q(): calculates k and q arrays from FAIR parameters
	:iirf_interp_function(): calculate the integrated impulse response target function
	:get_tau_sf(): calculate the carbon pool decay timeconstant scaling factor for a given target value of the integrated impulse response
	:printPara(): print the current value of a FAIR instance's parameters
	
	# # ------------ INSTANCE VARIABLES ------------ # #
	All instance parameters are optional, any unspecified parameters are assigned their default value. FAIR objects automatically calculate derived parameters.
	# ------------ SYNTAX ------------ #
	:para name: description (units)
	:type name: type

	:para emissions: array of annual emissions (GtC)
	:type emissions: 1D list/numpy array

	:para other_rf: Time series of other radiative forcing or float for 
	constant value (W/m^2)
	:type other_rf: float/numpy 1D array
	
	:para TCR: Transient climate response - global mean warming at the 
	time of doubled CO concentrations following a 1%/yr increase from 
	pre-industrial values (K)
	:type TCR: float/integer
	
	:para ECS: Equilibrium climate sensitivity - global mean warming 
	resulting from an instantaneous doubling of pre-industrial CO2 
	concentrations after allowing the climate system to reach a new
	equilibrium state (K)
	:type ECS: float/integer
	
	:para d: response time of each of the thermal components (years)
	:type d: 2D numpy array/list

	:para a: fraction of emissions received by each carbon reservoir (dimensionless)
	:type a: 4D numpy array/list

	:para tau: decay time constant of each carbon reservoir (years)
	:type tau: 4D numpy array/list

	:para r0: pre-industrial 100yr integrated impulse response 
	function (years)
	:type r0: float/integer

	:para rT: sensitivity of 100yr integrated impulse response 
	function to temperature (years / K)
	:type rT: float/integer

	:para rC: sensitivity of 100yr integrated impulse response 
	function to carbon uptake (years / GtC)
	:type rC: float/integer

	:para F_2x: radiative forcing due to a doubling of CO2 (W/m^2)
	:type F_2x: float/integer

	:para C_0: pre-industrial CO2 (ppm)
	:type C_0: float/integer

	:para ppm_gtc: conversion factor to go from atmospheric CO2 
	concentrations in ppm to emissions in GtC (GtC / ppm)
	:type ppm_gtc: float/integer

	:para iirf_max: maximum value of the 100yr integrated impulse response 
	function (years)
	:type iirf_max: float/integer
	"""

	# # ------------ FUNCTIONS ------------ # #		
	def __init__(self, emissions=np.array([0.,1.,2.,3.]), other_rf=0.0,\
				 TCR=1.6,ECS=2.75,d=np.array([4.1,239.0]),\
				 a=np.array([0.2173,0.2240,0.2824,0.2763]),\
				 tau=np.array([1000000,394.4,36.54,4.304]),r0=32.40,rC=0.019,\
				 rT=4.165,F_2x=3.74,C_0=278.0,ppm_gtc=2.123,iirf_max=97.0\
				 ):
		"""
		Initialises an instance of the FAIR class. 

		Sets attributes and properties based off the input arguments. Performs checks for ....[[]]

		# # ------------ SYNTAX ------------ # #
		:para name: description (units)
		:type name: type

		# # ------------ ARGUMENTS ------------ # #
		:para emissions: array of annual emissions (GtC)
		:type emissions: 1D list/numpy array

		:para other_rf: Time series of other radiative forcing or float for 
		constant value (W/m^2)
		:type other_rf: float/numpy 1D array
		
		:para TCR: Transient climate response - global mean warming at the 
		time of doubled CO concentrations following a 1%/yr increase from 
		pre-industrial values (K)
		:type TCR: float/integer
		
		:para ECS: Equilibrium climate sensitivity - global mean warming 
		resulting from an instantaneous doubling of pre-industrial CO2 
		concentrations after allowing the climate system to reach a new
		equilibrium state (K)
		:type ECS: float/integer
		
		:para d: response time of each of the thermal components (years)
		:type d: 2D numpy array/list

		:para a: fraction of emissions received by each carbon reservoir 
		(dimensionless)
		:type a: 4D numpy array/list

		:para tau: decay time constant of each carbon reservoir (years)
		:type tau: 4D numpy array/list

		:para r0: pre-industrial 100yr integrated impulse response 
		function (years)
		:type r0: float/integer

		:para rT: sensitivity of 100yr integrated impulse response 
		function to temperature (years / K)
		:type rT: float/integer

		:para rC: sensitivity of 100yr integrated impulse response 
		function to carbon uptake (years / GtC)
		:type rC: float/integer

		:para F_2x: radiative forcing due to a doubling of CO2 (W/m^2)
		:type F_2x: float/integer

		:para C_0: pre-industrial CO2 (ppm)
		:type C_0: float/integer

		:para ppm_gtc: conversion factor to go from atmospheric CO2 
		concentrations in ppm to emissions in GtC (GtC / ppm)
		:type ppm_gtc: float/integer

		:para iirf_max: maximum value of the 100yr integrated impulse response 
		function (years)
		:type iirf_max: float/integer

		# # ------------ DERIVED ATTRIBUTES ------------ # #
		:attr n: length of the emissions array (dimensionless)
		:type n: integer

		:attr k: realised warming in each temperature response after 70 
		years of 1%/yr CO2 increase as a fraction of equilibrium warming 
		(dimensionless)
		:type k: 2D numpy array

		:attr q: sensitivity of each temperature response to radiative forcing 
		(K W^-1 m^2)
		:type q: 2D numpy array
		"""

		#Assign the initialized parameters to the class data members
		if emissions is None or type(emissions) not in [list, np.ndarray]:
			raise ValueError("emissions must be supplied as an array or list")
		
		# ------------ ATTRIBUTES ------------ #
		self.emissions = np.array(emissions,dtype=float)
		self.other_rf = other_rf
		self.tau = tau
		self.r0 = r0
		self.rC = rC
		self.rT = rT
		self.C_0 = C_0
		self.ppm_gtc = ppm_gtc
		self.iirf_max = iirf_max
		self.a = a

		# ------------ PROPERTIES ------------ #
		# have to set 'hidden' properties the first time otherwise the setter 
		# functions will try to calculate q and k when not all their 
		# properties are set
		self._TCR = TCR
		self._ECS = ECS
		self._d = d
		self.F_2x = F_2x

		#Get the length of the emissions array
		self.n = len(emissions)

		# calculate k and q
		self.calc_k_q()

	def calc_k_q(self):
		"""
		Calculates k and q arrays from FAIR parameters.

		# # ------------ ARGUMENTS ------------ # #
		:para self.d: response time of each of the thermal components (years)
		:type self.d: 2D numpy array

		:para self.F_2x: radiative forcing due to a doubling of CO2 (W/m^2)
		:type self.F_2x: float/integer

		:para self.TCR: Transient climate response - global mean warming at 
		the time of doubled CO concentrations following a 1%/yr increase from 
		pre-industrial values (K)
		:type self.TCR: float/integer

		:para self.ECS: Equilibrium climate sensitivity - global mean warming 
		resulting from an instantaneous doubling of pre-industrial CO2 
		concentrations after allowing the climate system to reach a new 
		equilibrium state (K)
		:type self.ECS: float/integer

		# # ------------ DERIVED ATTRIBUTES ------------ # #
		:attr k: realised warming in each temperature response after 70 
		years of 1%/yr CO2 increase as a fraction of equilibrium warming 
		(dimensionless)
		:type k: 2D numpy array

		:attr q: sensitivity of each temperature response to radiative forcing 
		(K W^-1 m^2)
		:type q: 2D numpy array		
		"""

		self.k = 1.0 - (self.d/70.0)*(1.0 - np.exp(-70.0/self.d))
		self.q = np.transpose((1.0 / self.F_2x) * (1.0/(self.k[0]-self.k[1])) * np.array([self.TCR-self.ECS*self.k[1],self.ECS*self.k[0]-self.TCR]))

	def iirf_interp_funct(self,alp_b,iirf_targ):
		"""
		Calculate the integrated impulse response target function.

		The iIRF alculation is done for a given scaling of the carbon pool 
		decay timeconstants and FAIR instance. The difference between this 
		calculated value and the target 100 year integrated impulse response 
		function is returned.

		# # ------------ ARGUMENTS ------------ # #
		:para self.a: fraction of emissions received by each carbon reservoir 
		(dimensionless)
		:type self.a: 4D numpy array

		:para self.tau: decay time constant of each carbon reservoir (years)
		:type self.tau: 4D numpy array

		:para alp_b: scaling of the carbon pool decay timeconstants 
		(dimensionless)
		:type alp_b: float/integer

		:para iirf_targ: the target value of the 100yr integrated impulse 
		response function (years)
		:type iirf_targ: float/integer

		# # ------------ RETURNS ------------ # #
		:attr iirf_arr - iirf_targ: difference between 100yr iIRF for the FAIR 
		instance and value of alp_b and the target iIRF (years)
		:type iirf_arr - iirf_targ: float/integer
		"""

		iirf_arr = alp_b*(np.sum(self.a*self.tau*(1.0 - np.exp(-100.0/(self.tau*alp_b)))))
		return iirf_arr   -  iirf_targ
	
	def get_tau_sf(self,iirf_targ):
		"""
		Returns the solution for the scaling of the carbon decay timeconstants.

		The solution is found when the calculated iIRF matches the target iIRF.

		# # ------------ ARGUMENTS ------------ # #
		:para iirf_targ: the target value of the 100yr integrated impulse 
		response function (years)
		:type iirf_targ: float/integer

		:para self: instance of the FAIR class with the iirf_interp_funct 
		method
		:type self: FAIR instance

		:para self.sf: currently stored scaling of the carbon pool decay 
		timeconstant (dimensionless). It's used as a starting guess for the 
		solution.
		:type self.sf: float/integer
		"""

		if self.x == 1:
			return (root(self.iirf_interp_funct,0.16,args=(iirf_targ)))['x']
		else:
			return (root(self.iirf_interp_funct,self.sf,args=(iirf_targ)))['x']

	def time_step(self):
		"""
		Calculate FAIR output variables for the next timestep. 

		# # ------------ ARGUMENTS ------------ # #
		:para self.rC: sensitivity of 100yr integrated impulse response 
		function to carbon uptake (years / GtC)
		:type self.rC: float/integer

		:para self.C_acc_x: CO2 uptake up until the input timestep, 
		overwritten during calculation (GtC)
		:type self.C_acc_x: float/integer

		:para self.rT: sensitivity of 100yr integrated impulse response 
		function to temperature (years / K)
		:type self.rT: float/integer

		:para self.T_x: temperature anomaly in the input timestep, overwritten 
		during calculation (K)
		:type self.T_x: float/integer

		:para self.r0: pre-industrial 100yr integrated impulse response 
		function (years)
		:type self.r0: float/integer

		:para self.R_i_x: CO2 concentration in each carbon pool in the output 
		timestep, overwritten during calculation (ppm)
		:type self.R_i_x: 4D numpy array

		:para self.C_x: total CO2 concentration anomaly in the input timestep, 
		overwritten during calculation (ppm)
		:type self.C_x: float/integer

		:para self.emissions_x: output timestep CO2 emissions (GtC)
		:type self.emissions_x: float/integer

		:para self.emissions_x_old: input timestep CO2 emissions (GtC)
		:type self.emissions_x_old: float/integer

		:para self.ppm_gtc: conversion factor to go from atmospheric CO2 
		concentrations in ppm to emissions in GtC (GtC / ppm)
		:type self.ppm_gtc: float/integer

		:para self.other_rf: time series of other radiative forcing or float 
		for constant value (W/m^2)
		:type self.other_rf: float/numpy 1D array

		:para self.other_rf_x: non-CO2 radiative forcing in the output 
		timestep (W m^-2)
		:type self.other_rf_x: float/integer

		:para self.F_2x: forcing due to a doubling of CO2 (W/m^2)
		:type self.F_2x: float/integer

		:para self.C_0: pre-industrial CO2 (ppm)
		:type self.C_0: float/integer

		:para self.T_j_x: temperature anomaly in each response 'box' in the 
		input timestep, overwritten during calculation (K)
		:type self.T_j_x: 2D numpy array

		:para self.d: response time of each of the thermal components (years)
		:type self.d: 2D numpy array

		:para self.q: sensitivity of each temperature response to radiative 
		forcing (K W^-1 m^2)
		:type self.q: 2D numpy array

		# # ----------- DERIVED/UPDATED/OVERWRITTEN ATTRIBUTES ----------- # #
		:attr IIRF100_x: 100yr iIRF to be used in the output timestep's CO2 
		uptake calculations (years)
		:type IIRF100_x: float/integer

		:attr sf: carbon pool decay timeconstant scaling factor to be used in 
		the output timestep's CO2 uptake calculations (dimensionless)
		:type sf: float/integer

		:attr R_i_x: CO2 concentration in each carbon pool in the output 
		timestep, overwrites the previous value (ppm)
		:type R_i_x: 4D numpy array

		:attr C_x_old: total CO2 concentration anomaly in the input timestep 
		(ppm)
		:type C_x_old: float/integer

		:attr C_x: total CO2 concentration anomaly in the output timestep, 
		overwrites the previous value (ppm)
		:type C_x: float/integer

		:attr C_acc_x: CO2 uptake up until the output timestep, overwrites 
		previous value (GtC)
		:type C_acc_x: float/integer

		:attr RF_x: radiative forcing in output timestep (W m^-2)
		:type RF_x: float/integer

		:attr T_j_x: temperature anomaly in each response 'box' in the output 
		timestep, overwrites the previous value (K)
		:type T_j_x: 2D numpy array

		:attr T_x: temperature anomaly in the output timestep, overwrites the 
		previous value (K)
		:type T_x: float/integer
		"""

		self.IIRF100_x = self.rC * self.C_acc_x + self.rT * self.T_x + self.r0
		if self.IIRF100_x >= self.iirf_max:
			 self.IIRF100_x = self.iirf_max
		self.sf = self.get_tau_sf(self.IIRF100_x)

		tau_new = self.tau * self.sf

		self.R_i_x = self.R_i_x*np.exp(-1.0/tau_new) + self.a*(self.emissions_x) / self.ppm_gtc
		#Summ the boxes to get the total concentration anomaly
		self.C_x_old = self.C_x
		self.C_x = np.sum(self.R_i_x,axis=-1)
		#Calculate the additional carbon uptake
		self.C_acc_x =  self.C_acc_x + 0.5*(self.emissions_x +self.emissions_x_old) - (self.C_x - self.C_x_old)*self.ppm_gtc

		#Calculate the total radiative forCing
		if type(self.other_rf) == float:
			 self.RF_x = (self.F_2x/np.log(2.)) * np.log((self.C_x + self.C_0) /self.C_0) + self.other_rf
		else:
			 self.RF_x = (self.F_2x/np.log(2.)) * np.log((self.C_x + self.C_0) /self.C_0) + self.other_rf_x

		#Update the thermal response boxes
		self.T_j_x = self.T_j_x*np.exp(-1.0/self.d) + (self.q/self.d)*self.RF_x
		#Sum the thermal response boxes to get the total temperature anomlay
		self.T_x=np.sum(self.T_j_x,axis=-1)

	def run(self):
		"""
		Run the FAIR model. 

		The model calculates as many timesteps as there are values in the 
		emissions array and returns timeseries for global mean temperature 
		anomalies as well as atmospheric CO2 concentrations.

		# # ------------ ARGUMENTS ------------ # #
		:para self.n: length of emissions timeseries. Also defines the number 
		of timesteps in our run (timesteps - currently has to be years)
		:type self.n: 1D numpy array

		:para self.F_2x: forcing due to a doubling of CO2 (W/m^2)
		:type self.F_2x: float/integer

		:para self.C_0: pre-industrial CO2 (ppm)
		:type self.C_0: float/integer

		:para self.other_rf: time series of other radiative forcing or float 
		for constant value (W/m^2)
		:type self.other_rf: float/numpy 1D array

		:para self.d: response time of each of the thermal components (years)
		:type self.d: 2D numpy array

		:para self.rC: sensitivity of 100yr integrated impulse response 
		function to carbon uptake (years / GtC)
		:type self.rC: float/integer

		:para self.rT: sensitivity of 100yr integrated impulse response 
		function to temperature (years / K)
		:type self.rT: float/integer

		:para self.r0: pre-industrial 100yr integrated impulse response 
		function (years)
		:type self.r0: float/integer

		# # ------------ DERIVED ATTRIBUTES ------------ # #
		:attr R_i: CO2 concentrations in each of the four carbon pools at each 
		timestep (ppm)
		:type R_i: 2D (self.n x 4) numpy array

		:attr T_j: temperature anomaly in each response 'box' at each timestep 
		(K)
		:type T_j: 2D (self.n x 2) numpy array

		:attr C: total CO2 anomaly at each timestep. Converted to total 
		atmospheric CO2 concentrations after all timesteps have been 
		calculated by adding on the pre-industrial CO2 concentration, self.C_0 
		(ppm)
		:type C: 1D (self.n) numpy array 

		:attr T: total temperature anomaly at each timestep (K)
		:type T: 1D (self.n) numpy array

		:attr C_acc: cumulative uptake of carbon at each timestep (GtC)
		:type C_acc: 1D (self.n) numpy array

		:attr IRF100: 100yr integrated impulse response function at each 
		timestep (years)
		:type IRF100: 1D (self.n) numpy array

		:attr RF: radiative forcing at each timestep (W m^-2)
		:type RF: 1D (self.n) numpy array

		:attr R_i_x: CO2 concentration in each carbon pool in the timestep 
		under consideration. Overwritten each timestep (ppm)
		:type R_i_x: 4D numpy array

		:attr C_x: total CO2 concentration anomaly in the timestep under 
		consideration. Overwritten each timestep (ppm)
		:type R_i_x: 4D numpy array
		
		:attr RF_x: radiative forcing in timestep under consideration. 
		Overwritten each timestep (W m^-2)
		:type RF_x: float/integer

		:attr T_j_x: temperature anomaly in each response 'box' in the 
		timestep under consideration. Overwritten each timestep (K)
		:type T_j_x: 2D numpy array

		:attr T_x: total temperature anomaly in the timestep under 
		consideration. Overwritten each timestep (K)
		:type T_x: float/integer

		:attr C_acc_x: CO2 uptake up until the timestep under consideration. 
		Overwritten each timestep (GtC)
		:type C_acc_x: float/integer

		:attr .emmissions_x_old: CO2 emissions in the timestep one before the 
		one under consideration (GtC)
		:type .emmissions_x_old: float/integer

		:attr x: index of the timestep under consideration (same as timestep - 
		currently must be years)
		:type x: integer
		"""

		#Set up the output arrays
		self.R_i = np.zeros((self.n,4))
		self.T_j = np.zeros((self.n,2))
		self.C = np.zeros((self.n))
		self.T = np.zeros((self.n))
		self.C_acc =np.zeros((self.n))
		self.IRF100 = np.zeros((self.n))
		self.RF = np.zeros((self.n))
			
		#Initialise the carbon pools to be correct for first timestep in numerical method
		self.R_i_x = self.a * self.emissions[0] / self.ppm_gtc
		self.C_x = np.sum(self.R_i_x,axis=-1)
		if type(self.other_rf) == float:
			self.RF_x = (self.F_2x/np.log(2.)) * np.log((self.C_x + self.C_0) /self.C_0) + self.other_rf
		else:
			self.RF_x = (self.F_2x/np.log(2.)) * np.log((self.C_x + self.C_0) /self.C_0) + self.other_rf[0]
		#Update the thermal response boxes
		self.T_j_x = self.T_j[0]*np.exp(-1.0/self.d) + (self.q/self.d)*self.RF_x
		#Sum the thermal response boxes to get the total temperature anomaly
		self.T_x=np.sum(self.T_j_x,axis=-1)
		self.C_acc_x = 0.0
		self.emissions_x_old = self.emissions[0]
			
		self.R_i[0] = self.R_i_x
		self.T_j[0] = self.T_j_x
		self.C[0] = self.C_x
		self.T[0] = self.T_x
		self.C_acc[0] = self.C_acc_x
		self.IRF100[0] = self.rC * self.C_acc_x + self.rT * self.T_x + self.r0
		self.RF[0] = self.RF_x
			
		for x in range(1,self.n):
			
			self.x = x
			self.emissions_x = self.emissions[x]
			self.emissions_x_old = self.emissions[x-1]
			#Run a timestep
			self.time_step()
			#Save the timestep output
			self.R_i[x] = self.R_i_x
			self.T_j[x] = self.T_j_x
			self.C[x] = self.C_x
			self.T[x] = self.T_x
			self.C_acc[x] = self.C_acc_x
			self.IRF100[x] = self.IIRF100_x
			self.RF[x] = self.RF_x


		#Add on the pre-industrial concentrations to the carbon concentration output array
		self.C = self.C + self.C_0

	def printPara(self):
		"""
		Print out FAIR input parameters in human readable form. 

		# # ------------ PRINTED PARAMETERS ------------ # #
		self.:TCR, ECS, d, a, tau, r0, rC, rT, F_2x, C_0, ppm_gtc, iirf_max
		"""

		print 'TCR: ', self.TCR
		print 'ECS: ', self.ECS
		
		print 'd1: ', self.d[0]
		print 'd2: ', self.d[1]
		
		print 'a1: ', self.a[0]
		print 'a2: ', self.a[1]
		print 'a3: ', self.a[2]
		print 'a4: ', self.a[3]

		print 'tau1: ', self.tau[0]
		print 'tau2: ', self.tau[1]
		print 'tau3: ', self.tau[2]
		print 'tau4: ', self.tau[3]
		
		print 'r0: ', self.r0
		print 'rC: ', self.rC
		print 'rT: ', self.rT

		print 'F_2x: ', self.F_2x

		print 'C_0: ', self.C_0

		print 'ppm --> gtc: ', self.ppm_gtc

		print 'Max iirf: ', self.iirf_max

	# # ------------ PROPERTIES ------------ # #
	@property
	def TCR(self):
		"""
		Transient Climate Response (float/integer).

		Global mean warming at the time of doubled CO concentrations following 
		a 1%/yr increase from pre-industrial values (K). 

		Each time TCR is set k and q are recalculated.
		"""

		return self._TCR

	@TCR.setter
	def TCR(self, val):		
		if type(val) not in [int,float]:
			raise ValueError("Non-numeric TCR won't work")

		self._TCR = float(val)
		self.calc_k_q()
		
	@property
	def ECS(self):
		"""
		Equilibrium Climate Sensitivity (float/integer).

		Global mean warming resulting from an instantaneous doubling of 
		pre-industrial CO2 concentrations after allowing the climate system to 
		reach a new equilibrium state (K).

		Each time ECS is set k and q are recalculated.
		"""

		return self._ECS

	@ECS.setter
	def ECS(self, val):		
		if type(val) not in [int,float]:
			raise ValueError("Non-numeric ECS won't work")

		self._ECS = float(val)
		self.calc_k_q()
	
	@property
	def F_2x(self):
		"""
		Forcing due to a doubling of CO2 (float/integer).

		Units must be W m^-2.

		Each time F_2x is set k and q are recalculated.
		"""

		return self._F_2x

	@F_2x.setter
	def F_2x(self, val):		
		if type(val) not in [int,float]:
			raise ValueError("Non-numeric F_2x won't work")

		self._F_2x = float(val)
		self.calc_k_q()

	@property
	def d(self):
		"""
		Thermal response times (2D list/numpy array).

		Units must be years.

		Each time d is set k and q are recalculated.
		"""

		return self._d

	@d.setter
	def d(self, val):		
		if type(val) not in [list, np.ndarray] or len(val) != 2:
			raise ValueError("d must be a 2D array or list")

		self._d = np.array(val,dtype=float)
		self.calc_k_q()