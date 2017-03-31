# # # ------------ DESCRIPTION ------------ # # #
# Script to ...

# # # ------------ IMPORT REQUIRED MODULES ------------ # # #
import numpy as np
from scipy.optimize import root

# # # ------------ CODE ------------ # # #
class FAIR(object):
	"""Class for computing with the FAIR simple climate-carbon-cycle model.
		"""
	def __init__(self, emissions=np.array([0.,1.,2.,3.]), other_rf=0.0,TCR=1.75,ECS=2.5,d=np.array([4.1,239.0]),a=np.array([0.2173,0.2240,0.2824,0.2763]),tau=np.array([1000000,394.4,36.54,4.304]),r0=32.40,rC=0.019,rT=4.165,F_2x=3.74,C_0=278.0,ppm_gtc=2.123,iirf_max=97.0):
		"""FAIR init
			Args:
			:param emissions: Array of annual emissions in GtC
			:type emissions: numpy 1D array
			:param other_rf: Time series of other radiative forcing or float for constant value
			:type time_step: float/numpy 1D array
			:param tcrecs: Numpy array of the specified TCR and ECS values
			"""

		#Assign the initialized parameters to the class data members
		if emissions is not None and type(emissions) in [list, np.ndarray]:
			self.emissions = np.array(emissions,dtype=float)
		
		self.other_rf = other_rf

		# as we've defined TCR as a property, we store the actual variable value as a 'hidden' value by using an underscore first (to avoid getting in a loop with our getter function)
		self._TCR = TCR
		self._ECS = ECS
		self._d = d
		self._a = a
		self.tau = tau
		self.r0 = r0
		self.rC = rC
		self.rT = rT
		self._F_2x = F_2x
		self.C_0 = C_0
		self.ppm_gtc = ppm_gtc
		self.iirf_max = iirf_max
		
		#Get the length of the emissions array
		self.n = len(emissions)

		# calculate k and q
		
	# # ------------ PROPERTIES ------------ # #

	@property
	def TCR(self):
		return self._TCR

	@TCR.setter
	def TCR(self, val):		
		if type(val) not in [int,float]:
			raise ValueError("Non-numeric TCR won't work")

		self._TCR = val
		self.calc_k_q()
		
	@property
	def ECS(self):
		return self._ECS

	@ECS.setter
	def ECS(self, val):		
		if type(val) not in [int,float]:
			raise ValueError("Non-numeric ECS won't work")

		self._ECS = val
		self.calc_k_q()
	
	@property
	def F_2x(self):
		return self._F_2x

	@F_2x.setter
	def F_2x(self, val):		
		if type(val) not in [int,float]:
			raise ValueError("Non-numeric F_2x won't work")

		self._F_2x = val
		self.calc_k_q()

	@property
	def d(self):
		return self._d

	@d.setter
	def d(self, val):		
		if type(val) not in [list, np.ndarray] or len(val) != 4:
			raise ValueError("d must be a 4D array or list")

		self._d = val
		self.calc_k_q()

	@property
	def a(self):
		return self._a

	@a.setter
	def a(self, val):		
		if type(val) not in [list, np.ndarray] or len(val) != 4:
			raise ValueError("a must be a 4D array or list")

		if np.sum(val) != 1:
			raise ValueError("sum of a coefficients must be 1 to conserve carbon")

		self._a = val
		self.calc_k_q()

	# # ------------ FUNCTIONS ------------ # #		

	def calc_k_q(self):
		self.k = 1.0 - (self.d/70.0)*(1.0 - np.exp(-70.0/self.d))
		self.q = np.transpose((1.0 / self.F_2x) * (1.0/(self.k[0]-self.k[1])) * np.array([self.TCR-self.ECS*self.k[1],self.ECS*self.k[0]-self.TCR]))

	def iirf_interp_funct(self,alp_b,iirf_targ):

		iirf_arr = alp_b*(np.sum(self.a*self.tau*(1.0 - np.exp(-100.0/(self.tau*alp_b)))))
		return iirf_arr   -  iirf_targ
	
	def get_tau_sf(self,iirf_targ):
		
		if self.x == 1:
			return (root(self.iirf_interp_funct,0.16,args=(iirf_targ)))['x']
		else:
			return (root(self.iirf_interp_funct,self.sf,args=(iirf_targ)))['x']

	def time_step(self):

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
		self.T[0] = self.C_x
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
