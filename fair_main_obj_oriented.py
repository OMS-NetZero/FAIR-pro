# # # ------------ DESCRIPTION ------------ # # #
# Script to ...

# # # ------------ IMPORT REQUIRED MODULES ------------ # # #
import numpy as np
from scipy.optimize import root

# # # ------------ CODE ------------ # # #
class FAIR:
	"""
	Documentation string:
	FAIR class for running the forward emissions driven FAIR model.

	# # # ------------ KEY VARIABLES ------------ # # #
	variable: (type) Explanation/description

	# # # ------------ SET KEY VARIABLES ------------ # # #

	# # # ------------ KEY METHODS ------------ # # #
	# __init__: initialise an instance of the FAIR class

	# run: run the model

	# # # ------------ WORKFLOW SUMMARY ------------ # # #  

	"""

	def __init__(self,emissions):
		self.emissions = np.array(emissions,dtype=float)
		# set default parameters
		self.other_rf = 0.0
		self.TCR = 1.75
		self.ECS = 2.5
		self.d=np.array([4.1,239.0])
		self.a=np.array([0.2173,0.2240,0.2824,0.2763])
		self.tau=np.array([1000000,394.4,36.54,4.304])
		self.r0=32.40
		self.rC=0.019
		self.rT=4.165
		self.F_2x=3.74
		self.C_0=278.0
		self.ppm_gtc=2.123
		self.iirf_max=97.0

	def iirf_interp_funct(self,alp_b,a,tau,iirf,x):
		# it seems overly convoluted to me to have iirf and x as separate arguments to this function surely it would be much more flexible and sensible to just have the last argument be the target iirf (so this iirf_interp_funct could be used to just determine the scaling factor for a given iirf) rather than having to make sure you have an array and index the right element

		# ref eq. (7) of Millar et al ACP (2017)
	    
	    iirf_arr = alp_b*(np.sum(a*tau*(1.0 - np.exp(-100.0/(tau*alp_b)))))
	    return iirf_arr   -  iirf[x]

	def run(self):
		#Calculate the q1 and q2 model coefficients from the TCR, ECS and thermal response timescales.
		# ref eq. (4) and (5) of Millar et al ACP (2017)
		k = 1.0 - (self.d/70.0)*(1.0 - np.exp(-70.0/self.d))
		q = np.transpose( (1.0 / self.F_2x) * (1.0/(k[0]-k[1])) * np.array([self.TCR-self.ECS*k[1],self.ECS*k[0]-self.TCR]))

		#Set up the output timeseries variables
		# self.emissions must be a numpy array for this to work
		carbon_boxes_shape = tuple(list(self.emissions.shape) + [4])
		thermal_boxes_shape = tuple(list(self.emissions.shape) + [2])

		RF = np.zeros_like(self.emissions)
		C_acc = np.zeros_like(self.emissions)
		iirf = np.zeros_like(self.emissions)
		R_i = np.zeros(carbon_boxes_shape)
		T_j = np.zeros(thermal_boxes_shape)

		C = np.zeros_like(self.emissions)
		T = np.zeros_like(self.emissions)

		#Initialise the carbon pools to be correct for first timestep in numerical method
		R_i[0,:] = self.a * self.emissions[0,np.newaxis] / self.ppm_gtc
		C[0] = np.sum(R_i[0,:],axis=-1)
		if type(self.other_rf) == float:
			RF[0] = (self.F_2x/np.log(2.)) * np.log((C[0] + self.C_0) /self.C_0) + self.other_rf
		else:
			RF[0] = (self.F_2x/np.log(2.)) * np.log((C[0] + self.C_0) /self.C_0) + self.other_rf[x]
		#Update the thermal response boxes
		T_j[0,:] = (q/self.d)*(RF[0,np.newaxis])
		#Sum the thermal response boxes to get the total temperature anomlay
		T[0]=np.sum(T_j[0,:],axis=-1)

		for x in range(1,self.emissions.shape[-1]):
		  
			#Calculate the parametrised iIRF and check if it is over the maximum allowed value
			iirf[x] = self.rC * C_acc[x-1]  + self.rT*T[x-1]  + self.r0
			if iirf[x] >= self.iirf_max:
			  iirf[x] = self.iirf_max
		  
			#Linearly interpolate a solution for alpha
			if x == 1:
			  time_scale_sf = (root(self.iirf_interp_funct,0.16,args=(self.a,self.tau,iirf,x)))['x']
			else:
			  time_scale_sf = (root(self.iirf_interp_funct,time_scale_sf,args=(self.a,self.tau,iirf,x)))['x']

			#Multiply default timescales by scale factor
			tau_new = self.tau * time_scale_sf

			#Compute the updated concentrations box anomalies from the decay of the pervious year and the additional emisisons
			R_i[x,:] = R_i[x-1,:]*np.exp(-1.0/tau_new) + self.a*(self.emissions[x,np.newaxis]) / self.ppm_gtc
			#Summ the boxes to get the total concentration anomaly
			C[x] = np.sum(R_i[...,x,:],axis=-1)
			#Calculate the additional carbon uptake
			# Do we want to keep this 0.5 factor here? We seem to have rid ourselves of it everywhere else...
			# The way this timing stuff is handled in MAGICC is quite nice and may 
			# provide us with guidance on how to do it ourselves if we want
			C_acc[x] =  C_acc[x-1] + 0.5*(self.emissions[x] + self.emissions[x-1]) - (C[x] - C[x-1])*self.ppm_gtc

			#Calculate the total radiative forcing
			if type(self.other_rf) == float:
				RF[x] = (self.F_2x/np.log(2.)) * np.log((C[x] + self.C_0) /self.C_0) + self.other_rf
			else:
				RF[x] = (self.F_2x/np.log(2.)) * np.log((C[x] + self.C_0) /self.C_0) + self.other_rf[x]

			#Update the thermal response boxes
			T_j[x,:] = T_j[x-1,:]*np.exp(-1.0/self.d) + (q/self.d)*(RF[x,np.newaxis])
			#Sum the thermal response boxes to get the total temperature anomaly
			T[x]=np.sum(T_j[x,:],axis=-1)

		self.C = C + self.C_0
		self.T = T

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

	def printOut(self):
		for i,temp in enumerate(self.T):
			print temp
			print self.C[i]


emissions = [100,0,0,0,0,0,0,0,0,0]
t1 = FAIR(emissions)
# t1.printPara()
t1.run()
t1.printOut()


# print "FAIR.__doc__:", FAIR.__doc__
# print "FAIR.__name__:", FAIR.__name__
# print "FAIR.__module__:", FAIR.__module__
# print "FAIR.__bases__:", FAIR.__bases__
# print "FAIR.__dict__:", FAIR.__dict__