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
		self.emissions = emissions
		# set default parameters
		self.other_rf=0.0
		self.TCR = 1.75
		self.ECS = 2.5
		self.d=np.array([4.1,239.0])
		self.a=np.array([0.2173,0.2240,0.2824,0.2763])
		self.tau=np.array([1000000,394.4,36.54,4.304])
		self.r0=32.40
		self.rc=0.019
		self.rt=4.165
		self.F_2x=3.74
		self.C_0=278.0
		self.ppm_gtc=2.123
		self.iirf_max=97.0

	# def run(self):

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
		print 'rC: ', self.rc
		print 'rT: ', self.rt

		print 'F_2x: ', self.F_2x

		print 'C_0: ', self.C_0

		print 'ppm --> gtc: ', self.ppm_gtc

		print 'Max iirf: ', self.iirf_max


emissions = np.array([0,1,2,3,4,5,6])
t1 = FAIR(emissions)
t1.TCR = 3.0
print type(t1)
t1.printPara()
print t1.emissions