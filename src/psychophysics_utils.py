import numpy as np
from chainer import serializers
import scipy.optimize
import sys
import scipy.special


from architecture import readArchitecture
from net import Net, totalInputLength


def makeFreqDepStr(freq, depthDb):
	return makeFreqStr(freq)+"_Dep"+makeDepthDbStr(depthDb)


def makeFreqStr(freq):
	return "Fr{:.2f}".format(freq)


def makeDepthDbStr(depthDb):
	if np.isinf(depthDb): depthDbStr="Ninf"
	else: depthDbStr="{:.2f}".format(depthDb)
	return depthDbStr


class NetType:
	def __init__(self, datasetType, arch, controlType, epoch):
		self.datasetType=datasetType
		self.arch=arch
		self.controlType=controlType
		self.epoch=epoch
		
		self.numLayer=int(arch.split("_")[0][len("Layer"):])
		self.numUnit=int(arch.split("_")[1][len("Unit"):])
		
		
	def makePath(self, parent):
		p=parent/self.datasetType/self.arch/self.controlType/("Epoch"+str(self.epoch))
		return p
	
	
	def __str__(self):
		s=self.datasetType+"/"+self.arch+"/"+self.controlType+"/"+("Epoch"+str(self.epoch))
		return s
	
	
def loadNet(netType, dirTraining):
	dirNet=dirTraining/"Net"
	
	dirArch=dirNet/netType.datasetType/netType.arch
	fileArch=dirArch/"Architecture.txt"
	architecture=readArchitecture(fileArch)
	net=Net(None, architecture)
	inputLength=totalInputLength(architecture)
	
	fileNet=dirArch/"TrainedNet"/(str(netType.epoch)+".npz")
	print("loading", fileNet)
	serializers.load_npz(fileNet, net)
	return net, inputLength


def scaleRms(wave, targetRms):
	wave*=targetRms/(wave**2).mean(axis=-1, keepdims=True)**0.5 #isinstance(targetRms, float)


log_expit=lambda x: -np.logaddexp(0, -x)


class Sigmoid:
	NUM_PARAM=2
	
	def __init__(self, chance):
		self.lower=chance
		self.upper=1
		self.d=self.upper-self.lower
	
	def predict(self, x):
		return self.__call__(x)
	
	def __call__(self, x):
		return self._func(x, self.slope, self.x0)
	
	@staticmethod
	def _funcStatic(x, slope, x0, lower, upper):
		d=upper-lower
# 		y=d/(1+np.exp(-slope*(x-x0)))+lower
		log_yl=np.log(d)-np.logaddexp(0,-slope*(x-x0))
		yl=np.exp(log_yl)
		y=yl+lower
		return y
	
	def _func(self, x, slope, x0):
		return Sigmoid._funcStatic(x, slope, x0, self.lower, self.upper)

	def _jac(self, x, slope, x0):
		f0=Sigmoid._funcStatic(slope*(x-x0), 1, 0, 0, 1)
		slopeJac=self.d*f0*(1-f0)*(x-x0)
		x0Jac=self.d*f0*(1-f0)*(-slope)
		j=np.stack((slopeJac, x0Jac), axis=1)
		return j

	def fit(self, x, y):
		slopeInit=1
		x0Init=x.mean()
		paramInit=np.array((slopeInit, x0Init))
		bounds=(np.array((0, -np.inf)), np.array((np.inf, np.inf)))
# 		popt,_=scipy.optimize.curve_fit(self._func, x, y, paramInit, bounds=bounds, method="trf", jac=self._jac)
		popt,_=scipy.optimize.curve_fit(self._func, x, y, paramInit, bounds=bounds, method="trf", jac=self._jac, maxfev=100000)
		self.slope,self.x0=popt
		
	def fit_predict(self, x, y):
		self.fit(x, y)
		return self.predict(x)
	
	def inv(self, y):
		if self.slope==0: return np.nan*np.empty(y.shape)
		x=-np.log(self.d/(y-self.lower)-1)/self.slope+self.x0
		return x
	
	def setParams(self, params):
		self.slope,self.x0=params
		
	def getParams(self):
		return self.slope, self.x0
		
		
class AsymSigmoid:
	'''
	f(x)=lower+d/(c+exp(-slope(x-x0)))^p
	d=upper-lowe
	c=1
	@see: https://doi.org/10.14214/sf.653
	'''
	
	NUM_PARAM=3
	
	def __init__(self, chance):
		self.chance=chance
		self.lower=chance
		self.upper=1
		self.d=self.upper-self.lower
	
	def predict(self, x):
		return self.__call__(x)
	
	def __call__(self, x):
		return self._func(x, self.logSlope, self.x0, self.logP)
	
	@staticmethod
	def _funcStatic(x, logSlope, x0, logP, lower, upper):
		d=upper-lower
		slope=np.exp(logSlope)
		p=np.exp(logP)
# 		y=d/(1+np.exp(-slope*(x-x0)))**p+lower
		log_yl=np.log(d)-p*np.logaddexp(0,-slope*(x-x0))
		yl=np.exp(log_yl)
		y=yl+lower
		return y
	
	def _func(self, x, logSlope, x0, logP):
		return AsymSigmoid._funcStatic(x, logSlope, x0, logP, self.lower, self.upper)

	def _jac(self, x, logSlope, x0, logP):
		d=self.d
		m=x0
		slope=np.exp(logSlope)
		p=np.exp(logP)
		X=-slope*(x-m)
# 		E=1+expX
		expX=np.exp(X)
		logE=np.logaddexp(0,X)
		Q=-p
		dXds=-(x-m)
		dXdm=slope
		dEdX=expX
		dfdE=d*Q*np.exp((Q-1)*logE)
		dfds=dfdE*dEdX*dXds
		dfdm=dfdE*dEdX*dXdm
		dfdQ=d*np.exp(Q*logE)*logE
		dfdp=-dfdQ
		
		dsdls=slope
		dfdls=dfds*dsdls
		dpdlp=p
		dfdlp=dfdp*dpdlp
		
		j=np.stack((dfdls, dfdm, dfdlp), axis=1)
		return j

	def fit(self, x, y):
		logSlopeInit=0
		x0Init=x.mean()
		logPInit=0
		paramInit=np.array((logSlopeInit, x0Init, logPInit))
		popt,_=scipy.optimize.curve_fit(self._func, x, y, paramInit, method="lm", jac=self._jac)
		self.logSlope,self.x0,self.logP=popt
		

	def fitInitSym(self, x, y):
		sigmoid=Sigmoid(self.chance)
		sigmoid.fit(x, y)
		slopeInit=sigmoid.slope
		x0Init=sigmoid.x0
		
		logSlopeInit=np.log(slopeInit)
		logPInit=0
		paramInit=np.array((logSlopeInit, x0Init, logPInit))
# 		popt,_=scipy.optimize.curve_fit(self._func, x, y, paramInit, method="lm", jac=self._jac)
		try:
			popt,_=scipy.optimize.curve_fit(self._func, x, y, paramInit, method="lm", jac=self._jac, maxfev=100000)
			self.logSlope,self.x0,self.logP=popt
		except RuntimeError as e:
			self.logSlope,self.x0,self.logP=logSlopeInit, x0Init, logPInit
			
		
	def fit_predict(self, x, y):
		self.fit(x, y)
		return self.predict(x)
	
	
	def inv(self, y):
		d=self.d
		m=self.x0
		slope=np.exp(self.logSlope)
		p=np.exp(self.logP)
		Q=-p
		L=self.lower
		
# 		E=((y-L)/d)**(1/Q)
# 		X=np.log(E-1)
		logE=1/Q*np.log((y-L)/d)
		exmp1_logE=np.expm1(logE)
		if np.isinf(exmp1_logE) and exmp1_logE>0:
			X=logE+np.log(1-1/np.exp(logE))
		else:
			X=np.log(exmp1_logE)

		x=-X/slope+m
		
		return x
	
	
	def setParams(self, params):
		self.logSlope,self.x0,self.logP=params
		
	def getParams(self):
		return self.logSlope, self.x0, self.logP


def fitSigmoid(sigmoid, depthsDb, correctRatio, verbose):
	'''
	correctRatio: shape=(depth, )
	'''
	try:
		if isinstance(sigmoid, Sigmoid):
			sigmoid.fit(depthsDb, correctRatio)
		elif isinstance(sigmoid, AsymSigmoid):
			sigmoid.fitInitSym(depthsDb, correctRatio)
		
		return sigmoid.getParams()
	
	except RuntimeError as e:
		print("ERROR", verbose, e, sys.stderr)
		return None
	
	