import numpy as np
from numpy import newaxis, float64
import scipy.signal

from psychophysics_utils import scaleRms


def fade(wave, fadeSec, waveFs):
	fadeLen=int(fadeSec*waveFs)
	fadeWin=np.hanning(fadeLen*2)
	wave[...,:fadeLen]*=fadeWin[:fadeLen]
	wave[...,-fadeLen:]*=fadeWin[-fadeLen:]


def dbToAmp(db):
	amp=10**(db/20)
	return amp


def cat0(stim, inputLength):
	wave0=np.zeros((stim.shape[0], inputLength-1))
	stim=np.concatenate((wave0, stim, wave0), axis=-1)
	return stim


def bandpassFilter(wave, centerFreq, bandwidth, waveFs):
	'''
	@param wave: shape[-1]=length
	'''
	stimLen=wave.shape[-1]
	fftfreq=np.fft.rfftfreq(stimLen)*waveFs
	sp=np.fft.rfft(wave, axis=-1)
	f0,f1=centerFreq-bandwidth/2,centerFreq+bandwidth/2
	sp[..., (fftfreq<f0)|(fftfreq>f1)]=0
	wave=np.fft.irfft(sp, stimLen, axis=-1)
	
	return wave



class StimParam_Viemeister1979_CfN_BwInf:
	def __init__(self, waveFs):
		self.waveFs=waveFs
		
		self.freqRange=(2, 4000)
		self.freqSize=8
		
		self.depthDbRange=(-40, 0) #min, max
		self.depthSize=11
		self.sec=0.5
		self.fadeSec=0.05

		self.numDepth0=1 #2IAFC

		initStimParam(self)
		
	def makeStim(self, freqs, targetRms, inputLength, depthsDb):
		stim,freqs,batchSize=makeNoise(self, freqs, depthsDb)
		
		fade(stim, self.fadeSec, self.waveFs)
		scaleRms(stim, targetRms)
		
		depths=dbToAmp(depthsDb)
		stim*=(1+depths[:,newaxis]*np.sin(freqs[:,newaxis]*2*np.pi*self.times))
		
		stim=cat0(stim, inputLength)
		return stim
				
				
class StimParam_Dau1997_Cf5000:
	def __init__(self, bw, waveFs):
		self.waveFs=waveFs
		
		assert bw==3 or bw==31 or bw==314
		self.bw=bw
		
		if bw==3:
			self.freqRange=(3, 100)
		elif bw==31:
			self.freqRange=(3, 150)
		elif bw==314:
			self.freqRange=(3, 100)
		self.freqSize=8
		
		self.depthDbRange=(-40, 0) #min, max
		self.depthSize=11
		
		self.sec=1
		self.fadeSec=0.2
		cutoff=10000
		if waveFs is not None and cutoff/(waveFs/2)<1:
			self.low10k=scipy.signal.butter(4, cutoff/(waveFs/2))
		else:
			self.low10k=None
		self.cf=5000
		
		self.numDepth0=2 #3IAFC
		
		initStimParam(self)
		self.name+="_Bw"+str(bw)
		
	def makeStim(self, freqs, targetRms, inputLength, depthsDb):
		stim,freqs,batchSize=makeNoise(self, freqs, depthsDb)
		
		if self.bw==3 or self.bw==31:
			stim=bandpassFilter(stim, self.cf, self.bw, self.waveFs)
		
		depths=dbToAmp(depthsDb)
		stim*=(1+depths[:,newaxis]*np.sin(freqs[:,newaxis]*2*np.pi*self.times))
		if self.bw==314:
			stim=bandpassFilter(stim, self.cf, self.bw, self.waveFs)
			
		fade(stim, self.fadeSec, self.waveFs)
		scaleRms(stim, targetRms)
		stim=cat0(stim, inputLength)
		
		if self.low10k is not None:
			stim=scipy.signal.filtfilt(self.low10k[0], self.low10k[1], stim, axis=-1)
		return stim


class StimParam_Lorenzi2001_CfN_BwInf:
	def __init__(self, waveFs):
		self.waveFs=waveFs
		
		self.freqRange=(4, 256)
		self.freqSize=8
		
		self.depthDbRange=(-40, 0) #min, max
		self.depthSize=11
		self.sec=2
		self.fadeSec=0.025
		
		self.numDepth0=1 #2IAFC
				
		initStimParam(self)
		
	def makeStim(self, freqs, targetRms, inputLength, depthsDb):
		stim,freqs,batchSize=makeNoise(self, freqs, depthsDb)
		
		fade(stim, self.fadeSec, self.waveFs)
		scaleRms(stim, targetRms)
		
		phase=np.random.rand(batchSize)*2*np.pi
		
		depths=dbToAmp(depthsDb)
		c=(1+depths**2/2)**(-0.5)
		
		stim*=c[:,newaxis]*(1+depths[:,newaxis]*np.sin(freqs[:,newaxis]*2*np.pi*self.times+phase[:,newaxis]))
		
		stim=cat0(stim, inputLength)
		self.phase=phase
		return stim
	
	
def makeCarrierLorenzi2001Narrow(stimParam, freqs, depthsDb):
	stim,freqs,batchSize=makeNoise(stimParam, freqs, depthsDb)
	stim=bandpassFilter(stim, stimParam.cf, stimParam.bw, stimParam.waveFs)
	
	return stim


class StimParam_Lorenzi2001_Cf5000_Bw2:
	def __init__(self, waveFs):
		self.cf=5000
		self.bw=2
		
		self.waveFs=waveFs
		
		self.freqRange=(1, 256)
		self.freqSize=8
		
		self.depthDbRange=(-60, 0) #min, max
		self.depthSize=16
		self.sec=2
		self.fadeSec=0.05
		
		self.numDepth0=1 #2IAFC
				
		initStimParam(self)
		
		
	def makeStim(self, freqs, targetRms, inputLength, depthsDb):
		batchSize=len(depthsDb)
		stim=makeCarrierLorenzi2001Narrow(self, freqs, depthsDb)
		
		fade(stim, self.fadeSec, self.waveFs)
		scaleRms(stim, targetRms)
		
		phase=np.random.rand(batchSize)*2*np.pi
		
		depths=dbToAmp(depthsDb)
		c=(1+depths**2/2)**(-0.5)
		
		stim*=c[:,newaxis]*(1+depths[:,newaxis]*np.sin(freqs[:,newaxis]*2*np.pi*self.times+phase[:,newaxis]))
		
		stim=cat0(stim, inputLength)
		self.phase=phase
		return stim
	
	
def initStimParam(stimParam):
	stimParam.name=stimParam.__class__.__name__[len("StimParam_"):]

	stimParam.depthsDb=np.linspace(stimParam.depthDbRange[0], stimParam.depthDbRange[1], stimParam.depthSize)
	stimParam.freqs=np.logspace(np.log10(stimParam.freqRange[0]), np.log10(stimParam.freqRange[1]), stimParam.freqSize)
	stimParam.freqs[0]=stimParam.freqRange[0]
	stimParam.freqs[-1]=stimParam.freqRange[1]
	
	if stimParam.waveFs is not None:
		waveFs=stimParam.waveFs
		stimParam.length=int(waveFs*stimParam.sec)
		stimParam.times=np.arange(stimParam.length)/waveFs


def makeNoise(stimParam, freqs, depthsDb):
	batchSize=len(depthsDb)
	if not isinstance(freqs, np.ndarray): freqs=freqs*np.ones(batchSize, float64)
	assert len(freqs)==len(depthsDb)
	stim=np.random.randn(batchSize, stimParam.length)
	return stim, freqs, batchSize


def makeStimParam(name, waveFs):
	if name=="Viemeister1979_CfN_BwInf":
		return StimParam_Viemeister1979_CfN_BwInf(waveFs)
	
	elif name=="Dau1997_Cf5000_Bw314":
		return StimParam_Dau1997_Cf5000(314, waveFs)
	
	elif name=="Dau1997_Cf5000_Bw31":
		return StimParam_Dau1997_Cf5000(31, waveFs)
	
	elif name=="Dau1997_Cf5000_Bw31":
		return StimParam_Dau1997_Cf5000(31, waveFs)
	
	elif name=="Dau1997_Cf5000_Bw3":
		return StimParam_Dau1997_Cf5000(3, waveFs)
	
	elif name=="Lorenzi2001_CfN_BwInf":
		return StimParam_Lorenzi2001_CfN_BwInf(waveFs)
	
	elif name=="Lorenzi2001_Cf5000_Bw2":
		return StimParam_Lorenzi2001_Cf5000_Bw2(waveFs)


def makeStimParamNameAll():
	names=(
		"Lorenzi2001_Cf5000_Bw2",
		"Dau1997_Cf5000_Bw3",
		"Dau1997_Cf5000_Bw31",
		"Dau1997_Cf5000_Bw314",
		"Lorenzi2001_CfN_BwInf",
		"Viemeister1979_CfN_BwInf",
		)
	return names


def makeStimParamAll(waveFs=None):
	names=makeStimParamNameAll()
	stimParams=[makeStimParam(name, waveFs) for name in names]
	return stimParams
