from chainer import Chain, functions, links, Variable
import numpy as np
import itertools
import chainer
from numpy import newaxis, float32


class Net(Chain):
	def __init__(self, numLabels, architecture):
		'''
		numLabels: [(datasetName: numLabel), ]
		architecture: ((channel, input len, filter len), ...)
		'''
		super(Net, self).__init__()
		
		self.numLabels=numLabels
		self.architecture=architecture
		self.numLayer=len(architecture)
		self.act=functions.elu
		
		for li,st in enumerate(architecture):
			numChannel,inputLen,filterLen=st
			
			if li==0: inChannel=1
			else: inChannel=architecture[li-1][0]
			
			if filterLen==1:
				assert inputLen==1
				dil=1
			else:
				assert (inputLen-1)%(filterLen-1)==0
				dil=(inputLen-1)//(filterLen-1)
			
			conv=links.DilatedConvolution2D(inChannel, numChannel, (filterLen,1), 1, 0, (dil,1))
			super(Net, self).add_link("c"+str(li), conv)
		
		if numLabels is not None:
			for datasetName,numLabel in self.numLabels:
				full=links.Convolution2D(architecture[-1][0], numLabel, 1)
				super(Net, self).add_link("full_"+datasetName, full)
		
	def __call__(self, x, datasetNames=None):
		for li,st in enumerate(self.architecture):
			x=self["c"+str(li)](x)
			x=self.act(x)
		
		if self.numLabels is None: return x
		y={}
		for datasetName,numLabel in self.numLabels:
			if datasetNames is None or datasetName in datasetNames:
				yi=self["full_"+datasetName](x)
				y[datasetName]=yi
		return y
		
	def getNumChannel(self):
		for li,st in enumerate(self.architecture):
			nc,inputLen,filterLen=st
			if li==0:
				numChannel=nc
			else:
				assert numChannel==nc
		return numChannel


def totalInputLength(architecture):
	return np.array([st[1]-1 for st in architecture]).sum()+1


def initCupy(gpuId):
	numGpu=0
	if gpuId>=0:
		import cupy
		for gi in itertools.count():
			try:
				cupy.cuda.Device(gi).use()
			except:
				break
		numGpu=gi
		cupy.cuda.Device(gpuId).use()
	return numGpu


def compRepre(net, stim):
	'''
	@return layerRepre: shape=(layer, batch, channel, length)
	'''
	with chainer.using_config('train', False), chainer.using_config("enable_backprop", False):
		layerRepre=compLongRepresentationNoPad(net, stim) #shape=(layer, batch, channel, length)
	layerRepre+=1 #elu
	return layerRepre


def compLongRepresentationNoPad(net, waves, trimInputLen=True):
	'''
	from physiology/recording.py at https://github.com/cycentum/cascaded-am-tuning-for-sound-recognition
	@return repre: shape=(layer, batch, channel, length)
	'''
	xp=net.xp
	
	if trimInputLen:
		inputLen=totalInputLength(net.architecture)
		length=waves.shape[1]-(inputLen-1)
	
	x=waves
	x=x[:,newaxis,:,newaxis]
	x=xp.asarray(x, float32)
	x=Variable(x)
	repre=compRepresentationNoPad(net, x)
	for li,r in enumerate(repre):
		if xp!=np: r=xp.asnumpy(r)
		r=r[...,0]
		if trimInputLen: r=r[...,-length:]
		repre[li]=r
		
	repre=np.stack(repre, axis=0)
	return repre


def compRepresentationNoPad(net, x, targetLayer=None):
	'''
	from physiology/recording.py at https://github.com/cycentum/cascaded-am-tuning-for-sound-recognition
	@param x: shape=(batch, channel=1, length, 1)
	@return repre: len=layer, [shape=(batch, channel, length, 1),... ]
	'''
	repre=[]
	if targetLayer is not None:
		targetLayer=set(targetLayer)
		maxLayer=max(targetLayer)
	for li,st in enumerate(net.architecture):
		x=net["c"+str(li)](x)
# 		x=functions.elu(x)
		x=net.act(x)
		
		if targetLayer is None or li in targetLayer:
			repre.append(x.data)
			if targetLayer is not None and li==maxLayer: break 
	
	return repre
