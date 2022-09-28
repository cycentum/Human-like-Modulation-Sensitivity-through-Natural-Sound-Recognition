import itertools
import pickle
import numpy as np
from numpy import newaxis

try:
	import cupy
except:
	cupy=None

from path import DIR_TEMPLATE_CORREL, DIR_TRAINING
from psychophysics_utils import makeFreqStr, makeDepthDbStr, makeFreqDepStr, loadNet, AsymSigmoid, fitSigmoid, NetType
from psychophysics_stim_param import makeStimParam
from utils import checkRandState, corrcoef
from net import initCupy, compRepre
from params import WAVE_RMS_VAL, DATASET_WAVE_FS



def saveTemplate(stimParam, freqIndex, netType):
	freq=stimParam.freqs[freqIndex]
	print("freq", freq)
	
	dirResult=netType.makePath(DIR_TEMPLATE_CORREL)/stimParam.name/"Template"/makeFreqStr(freq)
	dirResult.mkdir(exist_ok=True, parents=True)
	
	net, inputLength=loadNet(netType, DIR_TRAINING)
	if gpu_id>=0: net.to_gpu(gpu_id)
	
	numChannel=net.getNumChannel()
	numLayer=net.numLayer
	
	stimRms=WAVE_RMS_VAL
	repreLen=stimParam.length+(inputLength-1)
	templateSize=2**7
		
	templateNumBatch=int(np.ceil(templateSize/netBatchSizeUpper))
	templateBatchIndex=np.array_split(np.arange(templateSize), templateNumBatch)
	
	templateDepthDb=(-np.inf, stimParam.depthsDb[-1])
	for depthDb in templateDepthDb:
		checkRandState(dirResult/("RandState_Dep"+makeDepthDbStr(depthDb)+".pkl"))
		stim=stimParam.makeStim(freq*np.ones(templateSize), stimRms, inputLength, depthDb*np.ones(templateSize))
		
		template=np.zeros((numLayer, numChannel, repreLen))
		for bi,index in enumerate(templateBatchIndex):
			print("dep", depthDb, bi, "/", templateNumBatch)
			
			batchStim=stim[index]
			layerRepre=compRepre(net, batchStim) #shape=(layer, batch, channel, length)
			template+=layerRepre.sum(axis=1)
		template/=templateSize
		
		fileTemplate=dirResult/("Template_Dep"+makeDepthDbStr(depthDb)+".pkl")
		with open(fileTemplate, "wb") as f: pickle.dump(template, f)


def saveTemplateCorrel(stimParam, freqIndex, depthIndex, netType):
	freq=stimParam.freqs[freqIndex]
	print("freq", freq)
	depthDb=stimParam.depthsDb[depthIndex]
	print("depthDb", depthDb)
	
	dirResult=netType.makePath(DIR_TEMPLATE_CORREL)/stimParam.name/"TemplateCorrelLayer"/makeFreqDepStr(freq, depthDb)
	checkRandState(dirResult/"RandState.pkl")
	
	net, inputLength=loadNet(netType, DIR_TRAINING)
	if gpu_id>=0: net.to_gpu(gpu_id)
	
	numChannel=net.getNumChannel()
	numLayer=net.numLayer
	
	stimRms=WAVE_RMS_VAL
	numInstance=2**7
	
	numInstanceDepth10=numInstance*(1+stimParam.numDepth0)
	depth10=np.concatenate((depthDb*np.ones((1, numInstance)), -np.inf*np.ones((stimParam.numDepth0, numInstance))), axis=0) #shape=(1+numDepth0, instance)
	depth10=depth10.reshape(numInstanceDepth10)
	
	numBatch=int(np.ceil(numInstanceDepth10/netBatchSizeUpper))
	batchIndex=np.array_split(np.arange(numInstanceDepth10), numBatch)
	
	stim=stimParam.makeStim(freq*np.ones(numInstanceDepth10), stimRms, inputLength, depth10)
	
	dirResultTemplate=netType.makePath(DIR_TEMPLATE_CORREL)/stimParam.name/"Template"/makeFreqStr(freq)
	templateDepthDb=(-np.inf, stimParam.depthsDb[-1])
	template={} #template[depthDb]=np.ndarray((numLayer, numChannel, repreLen))
	for dep in templateDepthDb:
		fileTemplate=dirResultTemplate/("Template_Dep"+makeDepthDbStr(dep)+".pkl")
		with open(fileTemplate, "rb") as f: t=pickle.load(f)
		template[dep]=t #shape=(numLayer, numChannel, repreLen)
		
	template=template[templateDepthDb[1]]-template[templateDepthDb[0]]
	
	templateCorrel=np.empty((numInstanceDepth10, numLayer))
	for bi,index in enumerate(batchIndex):
		print(bi, "/", numBatch)
		
		batchStim=stim[index]
		layerRepre=compRepre(net, batchStim) #shape=(layer, batch, channel, length)
		layerRepre=layerRepre.transpose(1,0,2,3) #shape=(batch, layer, channel, length)
		
		templateCorrel[index]=corrcoef(layerRepre, template[newaxis], axis=(-1,-2), normalized=False) #mean product

	fileTemplateCorrel=dirResult/"TemplateCorrel.pkl"
	with open(fileTemplateCorrel, "wb") as f: pickle.dump(templateCorrel, f) #np.ndarray((numInstanceDepth10, layer))
	
	
def saveNumCorrect(stimParam, freqIndex, depthIndex, netType):
	freq=stimParam.freqs[freqIndex]
	print("freq", freq)
	depthDb=stimParam.depthsDb[depthIndex]
	print("depthDb", depthDb)
	
	dirResult=netType.makePath(DIR_TEMPLATE_CORREL)/stimParam.name/"TemplateCorrelLayer"/makeFreqDepStr(freq, depthDb)
	
	fileTemplateCorrel=dirResult/"TemplateCorrel.pkl"
	with open(fileTemplateCorrel, "rb") as f: templateCorrel=pickle.load(f) #np.ndarray((numInstanceDepth10, layer))
	assert (~np.isnan(templateCorrel)).all()
	assert (~np.isinf(templateCorrel)).all()
	
	numInstance=2**7
	numLayer=templateCorrel.shape[1]

	templateCorrel=templateCorrel.reshape(1+stimParam.numDepth0, numInstance, numLayer) #shape=(1+numDepth0, instance, layer)
	correct=(templateCorrel[0:1]>templateCorrel[1:]).all(axis=0) #shape=(instance, layer
	numCorrect=correct.sum(axis=0) #shape=(layer, )
	
	fileNumCorrect=dirResult/"NumCorrect.pkl"
	with open(fileNumCorrect, "wb") as f: pickle.dump((numCorrect, numInstance), f)


def saveSigmoid(stimParam, netType):
	chance=1/(1+stimParam.numDepth0)
	sigmoid=AsymSigmoid(chance)
	
	dirResultTemplateCorrel=netType.makePath(DIR_TEMPLATE_CORREL)/stimParam.name/"TemplateCorrelLayer"
	for (fi,freq),(di,depthDb) in itertools.product(enumerate(stimParam.freqs), enumerate(stimParam.depthsDb)):
		fileNumCorrect=dirResultTemplateCorrel/makeFreqDepStr(freq, depthDb)/"NumCorrect.pkl"
		with open(fileNumCorrect, "rb") as f: numCorrect, numInstance=pickle.load(f)
		numLayer=numCorrect.shape[0]
		if fi==0 and di==0:
			correctRatio=np.empty((len(stimParam.freqs), len(stimParam.depthsDb), numLayer))
		correctRatio[fi,di]=numCorrect/numInstance
	
	params=np.nan*np.empty((numLayer, len(stimParam.freqs), sigmoid.NUM_PARAM))
	for li,(fi,freq) in itertools.product(range(numLayer), enumerate(stimParam.freqs)):
		sigmoidParam=fitSigmoid(sigmoid, stimParam.depthsDb, correctRatio[fi, :, li], str((li,fi,freq)))
		if sigmoidParam is not None:
			params[li, fi]=sigmoidParam

	fileSigmoid=netType.makePath(DIR_TEMPLATE_CORREL)/stimParam.name/"TemplateCorrelLayerSigmoid.pkl"
	with open(fileSigmoid, "wb") as f: pickle.dump(params, f)



if __name__=="__main__":
	gpu_id=0
# 	gpu_id=-1 #when no GPU is available

	NUM_GPU=initCupy(gpu_id)
	print("NUM_GPU", NUM_GPU)
	
	netBatchSizeUpper=8 #set according to your memory size

	datasetType="ESC"
# 	datasetType="TIMIT"

	CONTROL_TYPE="Original"
# 	CONTROL_TYPE="EnvSingleBand"
# 	CONTROL_TYPE="TFSSingleBand"
# 	CONTROL_TYPE="EnvMultiBand"
# 	CONTROL_TYPE="TFSMultiBand"
	
	architecture_numLayer=13 #can be one of {7, 9, 11, 13}
	architecture_numUnit=256 #can be one of {32, 64, 128, 256, 512}
	architecture_sample=0   #can be one of {0, 1, 2, 3}
	
	architectureName="Layer"+str(architecture_numLayer)+"_Unit"+str(architecture_numUnit)+"_Sample"+str(architecture_sample)
	
	epoch=13920
	
	netType=NetType(datasetType, architectureName, CONTROL_TYPE, epoch)
	
	waveFs=DATASET_WAVE_FS[datasetType]
	
	stimParamName="Viemeister1979_CfN_BwInf"
# 	stimParamName="Dau1997_Cf5000_Bw314"
# 	stimParamName="Dau1997_Cf5000_Bw31"
# 	stimParamName="Dau1997_Cf5000_Bw3"
# 	stimParamName="Lorenzi2001_CfN_BwInf"
# 	stimParamName="Lorenzi2001_Cf5000_Bw2"
	
	stimParam=makeStimParam(stimParamName, waveFs)
	
	print(stimParam.name)
	print("freqs", stimParam.freqs)
	print("depthsDb", stimParam.depthsDb)
	
	for freqIndex in range(len(stimParam.freqs)):
		saveTemplate(stimParam, freqIndex, netType)
	
		for depthIndex in range(len(stimParam.depthsDb)):
			saveTemplateCorrel(stimParam, freqIndex, depthIndex, netType)
			saveNumCorrect(stimParam, freqIndex, depthIndex, netType)
	
	saveSigmoid(stimParam, netType)
