import sys
import itertools
import pickle
import numpy as np
from numpy import newaxis
from training_utils import loadBestEpoch
from zipfile import ZipFile
from io import BytesIO

try:
	import cupy
except:
	print("Could not import cupy", sys.stderr)
	cupy=None

from path import DIR_PHYSIOLOGY, DIR_TRAINING, DIR_NET, DIR_REPO_NEUROPHYSIOLOGY
from psychophysics_utils import makeFreqStr, loadNet, NetType, scaleRms
from utils import checkRandState
from net import initCupy, compRepre
from params import WAVE_RMS_VAL, DATASET_WAVE_FS

sys.path.append(str(DIR_REPO_NEUROPHYSIOLOGY/"cascaded-am-tuning-for-sound-recognition"/"physiology"))
try:
	from mtf_analysis import compLayerMeasure, compRegionLayerSimilarity
	from am_meta_analysis import readCumulative
except:
	print("Could not import cascaded-am-tuning-for-sound-recognition")


REGIONS=("AN","CN","SOC","NLL", "IC", "MGB","cortex")


def vectorCosSin(freq, inputLen, times):
	period=1/freq
	angle=times%period/period*2*np.pi
	cos=np.cos(angle)
	sin=np.sin(angle)
	
	cos=cos[inputLen-1:]
	sin=sin[inputLen-1:]
	return cos,sin


def saveResponse(freqIndex, netType):
	freq=FREQS[freqIndex]
	print("freq", freq)
	
	dirResult=netType.makePath(DIR_PHYSIOLOGY)/"AveSyn"/makeFreqStr(freq)
	dirResult.mkdir(exist_ok=True, parents=True)
	
	net, inputLength=loadNet(netType, DIR_TRAINING)
	if gpu_id>=0: net.to_gpu(gpu_id)
	
	checkRandState(dirResult/"RandState.pkl")
	
	stimSec=4
	waveLen=int(stimSec*waveFs)
	times=np.arange(waveLen)/waveFs
	
	meanSize=2**3
	numBatch=int(np.ceil(meanSize/netBatchSizeUpper))
	batchIndex=np.array_split(np.arange(meanSize), numBatch)
	
	modDepth=1
	stimAll=np.random.randn(meanSize, waveLen)
	stimAll*=(1-modDepth*np.cos(freq*2*np.pi*times))
	scaleRms(stimAll, WAVE_RMS_VAL)
	
	cos,sin=vectorCosSin(freq, inputLength, times)
	
	numChannel=net.getNumChannel()
	numLayer=net.numLayer
	response=np.empty((2, numLayer, numChannel, meanSize))
	for bi,index in enumerate(batchIndex):
# 		print("batch", bi, "/", numBatch, flush=True)
		stim=stimAll[index]
		repre=compRepre(net, stim) #shape=(layer, batch, channel, length)
		
		ave=repre.mean(axis=-1)
		s=repre.sum(axis=-1)
		syn=(((repre*cos).sum(axis=-1)/s)**2+((repre*sin).sum(axis=-1)/s)**2)**0.5
		syn[s==0]=0
		resp=np.stack((ave,syn), axis=0) #shape=(type, layer, batch, channel)
		resp=resp.transpose(0, 1, 3, 2)
		response[..., index]=resp
	
	response=response.mean(axis=-1) #shape=((ave, syn), layer, channel)
	
	fileResponse=dirResult/"Response.pkl"
	with open(fileResponse, "wb") as f: pickle.dump(response, f)


def saveResponse0():
	dirResult=netType.makePath(DIR_PHYSIOLOGY)/"Ave0"
	
	net, inputLength=loadNet(netType, DIR_TRAINING)
	if gpu_id>=0: net.to_gpu(gpu_id)
	
	checkRandState(dirResult/"RandState.pkl")
	
	stimSec=4
	waveLen=int(stimSec*waveFs)
	times=np.arange(waveLen)/waveFs
	meanSize=2**3
	
	numBatch=int(np.ceil(meanSize/netBatchSizeUpper))
	batchIndex=np.array_split(np.arange(meanSize), numBatch)
	
	stimAll=np.random.randn(meanSize, waveLen)
	scaleRms(stimAll, WAVE_RMS_VAL)
	
	numChannel=net.getNumChannel()
	numLayer=net.numLayer
	responseAve=np.empty((numLayer, numChannel, meanSize))
	for bi,index in enumerate(batchIndex):
# 		print("batch", bi, "/", numBatch, flush=True)
		stim=stimAll[index]
		repre=compRepre(net, stim) #shape=(layer, batch, channel, length)
		
		ave=repre.mean(axis=-1) #shape=(layer, batch, channel)
		ave=ave.transpose(0, 2, 1) #shape=(layer, channel, batch)
		responseAve[..., index]=ave
	
	responseAve=responseAve.mean(axis=-1) #shape=(layer, channel)
	
	fileResponse=dirResult/"Response.pkl"
	with open(fileResponse, "wb") as f: pickle.dump(responseAve, f)


def loadResponse(dirNetType):
	response=[]
	fileZip=dirNetType/"AveSyn.zip"
	if fileZip.is_file():
		with ZipFile(fileZip, "r") as zf:
			for fi,freq in enumerate(FREQS):
				b=zf.read("AveSyn/"+makeFreqStr(freq)+"/Response.pkl")
				f=BytesIO(b)
				r=pickle.load(f)
				response.append(r)
	else:
		for fi,freq in enumerate(FREQS):
			dirResult=dirNetType/"AveSyn"/makeFreqStr(freq)
			fileResponse=dirResult/"Response.pkl"
			with open(fileResponse, "rb") as f: r=pickle.load(f) #shape=(tuningType, layer, channel)
			response.append(r)
	response=np.stack(response, axis=-1) #shape=(tuningType, layer, channel, freq)
	return response


def saveSimilarityMatrix():
	dirNetType=netType.makePath(DIR_PHYSIOLOGY)
	fileSim=dirNetType/"SimilarityMatrix.pkl"
	
	response=loadResponse(dirNetType)
	response=response.transpose(3, 0, 1, 2) #shape=(freq, tuningType, layer, channel)
	numLayer, numChannel=response.shape[-2:]
	
	with open(dirNetType/"Ave0"/"Response.pkl", "rb") as f: responseAve0=pickle.load(f)
	
	layerMeasures=compLayerMeasure(response, responseAve0)
	dirCumulatives=DIR_REPO_NEUROPHYSIOLOGY/"cascaded-am-tuning-for-sound-recognition"/"cascaded-am-tuning-for-sound-recognition"/"am-meta-analysis"/"cumulative"
	cumulatives=readCumulative([file for file in layerMeasures.keys() if "-" not in file], str(dirCumulatives))
	similarity=compRegionLayerSimilarity(layerMeasures, cumulatives, numLayer, REGIONS, numChannel)
	
	with open(fileSim, "wb") as f: pickle.dump(similarity, f)



if __name__=="__main__":
	gpu_id=0
# 	gpu_id=-1 #when no GPU is available

	NUM_GPU=initCupy(gpu_id)
	print("NUM_GPU", NUM_GPU)
	
	netBatchSizeUpper=4 #set according to your memory size

	datasetType="ESC"
# 	datasetType="TIMIT"

	architecture_numLayer=13 #can be one of {7, 9, 11, 13}
	architecture_numUnit=256 #can be one of {32, 64, 128, 256, 512}
	architecture_sample=0   #can be one of {0, 1, 2, 3}
	
	architectureName="Layer"+str(architecture_numLayer)+"_Unit"+str(architecture_numUnit)+"_Sample"+str(architecture_sample)
	
	breakValEpoch=96
	epoch=loadBestEpoch(DIR_NET, architectureName, datasetType, breakValEpoch, "Original")
	
	netType=NetType(datasetType, architectureName, "Original", epoch)
	
	waveFs=DATASET_WAVE_FS[datasetType]
	
	FREQS=np.logspace(np.log10(1), np.log10(2000), 2**7+1)
	
	for freqIndex in range(len(FREQS)):
		saveResponse(freqIndex, netType)
	saveResponse0()

	saveSimilarityMatrix()
