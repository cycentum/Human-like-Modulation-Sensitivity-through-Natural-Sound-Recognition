import sys
import itertools
import pickle
import numpy as np
from numpy import newaxis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
	import cupy
except:
	print("Could not import cupy", sys.stderr)
	cupy=None

from path import DIR_TIME_AVE, DIR_TRAINING, DIR_NET
from psychophysics_utils import makeFreqDepStr, loadNet, log_expit, AsymSigmoid, fitSigmoid, NetType
from psychophysics_stim_param import makeStimParam
from utils import checkRandState
from net import initCupy, compRepre
from params import WAVE_RMS_VAL, DATASET_WAVE_FS
from training_utils import loadBestEpoch


def saveTimeAve(stimParam, freqIndex, depthIndex, netType):
	freq=stimParam.freqs[freqIndex]
	print("freq", freq)
	depthDb=stimParam.depthsDb[depthIndex]
	print("depthDb", depthDb)
	
	dirResult=netType.makePath(DIR_TIME_AVE)/stimParam.name/"TimeAve"/makeFreqDepStr(freq, depthDb)
	dirResult.mkdir(exist_ok=True, parents=True)
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
	
	timeAve=np.empty((numLayer, numChannel, numInstanceDepth10))
	
	stimAll=stimParam.makeStim(freq*np.ones(numInstanceDepth10), stimRms, inputLength, depth10)
	for bi,index in enumerate(batchIndex):
# 		print("batch", bi, "/", numBatch)
		stim=stimAll[index]
		layerRepre=compRepre(net, stim) #shape=(layer, batch, channel, length)
		ave=layerRepre.mean(axis=-1)
		ave=ave.transpose(0, 2, 1) #shape=(layer, channel, batch)
		timeAve[:,:,index]=ave

	timeAve=timeAve.reshape(numLayer, numChannel, 1+stimParam.numDepth0, numInstance)

	fileAve=dirResult/"TimeAve.pkl"
	with open(fileAve, "wb") as f: pickle.dump(timeAve, f)

	
def logisticLayer(stimParam, freqIndex, depthIndex, netType):
	freq=stimParam.freqs[freqIndex]
	print("freq", freq)
	depthDb=stimParam.depthsDb[depthIndex]
	print("depthDb", depthDb)
	
	dirTimeAve=netType.makePath(DIR_TIME_AVE)/stimParam.name/"TimeAve"/makeFreqDepStr(freq, depthDb)
	dirLogistic=netType.makePath(DIR_TIME_AVE)/stimParam.name/"LogisticLayer"/makeFreqDepStr(freq, depthDb)
	dirLogistic.mkdir(exist_ok=True, parents=True)

	numInstance=2**7
	
	numFold=4 #val0, tra0: outer fold; val1, tra1: inner fold
	assert numInstance%numFold==0
	
	numInstanceVal0=numInstance//numFold
	numInstanceTra0=numInstance-numInstanceVal0
	assert numInstanceTra0%numFold==0
	label0=np.concatenate((np.ones((1, numInstanceTra0)), np.zeros((stimParam.numDepth0, numInstanceTra0))), axis=0) #shape=(1+numDepth0, instance)
	label0=label0.reshape((1+stimParam.numDepth0)*numInstanceTra0)
	
	numInstanceVal1=numInstanceTra0//numFold
	numInstanceTra1=numInstanceTra0-numInstanceVal1
	label1=np.concatenate((np.ones((1, numInstanceTra1)), np.zeros((stimParam.numDepth0, numInstanceTra1))), axis=0) #shape=(1+numDepth0, instance)
	label1=label1.reshape((1+stimParam.numDepth0)*numInstanceTra1)
	
	checkRandState(dirLogistic/("RandState_foldIndex.pkl"))
	fold0Index=np.array_split(np.arange(numInstance), numFold)
	fold1Index=[]
	for fold0 in range(numFold):
		val0Index=fold0Index[fold0]
		tra0Index=np.array(sorted(set(range(numInstance))-set(val0Index)))
		tra0Index=np.random.permutation(tra0Index)
		index=np.array_split(tra0Index, numFold)
		fold1Index.append(index)
	
	numLayer=netType.numLayer
	numUnit=netType.numUnit
	
	cs=np.logspace(-3,1,9,base=2)
	
	checkRandState(dirLogistic/("RandState.pkl"))
	
	fileTimeAve=dirTimeAve/"TimeAve.pkl"
	with open(fileTimeAve, "rb") as f: timeAve=pickle.load(f) #shape=(layer, channel, 1+numDepth0, instance)
	assert timeAve.shape==(numLayer, numUnit, 1+stimParam.numDepth0, numInstance)
	
	correctRatio1=np.empty((numLayer, numFold, len(cs)), float)
	output0=np.empty((numLayer, numFold, numInstanceVal0, 1+stimParam.numDepth0))
	for li in range(numLayer):
		for fold0 in range(numFold):
			print("layer", li, "fold0", fold0)
			
			val0Index=fold0Index[fold0]
			tra0Index=np.array(sorted(set(range(numInstance))-set(val0Index)))
			
			xTra0=timeAve[li][..., tra0Index].copy().reshape(numUnit, (1+stimParam.numDepth0)*numInstanceTra0).T
			xVal0=timeAve[li][..., val0Index].copy().reshape(numUnit, (1+stimParam.numDepth0)*numInstanceVal0).T

			scaler=StandardScaler()
			xTra0=scaler.fit_transform(xTra0)
			xVal0=scaler.transform(xVal0)
			
			output1=np.empty((numFold, len(cs), numInstanceVal1, 1+stimParam.numDepth0))
			for fold1 in range(numFold):
				val1Index=fold1Index[fold0][fold1]
				tra1Index=np.array(sorted(set(tra0Index)-set(val1Index)))
				
				xTra1=timeAve[li][..., tra1Index].copy().reshape(numUnit, (1+stimParam.numDepth0)*numInstanceTra1).T
				xVal1=timeAve[li][..., val1Index].copy().reshape(numUnit, (1+stimParam.numDepth0)*numInstanceVal1).T

				xTra1=scaler.transform(xTra1)
				xVal1=scaler.transform(xVal1)
				
				for ci,c in enumerate(cs):
					reg=LogisticRegression(penalty="l2", max_iter=10000, solver="lbfgs", class_weight='balanced', C=c)
					reg.fit(xTra1, label1)
					
					y=reg.decision_function(xVal1) #y.shape=(numInstance*(1+numDepths0), )
					y=y.reshape(1+stimParam.numDepth0, numInstanceVal1).T #shape=(numInstance, 1+numDepth0)
					output1[fold1,ci]=y
	
			logProb=log_expit(output1)
			correct=(logProb[..., 0, newaxis]>logProb[..., 1:]).all(axis=-1) #shape=(fold1, c, instance)
			correct=correct.mean(axis=-1) #shape=(fold1, c)
			correct=correct.mean(axis=0)
			correctRatio1[li, fold0]=correct
			
			bestCIndex=correct.argmax() #if tie, smallest c (because argmax returns the first index)
			bestC=cs[bestCIndex]
			
			reg=LogisticRegression(penalty="l2", max_iter=10000, solver="lbfgs", class_weight='balanced', C=bestC)
			reg.fit(xTra0, label0)
			
			y=reg.decision_function(xVal0) #y.shape=(numInstance*(1+numDepths0), )
			y=y.reshape(1+stimParam.numDepth0, numInstanceVal0).T #shape=(numInstance, 1+numDepth0)
			output0[li, fold0]=y
	
	logProb=log_expit(output0)
	correct=(logProb[..., 0, newaxis]>logProb[..., 1:]).all(axis=-1) #shape=(layer, fold, instance)
	correct=correct.mean(axis=-1) #shape=(layer, fold)
	correct=correct.mean(axis=1) #shape=(layer, )
	
	fileCorrect=dirLogistic/("CorrectRatio1.pkl")
	with open(fileCorrect, "wb") as f: pickle.dump((correctRatio1, cs), f)
	
	fileCorrect=dirLogistic/("CorrectRatio0.pkl")
	with open(fileCorrect, "wb") as f: pickle.dump(correct, f)
	
	
def logisticWhole(stimParam, freqIndex, depthIndex, netType):
	freq=stimParam.freqs[freqIndex]
	print("freq", freq)
	depthDb=stimParam.depthsDb[depthIndex]
	print("depthDb", depthDb)
	
	dirTimeAve=netType.makePath(DIR_TIME_AVE)/stimParam.name/"TimeAve"/makeFreqDepStr(freq, depthDb)
	dirLogistic=netType.makePath(DIR_TIME_AVE)/stimParam.name/"LogisticWhole"/makeFreqDepStr(freq, depthDb)
	dirLogistic.mkdir(exist_ok=True, parents=True)

	numInstance=2**7
	
	numFold=4 #val0, tra0: outer fold; val1, tra1: inner fold
	assert numInstance%numFold==0
	
	numInstanceVal0=numInstance//numFold
	numInstanceTra0=numInstance-numInstanceVal0
	assert numInstanceTra0%numFold==0
	label0=np.concatenate((np.ones((1, numInstanceTra0)), np.zeros((stimParam.numDepth0, numInstanceTra0))), axis=0) #shape=(1+numDepth0, instance)
	label0=label0.reshape((1+stimParam.numDepth0)*numInstanceTra0)
	
	numInstanceVal1=numInstanceTra0//numFold
	numInstanceTra1=numInstanceTra0-numInstanceVal1
	label1=np.concatenate((np.ones((1, numInstanceTra1)), np.zeros((stimParam.numDepth0, numInstanceTra1))), axis=0) #shape=(1+numDepth0, instance)
	label1=label1.reshape((1+stimParam.numDepth0)*numInstanceTra1)
	
	checkRandState(dirLogistic/("RandState_foldIndex.pkl"))
	fold0Index=np.array_split(np.arange(numInstance), numFold)
	fold1Index=[]
	for fold0 in range(numFold):
		val0Index=fold0Index[fold0]
		tra0Index=np.array(sorted(set(range(numInstance))-set(val0Index)))
		tra0Index=np.random.permutation(tra0Index)
		index=np.array_split(tra0Index, numFold)
		fold1Index.append(index)
	
	numLayer=netType.numLayer
	numUnit=netType.numUnit
	
	cs=np.logspace(-3,1,9,base=2)
	
	checkRandState(dirLogistic/("RandState.pkl"))
	
	fileTimeAve=dirTimeAve/"TimeAve.pkl"
	with open(fileTimeAve, "rb") as f: timeAve=pickle.load(f) #shape=(layer, channel, 1+numDepth0, instance)
	assert timeAve.shape==(numLayer, numUnit, 1+stimParam.numDepth0, numInstance)
	
	correctRatio1=np.empty((numFold, len(cs)), float)
	output0=np.empty((numFold, numInstanceVal0, 1+stimParam.numDepth0))
	for fold0 in range(numFold):
		print("fold0", fold0)
		
		val0Index=fold0Index[fold0]
		tra0Index=np.array(sorted(set(range(numInstance))-set(val0Index)))
		
		xTra0=timeAve[..., tra0Index].copy().reshape(numLayer*numUnit, (1+stimParam.numDepth0)*numInstanceTra0).T
		xVal0=timeAve[..., val0Index].copy().reshape(numLayer*numUnit, (1+stimParam.numDepth0)*numInstanceVal0).T

		scaler=StandardScaler()
		xTra0=scaler.fit_transform(xTra0)
		xVal0=scaler.transform(xVal0)
		
		output1=np.empty((numFold, len(cs), numInstanceVal1, 1+stimParam.numDepth0))
		for fold1 in range(numFold):
			val1Index=fold1Index[fold0][fold1]
			tra1Index=np.array(sorted(set(tra0Index)-set(val1Index)))
			
			xTra1=timeAve[..., tra1Index].copy().reshape(numLayer*numUnit, (1+stimParam.numDepth0)*numInstanceTra1).T
			xVal1=timeAve[..., val1Index].copy().reshape(numLayer*numUnit, (1+stimParam.numDepth0)*numInstanceVal1).T

			xTra1=scaler.transform(xTra1)
			xVal1=scaler.transform(xVal1)
			
			for ci,c in enumerate(cs):
				reg=LogisticRegression(penalty="l2", max_iter=10000, solver="lbfgs", class_weight='balanced', C=c)
				reg.fit(xTra1, label1)
				
				y=reg.decision_function(xVal1) #y.shape=(numInstance*(1+numDepths0), )
				y=y.reshape(1+stimParam.numDepth0, numInstanceVal1).T #shape=(numInstance, 1+numDepth0)
				output1[fold1,ci]=y

		logProb=log_expit(output1)
		correct=(logProb[..., 0, newaxis]>logProb[..., 1:]).all(axis=-1) #shape=(fold1, c, instance)
		correct=correct.mean(axis=-1) #shape=(fold1, c)
		correct=correct.mean(axis=0)
		correctRatio1[fold0]=correct
		
		bestCIndex=correct.argmax() #if tie, smallest c (because argmax returns the first index)
		bestC=cs[bestCIndex]
		
		reg=LogisticRegression(penalty="l2", max_iter=10000, solver="lbfgs", class_weight='balanced', C=bestC)
		reg.fit(xTra0, label0)
		
		y=reg.decision_function(xVal0) #y.shape=(numInstance*(1+numDepths0), )
		y=y.reshape(1+stimParam.numDepth0, numInstanceVal0).T #shape=(numInstance, 1+numDepth0)
		output0[fold0]=y
	
	logProb=log_expit(output0)
	correct=(logProb[..., 0, newaxis]>logProb[..., 1:]).all(axis=-1) #shape=(fold, instance)
	correct=correct.mean(axis=-1) #shape=(fold, )
	correct=correct.mean() #shape=(, )
	
	fileCorrect=dirLogistic/("CorrectRatio1.pkl")
	with open(fileCorrect, "wb") as f: pickle.dump((correctRatio1, cs), f)
	
	fileCorrect=dirLogistic/("CorrectRatio0.pkl")
	with open(fileCorrect, "wb") as f: pickle.dump(correct, f)


def loadCorrectRatioLayer(stimParam, netType, dirResult):
	'''
	return correctRatio: shape=(freq, depth, layer)
	'''
	
	dirLogistic=netType.makePath(dirResult)/stimParam.name/"LogisticLayer"
	
	numLayer=netType.numLayer
	correctRatio=np.empty((len(stimParam.freqs), len(stimParam.depthsDb), numLayer))
	for (fi,freq),(di,depthDb) in itertools.product(enumerate(stimParam.freqs), enumerate(stimParam.depthsDb)):
		fileCorrect=dirLogistic/makeFreqDepStr(freq, depthDb)/("CorrectRatio0.pkl")
		with open(fileCorrect, "rb") as f: c=pickle.load(f)
		correctRatio[fi,di]=c
	
	return correctRatio


def logisticLayerSigmoid(stimParam, netType):
	correctRatio=loadCorrectRatioLayer(stimParam, netType, DIR_TIME_AVE)
	
	chance=1/(1+stimParam.numDepth0)
	sigmoid=AsymSigmoid(chance)
	
	numLayer=netType.numLayer
	params=np.nan*np.empty((numLayer, len(stimParam.freqs), sigmoid.NUM_PARAM))
	for li,(fi,freq) in itertools.product(range(numLayer), enumerate(stimParam.freqs)):
		p=fitSigmoid(sigmoid, stimParam.depthsDb, correctRatio[fi, :, li], str((li,fi,freq)))
		params[li,fi]=p
	
	fileSigmoid=netType.makePath(DIR_TIME_AVE)/stimParam.name/"LogisticLayerSigmoid.pkl"
	with open(fileSigmoid, "wb") as f: pickle.dump(params, f)
	
	
def logisticWholeSigmoid(stimParam, netType):
	chance=1/(1+stimParam.numDepth0)
	sigmoid=AsymSigmoid(chance)
	
	dirLogistic=netType.makePath(DIR_TIME_AVE)/stimParam.name/"LogisticWhole"
	fileSigmoid=netType.makePath(DIR_TIME_AVE)/stimParam.name/"LogisticWholeSigmoid.pkl"

	correctRatio=np.empty((len(stimParam.freqs), len(stimParam.depthsDb)))
	for (fi,freq),(di,depthDb) in itertools.product(enumerate(stimParam.freqs), enumerate(stimParam.depthsDb)):
		fileCorrect=dirLogistic/makeFreqDepStr(freq, depthDb)/("CorrectRatio0.pkl")
		with open(fileCorrect, "rb") as f: c=pickle.load(f)
		correctRatio[fi,di]=c
	
	params=np.nan*np.empty((len(stimParam.freqs), sigmoid.NUM_PARAM))
	for fi,freq in enumerate(stimParam.freqs):
		p=fitSigmoid(sigmoid, stimParam.depthsDb, correctRatio[fi, :], str((fi,freq)))
		params[fi]=p
	
	with open(fileSigmoid, "wb") as f: pickle.dump(params, f)
	
	

if __name__=="__main__":
# 	gpu_id=0
	gpu_id=-1 #when no GPU is available

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
	
	breakValEpoch=96
	epoch=loadBestEpoch(DIR_NET, architectureName, datasetType, breakValEpoch, CONTROL_TYPE)
	
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
	
	for freqIndex, depthIndex in itertools.product(range(len(stimParam.freqs)), range(len(stimParam.depthsDb))):
		saveTimeAve(stimParam, freqIndex, depthIndex, netType)
		
	for freqIndex, depthIndex in itertools.product(range(len(stimParam.freqs)), range(len(stimParam.depthsDb))):
		logisticLayer(stimParam, freqIndex, depthIndex, netType)
	logisticLayerSigmoid(stimParam, netType)
		
	for freqIndex, depthIndex in itertools.product(range(len(stimParam.freqs)), range(len(stimParam.depthsDb))):
		logisticWhole(stimParam, freqIndex, depthIndex, netType)
	logisticWholeSigmoid(stimParam, netType)
