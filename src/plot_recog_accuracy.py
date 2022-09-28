import itertools
import pickle
from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict
# from scipy.stats import gaussian_kde
from operator import itemgetter

from path import DIR_NET
from training_utils import calcCorrectRatio, getBestEpoch, loadEpochCorrect
import utils_plot


def makeChance(datasetName):
	if datasetName=="ESC": chance=1/50
	elif datasetName=="TIMIT": chance=1/39
	return chance


def loadArchAccuracy(datasetNames, numLayers, numUnits, numSample, breakValEpoch):
	accuracy={}
	for datasetName,numLayer,numUnit,sample in itertools.product(datasetNames,numLayers,numUnits,range(numSample)):
		arch="Layer"+str(numLayer)+"_Unit"+str(numUnit)+"_Sample"+str(sample)
		dirResult=DIR_NET/datasetName/arch
		
		epochCorrect=loadEpochCorrect(DIR_NET, arch, datasetName, "Original")
		
		bestEpoch=getBestEpoch(epochCorrect, breakValEpoch)
		
		fileConfusion=dirResult/"ValConfusion"/(str(bestEpoch)+".pkl")
		with open(fileConfusion, "rb") as f: confusion=pickle.load(f)
		
		acc=calcCorrectRatio(confusion[datasetName])
		accuracy[datasetName,numLayer,numUnit,sample]=acc,arch
	
	return accuracy
		

if __name__=="__main__":
	import seaborn as sns

	datasetNames=("ESC", "TIMIT")
	numLayers=(7, 9, 11, 13)
	numUnits=(32, 64, 128, 256, 512)
	numSample=4
	
	accuracy={}
	
	for breakValEpoch in (32, 96):
		if breakValEpoch==32: nl=numLayers
		elif breakValEpoch==96: nl=(13,)
		accuracy[breakValEpoch]=loadArchAccuracy(datasetNames, nl, numUnits, numSample, breakValEpoch)
	
	layerAccuracy=defaultdict(list)
	for breakValEpoch in accuracy.keys():
		for (datasetName,numLayer,numUnit,sample),(acc,arch) in accuracy[breakValEpoch].items():
			layerAccuracy[breakValEpoch,datasetName,numLayer].append((acc,arch))
	
	numBest=4
	layerAveBest={}
	for (breakValEpoch,datasetName,numLayer),acc_arch in layerAccuracy.items():
		acc=list(map(itemgetter(0), acc_arch))
		assert len(acc)==len(numUnits)*numSample, (len(acc), len(numUnits), numSample)
		ave=np.sort(np.array(acc))[-numBest:].mean()
		layerAveBest[breakValEpoch,datasetName,numLayer]=ave
		
		if breakValEpoch==96:
			acc_arch=sorted(acc_arch, key=itemgetter(0))
			print(datasetName)
			for acc,arch in acc_arch:
				print("{:.3f}".format(acc), arch)
			print("{:.3f}".format(ave), "ave")
			print()
		
	for dni,datasetName in enumerate(datasetNames):
		ax=plt.subplot(1,len(datasetNames),dni+1)
		
		ax.set_title(utils_plot.DATASET_NAME[datasetName])
		ax.set_ylabel("Recognition accuracy")
		ax.set_xlabel("Number of layers")
		
		chance=makeChance(datasetName)
# 		ax.plot((numLayers[0], numLayers[-1]), (chance, chance), ":k")
		
# 		for numLayer in numLayers:
# 			acc=layerAccuracy[datasetName,numLayer]
# 			
# 			density=gaussian_kde(acc, 0.2)
# 			jitter=density(acc)*0.015
# 			jitter=0.2
# 			jitter=(np.random.rand(len(acc))*2-1)*jitter
# 			
# 			ax.plot(numLayer*np.ones(len(acc))+jitter, acc, ".k")
# 			
# 			ave=layerAveBest[datasetName,numLayer]
# 			ax.plot(numLayer+0.5, ave, "xk")
		
		x=[numLayer*np.ones(len(layerAccuracy[32,datasetName,numLayer]), np.int32) for numLayer in numLayers]
		x.append((numLayers[-1]+2)*np.ones(len(layerAccuracy[96,datasetName,13]), np.int32)) #(numLayer==13 and breakValEpoch==96) will be plotted at numLayer==15
		x=np.concatenate(x)
		
		y=[list(map(itemgetter(0), layerAccuracy[32,datasetName,numLayer])) for numLayer in numLayers]
		y.append(list(map(itemgetter(0), layerAccuracy[96,datasetName,13])))
		y=np.concatenate(y)
		
		sns.swarmplot(x,y,color=(0.5,0.5,0.5),size=4,ax=ax)
		
		x=[*numLayers, numLayers[-1]+2]
		y=[layerAveBest[32,datasetName,numLayer] for numLayer in numLayers]
		y.append(layerAveBest[96,datasetName,13])
		sns.pointplot(x,y,color=(1,0,0),ax=ax,markers=["X"],join=False)
		
		y=chance*np.ones(len(numLayers)+1)
		sns.pointplot(x,y,color=(0,0,0),ax=ax)

	plt.show()
	