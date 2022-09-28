import itertools
import pickle
from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict
from operator import itemgetter
import seaborn as sns

from path import DIR_NET
from training_utils import calcCorrectRatio
from plot_tmtf import makeNetTypes, makeArchs
from plot_recog_accuracy import makeChance
from utils_plot import DATASET_NAME


def loadAccuracy(datasetName, architectures, controlType):
	netTypes=makeNetTypes(datasetName, architectures, controlType)
	accuracy={}
	for netType in netTypes:
		dirResult=DIR_NET/datasetName/netType.arch
		if controlType!="Original":
			dirResult=dirResult/("Result_"+controlType)
		
		fileConfusion=dirResult/"ValConfusion"/(str(netType.epoch)+".pkl")
		with open(fileConfusion, "rb") as f: confusion=pickle.load(f)
		
		acc=calcCorrectRatio(confusion[datasetName])
		accuracy[netType]=acc
	
	return accuracy
			

if __name__=="__main__":
	datasetNames=("ESC", "TIMIT")
	controlTypes=("Original", "EnvSingleBand", "EnvMultiBand", "TFSSingleBand", "TFSMultiBand")
	
	datasetArchs={}
	for di,datasetName in enumerate(datasetNames):
		architectures=makeArchs(datasetName, 4)
		datasetArchs[datasetName]=architectures
	
	accuracy=defaultdict(list)
	for datasetName,controlType in itertools.product(datasetNames,controlTypes):
		acc=loadAccuracy(datasetName, datasetArchs[datasetName], controlType)
		for netType in acc:
			accuracy[datasetName,controlType,netType.arch]=acc[netType]
	
	for di,datasetName in enumerate(datasetNames):
		ax=plt.subplot(1,2,di+1)
		ax.set_title(DATASET_NAME[datasetName])
		ax.set_ylabel("Recognition accuracy")
		
		chance=makeChance(datasetName)
		
		archs=datasetArchs[datasetName]
		x=[]
		acc=[]
		for arch in archs:
			x.extend(list(range(len(controlTypes))))
			acc.extend([accuracy[datasetName,controlType,arch] for controlType in controlTypes])
		
# 		sns.swarmplot(x,acc,color=(0.5,0.5,0.5),ax=ax)
		
		acc=[]
		for controlType in controlTypes:
			ave=np.array([accuracy[datasetName,controlType,arch] for arch in archs]).mean()
			acc.append(ave)
			print(datasetName, controlType, "{:.3f}".format(ave), sep="\t")
		
		x=list(range(len(controlTypes)))
# 		sns.pointplot(x,acc,color=(0,0,0),ax=ax,markers=["X"],join=False)
		sns.barplot(x,acc,color=(0,0,0),ax=ax)
		
		y=chance*np.ones(len(controlTypes))
		sns.pointplot(x,y,color=(0.5,0.5,0.5),ax=ax, markersize=0)
		
		ax.set_xticks(x)
		ax.tick_params(axis="x", labelrotation=90)
		ax.set_xticklabels(controlTypes)
	
	for ci,controlType in enumerate(controlTypes):
		print(ci, controlType)
	
	plt.tight_layout()
	plt.show()