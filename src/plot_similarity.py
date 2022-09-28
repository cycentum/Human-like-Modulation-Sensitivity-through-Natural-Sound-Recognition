from matplotlib import pyplot as plt
import numpy as np
from operator import itemgetter

from path import DIR_TIME_AVE, DIR_TEMPLATE_CORREL, DIR_HUMAN
from psychophysics_stim_param import makeStimParamAll
from utils_plot import defaultColors
from human import loadHuman, meanInterpHuman
from plot_tmtf import makeNetTypes, loadTmtf, makeArchs, compAveTmtf
from utils_plot import DATASET_NAME


def catTmtf(tmtf, stimParams):
	cat=[tmtf[stimParam.name] for stimParam in stimParams]
	cat=np.concatenate(cat, axis=-1)
	return cat


def compCorrel(catModel, catHumans):
	numLayer=catModel.shape[0]
	correl=np.empty(numLayer)
	for layer in range(numLayer):
		correl[layer]=np.corrcoef(catModel[layer], catHumans)[0,1]
	correl[np.isnan(correl)]=0
	return correl


def compRMSError(catModel, catHumans):
	error=((catModel-catHumans)**2).mean(axis=1)**0.5
	return error


def compNetDifference(catModel, catHumans):
	error=((catModel-catHumans)).mean(axis=1)
	return error


def compSimilarity(datasetType, architectures, controlType, dirResult, catHumans, stimParams):
	netTypes=makeNetTypes(datasetType, architectures, controlType)
	tmtf=dict([(netType,loadTmtf(netType, dirResult, stimParams)) for netType in netTypes])
	aveTmtf=compAveTmtf(tmtf, netTypes, stimParams)["ave"]
	catModel=catTmtf(aveTmtf, stimParams)
	
	correl=compCorrel(catModel, catHumans)
	rmsError=compRMSError(catModel, catHumans)
	netDifference=compNetDifference(catModel, catHumans)
	return correl,rmsError, netDifference


def expandLim(lim):
	dif=lim[1]-lim[0]
	lim0=lim[0]-dif*0.1
	lim1=lim[1]+dif*0.1
	return (lim0, lim1)


def makeAxes(title, xticks, lim, ticks0=None, ticks1=None, ticks2=None):
	'''
	lim: ((correl.min(), correl.max()), (rmsError.min(), rmsError.max()))
	'''
# 	fig,axes=plt.subplots(3, 1, num=title, figsize=(4,5))
	fig,axes=plt.subplots(2, 1, num=title, figsize=(4,5))
	axes[0].set_ylabel("Correlation coefficient")
	axes[0].set_xlabel("Layer")
	axes[0].set_title("Relative similarity")
	
	axes[1].set_ylabel("RMS error (dB)")
	axes[1].set_xlabel("Layer")
	axes[1].set_title("Absolute dissimilarity")

# 	axes[2].set_ylabel("Difference (dB)")
# 	axes[2].set_xlabel("Layer")
# 	axes[2].set_title("Net difference")

# 	axes[1].set_ylabel("Difference (dB)")
# 	axes[1].set_xlabel("Layer")
# 	axes[1].set_title("Net difference")
	for ax in axes:
		ax.set_xticks(xticks)
	
# 	axes[0].set_ylim(*expandLim(lim[0]))
# 	axes[1].set_ylim(*expandLim(lim[1]))

	if ticks0 is not None:
		axes[0].set_yticks(ticks0)
	if ticks1 is not None:
		axes[1].set_yticks(ticks1)
	if ticks2 is not None:
		axes[2].set_yticks(ticks1)
	
	return fig, axes
	
	
def showLegend(axes):
# 	for ax in axes:
# 		ax.legend()
	pass
	

def calcMinMax(similarity):
	correlMin=min([correl.min() for correl,rmsError,netDifference in similarity])
	correlMax=max([correl.max() for correl,rmsError,netDifference in similarity])
	rmsErrorMin=min([rmsError.min() for correl,rmsError,netDifference in similarity])
	rmsErrorMax=max([rmsError.max() for correl,rmsError,netDifference in similarity])
	netDifferenceMin=min([netDifference.min() for correl,rmsError,netDifference in similarity])
	netDifferenceMax=max([netDifference.max() for correl,rmsError,netDifference in similarity])
	return ((correlMin, correlMax), (rmsErrorMin, rmsErrorMax), (netDifferenceMin, netDifferenceMax))
	

def plotSimilarity(axes, similarity, fmt, **kwargs):
	correl,rmsError,netDifference=similarity
	numLayer=correl.shape[0]
	axes[0].plot(np.arange(numLayer)+1, correl, fmt, **kwargs)
	axes[1].plot(np.arange(numLayer)+1, rmsError, fmt, **kwargs)
# 	axes[2].plot(np.arange(numLayer)+1, netDifference, fmt, **kwargs)
# 	axes[1].plot(np.arange(numLayer)+1, netDifference, fmt, **kwargs)
# 	print("min", correl.min(), rmsError.min(), sep="\t")
# 	print("max", correl.max(), rmsError.max(), sep="\t")


if __name__=="__main__":
	stimParams=makeStimParamAll()
	
	humans=loadHuman(DIR_HUMAN)
	aveHuman=meanInterpHuman(humans, stimParams)
	catHuman=catTmtf(aveHuman, stimParams)
	
	controlTypes=("EnvSingleBand", "EnvMultiBand", "TFSSingleBand", "TFSMultiBand")
	markers=("1","2","3","4")
	
	similarity_original={}
	similarity_init={}
	similarity_control={}
	similarity_template_correl={}
	
	for datasetType in ("ESC", "TIMIT"):
		architectures=makeArchs(datasetType, 4)
		
		dirResult=DIR_TIME_AVE
		controlType="Original"
		similarity_original[datasetType]=compSimilarity(datasetType, architectures, controlType, dirResult, catHuman, stimParams)
	
		controlType="Init"
		similarity_init[datasetType]=compSimilarity(datasetType, architectures, controlType, dirResult, catHuman, stimParams)
		
		for ci,controlType in enumerate(controlTypes):
			similarity_control[datasetType, controlType]=compSimilarity(datasetType, architectures, controlType, dirResult, catHuman, stimParams)
	
		controlType="Original"
		dirResult=DIR_TEMPLATE_CORREL
		similarity_template_correl[datasetType]=compSimilarity(datasetType, architectures, controlType, dirResult, catHuman, stimParams)
	
	
	for datasetType in ("ESC", "TIMIT"):
		fig,axes=makeAxes(DATASET_NAME[datasetType], np.arange(0,13,2)+1, calcMinMax([*similarity_original.values(), *similarity_init.values()]))
		plotSimilarity(axes, similarity_original[datasetType], "o-", color=defaultColors(0), label="Optimized")
		plotSimilarity(axes, similarity_init[datasetType], "s--", color=defaultColors(1), label="Not optimized")
		showLegend(axes)
		plt.tight_layout()
		
		fig,axes=makeAxes(DATASET_NAME[datasetType]+", Control", np.arange(0,13,2)+1, calcMinMax([*similarity_original.values(), *similarity_init.values(), *similarity_control.values()]))
		if datasetType=="TIMIT":
			axes[1].set_yticks(np.arange(-6, 4, 2))
		plotSimilarity(axes, similarity_original[datasetType], "o-", color=(0.7,0.7,0.7), label="Original")
		plotSimilarity(axes, similarity_init[datasetType], "s--", color=(0.7,0.7,0.7), label="Not optimized")
		for ci,controlType in enumerate(controlTypes):
			plotSimilarity(axes, similarity_control[datasetType, controlType], markers[ci]+"-", color=defaultColors(ci), label=controlType, markersize=9)
			
	# 		if datasetType=="ESC" and controlType in ("EnvSingleBand", "EnvMultiBand", "TFSSingleBand", ):
	# 			threshold=0.1
	# 			netTypeAccuracy=loadAccuracy(datasetType, architectures, controlType)
	# 			targetArchitectures=[netType.arch for netType,acc in netTypeAccuracy.items() if acc>=threshold]
	# 			targetArchitectures=[arch for arch in architectures if arch in targetArchitectures]
	# 			assert len(targetArchitectures)==len(architectures)-1
	# 			
	# 			similarity_control=compSimilarity(datasetType, targetArchitectures, controlType, dirResult, catHuman, stimParams)
	# 			plotSimilarity(axes, similarity_control, ".:", color=defaultColors(ci), label=controlType+"(excluding low accuracy model)")
			
		showLegend(axes)
		plt.tight_layout()
		
		fig,axes=makeAxes(DATASET_NAME[datasetType]+", Template correl", np.arange(0,13,2)+1, calcMinMax([*similarity_original.values(), *similarity_init.values(), *similarity_template_correl.values()]), (-0.2, 0.0, 0.2, 0.4, 0.6, 0.8))
		plotSimilarity(axes, similarity_original[datasetType], "o-", color=(0.7,0.7,0.7) if datasetType=="ESC" else defaultColors(0), label="Time ave, Optimized")
		plotSimilarity(axes, similarity_init[datasetType], "s--", color=(0.7,0.7,0.7) if datasetType=="ESC" else defaultColors(1), label="Time ave, Not optimized")
		plotSimilarity(axes, similarity_template_correl[datasetType], "o-", color="k", fillstyle="none", label="Template correl, Optimized")
		showLegend(axes)
		plt.tight_layout()
		
	plt.show()
		
		