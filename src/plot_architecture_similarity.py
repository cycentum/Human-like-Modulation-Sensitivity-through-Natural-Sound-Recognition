from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from operator import itemgetter
import itertools
from scipy.stats import pearsonr
import itertools

from path import DIR_TIME_AVE, DIR_HUMAN
from psychophysics_stim_param import makeStimParamAll
from human import loadHuman, meanInterpHuman
from plot_recog_accuracy import loadArchAccuracy
from plot_similarity import catTmtf, compSimilarity, makeAxes, showLegend, calcMinMax, expandLim
from utils_plot import DATASET_NAME


if __name__=="__main__":
	stimParams=makeStimParamAll()
	
	humans=loadHuman(DIR_HUMAN)
	aveHuman=meanInterpHuman(humans, stimParams)
	catHuman=catTmtf(aveHuman, stimParams)
	
	numLayer=13
	numUnits=(32, 64, 128, 256, 512)
	numSample=4
	
	archAccuracy={}
	minAccuracy={}
	maxAccuracy={}
	similarity_archs={}
	for datasetType in ("ESC", "TIMIT"):
		archAccuracy[datasetType]=loadArchAccuracy((datasetType,), (numLayer,), numUnits, numSample, 96)
		archAccuracy[datasetType]=list(archAccuracy[datasetType].values())
		archAccuracy[datasetType]=sorted(archAccuracy[datasetType], key=itemgetter(0))
	
		minAccuracy[datasetType]=archAccuracy[datasetType][0][0]
		maxAccuracy[datasetType]=archAccuracy[datasetType][-1][0]
	
		similarity_archs[datasetType]=[compSimilarity(datasetType, [arch, ], "Original", DIR_TIME_AVE, catHuman, stimParams) for accuracy,arch in archAccuracy[datasetType]]
	
	similarityMinMax=calcMinMax(list(itertools.chain.from_iterable(similarity_archs.values())))
	for datasetType in ("ESC", "TIMIT"):
		numLayer=similarity_archs[datasetType][0][0].shape[0]
		fig,axes=makeAxes(DATASET_NAME[datasetType]+", all architectures", np.arange(0,numLayer,2)+1, similarityMinMax, np.arange(-0.2, 1.1, 0.2), np.arange(4, 16+1, 2))
# 		fig,axes=makeAxes(DATASET_NAME[datasetType]+", all architectures", np.arange(0,numLayer,2)+1, similarityMinMax)
		cmap=plt.get_cmap("cividis")
		cmap=matplotlib.cm.ScalarMappable(matplotlib.colors.Normalize(minAccuracy[datasetType], maxAccuracy[datasetType]), cmap)
		for axi,ax in enumerate(axes):
			for (accuracy,arch), sim in zip(archAccuracy[datasetType], similarity_archs[datasetType]):
				ax.plot(np.arange(numLayer)+1, sim[axi], "-", color=cmap.to_rgba(accuracy), linewidth=1)
		
# 			plt.colorbar(cmap, label="Recognition accuracy", ax=ax)
		plt.tight_layout()
		
		print(datasetType)
		print("min", min(similarity_archs[datasetType], key=lambda x:x[0].min())[0].min(), min(similarity_archs[datasetType], key=lambda x:x[1].min())[1].min(), sep="\t")
		print("max", max(similarity_archs[datasetType], key=lambda x:x[0].max())[0].max(), max(similarity_archs[datasetType], key=lambda x:x[1].max())[1].max(), sep="\t")
	
	correl={}
	p={}
	for datasetType in ("ESC", "TIMIT"):
		accuracy=list(map(itemgetter(0), archAccuracy[datasetType]))
		
		similarity_archs_array=np.array(similarity_archs[datasetType])
		correl[datasetType]=np.empty((numLayer,2))
		p[datasetType]=np.empty((numLayer,2))
		for layer,(i,name) in itertools.product(range(numLayer), enumerate(("Correlation coefficient", "RMS Error (dB)"))):
			correl[datasetType][layer,i],p[datasetType][layer,i]=pearsonr(similarity_archs_array[:,i,layer], accuracy)
			print(layer, name, p[datasetType][layer,i]*numLayer, sep="\t")
	
	correlMin=min(correl[datasetType].min() for datasetType in ("ESC", "TIMIT"))
	correlMax=max(correl[datasetType].max() for datasetType in ("ESC", "TIMIT"))
	correlMin,correlMax=expandLim((correlMin, correlMax))
	for datasetType in ("ESC", "TIMIT"):	
		fig,axes=makeAxes(DATASET_NAME[datasetType]+", Correlation between TMTF (dis)similarity and recognition accuracy", np.arange(0,numLayer,2)+1, ((correlMin,correlMax),(correlMin,correlMax)))
		for ax in axes:
			ax.set_ylabel("Correlation coefficient")
# 			ax.set_ylim(correlMin, correlMax)
			ax.set_ylim(-1.05, 1.05)
		
		threshold=0.01
		for axi,ax in enumerate(axes):
			ax.plot(np.arange(numLayer)+1, correl[datasetType][:,axi], ".-k")
			
			significant=p[datasetType][:,axi]<threshold/numLayer
			ax.plot((np.arange(numLayer)+1)[significant], np.zeros(numLayer)[significant], "xk", label="p < 0.01 (Bonferroni)")
		showLegend(axes)
		plt.tight_layout()
		
	plt.show()
