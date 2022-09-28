from matplotlib import pyplot as plt
import matplotlib
import pickle
import numpy as np
import itertools
from operator import itemgetter
from collections import defaultdict
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from path import DIR_TIME_AVE, DIR_TEMPLATE_CORREL, DIR_NET, DIR_HUMAN
from psychophysics_utils import AsymSigmoid, NetType
from training_utils import loadBestEpoch
from psychophysics_stim_param import makeStimParamAll
from utils_plot import defaultColors
from plot_recog_accuracy import loadArchAccuracy
from utils_plot import STIM_PARAM_MARKER, DATASET_NAME, getStimParamName
from human import loadHuman


def makeArchs(datasetType, numArch, returnAccuracy=False):
	numLayer=13
	numUnits=(32, 64, 128, 256, 512)
	numSample=4
	
	archAccuracy=loadArchAccuracy((datasetType,), (numLayer,), numUnits, numSample, 96)
	archAccuracy=list(archAccuracy.values())
	archAccuracy=sorted(archAccuracy, key=itemgetter(0))
	if not returnAccuracy:
		archAccuracy=list(map(itemgetter(1), archAccuracy))
		
	if numArch>0:
		return archAccuracy[-1:-numArch-1:-1]
	return archAccuracy


def makeNetTypes(datasetType, architectures, CONTROL_TYPE):
	breakValEpoch=96
	
	if CONTROL_TYPE=="Init":
		netTypes=[NetType(datasetType, arch, "Original", 0) for arch in architectures]
	else:
		netTypes=[NetType(datasetType, arch, CONTROL_TYPE, loadBestEpoch(DIR_NET, arch, datasetType, breakValEpoch, CONTROL_TYPE)) for arch in architectures]
		
	return netTypes


def loadSigmoidParams(dirResult, netType, stimParam):
	if dirResult==DIR_TIME_AVE: filename="LogisticLayerSigmoid.pkl"
	elif dirResult==DIR_TEMPLATE_CORREL: filename="TemplateCorrelLayerSigmoid.pkl"

	fileSigmoid=netType.makePath(dirResult)/stimParam.name/filename
	with open(fileSigmoid, "rb") as f: sigmoidParams=pickle.load(f) #shape=(layer, freq, param)
	
	return sigmoidParams


def loadTmtf(netType, dirResult, stimParams):
	threshold=0.707
	
	tmtf={}
	for stimParam in stimParams:
		sigmoidParams=loadSigmoidParams(dirResult, netType, stimParam)
		
		chance=1/(1+stimParam.numDepth0)
		sigmoid=AsymSigmoid(chance)
		
		tmtf[stimParam.name]=np.nan*np.empty((netType.numLayer, len(stimParam.freqs)))
		for layer,fi in itertools.product(range(netType.numLayer), range(len(stimParam.freqs))):
			sigmoid.setParams(sigmoidParams[layer,fi])
			value=sigmoid.inv(threshold)
			value=np.clip(value, stimParam.depthsDb.min(), stimParam.depthsDb.max())
			tmtf[stimParam.name][layer,fi]=value
		
	return tmtf


def makeLayerAxs(layers, humans, figTitle):
	fig,axs=plt.subplots(len(layers), len(stimParams), num=figTitle, sharex="col", sharey="all", figsize=(8, 13))
	
	for (spi,stimParam),(li,layer) in itertools.product(enumerate(stimParams), enumerate(layers)):
		ax=axs[li,spi]
		ax.set_xscale("log")
		ax.set_ylim(-40, 0)
		ax.set_yticks((-40, -30, -20, -10, 0))
		ax.set_xticks((1,10,100,1000))
		ax.tick_params(labelsize=8, colors=(0.3,0.3,0.3))
		for s in ("left","top","right","bottom"): ax.spines[s].set_color((1.0,1.0,1.0))
		
		if li==0:
			ax.set_title(getStimParamName(stimParam.name), fontsize=10.3)

# 		if spi==0:
# 			ax.set_title("Layer "+str(len(layers)-1-layer+1), x=-1.2, y=0.4)
			
		if spi!=0:
			ax.tick_params(axis="y", length=0)
			
		if li==len(layers)-1 and spi==0:
			ax.set_xlabel("AM rate (Hz)", fontsize=9, color=(0.3,0.3,0.3))
			
		if spi==0 and li==len(layers)-1:
			ax.set_ylabel("Threshold (dB)", fontsize=9, color=(0.3,0.3,0.3))
	
	plt.subplots_adjust(left=0.3, right=0.98, top=0.95, bottom=0.05, hspace=0.3, wspace=0.1)
		
	for spi,stimParam in enumerate(stimParams):
		human=humans[stimParam.name]
		for li,layer in enumerate(layers):
			ax=axs[li, spi]
			for hi,h in enumerate(human):
				ax.plot(h[:,0],h[:,1],":", color=(0.,0.,0.), linewidth=1)
# 				ax.plot(h[:,0],h[:,1],".-", color=(0.,0.,0.), linewidth=1, markersize=2)
	
	for li,layer in enumerate(layers):
		posX=[(ax._position.x0, ax._position.x1) for ax in axs[li]]
		posY=(axs[li][0]._position.y0, axs[li][0]._position.y1)
		c=(0.5,0.5,0.5)
		rect=plt.Rectangle((posX[0][0],posY[0]),posX[-1][1]-posX[0][0],posY[1]-posY[0], fill=False, color=c, zorder=1000, transform=fig.transFigure, figure=fig)
		fig.patches.append(rect)
		for axi,ax in enumerate(axs[li][:-1]):
			x=(posX[axi][1]+posX[axi+1][0])/2
			path=Path(np.array([(x,posY[0]),(x,posY[1])]))
			fig.patches.append(PathPatch(path, color=c, zorder=1000, transform=fig.transFigure, figure=fig))
		
		y=(posY[0]+posY[1])/2-0.01
		x=0.15
		fig.text(x,y,"Layer "+str(len(layers)-1-layer+1),fontsize=10.3, color="k")
		
	
	
	return axs
			
			
def plotLayerTmtf(axs, layers, tmtf, netTypes, stimParams, fmt, **kwargs):
	for (li,layer),(spi,stimParam) in itertools.product(enumerate(layers), enumerate(stimParams)):
# 		ax=axs[spi, li]
		ax=axs[len(layers)-1-li, spi]
		for netType in netTypes:
			ax.plot(stimParam.freqs, tmtf[netType][stimParam.name][layer], fmt, **kwargs)


def compAveTmtf(tmtf, netTypes, stimParams):
	ave=defaultdict(list)
	for netType,stimParam in itertools.product(netTypes, stimParams):
		ave[stimParam.name].append(tmtf[netType][stimParam.name])
	
	for stimParam in stimParams:
		ave[stimParam.name]=np.stack(ave[stimParam.name], axis=0).mean(axis=0)
	
	ave={"ave":ave}
	return ave


if __name__=="__main__":
# 	datasetType="ESC"
	datasetType="TIMIT"
	
	humans=loadHuman(DIR_HUMAN)
	
	stimParams=makeStimParamAll()
	
	architectures=makeArchs(datasetType, 4)
	
	dirResult=DIR_TIME_AVE
	
	numLayer=13
	layers=np.arange(numLayer)
	axs=makeLayerAxs(layers, humans, DATASET_NAME[datasetType]+", Optimized")
	
	controlType="Init"
	netTypes=makeNetTypes(datasetType, architectures, controlType)
	tmtf=dict([(netType,loadTmtf(netType, dirResult, stimParams)) for netType in netTypes])
# 	plotLayerTmtf(axs, layers, tmtf, netTypes, stimParams, "s--", color=defaultColors(1, 0.5), markersize=3)
	tmtf=compAveTmtf(tmtf, netTypes, stimParams)
	plotLayerTmtf(axs, layers, tmtf, ["ave"], stimParams, "s--", color=defaultColors(1), markersize=3, linewidth=1)
	
	controlType="Original"
	netTypes=makeNetTypes(datasetType, architectures, controlType)
	tmtf=dict([(netType,loadTmtf(netType, dirResult, stimParams)) for netType in netTypes])
# 	plotLayerTmtf(axs, layers, tmtf, netTypes, stimParams, "o-", color=defaultColors(0, 0.5), markersize=3)
	tmtf=compAveTmtf(tmtf, netTypes, stimParams)
	plotLayerTmtf(axs, layers, tmtf, ["ave"], stimParams, "o-", color=defaultColors(0), markersize=3, linewidth=1)
	
	## template correl
	if datasetType=="TIMIT":
		dirResult=DIR_TEMPLATE_CORREL
		netTypes=makeNetTypes(datasetType, architectures, controlType)
		tmtf=dict([(netType,loadTmtf(netType, dirResult, stimParams)) for netType in netTypes])
	# 	plotLayerTmtf(axs, layers, tmtf, netTypes, stimParams, "o-", color="k", markersize=3)
		tmtf=compAveTmtf(tmtf, netTypes, stimParams)
		plotLayerTmtf(axs, layers, tmtf, ["ave"], stimParams, "o-", color="k", markersize=3, fillstyle="none", linewidth=1)
		dirResult=DIR_TIME_AVE

	controlType="Original"
	archAccuracy=makeArchs(datasetType, 0, True)
	accuracy=np.array(list(map(itemgetter(0), archAccuracy)))
	architectures=list(map(itemgetter(1), archAccuracy))
	netTypes=makeNetTypes(datasetType, architectures, controlType)
	tmtf=dict([(netType,loadTmtf(netType, dirResult, stimParams)) for netType in netTypes])
	
	minAccuracy=accuracy.min()
	maxAccuracy=accuracy.max()
	cmap=plt.get_cmap("cividis")
	cmap=matplotlib.cm.ScalarMappable(matplotlib.colors.Normalize(minAccuracy, maxAccuracy), cmap)

	axs=makeLayerAxs(layers, humans, DATASET_NAME[datasetType]+", All models")
	
	for acc,netType in zip(accuracy, netTypes):
		plotLayerTmtf(axs, layers, {netType:tmtf[netType]}, [netType], stimParams, "-", color=cmap.to_rgba(acc), linewidth=1)
	
	
	architectures=makeArchs(datasetType, 4)
	
	markers=("1","2","3","4")
	axs=makeLayerAxs(layers, humans, DATASET_NAME[datasetType]+", Control")
	for i,controlType in enumerate(("EnvSingleBand", "EnvMultiBand", "TFSSingleBand", "TFSMultiBand")):
		netTypes=makeNetTypes(datasetType, architectures, controlType)
		tmtf=dict([(netType,loadTmtf(netType, dirResult, stimParams)) for netType in netTypes])
# 		plotLayerTmtf(axs, layers, tmtf, netTypes, stimParams, ".-", color=defaultColors(i), markersize=3)
		tmtf=compAveTmtf(tmtf, netTypes, stimParams)
		plotLayerTmtf(axs, layers, tmtf, ["ave"], stimParams, markers[i]+"-", color=defaultColors(i), markersize=6, linewidth=1)
	

	if datasetType=="ESC":
		dirResult=DIR_TEMPLATE_CORREL
		
		axs=makeLayerAxs(layers, humans, DATASET_NAME[datasetType]+", Template correlation")
		
		controlType="Original"
		netTypes=makeNetTypes(datasetType, architectures, controlType)
		tmtf=dict([(netType,loadTmtf(netType, dirResult, stimParams)) for netType in netTypes])
	# 	plotLayerTmtf(axs, layers, tmtf, netTypes, stimParams, "o-", color="k", markersize=3)
		tmtf=compAveTmtf(tmtf, netTypes, stimParams)
		plotLayerTmtf(axs, layers, tmtf, ["ave"], stimParams, "o-", color="k", markersize=3, fillstyle="none", linewidth=1)

	plt.show()
