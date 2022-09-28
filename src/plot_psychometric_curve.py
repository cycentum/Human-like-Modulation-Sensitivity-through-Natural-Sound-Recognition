from matplotlib import pyplot as plt
import pickle
import numpy as np
import itertools
from operator import itemgetter

from path import DIR_TIME_AVE, DIR_TRAINING, DIR_NET
from psychophysics_utils import makeFreqStr, loadNet, log_expit, AsymSigmoid, fitSigmoid, NetType
from psychophysics_stim_param import makeStimParam
from utils import checkRandState
from net import initCupy, compRepre
from params import WAVE_RMS_VAL, DATASET_WAVE_FS
from training_utils import loadBestEpoch
from run_psychophysics_time_ave import loadCorrectRatioLayer
from utils_plot import DATASET_NAME
from plot_tmtf import loadSigmoidParams



if __name__=="__main__":
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
	
	correctRatio=loadCorrectRatioLayer(stimParam, netType, DIR_TIME_AVE) #shape=(freq, depth, layer)
	
	sigmoidParams=loadSigmoidParams(DIR_TIME_AVE, netType, stimParam)
	
	chance=1/(1+stimParam.numDepth0)
	sigmoid=AsymSigmoid(chance)
	
	
	layer=9
	
	threshold=0.707
	
	mi=correctRatio[:,:,layer].min()
	
	plt.figure(DATASET_NAME[datasetType]+", "+architectureName+", "+CONTROL_TYPE+", "+stimParamName+", layer="+str(layer+1))
	for fi,freq in enumerate(stimParam.freqs):
		ax=plt.subplot(2,4,fi+1)
		ax.set_title("{:.3f} Hz".format(freq))
		
		ax.plot(stimParam.depthsDb, correctRatio[fi,:,layer], ".k")
		
		sigmoid.setParams(sigmoidParams[layer,fi])
		x=np.linspace(stimParam.depthsDb.min(), stimParam.depthsDb.max(), 100)
		y=sigmoid(x)
		ax.plot(x, y, "k")
		
		depthAtThreshold=sigmoid.inv(threshold)
		ax.plot((stimParam.depthsDb.min(), stimParam.depthsDb.max()), (threshold, threshold), ":", color=(0.5,0.5,0.5))
		ax.plot((depthAtThreshold, depthAtThreshold), (mi-0.1, 1.1), ":", color=(0.5,0.5,0.5))
		
		ax.set_ylim(mi-0.1, 1.1)
		ax.set_xlabel("AM depth (dB)")
		ax.set_ylabel("Proportion correct")
	
	plt.show()