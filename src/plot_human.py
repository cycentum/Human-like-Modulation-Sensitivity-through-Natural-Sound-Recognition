from matplotlib import pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import numpy as np
	
from path import DIR_HUMAN
from utils_plot import defaultColors, STIM_PARAM_MARKER, getStimParamName
from human import loadHuman
from psychophysics_stim_param import makeStimParamNameAll

if __name__=="__main__":
	humans=loadHuman(DIR_HUMAN)
	
	stimParamNames=makeStimParamNameAll()
	
	fig, axs=plt.subplots(1, len(stimParamNames), figsize=(8, 2), sharey="all")
	for spi,stimParamName in enumerate(stimParamNames):
		ax=axs[spi]
		ax.set_xlabel("AM rate (Hz)", fontsize=9, color=(0.3,0.3,0.3))
		ax.set_xscale("log")
		if spi==0:
			ax.set_ylabel("Threshold (dB)", fontsize=9, color=(0.3,0.3,0.3))
			ax.set_yticks((-40, -30, -20, -10, 0))
		else:
			ax.tick_params(axis="y", length=0)
		ax.set_ylim(-40,0)
		ax.set_xticks((1,10,100,1000))
		ax.tick_params(labelsize=8, colors=(0.3,0.3,0.3))
		for s in ("left","top","right","bottom"): ax.spines[s].set_color((1.0,1.0,1.0))
		ax.set_title(getStimParamName(stimParamName))
		
		human=humans[stimParamName]
		for hi,h in enumerate(human):
			if hi==0: label=stimParamName
			else: label=None
# 			ax.plot(h[:,0],h[:,1],STIM_PARAM_MARKER[spi]+"-",color=defaultColors(spi),label=getStimParamName(label))
			ax.plot(h[:,0],h[:,1],".-k")
# 	ax.legend()
	
	plt.subplots_adjust(left=0.1, right=0.98, top=0.8, bottom=0.3, hspace=0.3, wspace=0.1)
	
	posX=[(ax._position.x0, ax._position.x1) for ax in fig.axes]
	posY=(fig.axes[0]._position.y0, fig.axes[0]._position.y1)
	c=(0.5,0.5,0.5)
	rect=plt.Rectangle((posX[0][0],posY[0]),posX[-1][1]-posX[0][0],posY[1]-posY[0], fill=False, color=c, zorder=1000, transform=fig.transFigure, figure=fig)
	fig.patches.append(rect)
	for axi,ax in enumerate(axs[:-1]):
		x=(posX[axi][1]+posX[axi+1][0])/2
		path=Path(np.array([(x,posY[0]),(x,posY[1])]))
		fig.patches.append(PathPatch(path, color=c, zorder=1000, transform=fig.transFigure, figure=fig))
	
	plt.show()
