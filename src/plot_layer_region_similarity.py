from matplotlib import pyplot as plt
import numpy as np
import pickle

from path import DIR_PHYSIOLOGY
from plot_tmtf import makeNetTypes, makeArchs
from utils_plot import DATASET_NAME


if __name__=="__main__":
	datasetTypes=("ESC", "TIMIT")

	REGIONS=("AN","CN","SOC","NLL", "IC", "MGB","AC")
			
	similarity={}
	for datasetType in datasetTypes:
		architectures=makeArchs(datasetType, 4)
		netTypes=makeNetTypes(datasetType, architectures, "Original")
		
		sim=[]
		for netType in netTypes:
				dirNetType=netType.makePath(DIR_PHYSIOLOGY)
				fileSim=dirNetType/"SimilarityMatrix.pkl"
				with open(fileSim, "rb") as f: s=pickle.load(f)
				sim.append(s)
		sim=np.stack(sim).mean(axis=0)
		similarity[datasetType]=sim
		
	for di,datasetType in enumerate(datasetTypes):
		ax=plt.subplot(1,2,di+1)
		ax.set_title(DATASET_NAME[datasetType])
		ax.set_xlabel("Layer")
		ax.set_ylabel("Brain region")
		
		sim=similarity[datasetType]
		mi=sim.min()
		ma=sim.max()
		print(datasetType)
		print("min", mi)
		print("max", ma)
		
		ax.axis("equal")
		heatmap=ax.pcolormesh(sim, cmap="cividis")
		plt.colorbar(heatmap, label="Similarity")
		
		ax.set_xticks(np.arange(sim.shape[1]) + 0.5)
		ax.set_yticks(np.arange(sim.shape[0]) + 0.5)
		
		ax.set_yticklabels(REGIONS)
		ax.set_xticklabels(np.arange(sim.shape[1])+1)

	plt.show()