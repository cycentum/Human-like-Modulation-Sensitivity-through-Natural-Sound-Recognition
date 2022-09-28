import itertools

from architecture import sampleArchitecture, architectureStr, readArchitecture
from params import WAVE_FS_ESC, WAVE_FS_TIMIT, DATASET_WAVE_FS
from path import DIR_TRAINING

def saveArchitectures():
	numLayers=(7, 9, 11, 13)
	numUnits=(32, 64, 128, 256, 512)

	filterLenUpper=8
	
	numSample=4
	
	totalInputSecUpper=0.2
	
	for datasetName in ("ESC", "TIMIT"):
		totalInputLenUpper=int(totalInputSecUpper*DATASET_WAVE_FS[datasetName])
			
		for numLayer, numChannel, sample in itertools.product(numLayers, numUnits, range(numSample)):
			dirNet=DIR_NET/datasetName/("Layer"+str(numLayer)+"_Unit"+str(numChannel)+"_Sample"+str(sample))
			fileArch=dirNet/"Architecture.txt"
			if fileArch.is_file():
				print(dirNet.name, "Already exists.")
				continue
			
			arch=sampleArchitecture(numLayer, totalInputLenUpper, numChannel, filterLenUpper)
			dirNet.mkdir(exist_ok=True, parents=True)
			print("Saving", arch)
			with open(fileArch, "w") as f:
				print(architectureStr(arch), file=f)


if __name__=="__main__":
	DIR_NET=DIR_TRAINING/"Net"
	
	import numpy as np
	from numpy import uint32
	np.random.seed(202105281138%np.iinfo(uint32).max) #ESC & TIMIT, numSample=4
	
	saveArchitectures()
	