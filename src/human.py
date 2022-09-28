import pickle
import numpy as np


HUMAN_DIRS={
	"Viemeister1979_CfN_BwInf":"Viemeister1979Fig2",
	"Dau1997_Cf5000_Bw314":"Dau1997Fig5",
	"Dau1997_Cf5000_Bw31":"Dau1997Fig4",
	"Dau1997_Cf5000_Bw3":"Dau1997Fig3",
	"Lorenzi2001_CfN_BwInf":("Lorenzi2001BroadFig2",4,3),
	"Lorenzi2001_Cf5000_Bw2":("Lorenzi2001NarrowFig5",3,0),
	}

def loadHuman(DIR_HUMAN):
	human={}
	for stimParamName,d in HUMAN_DIRS.items():
		if isinstance(d, str):
			dirName=d
			size=0
		elif isinstance(d, tuple):
			dirName,size,index=d
		
		di=DIR_HUMAN/dirName
		if size==0:
			file=di/"Values.pkl"
			with open(file, "rb") as f: data=pickle.load(f)
			human[stimParamName]=data
		else:
			data=[]
			for fi in range(size):
				file=di/("Values_"+str(fi)+".pkl")
				with open(file, "rb") as f: da=pickle.load(f)
				data.append(da[index])
			human[stimParamName]=data
	return human


def meanInterpHuman(humans, stimParams):
	meanInterp={}
	for stimParam in stimParams:
		human=humans[stimParam.name]
		freqMin=min([h[:,0].min() for h in human])
		freqMax=max([h[:,0].max() for h in human])
		assert freqMin==stimParam.freqs.min()
		assert freqMax==stimParam.freqs.max()
		
		newHumans=np.empty((len(human), len(stimParam.freqs)))
		for hi,h in enumerate(human):
			newHumans[hi]=np.interp(np.log10(stimParam.freqs), np.log10(h[:,0]), h[:,1])
		newHumans=newHumans.mean(axis=0)
		meanInterp[stimParam.name]=newHumans
	return meanInterp
