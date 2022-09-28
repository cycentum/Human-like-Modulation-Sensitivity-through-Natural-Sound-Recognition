import itertools
import pickle


def makeBreakEpoch(breakValEpoch):
	categoryBatchSize=1
	valEpoch=(2**4)//categoryBatchSize
	breakEpoch=breakValEpoch*valEpoch
	return valEpoch,breakEpoch


def getLastEpoch(epochCorrect, breakValEpoch):
	valEpoch, breakEpoch=makeBreakEpoch(breakValEpoch)
	
	bestCorrect=0
	for epoch in itertools.count(0):
		if epoch%valEpoch==0:
			assert epoch in epochCorrect
			if epochCorrect[epoch]>bestCorrect:
				bestEpoch=epoch
				bestCorrect=epochCorrect[epoch]
			
			if epoch>=bestEpoch+breakEpoch:
				break
	lastEpoch=epoch
	return lastEpoch, bestCorrect, bestEpoch


def getBestEpoch(epochCorrect, breakValEpoch):
	lastEpoch, bestCorrect, bestEpoch=getLastEpoch(epochCorrect, breakValEpoch)
	return bestEpoch


def loadEpochCorrect(DIR_NET, archName, datasetName, controlType):
	dirResult=DIR_NET/datasetName/archName
	if controlType!="Original":
		dirResult=dirResult/("Result_"+controlType)
	
	fileEpochCorrect=dirResult/"EpochCorrect.pkl"
	with open(fileEpochCorrect, "rb") as f: epochCorrect=pickle.load(f)
	return epochCorrect


def loadBestEpoch(DIR_NET, archName, datasetName, breakValEpoch, controlType):
	epochCorrect=loadEpochCorrect(DIR_NET, archName, datasetName, controlType)
	bestEpoch=getBestEpoch(epochCorrect, breakValEpoch)
	return bestEpoch


def calcCorrectRatio(confusion):
	'''
	confusion: [true, ans]
	'''
	correct=(confusion.diagonal()/confusion.sum(axis=1)).mean()
	return correct
