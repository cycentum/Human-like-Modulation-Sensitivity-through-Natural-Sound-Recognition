import sys
import itertools
import pickle
import numpy as np
from dotmap import DotMap
from numpy import newaxis, float32, float64, int32, int64, int16, int8, uint8, uint32, complex64, complex128, uint16
from collections import defaultdict, deque
from operator import itemgetter, attrgetter
from chainer import links, functions, Variable, optimizers, serializers
import chainer
import scipy.signal

try:
	import cupy
except:
	print("Could not import cupy", sys.stderr)
	cupy=None

from net import Net, totalInputLength, initCupy
from architecture import sampleArchitecture, architectureStr, readArchitecture
from utils import checkRandState
from params import WAVE_FS_ESC, WAVE_FS_TIMIT, DATASET_WAVE_FS, WAVE_RMS_TRA, WAVE_RMS_VAL
from path import DIR_TRAINING
from training_utils import getLastEpoch, calcCorrectRatio


def makeXp():
	if gpu_id>=0: return cupy
	return np
	
	
def asnp(x):
	if isinstance(x, np.ndarray): return x
	return cupy.asnumpy(x)


def makeSplitIndx(totalSize, blockSizeUpper):
	numBlock=int(np.ceil(totalSize/blockSizeUpper))
	index=np.array_split(np.arange(totalSize), numBlock)
	return index


def fade(wave, win):
	fadeLen=len(win)//2
	if len(wave)>fadeLen:
		wave[:fadeLen//2]*=win[:fadeLen//2]
		wave[-fadeLen//2:]*=win[-fadeLen//2:]
	else:
		wave[:len(wave)//2]*=win[:len(wave)//2]
		wave[len(wave)//2:]*=win[-len(wave[len(wave)//2:]):]
	return wave


class DataEsc:
	def __init__(self, data, categories):
		self.data=data
		self.categories=categories

		self.names=sorted(self.data.keys())
		
		for name in self.names:
			d=self.data[name]
			if d.fold in (0,1,2): d.foldIndex=0
			elif d.fold==3: d.foldIndex=1
			elif d.fold==4: d.foldIndex=2
		
		self.categoryData=defaultdict(list)
		for name in self.names:
			d=self.data[name]
			self.categoryData[d.fold, d.category].append(d)
			
		categoryIndex=dict([(category,ci) for ci,category in enumerate(self.categories)])
		for name in self.names:
			d=self.data[name]
			d.timeCategory=[((0,len(d.wave)), categoryIndex[d.category])]
			
		self.waveFs=WAVE_FS_ESC

	
	def __getitem__(self, key):
		return self.data[key]


	def numCategory(self):
		return len(self.categories)
	

	@staticmethod
	def load(dirData):
		if CONTROL_TYPE=="Original":
			filename="ESC.pkl"
		else:
			filename="ESC_"+CONTROL_TYPE+".pkl"
		
		file=dirData/filename
		print("Loading", file)
		with open(file, "rb") as f: data,categories=pickle.load(f)
		
		return DataEsc(data, categories)
		
	
class DataTimit:
	def __init__(self, data, categories, resampleToEsc):
		self.data=data
		self.categories=categories
		self.resampleToEsc=resampleToEsc
		
		self.names=sorted(self.data.keys())	
		
		for name in self.names:
			d=self.data[name]
			if d.group=="TRAIN": d.foldIndex=0
			elif d.group=="TEST_NONCORE": d.foldIndex=1
			elif d.group=="TEST_CORE": d.foldIndex=2
			else: raise Exception()
		
		categoryIndex=dict([(category,ci) for ci,category in enumerate(self.categories)])
		for name in self.names:
			d=self.data[name]
			d.timeCategory=[]
			for t,c in d.category:
				d.timeCategory.append((t,categoryIndex[c]))
		
		if resampleToEsc:
			self.waveFs=WAVE_FS_ESC
			for name in self.names:
				d=self.data[name]
				d.wave=scipy.signal.resample_poly(d.wave, WAVE_FS_ESC, WAVE_FS_TIMIT)
				for i,((t0,t1),c) in enumerate(d.timeCategory):
					t0=t0*WAVE_FS_ESC//WAVE_FS_TIMIT
					t1=t1*WAVE_FS_ESC//WAVE_FS_TIMIT
					d.timeCategory[i]=((t0,t1),c)
		else:
			self.waveFs=WAVE_FS_TIMIT
			
	
	def __getitem__(self, key):
		return self.data[key]

	
	def numCategory(self):
		return len(self.categories)
	
	
	@staticmethod
	def load(dirData, resampleToEsc):
		if CONTROL_TYPE=="Original":
			filename="TIMIT.pkl"
		else:
			filename="TIMIT_"+CONTROL_TYPE+".pkl"
			
		file=dirData/filename
		print("Loading", file)
		with open(file, "rb") as f: data,categories=pickle.load(f)
		
		return DataTimit(data, categories, resampleToEsc)


class DataSet:
	def __init__(self, datasets):
		self.datasets=datasets
		if len(datasets)==2:
			self.datasetNames=("ESC", "TIMIT")
		else:
			self.datasetNames=tuple(datasets.keys())
			
		self.datasetIndex=dict([(n,ni) for ni,n in enumerate(self.datasetNames)])
		
		highFreq=20
		for datasetName in self.datasetNames:
			dataset=self[datasetName]
			
			high=scipy.signal.butter(1, highFreq/(dataset.waveFs/2), "high")
			
			fadeLen=int(FADE_SEC*dataset.waveFs)
			fadeWin=np.hanning(fadeLen*2)
			
			for name in dataset.names:
				d=dataset.data[name]
				
				if d.wave.dtype==int16:
					d.wave=(d.wave/np.iinfo(int16).max).astype(float64)
				d.wave=fade(d.wave, fadeWin)
				d.wave=scipy.signal.filtfilt(high[0], high[1], d.wave)
				d.wave=d.wave.astype(float32)
				
				
	def makeCategoryInterval(self, foldIndex):
		for datasetName in self.datasetNames:
			dataset=self[datasetName]
			dataset.categoryInterval=defaultdict(list)
			for name in dataset.names:
				d=dataset.data[name]
				if d.foldIndex in foldIndex:
					for t,c in d.timeCategory:
						dataset.categoryInterval[d.foldIndex, c].append((name, t))
	
	
	def __getitem__(self, key):
		return self.datasets[key]
	
	
	def numCategory(self, datasetName):
		return len(self.datasets[datasetName].categories)
	
	
	def makeBatchValData(self, foldIndex, batchSize, inputMargin, rms):
		foldData=[]
		for dsi,datasetName in enumerate(self.datasetNames):
			dataset=self[datasetName]
			for di,name in enumerate(dataset.names):
				d=dataset[name]
				if d.foldIndex==foldIndex:
					foldData.append((datasetName, name, (len(d.wave), dsi, di)))
		foldData=sorted(foldData, key=itemgetter(2), reverse=True) #sort by length; if equal length, sort by index
		
		index=makeSplitIndx(len(foldData), batchSize)
		
		batchData=[]
		for ind in index:
			waves=[]
			for i in ind:
				datasetName,dataName,_=foldData[i]
				d=self[datasetName][dataName]
				
				wave=d.wave.copy()
				wave=makeInput(wave, addMargin((0,len(wave)), inputMargin), rms)
				waves.append(wave)
			
			length=max([len(wave) for wave in waves])
			x=np.zeros((len(ind), length), float32)
			for wi,wave in enumerate(waves): x[wi,:len(wave)]=wave
			x=x[:,newaxis,:,newaxis]
			
			batchData.append(DotMap())
			batchData[-1].x=x
			
			batchData[-1].names=[]
			batchData[-1].datasetNames=set()
			for i in ind:
				datasetName,dataName,_=foldData[i]
				batchData[-1].names.append((datasetName,dataName))
				batchData[-1].datasetNames.add(datasetName)
				
		return batchData
	
	
	@staticmethod
	def load(dirData, datasetType):
		resampleToEsc=datasetType=="Both"
		datasets=DotMap()
		if datasetType=="Both" or datasetType=="ESC":
			dataEsc=DataEsc.load(dirData)
			datasets.ESC=dataEsc
		if datasetType=="Both" or datasetType=="TIMIT":
			dataTimit=DataTimit.load(dirData, resampleToEsc)
			datasets.TIMIT=dataTimit
		return DataSet(datasets)



def makeInputMargin(inputLen):
	inputMargin1=(inputLen-1)//2
	inputMargin0=inputLen-1-inputMargin1
	return inputMargin0, inputMargin1


def addMargin(t01, inputMargin):
	t0=t01[0]-inputMargin[0]
	t1=t01[1]+inputMargin[1]
	return t0,t1


def sampleRms(size):
	rms=np.random.rand(size)*(WAVE_RMS_TRA[1]-WAVE_RMS_TRA[0])+WAVE_RMS_TRA[0]
	return rms


def compRms(wave):
	rms=(wave**2).mean()**0.5
	return rms


def scaleRms(wave, targetRms):
	rms0=compRms(wave)
	waveNew=wave*targetRms/rms0
	return waveNew

def makeInput(wave, time01, rms):
	wave=scaleRms(wave, rms)
	
	t0=max(time01[0], 0)
	t1=min(time01[1], len(wave))
	x=wave[t0:t1]
	if len(wave)-time01[1]<0: x=np.concatenate((x, np.zeros(-(len(wave)-time01[1]), x.dtype)))
	if time01[0]<0: x=np.concatenate((np.zeros(-time01[0], x.dtype), x))
	
	assert x.dtype==float32
	
	return x


def evaluateSingle(datasets, net, batchValData):
	xp=makeXp()
	
	confusion=dict([(datasetName, np.zeros((datasets.numCategory(datasetName), datasets.numCategory(datasetName)), int32)) for datasetName in datasets.datasetNames])
	with chainer.using_config('train', False), chainer.using_config("enable_backprop", False):
		for bi,batchData in enumerate(batchValData):
# 			print("val", bi, "/", len(batchValData))

			x=batchData.x
			x=xp.asarray(x)
			x=Variable(x)
			y=net(x, batchData.datasetNames)
			
			for di,(datasetName,dataName) in enumerate(batchData.names):
				yi=y[datasetName].data[di,:,:,0]
				for (t0,t1),c in datasets[datasetName][dataName].timeCategory:
					yit=yi[:, t0:t1].mean(axis=1)
					ans=int(asnp(yit.argmax()))
					confusion[datasetName][c,ans]+=1
	return confusion


def trainSingle(datasets, traFoldIndex, inputMargin, net, opt, traCategoryIndex, traBatchSize, categoryBatchSize):
	xp=makeXp()
	net.cleargrads()
	meanEr={}
	for datasetName in datasets.datasetNames:
		dataset=datasets[datasetName]
		waves=[]
		for ci in range(dataset.numCategory()):
			while len(traCategoryIndex[datasetName, ci])<categoryBatchSize:
				traCategoryIndex[datasetName, ci].extend(np.random.permutation(len(dataset.categoryInterval[traFoldIndex, ci])))
			for i in range(categoryBatchSize):
				index=traCategoryIndex[datasetName, ci].popleft()
				name,(t0,t1)=dataset.categoryInterval[traFoldIndex, ci][index]
				d=dataset[name]
				
				t=np.random.randint(t0,t1)
				
				wave=d.wave.copy()
				rms=float(sampleRms(1))
				wave=makeInput(wave, addMargin((t,t+1), inputMargin), rms)
				
				waves.append(wave)
				
		waves=np.stack(waves, axis=0) #shape=(batch, length)
		
		trues=np.repeat(np.arange(dataset.numCategory()), categoryBatchSize).astype(int32)
		
		meanEr[datasetName]=0
		batchIndex=makeSplitIndx(len(waves), traBatchSize)
		for bi,index in enumerate(batchIndex):
# 			print("tra", datasetName, bi, "/", len(batchIndex))
			
			batchWaves=waves[index]
	
			x=xp.asarray(batchWaves[:,newaxis,:,newaxis])
			x=Variable(x)
			y=net(x, datasetName)
				
			batchTrues=trues[index][..., newaxis, newaxis]
			batchTrues=xp.asarray(batchTrues)
			
			e=functions.softmax_cross_entropy(y[datasetName], batchTrues)
			e*=len(batchTrues)
			e.backward()
			e.unchain_backward()
			for yi in y.values(): yi.unchain_backward()
			
			meanEr[datasetName]+=float(e.data)
		
		meanEr[datasetName]/=dataset.numCategory()*categoryBatchSize

	opt.update()
	
	return meanEr


class TrainingState:
	def __init__(self, net, opt, epoch, traCategoryIndex, randState, epochConfusion, epochCorrect, epochTraError, epochTime):
		self.net=net
		self.opt=opt
		self.epoch=epoch
		self.traCategoryIndex=traCategoryIndex
		self.randState=randState
		self.epochConfusion=epochConfusion
		self.epochCorrect=epochCorrect
		self.epochTraError=epochTraError
		self.epochTime=epochTime
	
	
	def getAll(self):
		return self.net, self.opt, self.epoch, self.traCategoryIndex, self.randState, self.epochConfusion, self.epochCorrect, self.epochTraError, self.epochTime


	def save(self, dirState):
		dirState.mkdir(exist_ok=True, parents=True)
		files=TrainingState._files(dirState)
		serializers.save_npz(files.net, self.net)
		serializers.save_npz(files.opt, self.opt)
		with open(files.state, "wb") as f:
			pickle.dump((self.epoch, self.traCategoryIndex, self.randState, self.epochConfusion, self.epochCorrect, self.epochTraError, self.epochTime), f)
	
	@staticmethod
	def load(dirState, net, opt):
		files=TrainingState._files(dirState)
		serializers.load_npz(files.net, net)
		serializers.load_npz(files.opt, opt)
		with open(files.state, "rb") as f:
			epoch, traCategoryIndex, randState, epochConfusion, epochCorrect, epochTraError, epochTime=pickle.load(f)
		return TrainingState(net, opt, epoch, traCategoryIndex, randState, epochConfusion, epochCorrect, epochTraError, epochTime)
	
	@staticmethod
	def _files(dirState):
		files=DotMap(dirState)
		files.net=dirState/"net.npz"
		files.opt=dirState/"opt.npz"
		files.state=dirState/"state.pkl"
		return files


def calcMeanCorrect(correct, datasetNames):
	meanCorrect=np.array([correct[datasetName] for datasetName in datasetNames]).mean()
	return meanCorrect


def train(datasetType, archName, breakValEpoch):
	dirArch=DIR_NET/datasetType/archName
	if CONTROL_TYPE=="Original":
		dirResult=dirArch
	else:
		dirResult=dirArch/("Result_"+CONTROL_TYPE)
		
	arch=readArchitecture(dirArch/"Architecture.txt")
	
	inputLen=totalInputLength(arch)
	inputMargin=makeInputMargin(inputLen)
	
	traFoldIndex=0
	valFoldIndex=1
	
	datasets=DataSet.load(DIR_DATA, datasetType)
	datasets.makeCategoryInterval((traFoldIndex,))
	
	fileRand=dirArch/"RandState.pkl"
	checkRandState(fileRand)
	
	net=Net([(datasetName, datasets.numCategory(datasetName)) for datasetName in datasets.datasetNames], arch)
	lr=1e-4
	print("lr", lr, flush=True)
	opt=optimizers.Adam(lr)
	opt.setup(net)
	
	dirState=dirResult/"TrainingState"
	fileCheck=dirState/"Check"
	if fileCheck.is_file():
		state=TrainingState.load(dirState, net, opt)
		net, opt, epoch0, traCategoryIndex, randState, epochConfusion, epochCorrect, epochTraError, epochTime=state.getAll()
		epoch0+=1
		np.random.set_state(randState)
		
		bestEpoch=0
		bestCorrect=0
		confEpochs=sorted(epochCorrect.keys())
		for e in confEpochs:
			meanCorrect=epochCorrect[e]
			if meanCorrect>bestCorrect:
				bestEpoch=e
				bestCorrect=meanCorrect
	else:
		traCategoryIndex={}
		for datasetName in datasets.datasetNames:
			for ci in range(datasets.numCategory(datasetName)):
				traCategoryIndex[datasetName, ci]=deque()
				
		epoch0=0
		epochConfusion={}
		epochCorrect={}
		epochTraError=[]
		epochTime=[]
		bestCorrect=0
		
	if gpu_id>=0: net.to_gpu(gpu_id)
		
	categoryBatchSize=1
	valEpoch=(2**4)//categoryBatchSize
	
	breakEpoch=breakValEpoch*valEpoch
	
	batchValData=datasets.makeBatchValData(valFoldIndex, valBatchSize, inputMargin, WAVE_RMS_VAL)
	
	dirTrainedNet=dirResult/"TrainedNet"
	dirTrainedNet.mkdir(exist_ok=True, parents=True)
	
	for epoch in itertools.count(epoch0):
		print("Ep", epoch)
		if epoch%valEpoch==0:
			confusion=evaluateSingle(datasets, net, batchValData)
			correct=dict([(datasetName, calcCorrectRatio(conf)) for (datasetName, conf) in confusion.items()])
			print("Ep", epoch, correct)
			
			meanCorrect=calcMeanCorrect(correct, datasets.datasetNames)
			if len(epochConfusion)==0 or meanCorrect>bestCorrect:
				bestEpoch=epoch
				bestCorrect=meanCorrect
				fileTrainedNet=dirTrainedNet/(str(epoch)+".npz")
				with open(fileTrainedNet, "wb") as f: serializers.save_npz(f, net)
			
			epochConfusion[epoch]=confusion
			epochCorrect[epoch]=meanCorrect
			
			if epoch>=bestEpoch+breakEpoch:
				with open(dirResult/"EpochConfusion.pkl", "wb") as f: pickle.dump(epochConfusion, f)
				with open(dirResult/"EpochCorrect.pkl", "wb") as f: pickle.dump(epochCorrect, f)
				with open(dirResult/"EpochTraError.pkl", "wb") as f: pickle.dump(epochTraError, f)
				with open(dirResult/"EpochTime.pkl", "wb") as f: pickle.dump(epochTime, f)
				break
		
		traError=trainSingle(datasets, traFoldIndex, inputMargin, net, opt, traCategoryIndex, traBatchSize, categoryBatchSize)
		epochTraError.append(traError)
	
	state=TrainingState(net, opt, epoch, traCategoryIndex, np.random.get_state(), epochConfusion, epochCorrect, epochTraError, epochTime)
	state.save(dirState)
	with open(fileCheck, "wb") as f: pass
	
	
def evaluateEpoch(epoch, datasets, dirResult, arch):
	inputLen=totalInputLength(arch)
	inputMargin=makeInputMargin(inputLen)
	
	net=Net([(datasetName, datasets.numCategory(datasetName)) for datasetName in datasets.datasetNames], arch)
	
	fileEpochConfusion=dirResult/"EpochConfusion.pkl"
	if fileEpochConfusion.is_file():
		with open(fileEpochConfusion, "rb") as f: epochConfusion=pickle.load(f)
	else:
		with open(dirResult/"EpochCorrect.pkl", "rb") as f: epochCorrect=pickle.load(f)
	
	fileConf=dirResult/"ValConfusion"/(str(epoch)+".pkl")
	if fileConf.is_file(): return
	fileConf.parent.mkdir(exist_ok=True, parents=True)
	
	fileNet=dirResult/"TrainedNet"/(str(epoch)+".npz")
	with open(fileNet, "rb") as f: serializers.load_npz(f, net)
	if gpu_id>=0: net.to_gpu(gpu_id)
	
	valFoldIndex=1
	batchValData=datasets.makeBatchValData(valFoldIndex, valBatchSize, inputMargin, WAVE_RMS_VAL)
	valConfusion=evaluateSingle(datasets, net, batchValData)
	
	if fileEpochConfusion.is_file():
		for datasetName in valConfusion.keys():
			assert (epochConfusion[epoch][datasetName]==valConfusion[datasetName]).all()
	else:
		correct=dict([(datasetName, calcCorrectRatio(conf)) for (datasetName, conf) in valConfusion.items()])
		meanCorrect=calcMeanCorrect(correct, datasets.datasetNames)
		assert epochCorrect[epoch]==meanCorrect
	
	valFoldIndex=2
	batchValData=datasets.makeBatchValData(valFoldIndex, valBatchSize, inputMargin, WAVE_RMS_VAL)
	valConfusion=evaluateSingle(datasets, net, batchValData)
	
	with open(fileConf, "wb") as f: pickle.dump(valConfusion, f)
	
	correct=dict([(datasetName, calcCorrectRatio(conf)) for (datasetName, conf) in valConfusion.items()])
	print("ep", epoch, "val correct", correct, sep="\t", flush=True)
	
	
def evaluate(datasetType, archName, breakValEpoch):
	dirArch=DIR_NET/datasetType/archName
	
	if CONTROL_TYPE=="Original":
		dirResult=dirArch
	else:
		dirResult=dirArch/("Result_"+CONTROL_TYPE)
	
	with open(dirResult/"EpochCorrect.pkl", "rb") as f: epochCorrect=pickle.load(f)
	
	categoryBatchSize=1
	valEpoch=(2**4)//categoryBatchSize
	breakEpoch=breakValEpoch*valEpoch
	lastEpoch, bestCorrect, bestEpoch=getLastEpoch(epochCorrect, valEpoch, breakEpoch)
	print("Best epoch", bestEpoch)
	
	datasets=DataSet.load(DIR_DATA, datasetType)
	arch=readArchitecture(dirArch/"Architecture.txt")
	
	evaluateEpoch(bestEpoch, datasets, dirResult, arch)
	evaluateEpoch(0, datasets, dirResult, arch)



if __name__=="__main__":
	DIR_DATA=DIR_TRAINING/"Data"
	DIR_NET=DIR_TRAINING/"Net"
	
	FADE_SEC=0.01
	
	gpu_id=0
# 	gpu_id=-1 #when no GPU is available

	NUM_GPU=initCupy(gpu_id)
	print("NUM_GPU", NUM_GPU)
	
	traBatchSize=2**8 #set according to your memory size
	valBatchSize=2**2 #set according to your memory size
	
	datasetType="ESC"
# 	datasetType="TIMIT"

	CONTROL_TYPE="Original"
# 	CONTROL_TYPE="EnvSingleBand"
# 	CONTROL_TYPE="TFSSingleBand"
# 	CONTROL_TYPE="EnvMultiBand"
# 	CONTROL_TYPE="TFSMultiBand"
	
	architecture_numLayer=7 #can be one of {7, 9, 11, 13}
	architecture_numUnit=32 #can be one of {32, 64, 128, 256, 512}
	architecture_sample=0   #can be one of {0, 1, 2, 3}
	
	architectureName="Layer"+str(architecture_numLayer)+"_Unit"+str(architecture_numUnit)+"_Sample"+str(architecture_sample)
	
# 	breakValEpoch=32
	breakValEpoch=96
	
	train(datasetType, architectureName, breakValEpoch)
	evaluate(datasetType, architectureName, breakValEpoch)
