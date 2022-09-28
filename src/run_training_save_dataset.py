import itertools
import soundfile
from numpy import int16, float64
from dotmap import DotMap
import pickle
import numpy as np
from collections import defaultdict
import scipy.signal

import cochleagram
import utils
from params import WAVE_FS_ESC, WAVE_FS_TIMIT, DATASET_WAVE_FS
from path import DIR_TRAINING, DIR_DATASET_ESC, DIR_DATASET_TIMIT


def trim0(wave):
	for i0 in range(len(wave)):
		if wave[i0]!=0:
			break
	
	for i1 in range(len(wave), -1, -1):
		if wave[i1-1]!=0:
			break
	
	wave=wave[i0:i1]
	return wave, (i0,i1)


def cat0(wave, targetLength):
	assert targetLength>=len(wave)
	if targetLength>len(wave):
		wave0=np.concatenate((wave, np.zeros(targetLength-len(wave), wave.dtype)))
	else:
		wave0=wave
	return wave0


def compRms(wave):
	rms=(wave**2).mean()**0.5
	return rms


def scaleRms(wave, targetRms):
	rms0=compRms(wave)
	waveNew=wave*targetRms/rms0
	return waveNew


def save_ESC():
	fileData=DIR_DATA/"ESC.pkl"
	if fileData.is_file():
		print(fileData, "already exists.")
		return
	
	dirWaves=DIR_DATASET_ESC/"audio"
	data={}
	for file in dirWaves.iterdir():
		print(file)
		
		wave,fs=soundfile.read(file, dtype=int16)
		assert fs==WAVE_FS_ESC
		assert len(wave.shape)==1
		wave,_=trim0(wave)
		
		d=DotMap()
		data[file.name]=d
		
		d.wave=wave
		d.name=file.name
	
	categories={}
	fileMeta=DIR_DATASET_ESC/"meta"/"esc50.csv"
	with open(fileMeta, "r") as f:
		for li,line in enumerate(f):
			if li==0: continue
			filename,fold,target,category,esc10,src_file,take=line.rstrip().split(",")
			target=int(target)
			if target not in categories:
				categories[target]=category
			assert categories[target]==category
			
			d=data[filename]
			d.fold=int(fold)-1
			d.category=category
	
	categoryId=sorted(categories)
	assert (np.diff(np.array(categoryId))==1).all() and categoryId[0]==0 and len(categoryId)==50
	categories=[categories[i] for i in range(len(categories))]
	
	fileData.parent.mkdir(exist_ok=True, parents=True)
	with open(fileData, "wb") as f: pickle.dump((data, categories), f)


def save_TIMIT():
	fileData=DIR_DATA/"TIMIT.pkl"
	if fileData.is_file():
		print(fileData, "already exists.")
		return
	
	dirWaves=DIR_DATASET_TIMIT
	
	groupName=("TRAIN", "TEST")
	categoryType="PHN"
	
	CORE_TEST_SPEAKER={
		"MDAB0",
		"MWBT0",
		"FELC0",
		"MTAS1",
		"MWEW0",
		"FPAS0",
		"MJMP0",
		"MLNT0",
		"FPKT0",
		"MLLL0",
		"MTLS0",
		"FJLM0",
		"MBPM0",
		"MKLT0",
		"FNLP0",
		"MCMJ0",
		"MJDH0",
		"FMGD0",
		"MGRT0",
		"MNJM0",
		"FDHC0",
		"MJLN0",
		"MPAM0",
		"FMLD0",
		}
	assert len(CORE_TEST_SPEAKER)==24
	
	data={}
	for gn in groupName:
		dirGroup=dirWaves/gn
		for dirDialect in dirGroup.iterdir():
			for dirSpeaker in dirDialect.iterdir():
				print(dirSpeaker)
				speaker=dirSpeaker.name
				nameFiles=defaultdict(dict)
				for file in dirSpeaker.iterdir():
					name=file.name[:-4]
					ext=file.name[-3:]
					nameFiles[name][ext]=file
					
				for name in nameFiles:
					if name.startswith("SA"): continue
					if gn=="TEST" and speaker not in CORE_TEST_SPEAKER and name.startswith("SX"): continue
					
					fileWave=nameFiles[name]["WAV"]
					wave,fs=soundfile.read(fileWave, dtype=int16)
					assert fs==WAVE_FS_TIMIT
					
					_,t01=trim0(wave)
					assert t01[0]<10 and t01[1]>=len(wave)-10
					
					d=DotMap()
					d.name=gn+"_"+dirDialect.name+"_"+dirSpeaker.name+"_"+name
					data[d.name]=d
					d.wave=wave
					
					if gn=="TRAIN": d.group=gn
					elif speaker in CORE_TEST_SPEAKER: d.group="TEST_CORE"
					else: d.group="TEST_NONCORE"
					
					fileCategory=nameFiles[name][categoryType]
					category=[]
					d.category=category
					with open(fileCategory, "r") as f:
						for line in f:
							line=line.rstrip().split()
							if len(line)==0: continue
							t0,t1,c=line
							t0=int(t0)
							t1=int(t1)
							category.append(((t0,t1), c))
	
	#collapse categories to 39
	excluded={"q"}
	
	labelGroup=(
		("aa", "ao"),
		("ah", "ax", "ax-h"),
		("er", "axr"),
		("hh", "hv"),
		("ih", "ix"),
		("l", "el"),
		("m", "em"),
		("n", "en", "nx"),
		("ng", "eng"),
		("sh", "zh"),
		("uw", "ux"),
		("sil", "pcl", "tcl", "kcl", "bcl", "dcl", "gcl", "h#", "pau", "epi"),
	)
	labelMap={}
	for lg in labelGroup:
		for la in lg:
			labelMap[la]=lg[0]
	
	for d in data.values():
		newCategory=[]
		for t,c in d.category:
			if c in excluded: continue
			if c in labelMap: c=labelMap[c]
			newCategory.append((t,c))
		d.category=newCategory
	
	categories=set()
	for d in data.values():
		for t,c in d.category:
			categories.add(c)
	
	# Lee KF, Hon HW (1989)
	sortedCategories=('iy',
		'ih',
		'eh',
		'ae',
		'ah',
		'uw',
		'uh',
		'aa',
		'ey',
		'ay',
		'oy',
		'aw',
		'ow',
		'l',
		'r',
		'y',
		'w',
		'er',
		'm',
		'n',
		'ng',
		'ch',
		'jh',
		'dh',
		'b',
		'd',
		'dx',
		'g',
		'p',
		't',
		'k',
		'z',
		'v',
		'f',
		'th',
		's',
		'sh',
		'hh',
		'sil',
		)
	assert tuple(sorted(categories))==tuple(sorted(sortedCategories))

	fileData.parent.mkdir(exist_ok=True, parents=True)
	with open(fileData, "wb") as f: pickle.dump((data, sortedCategories), f)


def makeEnvTfsControl(wave, waveFs, controlType):
	if wave.dtype==int16:
		wave=(wave/np.iinfo(int16).max).astype(float64)
	
	length=len(wave)
	if "MultiBand" in controlType:
		freq0=20
		erb0=cochleagram.freq2erb(freq0)
		erb1=cochleagram.freq2erb(waveFs/2)
		erbs=np.arange(int(erb1-erb0)+1)+erb0
		erbs=erbs[erbs<=erb1]
		freq1=cochleagram.erb2freq(erbs[-1])
		numFilters=len(erbs)
		
		length2=utils.smallestPow2Above(length)
		wave2=cat0(wave, length2)
		
		filt, centerFreq, allFreq=cochleagram.make_erb_cos_filters(length2, waveFs, numFilters-2, freq0, freq1)
		
		coch=cochleagram.generate_subbands(wave2, filt)
			
	elif "SingleBand" in controlType:
		coch=wave[None,:]
	
	analytic=scipy.signal.hilbert(coch, axis=1)
	analytic=analytic[:, :length]
	env=abs(analytic)
	phase=np.angle(analytic)
	
	if "Env" in controlType:
		wn=np.random.randn(length)
		wn=scaleRms(wn, compRms(wave))
		if "MultiBand" in controlType:
			wn2=cat0(wn, length2)
			wnCoch=cochleagram.generate_subbands(wn2, filt)
		else:
			wnCoch=wn[None, :]
		wnAna=scipy.signal.hilbert(wnCoch, axis=1)
		wnAna=wnAna[:, :length]
		wnEnv=abs(wnAna)
		wnPhase=np.angle(wnAna)
			
	if "TFS" in controlType:
		newPhase=phase
		newEnv=(env**2).mean(axis=1, keepdims=True)**0.5
	
	elif "Env" in controlType:
		newEnv=env
		newPhase=wnPhase
	
	newCoch=newEnv*np.exp(1j*newPhase).real
	newWave=newCoch.sum(axis=0)
	
	return newWave

		
def saveEnvTfsControl(datasetType, controlType):
	fileDataControl=DIR_DATA/(datasetType+"_"+controlType+".pkl")
	if fileDataControl.is_file():
		print(fileDataControl, "already exists.")
		return
	
	fileData=DIR_DATA/(datasetType+".pkl")
	with open(fileData, "rb") as f: data, categories=pickle.load(f)

	for i,(filename,d) in enumerate(data.items()):
		print(filename, i, "/", len(data))
		d.wave=makeEnvTfsControl(d.wave, DATASET_WAVE_FS[datasetType], controlType)
	
	with open(fileDataControl, "wb") as f: pickle.dump((data, categories), f)
		

if __name__=="__main__":
	DIR_DATA=DIR_TRAINING/"Data"
	
	save_ESC()
	save_TIMIT()
	
	for datasetType,controlType_EnvTFS,controlType_band in itertools.product(("ESC", "TIMIT"), ("Env", "TFS"), ("Single", "Multi")):
		controlType=controlType_EnvTFS+controlType_band+"Band"
		print(datasetType, controlType)
		saveEnvTfsControl(datasetType, controlType)
