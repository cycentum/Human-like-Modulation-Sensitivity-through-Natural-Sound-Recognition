from matplotlib import pyplot as plt


DATASET_NAME={"ESC":"Everyday sound", "TIMIT":"Speech sound"}

def defaultColors(index=None, alpha=1):
	'''
	@param alpha: [0,1]
	'''
	colorStr = plt.rcParams['axes.prop_cycle'].by_key()['color']
	if index is None:
		colors=[]
		for cs in colorStr:
			r=int(cs[1:3],16)/255
			g=int(cs[3:5],16)/255
			b=int(cs[5:7],16)/255
			colors.append((r,g,b,alpha))
		return colors

	else:
		cs=colorStr[index%len(colorStr)]
		r=int(cs[1:3],16)/255
		g=int(cs[3:5],16)/255
		b=int(cs[5:7],16)/255
		return(r,g,b,alpha)


STIM_PARAM_MARKER=("+", "x", "1", "2", "3", "4")

def getStimParamName(stimParamName):
	if stimParamName is None: return None
	if stimParamName=="Viemeister1979_CfN_BwInf":
		name="Broad (short)"
	elif stimParamName.startswith("Dau1997_Cf5000"):
		name=stimParamName[len("Dau1997_Cf5000_Bw"):]+" Hz"
	elif stimParamName.startswith("Lorenzi2001"):
		if "BwInf" in stimParamName:
			name="Broad (long)"
		elif "Bw2" in stimParamName:
			name="2 Hz"
# 	return phase+" phase - "+bw
# 	return bw+" - "+phase+" phase"
# 	return bw+"\n"+phase
	return name
