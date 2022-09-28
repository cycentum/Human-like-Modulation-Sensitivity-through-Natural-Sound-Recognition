'''
Translated from Matlab codes of Sound Texture Synthesis Toolbox http://mcdermottlab.mit.edu/
'''

import numpy as np


def erb2freq(n_erb):
	freq_Hz = 24.7*9.265*(np.exp(n_erb/9.265)-1)
	return freq_Hz


def freq2erb(freq_Hz):
	n_erb = 9.265*np.log(1+freq_Hz/(24.7*9.265));
	return n_erb

def make_erb_cos_filters(signal_length, fs, N, low_lim, hi_lim):
	'''
	@return filter, centerFreq, allFreq
	'''
	freqs=np.fft.rfftfreq(signal_length)
	freqs=freqs*fs
	nfreqs=freqs.shape[0]-1 #does not include DC
	max_freq=freqs[freqs.shape[0]-1]
	
	cos_filts = np.zeros([nfreqs + 1, N])
	
	if hi_lim > fs / 2 :
		hi_lim = max_freq
	
	# make cutoffs evenly spaced on an erb scale
	cutoffs=np.zeros(N+2)
	for fi in range(0, N+2):
		cutoffs[fi] = erb2freq((freq2erb(hi_lim) - freq2erb(low_lim)) * fi / (N + 1) + freq2erb(low_lim));
	
	for k in range(0, N):
		l = cutoffs[k]
		h = cutoffs[k + 2] # adjacent filters overlap by 50 #
		lInds=np.where(freqs > l)[0]
		hInds=np.where(freqs < h)[0]
		if(lInds.shape[0]==0 or hInds.shape[0]==0):
			continue
		l_ind = np.min(lInds)
		h_ind = np.max(hInds)
		avg = (freq2erb(l) + freq2erb(h)) / 2
		rnge = (freq2erb(h) - freq2erb(l))
		cos_filts[np.arange(l_ind, h_ind+1), k] = np.cos((freq2erb(freqs[np.arange(l_ind, h_ind+1)]) - avg) / rnge * np.pi) # map cutoffs to - pi / 2, pi / 2 interval
	
	filts = np.zeros([nfreqs + 1, N + 2]);
	filts[:, 1:N+1] = cos_filts;
	hInds=np.where(freqs < cutoffs[1])[0]
	if(hInds.shape[0]>0):
		h_ind = np.max(hInds); # lowpass filter goes up to peak of first cos filter
		filts[0:h_ind+1, 0] = np.sqrt(1 - filts[0:h_ind+1, 1] ** 2);
	lInds=np.where(freqs > cutoffs[N])[0]
	if(lInds.shape[0]>0):
		l_ind = np.min(lInds); # highpass filter goes down to peak of last cos filter
		filts[l_ind:nfreqs+1, N+1] = np.sqrt(1 - filts[l_ind:nfreqs+1, N] ** 2);

	filts=np.transpose(filts)

	return (filts, cutoffs, freqs)


def make_constQ_cos_filters(signal_length, sr, N, low_lim, hi_lim, Q):
	freqs=np.fft.rfftfreq(signal_length)
	freqs=freqs*sr
	nfreqs=freqs.shape[0]-1 #does not include DC
	max_freq=freqs[freqs.shape[0]-1]
	
	cos_filts = np.zeros([nfreqs+1,N]);
	
	if hi_lim>sr/2:
		hi_lim = max_freq;
	
	#make center frequencies evenly spaced on a log scale
	#want highest cos filter to go up to hi_lim
	Cfs=np.zeros(N)
	for ci in range(0, N):
		Cfs[ci] = 2 ** ((np.log2(hi_lim)-np.log2(low_lim))*ci/(N-1)+np.log2(low_lim))
	
	#easy-to-implement version: filters are symmetric on linear scale
	for k in range(0, N):
		bw = Cfs[k]/Q
		l = Cfs[k]-bw #so that half power point is at Cf-bw/2
		h = Cfs[k]+bw
		lInds=np.where(freqs > l)[0]
		hInds=np.where(freqs < h)[0]
		if(lInds.shape[0]==0 or hInds.shape[0]==0):
			continue
		l_ind = np.min(lInds)
		h_ind = np.max(hInds)
		if(l_ind>=h_ind+1):
			continue
		avg = Cfs[k]; #(log2(l+1)+log2(h+1))/2;
		rnge = h-l;#(log2(h+1)-log2(l+1));
		cos_filts[l_ind:h_ind+1,k] = np.cos((freqs[l_ind:h_ind+1] - avg)/rnge*np.pi) #map cutoffs to -pi/2, pi/2 interval
	
	temp=np.sum(cos_filts**2, 1)
	index=(freqs>=Cfs[3]) & (freqs<=Cfs[-4])
	temp=temp[index]
	if temp.shape[0]==0:
		filts=np.array([])
	else:
		filts=cos_filts/np.sqrt(np.mean(temp));
	
	filts=np.transpose(filts)
	
	return (filts, Cfs, freqs)


def generate_subbands(signal, filts):
	fft_sample = np.fft.rfft(signal);
# 	fft_subbands = filts* np.array((np.ones([filts.shape[0], 1]) * np.matrix(fft_sample)));#multiply by array of column replicas of fft_sample
	fft_subbands = filts* fft_sample;#multiply by array of column replicas of fft_sample
	subbands = np.fft.irfft(fft_subbands); #ifft works on columns; imag part is small, probably discretization error?
	return subbands



def collapse_subbands(subbands, filts):
	fft_subbands=np.fft.rfft(subbands, axis=1)
	fft_subbands*=filts
	subbands=np.fft.irfft(fft_subbands, axis=1)
	signal = np.sum(subbands,axis=0)
	return signal

