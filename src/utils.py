import numpy as np
import pickle


def smallestPow2Above(x):
	'''
	@return 2**y: 2**(y-1) < x <= 2**y
	'''
	return 2**(np.ceil(np.log2(x)).astype(np.int32))


def checkRandState(file):
	if file.is_file():
		print("Loading existing rand state:", file)
		with open(file, "rb") as f: randState=pickle.load(f)
		np.random.set_state(randState)
	else:
		print("Saving new rand state:", file)
		randState=np.random.get_state()
		with open(file, "wb") as f: pickle.dump(randState, f)


def corrcoef(x0, x1, axis=None, normalized=True):
	'''
	@param axis: must be continuous from the last axis
	'''
	if axis is None: axis=tuple(range(x0.ndim))
	if isinstance(axis, int): axis=(axis,)
	axis=tuple(a%x0.ndim for a in axis)
	
	#verify axis
	ma=max(axis)
	assert ma==x0.ndim-1
	mi=min(axis)
	for ai in range(mi,ma+1):
		assert ai in axis
	
	x=(x0,x1)
	if normalized:
		m=[xi.mean(axis=axis, keepdims=True) for xi in x]
		x_m=[xi-mi for xi,mi in zip(x,m)]
	else:
		x_m=x
# 	prod=(x_m[0]*x_m[1]).mean(axis=axis)
	meanSize=np.maximum(np.array(x0.shape), np.array(x1.shape))[np.array(axis)].prod()
	prod=np.einsum(x_m[0], [Ellipsis,*axis], x_m[1], [Ellipsis,*axis])/meanSize
	if normalized:
		s=[xi.std(axis=axis) for xi in x]
		cc=prod/(s[0]*s[1])
	else:
		cc=prod
	return cc
