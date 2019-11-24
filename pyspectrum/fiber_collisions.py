import logging
import functools
import copy
import scipy
from scipy import constants,special
import re
import utils

def damping_kernel(k,kmin=1e-5,kmax=1.):
	ones = scipy.ones_like(k)
	mask = k>kmax
	ones[mask] *= scipy.exp(-(k[mask]/kmax-1)**2)
	mask = k<kmin
	ones[mask] *= scipy.exp(-(kmin/k[mask]-1)**2)
	return ones
	
def damping(func):
	@functools.wraps(func)
	def wrapper(self,*args,**kwargs):
		return self.damping*func(self,*args,**kwargs)
	return wrapper

def weights_trapz(x):
	trapzw = x*scipy.concatenate([[x[1]-x[0]],x[2:]-x[:-2],[x[-1]-x[-2]]])/2.
	return trapzw

def W2D(x):
	return 2.*special.j1(x)/x

# Appendix Hahn 2016 arXiv:1609.01714v1
def H(*ells):
	if ells == (2,0):
		return lambda x: x**2 - 1.
	if ells == (4,0):
		return lambda x: 7./4.*x**4 - 5./2.*x**2 + 3./4.
	if ells == (4,2):
		return lambda x: x**4 - x**2
	if ells == (6,0):
		return lambda x: 33./8.*x**6 - 63./8.*x**4 + 35./8.*x**2 - 5./8.
	if ells == (6,2):
		return lambda x: 11./4.*x**6 - 9./2.*x**4 + 7./4.*x**2
	if ells == (6,4):
		return lambda x: x**6 - x**4

class FiberCollisions(object):

	TYPE_FLOAT = scipy.float64
	logger = logging.getLogger('FiberCollisions')

	def __init__(self,**params):
		self.params = params

	def index(self,pole):
		return self.poles.index(pole)
	
	def __contains__(self,pole):
		return pole in self.poles
		
	def __iter__(self):
		return self.poles.__iter__()
		
	@property
	def ellsout(self):
		return self.params['ellsout']

	@property
	def ellsin(self):
		return self.params['ellsin']

	@utils.classparams
	def setup(self,k,ellsin=[0],ellsout=[0],fs=0.,Dfc=1.):
		self.set_grid()
		self.set_uncorrelated()
		self.set_correlated()

	@utils.classparams
	def set_grid(self,k,q=[0,scipy.inf]):
		for key in ['k']: setattr(self,key,scipy.array(self.params[key],dtype=self.TYPE_FLOAT))
		self.kmask = (self.k >= q[0]) & (self.k <= q[-1])
		self.q = self.k[self.kmask]
		self.damping = damping_kernel(self.k)

	@utils.classparams
	def set_uncorrelated(self,fs=0.,Dfc=1.):
		self.kernel_uncorrelated = []
		for ell in self.ellsout:
			tmp = fs*(2.*ell+1.)*special.legendre(ell)(0.)*(constants.pi*Dfc)**2/self.k*W2D(self.k*Dfc)
			self.kernel_uncorrelated.append(tmp)
		self.kernel_uncorrelated = scipy.array(self.kernel_uncorrelated)

	@utils.classparams
	def set_correlated(self,fs=0.,Dfc=1.):
		self.kernel_correlated = []
		kk,qq = scipy.meshgrid(self.k,self.q,indexing='ij')
		for ellout in self.ellsout:
			self.kernel_correlated.append([])
			for ellin in self.ellsin:
				if ellin == ellout:
					tmp = qq/kk
					tmp[tmp>1.] = 1.
					fll = tmp*W2D(qq*Dfc)*(scipy.amin([kk,qq],axis=0)/scipy.amax([kk,qq],axis=0))**ellout
				else:
					tmp = qq/kk
					tmp[tmp>1.] = 1.
					tmp = tmp*W2D(qq*Dfc)*(2.*ellout+1.)/2.*H(max(ellout,ellin),min(ellout,ellin))(scipy.amin([kk,qq],axis=0)/scipy.amax([kk,qq],axis=0))
					fll = scipy.zeros_like(tmp)
					nonzero = ((ellout>=ellin) & (kk>=qq)) | ((ellout<=ellin) & (kk<=qq))
					fll[nonzero] = tmp[nonzero]
				self.kernel_correlated[-1].append(fll)
		self.kernel_correlated = fs*Dfc**2/2.*scipy.array(self.kernel_correlated)*weights_trapz(self.q)

	@damping
	def correlated(self,Pl):
		Pl = Pl[:,self.kmask]
		return scipy.sum(self.kernel_correlated*Pl[None,:,None,:],axis=(-3,-1))

	@damping
	def uncorrelated(self):
		return self.kernel_uncorrelated

	def __call__(self,Pl):
		return self.correlated(Pl) + self.uncorrelated()

	@utils.getstateclass
	def getstate(self,state):
		for key in ['k','q','kmask','kernel_uncorrelated','kernel_correlated','damping']:
			if hasattr(self,key): state[key] = getattr(self,key)
		return state

	@utils.setstateclass
	def setstate(self,state):
		pass

	@classmethod
	@utils.loadstateclass
	def loadstate(self,state):
		self.setstate(state)
		return self

	@classmethod
	@utils.loadclass
	def load(self,state):
		self.setstate(state)
		return self

	@utils.saveclass
	def save(self):
		return self.getstate()
