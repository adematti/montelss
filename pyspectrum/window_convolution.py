import functools
import logging
import numpy
import scipy
from scipy import special,constants,integrate
import math
from fractions import Fraction
from FFTlog import FFTlogBessel
from window_function import WindowFunction
import utils

def G(p):
	"""Return the function G(p), as defined in Wilson et al 2015.
	See also: WA Al-Salam 1953
	Taken from nbodykit.

	Returns
	-------
	numer, denom: int
		the numerator and denominator

	"""
	toret = 1
	for p in range(1,p+1): toret *= (2*p - 1)
	return	toret,math.factorial(p)

def coefficients(ellout,ellin,front_coeff=True,as_string=False):
	
	coeffs = []
	qvals = []
	retstr = []
	
	for p in range(min(ellin,ellout)+1):

		numer = []
		denom = []

		# numerator of product of G(x)
		for r in [G(ellout-p), G(p), G(ellin-p)]:
			numer.append(r[0])
			denom.append(r[1])

		# divide by this
		a,b = G(ellin+ellout-p)
		numer.append(b)
		denom.append(a)

		numer.append((2*(ellin+ellout) - 4*p + 1))
		denom.append((2*(ellin+ellout) - 2*p + 1))

		q = ellin+ellout-2*p
		if front_coeff:
			numer.append((2*ellout+1))
			denom.append((2*q+1))

		numer = Fraction(scipy.prod(numer))
		denom = Fraction(scipy.prod(denom))
		if not as_string:
			coeffs.append(numer*1./denom)
			qvals.append(q)
		else:
			retstr.append('l{:d} {}'.format(q,numer/denom))

	if not as_string:
		return qvals[::-1], coeffs[::-1]
	else:
		return retstr[::-1]
	
class MultipoleToMultipole(object):
	
	TYPE_FLOAT = scipy.float64
	logger = logging.getLogger('MultiToMulti')

	def __init__(self,ellsin,ellsout,w):
		self.logger.info('Setting multipoles to multipoles transforms.')
		self.conversion = scipy.empty((len(ellsout),len(ellsin))+w(0,0).shape,dtype=w(0,0).dtype)
		for illout,ellout in enumerate(ellsout):
			for illin,ellin in enumerate(ellsin):
				ells,coeffs = coefficients(ellout,ellin[0]) #case ellin = (ell,n)
				self.conversion[illout][illin] = scipy.sum([coeff*w(ell,ellin[-1]) for ell,coeff in zip(ells,coeffs)],axis=0)
	
	def transform(self,func):
		tmp = func*self.conversion
		return scipy.sum(func*self.conversion,axis=1)

		
	@utils.getstateclass
	def getstate(self,state):
		for key in ['conversion']:
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

class ConvolutionMultipole(object):
	
	TYPE_FLOAT = scipy.float64
	logger = logging.getLogger('Convolution')

	def __init__(self,**params):
		self.params = params

	@utils.classparams
	def set_grid(self,s,ellsin,ellsout):
		for key in ['s']: setattr(self,key,scipy.asarray(self.params[key],dtype=self.TYPE_FLOAT))
		self.multitomulti = MultipoleToMultipole(self.ellsin,self.ellsout,lambda ell,n: self.window[n](self.s,ell))

	@utils.classparams
	def set_window(self,path_window={}):
		self.window = {n: WindowFunction.load(path_window[n]) for n in path_window}
		self.ns = sorted(self.window.keys())
		assert self.ns[0] == 0
		self.los = self.window[0].los[0]
		for n in self.ns: assert self.window[n].los == (self.los,n)
		self.normwindow = self.window[0].norm
		return self.los, self.ns

	def convolve(self,func):
		return self.multitomulti.transform(func)

	@property
	def ellsin(self):
		return self.params['ellsin']

	@property
	def ellsout(self):
		return self.params['ellsout']

	def indexin(self,ell):
		return self.ellsin.index(ell)

	def indexout(self,ell):
		return self.ellsout.index(ell)

	@utils.getstateclass
	def getstate(self,state):
		for key in ['s','k']:
			if hasattr(self,key): state[key] = getattr(self,key)
		for key in ['multitomulti']:
			if hasattr(self,key): state[key] = getattr(self,key).getstate()
		return state

	@utils.setstateclass
	def setstate(self,state):
		if 'multitomulti' in state: self.multitomulti = MultipoleToMultipole.loadstate(state['multitomulti'])

	@classmethod
	@utils.loadstateclass
	def loadstate(self,state):
		self.setstate(state)
		return self	

	def copy(self):
		return self.__class__.loadstate(self.getstate())

class BaseWideAngleCorrection(object):

	TYPE_FLOAT = scipy.float64
	logger = logging.getLogger('BaseWideAngleCorrection')

	# Beutler 2018 (arXiv: 1810.05051v1) eq. 2.10-2.18, taking care about 2.13 definition (-i)**ell

	def __init__(self,ellmax=4,nmax=0,ells=None):
		if ells is None: ells = range(ellmax+1)
		self.ells = {0:[ell for ell in ells if ell%2==0]}
		self.ellslin = set()
		self.ellsref = set(self.ells[0])
		self.ns = [0] + sorted(self._required_lin.keys())
		self.ns = [n for n in self.ns if n <= nmax]
		for n in self.ns[1:]:
			self.ells[n] = sorted(set(self._required_lin[n].keys()) & set(ells))
			for ell in self.ells[n]:
				self.ellslin |= set(self._required_lin[n][ell])
				self.ellsref |= set(self._required_ref[n][ell])
		self.logger.info('Wide-angle corrections: {}.'.format(str(self.ells)))
	
	@property
	def flatells(self):
		return [(ell,n) for n in self.ns for ell in self.ells[n]]

	def flatellsn(self,n):
		return [(ell,n) for ell in self.ells[n]]

	def __call__(self,s,**kwargs):
		return scipy.array([self.Xin(elln,**kwargs)*s**elln[-1] for elln in self.flatells])

	def mask(self,n=None,ells=None):
		flatells = scipy.array(self.flatells)
		if n is None: return scipy.in1d(flatells[:,0],ells)
		if ells is None: return flatells[:,-1] == n
		return (flatells[:,-1] == n) & scipy.in1d(flatells[:,0],ells)

	@utils.getstateclass
	def getstate(self,state):
		for key in ['ells','ns','ellsref','ellslin']:
			if hasattr(self,key): state[key] = getattr(self,key)
		return state

	@utils.setstateclass
	def setstate(self,state):
		return self

	@classmethod
	@utils.loadstateclass	
	def loadstate(self,state):
		self.setstate(state)
		return self

class BisectorCorrection(BaseWideAngleCorrection):

	_required_lin = {2:{0:[0,2],2:[0,2,4],4:[2,4]}}
	_required_ref = {2:{0:[],2:[],4:[]}}
	logger = logging.getLogger('BisectorCorrection')

	def __init__(self,*args,**kwargs):
		super(BisectorCorrection,self).__init__(*args,**kwargs)

	def Xin(self,elln,beta=None):
		if elln[-1] == 0: return self.Xiref(elln[0])
		assert beta is not None
		if elln == (0,2): return -4./45*beta**2*self.Xilin(0) + (1./5.*beta+1./45.*beta**2)*self.Xilin(2)
		if elln == (2,2): return 4./45.*beta**2*self.Xilin(0) - (3./7.*beta+53./441.*beta**2)*self.Xilin(2) - 4./245.*beta**2*self.Xilin(4)
		if elln == (4,2): return (8./35.*beta+24./245.*beta**2)*self.Xilin(2) + 4./245.*beta**2*self.Xilin(4)

class EndpointCorrection(BaseWideAngleCorrection):

	_required_lin = {1:{1:[],3:[]},2:{0:[0,2],2:[0,2,4],4:[0,2,4]}}
	_required_ref = {1:{1:[2],3:[2,4]},2:{0:[2],2:[2,4],4:[2,4]}}
	logger = logging.getLogger('EndpointCorrection')

	def __init__(self,*args,**kwargs):
		super(EndpointCorrection,self).__init__(*args,**kwargs)

	def Xin(self,elln,beta=None):
		if elln[-1] == 0: return self.Xiref(elln[0])
		assert beta is not None
		if elln == (1,1): return -3./5.*self.Xiref(2)
		if elln == (3,1): return 3./5.*self.Xiref(2) - 10./9.*self.Xiref(4)
		if elln == (0,2): return -4./45*beta**2*self.Xilin(0) + (1./5.*beta+1./45.*beta**2)*self.Xilin(2) + 1./5.*self.Xiref(2)
		if elln == (2,2): return 4./45.*beta**2*self.Xilin(0) - (3./7.*beta+53./441.*beta**2)*self.Xilin(2) - 4./245.*beta**2*self.Xilin(4) - 2./7.*self.Xiref(2) + 5./7.*self.Xiref(4)
		if elln == (4,2): return (8./35.*beta+24./245.*beta**2)*self.Xilin(2) + 4./245.*beta**2*self.Xilin(4) + 3./35.*self.Xiref(2) - 90./77.*self.Xiref(4)

class MidpointCorrection(BaseWideAngleCorrection):

	_required_lin = {2:{0:[0,2],2:[0,2,4],4:[2,4]}}
	_required_ref = {2:{0:[],2:[],4:[]}}
	logger = logging.getLogger('MidpointCorrection')

	def __init__(self,*args,**kwargs):
		super(MidpointCorrection,self).__init__(*args,**kwargs)

	def Xin(self,elln,beta=None):
		if elln[-1] == 0: return self.Xiref(elln[0])
		assert beta is not None
		if elln == (0,2): return -4./45*beta**2*self.Xilin(0) + (1./15.*beta-11./315.*beta**2)*self.Xilin(2)
		if elln == (2,2): return 4./45.*beta**2*self.Xilin(0) - (11./21.*beta+71./441.*beta**2)*self.Xilin(2) - 52./735.*beta**2*self.Xilin(4)
		if elln == (4,2): return (16./35.*beta+48./245.*beta**2)*self.Xilin(2) + 4./2695.*beta**2*self.Xilin(4)

class LOSCorrections(object):

	TYPE_FLOAT = scipy.float64
	logger = logging.getLogger('LOSCorrections')

	def __init__(self,**params):
		self.params = params
		self.corrections = {}

	def add(self,name,los='midpoint',**kwargs):
		self.corrections[name] = globals()[los[0].upper() + los[1:] + 'Correction'](**kwargs)
		self.set_xi()
		return self.corrections[name].flatells

	def Xiref(self,ell):
		return self.xiref[self.torealref.index(ell)]

	def Xilin(self,ell):
		return self.xilin[self.toreallin.index(ell)]

	def set_xi(self):
		self.ellsref = set()
		self.ellslin = set()
		for _,correction in self.corrections.items():
			correction.Xiref = self.Xiref
			correction.Xilin = self.Xilin
			self.ellsref |= correction.ellsref
			self.ellslin |= correction.ellslin
		self.ellsref = sorted(list(self.ellsref))
		self.ellslin = sorted(list(self.ellslin))

	@utils.classparams
	def set_grid(self,s,k,fftlog={}):
		for key in ['s','k']: setattr(self,key,scipy.asarray(self.params[key],dtype=self.TYPE_FLOAT))
		self.torealref = FFTlogBessel(**self.params['fftlog'])
		self.torealref.setup(s=self.s,k=self.k,ells=self.ellsref,direction='backward')
		self.s,self.k = self.torealref.s,self.torealref.k
		if self.ellslin:
			self.toreallin = FFTlogBessel(**self.params['fftlog'])
			self.toreallin.setup(s=self.s,k=self.k,ells=self.ellslin,lowringing=False,direction='backward')

	@utils.classparams
	def setup(self,s,k,fftlog={}):
		self.set_xi()
		self.set_grid()

	def integrate(self,spectrumref,spectrumlin=None,f=1.,b1=1.,beta=None,**kwargs):
		self.xin_kwargs = {'beta':f/b1 if beta is None else beta}
		self.xiref = self.torealref.integrate(spectrumref)
		if self.ellslin: self.xilin = self.toreallin.integrate(spectrumlin)

	def __getitem__(self,name):
		return self.corrections[name](self.s,**self.xin_kwargs)
	
	def mask(self,name,**kwargs):
		return self.corrections[name].mask(**kwargs)

	def ells(self,name):
		return self.corrections[name].ells

	def flatells(self,name):
		return self.corrections[name].flatells

	def flatellsn(self,name,n=0):
		return self.corrections[name].flatellsn(n)

	def ns(self,name):
		return self.corrections[name].ns

	@utils.getstateclass
	def getstate(self,state):
		for key in ['s','k']:
			if hasattr(self,key): state[key] = getattr(self,key)
		for key in ['torealref','toreallin']:
			if hasattr(self,key): state[key] = getattr(self,key).getstate()
		state['corrections'] = {}
		for key in self.corrections:
			tmp = self.corrections[key]
			state['corrections'][key] = {'__class__':tmp.__class__.__name__, '__dict__':tmp.getstate()}
		return state

	@utils.setstateclass
	def setstate(self,state):
		for key in ['torealref','toreallin']:
			if key in state: setattr(self,key,FFTlogBessel.loadstate(state[key]))
		if 'corrections' in state:
			self.corrections = {key: globals()[val['__class__']].loadstate(val['__dict__']) for key,val in state['corrections'].items()}
		self.set_xi()

	@classmethod
	@utils.loadstateclass
	def loadstate(self,state):
		self.setstate(state)
		return self
