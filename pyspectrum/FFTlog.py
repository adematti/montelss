import logging
import numpy
import scipy
from scipy import constants
import fftlog
import utils

class FFTlog(object):
	
	logger = logging.getLogger('FFTlog')
	
	def __init__(self,**params):
		self.params = params
	
	@staticmethod
	def calc_log_binning(x,nx=None):
		nonzero = x>0.
		logxmin,logxmax = scipy.log10(x[nonzero].min()),scipy.log10(x[nonzero].max())
		if not nx: nx = scipy.sum(nonzero)
		logxc = (logxmin + logxmax)/2.
		dlogx = (logxmax - logxmin)/(nx-1.)
		return logxc,dlogx,nx
	
	@staticmethod	
	def logspace(logkc,dlogk,nk):
		return 10**(logkc+(scipy.arange(1,nk+1)-(nk+1)/2.0)*dlogk)
	
	@utils.classparams
	def setup(self,s,k,orders,bias=0,lowringing=False,direction='forward'):
		self.logger.debug('Initialization...')
		self.logsc,self.dlogs,self.ns = self.__class__.calc_log_binning(s)
		logkc = self.__class__.calc_log_binning(k,self.ns)[0]
		kcsc = 10**(self.logsc+logkc)
		self.logger.info('Proposed central k * central s = {:.4g}; ns = {:d}, s-range = {:.4g} - {:.4g}.'.format(kcsc,self.ns,self.s[0],self.s[-1]))
		self.preset,self.kcsc = [],[]
		for order in orders:
			nkcsc,save,ok = fftlog.fhti(self.ns,order,self.dlogs*scipy.log(10.),bias,kcsc,int(lowringing))
			self.kcsc += [nkcsc]
			txt = 'For order {:.3g}, new central k * central s = {:.4g}, ok = {}.'.format(order,nkcsc,ok==True)
			self.logger.debug(txt)
			if not ok: raise ValueError(txt)
		self.kcsc = scipy.mean(self.kcsc)
		self.logger.info('New central k * central s = {:.4g}; nk = {:d}, k-range = {:.4g} - {:.4g}.'.format(self.kcsc,self.nk,self.k[0],self.k[-1]))
		if not lowringing:
			assert self.kcsc == kcsc, 'New k/s is not the same as given in input.'
		for order in orders:
			save,ok = fftlog.fhti(self.ns,order,self.dlogs*scipy.log(10.),bias,self.kcsc,0)[1:]
			self.preset += [save]
			if not ok: raise ValueError('FFTlog fails to compute order {:.3g}.'.format(order))
	
	@utils.getstateclass
	def getstate(self,state):
		for key in ['logsc','dlogs','ns','kcsc','preset']:
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
	
	@property
	def logkc(self):
		return scipy.log10(self.kcsc) - self.logsc
	
	@property
	def s(self):
		return self.__class__.logspace(self.logsc,self.dlogs,self.ns)
		
	@property
	def k(self):
		return self.__class__.logspace(self.logkc,self.dlogk,self.nk)
		
	@property
	def nk(self):
		return self.ns
		
	@property
	def dlogk(self):
		return self.dlogs
	
	@property
	def direction(self):
		return (-1)**(self.params['direction'] != 'forward')

	def integrate(self,integrand):
		res = []
		for integ,preset in zip(integrand,self.preset):
			if scipy.isrealobj(integ): res.append(fftlog.fht(integ,preset,self.direction))
			else: res.append(fftlog.fht(integ.real,preset,self.direction)+1j*fftlog.fht(integ.imag,preset,self.direction))
		return scipy.asarray(res)
	
	@classmethod
	def transform(cls,integrand,s,k,**kwargs):
		self = cls(**kwargs)
		self.setup(s,k)
		return self.integrate(integrand)

class FFTlogBessel(FFTlog):

	logger = logging.getLogger('FFTlogBessel')

	def getstate(self):
		state = super(FFTlogBessel,self).getstate()
		for key in ['norm','integrand','ells']: 
			if hasattr(self,key): state[key] = getattr(self,key)
		return state
	
	@utils.classparams
	def setup(self,s,k,ells,**kwargs):
		ells = scipy.asarray(self.ells)
		super(FFTlogBessel,self).setup(s,k,ells+1./2.,**kwargs)
		if self.direction == 1:
			self.norm = (2*constants.pi)**(3./2.)*self.k**(-3./2.)*((-1j)**ells)[:,None]
			self.integrand = scipy.array([self.s**(3./2.)]*len(ells))
		else:
			self.norm = (2.*constants.pi)**(-3./2.)*self.s**(-3./2.)*(1j**ells)[:,None]
			self.integrand = scipy.array([self.k**(3./2)]*len(ells))
		if scipy.isreal(self.norm).all(): self.norm = self.norm.real

	def integrate(self,func):
		tmp = super(FFTlogBessel,self).integrate(self.integrand*func)
		return self.norm*tmp
	
	@property
	def ells(self):
		return self.params['ells']
	
	@property
	def nells(self):
		return len(self.ells)
	
	def index(self,ell):
		return self.ells.index(ell)
