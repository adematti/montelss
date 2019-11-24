import functools
import logging
import numpy
import scipy
from scipy import constants,integrate,special
from mu_function import MuFunction
import utils
   		
def integrate_legendre(func):
	@functools.wraps(func)
	def wrapper(self,*args,**kwargs):
		kernel = self.kernelcorrmu if kwargs.get('corrmu',False) else self.kernel
		integrand = scipy.expand_dims(func(self,self.kk,self.mumu,*args,**kwargs),axis=-3)
		return integrate.trapz(integrand*kernel,x=self.mu,axis=-1)
	return wrapper

class EffectAP(object):

	TYPE_INT = scipy.int16
	TYPE_FLOAT = scipy.float64
	logger = logging.getLogger('EffectAP')

	def __init__(self,**params):
		self.params = params

	@staticmethod
	def k_mu_ap(k,mu,qpar,qper):
		F = qpar/qper
		factor_ap = scipy.sqrt(1+mu**2*(1./F**2-1))
		# Beutler 2016 (arXiv: 1607.03150v1) eq 44
		kap = k/qper*factor_ap
		# Beutler 2016 (arXiv: 1607.03150v1) eq 45
		muap = mu/F/factor_ap
		jacob = 1./(qper**2*qpar)
		return jacob,kap,muap

	@integrate_legendre
	def spectrum_multipoles_no_ap(self,*args,**kwargs):
		return self.input_model_ref(*args,**kwargs)
	
	def spectrum_multipoles(self,*args,**kwargs):
		qpar,qper = kwargs.get('qpar',1.),kwargs.get('qper',1.)
		kwargs['kobs'] = self.kk
   		jacob,kap,muap = self.__class__.k_mu_ap(self.kk,self.mumu,qpar,qper)
   		kernel = self.kernelcorrmu if kwargs.get('corrmu',False) else self.kernel
   		integral = scipy.expand_dims(jacob*self.input_model_ref(kap,muap,*args,**kwargs),axis=-3)
   		multi = integrate.trapz(integral*kernel,x=self.mu,axis=-1)
		if self.input_model_lin is not None:
			integral = jacob*self.input_model_lin(kap,*args,**kwargs)
			modellin = integrate.trapz(integral*kernel[0],x=self.mu,axis=-1)
			return multi,modellin
		return multi,None
			
	@utils.classparams
	def setup(self):
		self.set_grid()

	@utils.classparams
	def set_grid(self,k,mu,ells,path_mu=None):
	
		assert ells[0] == 0 # for modellin

		for key in ['k','mu']: setattr(self,key,scipy.array(self.params[key],dtype=self.TYPE_FLOAT))
		self.kk,self.mumu = scipy.meshgrid(self.k,self.mu,sparse=False,indexing='ij')
		murange = self.mu[-1]-self.mu[0]
		self.logger.info('Setting grid {:d} (k) x {:d} (mu).'.format(self.kk.shape[0],self.mumu.shape[-1]))
		
		self.kernel = scipy.asarray([(2.*ell+1.)*special.legendre(ell)(self.mumu) for ell in self.ells])/murange
		
		if path_mu is not None:
			self.logger.info('Loading provided mu window function.')
			window = MuFunction.load(path_mu)
			self.logger.info('With resolution {:d} (k) x {:d} (mu).'.format(len(window.k),len(window.mu)))
			window = window(self.k,self.mu)
			self.kernelcorrmu = scipy.asarray([(2.*ell+1.)*window*special.legendre(ell)(self.mumu) for ell in self.ells])/murange
	
	def set_input_model(self,modelref,modellin=None):
		self.input_model_ref = modelref
		self.input_model_lin = modellin
	
	@property
	def ells(self):
		return self.params['ells']

	@utils.getstateclass
	def getstate(self,state):
		for key in ['k','mu','kk','mumu','kernel','kernelcorrmu']:
			if hasattr(self,key): state[key] = getattr(self,key)
		return state

	@utils.setstateclass
	def setstate(self,state):
		pass

	@classmethod
	@utils.loadclass
	def load(self,state):
		self.setstate(state)
		return self

	@utils.saveclass
	def save(self):
		return self.getstate()
