import logging
import functools
import scipy
from scipy import special,constants,integrate
from cosmology import Cosmology
import pyregpt
from pyregpt import *
import utils

def damping_kernel(k,kmin=1e-5,kmax=1.):
	ones = scipy.ones_like(k)
	mask = k>kmax
	ones[mask] *= scipy.exp(-(k[mask]/kmax-1)**2)
	mask = k<kmin
	ones[mask] *= scipy.exp(-(kmin/k[mask]-1)**2)
	return ones
	
def damping(kmin=1e-5,kmax=1.):
	def decorator(func):
		@functools.wraps(func)
		def wrapper(self,k,*args,**kwargs):
			damping = scipy.interp(k,self.spectrum_lin.k,damping_kernel(self.spectrum_lin.k,kmin=kmin,kmax=kmax),left=0.,right=0.)
			return damping*func(self,k,*args,**kwargs)
		return wrapper
	return decorator

class BasePowerSpectrumModel(object):

	TYPE_FLOAT = scipy.float64
	TERMS = ['spectrum_lin','spectrum_nowiggle']
	logger = logging.getLogger('Model')
	
	def __init__(self,**params):
		self.params = params

	def setup(self):
		self.set_cosmology()
		self.set_spectrum_lin()

	@utils.classparams
	def set_cosmology(self,cosmology=None,Class={}):
		if cosmology is not None:
			self.logger.info('Setting {} cosmology.'.format(cosmology))
			self.cosmo = getattr(Cosmology,cosmology)()
		else:
			self.cosmo = Cosmology()
		self.cosmo.set_params(**Class)

	@utils.classparams
	def set_spectrum_lin(self,kin,redshift=0.,cb=True):
		if cb:
			self.logger.info('Computing power spectrum of cold dark matter and baryons.')
			fun_pk = self.cosmo.compute_pk_cb_lin
			fun_sigma8 = self.cosmo.sigma8_cb
		else:
			self.logger.info('Computing matter power spectrum.')
			fun_pk = self.cosmo.compute_pk_lin
			fun_sigma8 = self.cosmo.sigma8
		self.spectrum_lin = SpectrumLin()
		self.spectrum_lin['k'] = scipy.asarray(kin,dtype=self.TYPE_FLOAT)
		self.logger.info('Calculating the linear power spectrum at redshift z = {:.4f}.'.format(redshift))
		self.logger.info('Cosmological parameters: {}.'.format(self.cosmo.params))
		self.spectrum_lin['pk'] = fun_pk(self.spectrum_lin['k'],z=redshift)
		self.sigma8 = fun_sigma8()
		self.spectrum_nowiggle = SpectrumNoWiggle(k=self.spectrum_lin['k'])
		self.spectrum_nowiggle.run_terms(pk=self.spectrum_lin['pk'],n_s=self.cosmo.n_s(),h=self.cosmo.h(),Omega_m=self.cosmo.Omega_m(),Omega_b=self.cosmo.Omega_b(),T_cmb=self.cosmo.T_cmb())

	def to_redshift(self,redshift=0.):
		# WARNING: Unable to tackle correctly A and B terms
		redshift_old = self.params['redshift']
		growth_factor_old = self.cosmo.scale_independent_growth_factor(redshift_old)
		growth_factor_new = self.cosmo.scale_independent_growth_factor(redshift)
		scale = growth_factor_new/growth_factor_old
		self.logger.info('Rescaling spectrum, bias and A & B terms by {:.4g} to account for redshift change from {:.4g} to {:.4g}.'.format(scale,redshift_old,redshift))
		self.sigma8 *= scale
		for term in self.TERMS:
			getattr(self,term).rescale(scale)
		return self

	@classmethod
	@utils.loadclass
	def load(self,state):
		self.setstate(state)
		return self

	@utils.saveclass
	def save(self):
		return self.getstate()

	def __getstate__(self):
		return self.getstate()
	
	def __setstate__(self,state):
		self.setstate(state)

	@utils.getstateclass
	def getstate(self,state):
		for key in ['sigma8']:
			if hasattr(self,key): state[key] = getattr(self,key)
		#for key in ['cosmo']:
		#	if hasattr(self,key): state[key] = getattr(self,key).getstate()
		for key in self.TERMS:
			if hasattr(self,key):
				tmp = getattr(self,key)
				state[key] =  {'__class__':tmp.__class__.__name__, '__dict__':tmp.as_dict()}
		return state

	@utils.setstateclass
	def setstate(self,state):
		#for key in ['cosmo']:
		#	key in state:
		#		setattr(self,key,Cosmology.loadstate(tmp))
		for key in self.TERMS:
			if key in state:
				cls = getattr(pyregpt,state[key]['__class__'])
				setattr(self,key,cls(**state[key]['__dict__']))
		self.set_regpt()

	def set_regpt(self):
		if hasattr(self,'spectrum_lin'):
			self.pyregpt = PyRegPT()
			self.pyregpt.set_spectrum_lin(self.spectrum_lin)

	def spectrum_galaxy_tree_real(self,k,b1=1.3,rsigma8=1.,**kwargs):
		return b1**2*self.spectrum_lin.pk_interp(k,Dgrowth=rsigma8,left=0.,right=0.)

	def DFoG(self,k,mu=1.,f=1.,sigmav=1.,avir=0.,sigmaerr=0.,FoG='gaussian'):
		if FoG == 'gaussian': return scipy.exp(-(f*k*mu*sigmav)**2)
		elif FoG == 'lorentzian1': return (1.+(k*mu*sigmav)**2)**(-1)
		elif FoG == 'lorentzian2': return (1.+(k*mu*sigmav)**2/2.)**(-2)
		elif FoG == 'infall':
			# Hikage 2015 (arXiv: 1506.01100v2) eq. 10
			kmu = k*mu
			diff = sigmav**2-avir**2/3.
			if diff < 0: diff = 0.
			toret = scipy.exp(-kmu**2*diff/2.)*scipy.sin(kmu*avir)/(kmu*avir)
			toret[kmu==0.] = 1.
			return toret
		elif FoG == 'skewed':
			kmu2 = (k*mu)**2
			return 1./scipy.sqrt(1.+kmu2*avir**2)*scipy.exp(-kmu2*sigmav**2/(1.+kmu2*avir**2))*scipy.exp(-kmu2*sigmaerr**2)
		raise ValueError('You have to choose between gaussian, lorentzian[1,2], infall and skewed damping.')

	def sum_FoG(self,k,mu,FoG='gaussian',sigmav=1.,wFoG=None,**kwargs):
		if wFoG is not None:
			wFoG = wFoG + [1.-sum(wFoG)]
			#print wFoG,sigmav,FoG
			return scipy.sum([wFoG_*scipy.sqrt(self.DFoG(k,mu,sigmav=sigmav_,FoG=FoG_,**kwargs)) for wFoG_,sigmav_,FoG_ in zip(wFoG,sigmav,FoG)],axis=0)**2
		return self.DFoG(k,mu,sigmav=sigmav,FoG=FoG,**kwargs)

	def spectrum_no_wiggle(self,k,mu,f=0.8,b1=1.3,sigmav=4.,FoG='gaussian',rsigma8=1.,**kwargs):
		Pdd = self.spectrum_nowiggle.pk_interp(k,Dgrowth=rsigma8,left=0.,right=0.)
		fmu2 = f*mu**2
		#return b1**2*Pdd+2*fmu2*b1*Pdd+fmu2**2*Pdd
		return self.DFoG(k,mu,f=f,sigmav=sigmav,FoG=FoG)*(b1**2+2*fmu2*b1+fmu2**2)*Pdd
	
	def spectrum_galaxy_tree(self,k,mu,f=0.8,b1=1.3,sigmav=4.,FoG='gaussian',rsigma8=1.,Ng=0.,**kwargs):
		Pdd = self.spectrum_lin.pk_interp(k,Dgrowth=rsigma8,left=0.,right=0.)
		fmu2 = f*mu**2
		return self.DFoG(k,mu,f=f,sigmav=sigmav,FoG=FoG)*((b1**2+2*fmu2*b1+fmu2**2)*Pdd+Ng)
