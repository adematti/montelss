import logging
import copy
import scipy
from numpy import linalg
#from scipy import linalg
from numpy.linalg import LinAlgError
import montelss
import pyspectrum
import utils

class BaseLikelihoodClustering(montelss.Likelihood):
	
	logger = logging.getLogger('BaseLikelihoodClustering')
	
	@utils.classparams
	def set_data(self,path_data,xrange=None,remove_sn=False):
		
		self.logger.info('Setting data: {}.'.format(path_data))
		if self.remove_sn: self.logger.info('Removing shotnoise.')
		else: self.logger.info('Keeping shotnoise.')
		get_data = utils.load_data(path_data,estimator=self.estimator,remove_sn=self.remove_sn)
		self.shotnoise = get_data('shotnoise')
		#self.shotnoise = 5163.38 if 'NGC' in self.id else 4675.43
		self.logger.info('Shotnoise is {:.2f}.'.format(self.shotnoise))
		#self.norm = get_data('norm')/get_data('data.W')**2
		self.norm = None
		self.xdata,self.adata = [],[]
		for ell in self.ells:
			self.logger.info('Setting {0}-range {1[0]:.4g} - {1[1]:.4g} of data for ell = {2}.'.format(self.xlabel,self.xrange[ell],ell))
			self.xdata.append(get_data('x',ell=ell,mode=self.mode,xlim=tuple(self.xrange[ell])))
			self.adata.append(get_data('y',ell=ell,mode=self.mode,xlim=tuple(self.xrange[ell])))
		self.data = scipy.concatenate(self.adata)

	@utils.classparams
	def set_data_covariance(self,path_covariance,xrange=None,remove_sn=False):

		self.logger.info('Setting mean: {}.'.format(path_covariance))
		if self.remove_sn: self.logger.info('Removing shotnoise.')
		else: self.logger.info('Keeping shotnoise.')
		get_cov = utils.load_covariance(path_covariance,remove_sn=self.remove_sn)
		nobs = get_cov('nobs')
		self.shotnoise = get_cov('shotnoise',estimator=self.estimator,recon=self.recon)
		#self.shotnoise = 5524.98
		self.logger.info('Shotnoise is {:.2f}.'.format(self.shotnoise))
		#self.norm = get_cov('norm')/get_cov('data.W')**2
		self.norm = None
		self.xdata,self.adata = [],[]
		for ell in self.ells:
			self.logger.info('Setting {0}-range {1[0]:.4g} - {1[1]:.4g} of mean for ell = {2}.'.format(self.xlabel,self.xrange[ell],ell))
			self.xdata.append(get_cov('x',estimator=self.estimator,recon=self.recon,mode=self.mode,ell=ell,xlim=tuple(self.xrange[ell])))
			self.adata.append(get_cov('mean',estimator=self.estimator,recon=self.recon,mode=self.mode,ell=ell,xlim=tuple(self.xrange[ell])))
		self.data = scipy.concatenate(self.adata)
		self.invcovariance *= nobs
		self.astddev /= scipy.sqrt(nobs)
		self.logger.info('Dividing covariance matrix by the number of observations: {:d}.'.format(nobs))

	@utils.classparams	
	def set_covariance(self,path_covariance,xrange=None,invertcovariance=[],scalecovariance=None):

		self.logger.info('Setting covariance matrix: {}.'.format(path_covariance))
		#if self.remove_sn: self.logger.info('Removing shotnoise.')
		#else: self.logger.info('Keeping shotnoise.')
		#get_cov = utils.load_covariance(path_covariance,remove_sn=self.remove_sn)
		get_cov = utils.load_covariance(path_covariance,remove_sn=True)
		self.nobs = get_cov('nobs')
		xmask,self.xcovariance,self.mean,self.astddev = [],[],[],[]
		for ell in self.ells:
			x = get_cov('x',estimator=self.estimator,recon=self.recon,mode=self.mode,ell=ell)
			self.logger.info('Setting {0}-range {1[0]:.4g} - {1[1]:.4g} of covariance for ell = {2}.'.format(self.xlabel,self.xrange[ell],ell))
			xmask.append((x>=self.xrange[ell][0]) & (x<=self.xrange[ell][-1]))
			self.xcovariance.append(x[xmask[-1]])
			self.astddev.append(get_cov('stddev',estimator=self.estimator,recon=self.recon,mode=self.mode,ell=ell)[xmask[-1]])
			self.mean.append(get_cov('mean',estimator=self.estimator,recon=self.recon,mode=self.mode,ell=ell)[xmask[-1]])

		self.nbins = sum(map(len,self.xcovariance))
		self.mean = scipy.concatenate(self.mean)
		self.logger.info('Covariance parameters: (nbins,nobs) = ({:d},{}).'.format(self.nbins,self.nobs))
		self.covariance = get_cov('covariance',estimator=self.estimator,recon=self.recon,mode=self.mode,ell=self.ells)

		self.set_invcovariance(xmask)

	@utils.classparams
	def set_invcovariance(self,xmask,invertcovariance=[],scalecovariance=None):

		del self.params['xmask']
		xmaskall = scipy.concatenate(xmask)

		if self.scale_data_covariance is not None:
			self.logger.info('Scaling covariance by {:.4f}.'.format(scalecovariance))
			self.covariance *= scalecovariance
		self.stddev = scipy.diag(self.covariance[scipy.ix_(xmaskall,xmaskall)])

		error_message = 'The covariance matrix is ill-conditionned. You may want to try the option sliced.'
		if 'sliced' in self.invert_covariance:
			self.logger.info('Slicing covariance.')
			self.covariance = self.covariance[scipy.ix_(xmaskall,xmaskall)]
			error_message = 'The covariance matrix is ill-conditionned. You may want to try the option block.'

		self.covariance = self.covariance.astype(scipy.float64) #make sure we have enough precision

		if 'diagonal' in self.invert_covariance:
			self.logger.info('Inverting diagonal matrix/blocks.')
			def inv(A):
				return scipy.diag(1./scipy.diag(A))
		elif 'cholesky' in self.invert_covariance:
			self.logger.info('Inverting using Choleskys decomposition.')
			def inv(A):
				c = linalg.inv(linalg.cholesky(A)) #using Cholesky's decomposition
				return scipy.dot(c.T,c)
		else:
			self.logger.info('Inverting using linalg inversion.')
			def inv(A):
				return linalg.inv(A)

		if 'block' in self.invert_covariance:
			self.logger.info('Inverting by block.')
			if 'sliced' in self.invert_covariance: blocksize = scipy.cumsum([0] + map(scipy.sum,xmask))
			else: blocksize = scipy.cumsum([0] + map(len,xmask))
			blocks = [[self.covariance[i1:i2,j1:j2] for j1,j2 in zip(blocksize[:-1],blocksize[1:])] for i1,i2 in zip(blocksize[:-1],blocksize[1:])]
			self.invcovariance = utils.blockinv(blocks,inv=inv)
			error_message = 'The covariance matrix is ill-conditionned. You have to provide a better estimate.'
		else:
			self.invcovariance = inv(self.covariance)

		diff = self.covariance.dot(self.invcovariance)-scipy.eye(self.covariance.shape[0])
		diff = scipy.absolute(diff).max()
		self.logger.info('Inversion computed to absolute precision {:.4g}.'.format(diff))
		if diff > 1.: raise LinAlgError(error_message)

		if 'sliced' not in self.invert_covariance:
			self.logger.info('Slicing covariance.')
			self.invcovariance = self.invcovariance[scipy.ix_(xmaskall,xmaskall)]
	
	def fisher_info(self):
		return self.mean.dot(self.precision).dot(self.mean.T)
	
	def D(self):
		return (self.nbins+1.)/(self.nobs-1.)
	
	def A(self):
		return 2./(self.nobs-self.nbins-1.)/(self.nobs-self.nbins-4.)

	def B(self):
		return (self.nobs-self.nbins-2.)/2.*self.A()

	def C1(self,f=1.):
		return self.A()*f + self.B()*f*(self.nbins+1.)

	def C2(self,f=1.):
		return self.A()*f**2 + self.B()*f*(self.neff+f)

	def set_lkl(self):

		self.precision = self.invcovariance.copy()
		# Percival et al. 2013 arXiv: 1312.4841v1
		if 'hartlap' in self.invert_covariance:
			factor = 1.-self.D()
			self.logger.info('Applying hartlap factor {:.4f} to precision matrix.'.format(factor))
			self.logger.info('Using parameters: (nbins,nobs) = ({:d},{}).'.format(self.nbins,self.nobs))
			if not (factor > 0.) & (factor < 1.):
				raise ValueError('There is something wrong with the Hartlap factor. Are you sure the covariance matrix is built from mocks?')
			self.precision *= factor
		if self.type_lkl == 'tdistrib':
			self.logger.info('Setting t-likelihood for {} {}.'.format(self.__class__.__name__,self.id))
			def lnlkl(**kwargs):
				delta = self.data-self._model
				return -self.nobs/2.*scipy.log(1. + delta.dot(self.invcovariance).dot(delta.T)/(self.nobs - 1.))
		else:
			self.logger.info('Setting gaussian likelihood for {} {}.'.format(self.__class__.__name__,self.id))
			def lnlkl(**kwargs):
				delta = self.data-self._model
				return -0.5*delta.dot(self.precision).dot(delta.T)
		self.lnlkl = lnlkl
	
	def set_neff(self,fisher_info):
		info = self.fisher_info()
		self.neff = 0.
		for par in self.vary:
			ff = info/fisher_info[par]
			self.logger.info('Fisher fraction of {:.4f} for parameter {}.'.format(ff,par))
			self.neff += ff
		self.logger.info('Effective number of parameters in {} is {:.4f}.'.format(self.id,self.neff))

	def set_scale_parameter_covariance(self,C1,C2,D):
		self.scale_parameter_covariance = {par:1. for par in self.vary}
		self.scale_mock_parameter_covariance = {par:1. for par in self.vary}
		# Percival et al. 2013 arXiv: 1312.4841v1
		if (self.type_lkl != 'tdistrib') and ('hartlap' in self.invert_covariance):
			for par in self.vary:
				self.scale_parameter_covariance[par] = (1. + C1[par] - C2[par])/(1. + C2[par])
				self.logger.info('I will apply factor {:.4f} to variance of parameter {}.'.format(self.scale_parameter_covariance[par],par))
				self.scale_mock_parameter_covariance[par] = self.scale_parameter_covariance[par]/D[par]
				self.logger.info('You will have to apply factor {:.4f} to mock variance of parameter {}.'.format(self.scale_mock_parameter_covariance[par],par))
		
	@property
	def xlabel(self):
		if 'spectrum' in self.estimator: return 'k'
		return 's'

	@property
	def xrange(self):
		return self.params['xrange']

	@property
	def x(self):
		return self.xdata

	@property
	def remove_sn(self):
		return self.params['remove_sn']

	@property
	def invert_covariance(self):
		return self.params['invertcovariance']

	@property
	def scale_data_covariance(self):
		return self.params['scalecovariance']

	@property
	def type_lkl(self):
		return self.params.get('typelkl','chi2')

	@property
	def estimator(self):
		"""
		Placeholder to remind that this function needs to be defined for a new likelihood.
		Raises
		------
		NotImplementedError
		"""
		raise NotImplementedError('Must implement property estimator() in your {}.'.format(self.__class__.__name__))

	@property
	def recon(self):
		"""
		Placeholder to remind that this function needs to be defined for a new likelihood.
		Raises
		------
		NotImplementedError
		"""
		raise NotImplementedError('Must implement property recon() in your {}.'.format(self.__class__.__name__))

	@property
	def mode(self):
		"""
		Placeholder to remind that this function needs to be defined for a new likelihood.
		Raises
		------
		NotImplementedError
		"""
		raise NotImplementedError('Must implement property mode() in your {}.'.format(self.__class__.__name__))


	def check_compatibility(self):
		for ill,ell in enumerate(self.ells):
			if not scipy.allclose(self.xdata[ill],self.xcovariance[ill],rtol=1e-05,atol=1e-03,equal_nan=False):
				raise ValueError('Data and covariance matrix bins are not compatible for ell = {}.'.format(ell))

	@utils.classparams
	def setup(self,krange,path_covariance=None,invertcovariance=[]):
		self.set_model()
		self.set_ells()
		self.set_covariance(path_covariance)

	def set_model(self):
		"""
		Placeholder to remind that this function needs to be defined for a new likelihood.
		Raises
		------
		NotImplementedError
		"""
		raise NotImplementedError('Must implement method set_model() in your {}.'.format(self.__class__.__name__))

	def set_ells(self):
		self.ells = [ell for ell in self.geometry.ellsout if ell in self.xrange]
		self.ills = [ill for ill,ell in enumerate(self.geometry.ellsout) if ell in self.xrange]
		self.logger.info('I will fit ells = {}.'.format(self.ells))
		for ell in self.krange:
			if not ell in self.ells:
				self.logger.warning('I cannot fit ell = {:d} as it is not predicted by {}.'.format(ell,self.geometry.__class__.__name__))

	def init(self,path_data=None,path_covariance=None):
		if not hasattr(self,'parid'): self.parid = self.id
		if not self.sorted: self.sorted = pyspectrum.utils.sorted_parameters(utils.get_parameters(self.fitargs))
		self._kwargs,self._amodel = {},[]
		self.set_model()
		if path_data is None:
			if path_covariance is None:
				self.logger.info('Using preloaded data {} and covariance {}.'.format(self.params['path_data'],self.params['path_covariance']))
			else:
				self.set_data_covariance(path_covariance)
				self.logger.info('Using preloaded covariance {}.'.format(self.params['path_covariance']))
		else:
			if path_covariance is not None: self.set_covariance(path_covariance)
			else: self.logger.info('Using preloaded covariance {}.'.format(self.params['path_covariance']))
			self.set_data(path_data)
		#self.check_compatibility()
		if 'spectrum' in self.estimator:
			self.geometry.set_kout(kout={ell:self.xdata[ill] for ill,ell in enumerate(self.ells)})
		else:
			self.geometry.set_sout(sout={ell:self.xdata[ill] for ill,ell in enumerate(self.ells)})
		self.set_lkl()

	def prepare(self,others):
		fisher_info = {}
		for other in others:
			if isinstance(other,self.__class__):
				info = other.fisher_info()
				for par in other.vary:
					if par in fisher_info:
						fisher_info[par] += info
					else:
						fisher_info[par] = info
		C1,C2,D = {par:0. for par in self.vary},{par:0. for par in self.vary},{par:0. for par in self.vary}
		for other in others:
			if isinstance(other,self.__class__):
				info = other.fisher_info()
				other.set_neff(fisher_info)
				for par in other.vary:
					if par in self.vary:
						f = info/fisher_info[par]
						C1[par] += other.C1(f=f)
						C2[par] += other.C2(f=f)
						D[par] += f*(1.-other.D())
		self.set_scale_parameter_covariance(C1,C2,D)

	def getstate(self):
		state = super(BaseLikelihoodClustering,self).getstate()
		for key in ['xdata','adata','data','xcovariance','astddev','mean','invcovariance','nobs','ells','ills']:
			if hasattr(self,key): state[key] = getattr(self,key)
		state['__class__'] = self.__class__.__name__
		return state

	def to_model_par(self,key):
		return key.replace('_{}'.format(self.parid),'')

	def to_model_kwargs(self,**kwargs):
		toret = {}
		for par in self.parameters:
			mpar = self.to_model_par(par)
			toret[mpar] = kwargs[par]
		toret.update(self.other_kwargs)
		return toret
	
	def set_fitargs(self,fitargs,peculiar=[]):
		params = utils.get_parameters(fitargs)
		for key in peculiar:
			for prefix in [''] + utils.MINUIT_PREFIX + utils.OTHER_PREFIX:
				if prefix+key in fitargs:
					if prefix == 'latex_':
						fitargs['{}{}_{}'.format(prefix,key,self.parid)] = '{}^{{\\rm {}}}'.format(fitargs.pop(prefix+key),self.parid)
					else:
						fitargs['{}{}_{}'.format(prefix,key,self.parid)] = fitargs.pop(prefix+key)
		self.fitargs = fitargs

	def amodel(self,**kwargs):
		mkwargs = self.to_model_kwargs(**kwargs)
		if mkwargs == self._kwargs:
			return self._amodel
		for new,old in zip(['Ns'],['As']):
			if old in mkwargs: mkwargs[new] = mkwargs.pop(old)*self.shotnoise
		mkwargs['normwindow'] = self.norm
		self._amodel = self._amodel_(**mkwargs)
		self._kwargs = mkwargs
		return self._amodel

	def model(self,**kwargs):
		self._model = scipy.concatenate(self.amodel(**kwargs),axis=-1)
		return self._model

	def derived_parameters(self,values,errors={}):
		sigma8 = self.model_sp.sigma8*values.get('rsigma8',1.)
		dvalues = {par:1.*val for par,val in values.items() if par in self.fitargs}
		def mults8(key):
			for par in ['f','b']:
				if key.startswith(par) and not key.startswith('beta') and 'sigma8' not in key: return True
			return False
		for key in dvalues.keys():
			if mults8(key): dvalues['{}sigma8'.format(key)] = values[key]*sigma8
		derrors = {par:1.*val for par,val in errors.items() if par in self.fitargs}
		for key in derrors.keys():
			if key in self.vary: derrors[key] = scipy.sqrt(self.scale_parameter_covariance[key])*errors[key]
			if mults8(key): derrors['{}sigma8'.format(key)] = derrors[key]*sigma8
		dvalues['sigma8'] = sigma8
		tmp = values.values()[0]
		if not scipy.isscalar(tmp): dvalues['sigma8'] = scipy.ones_like(tmp)*dvalues['sigma8'] # to get the correct shape
		derrors['sigma8'] = 0.*dvalues['sigma8']
		dlatex = {par:val for par,val in self.latex.items()}
		for key in dlatex.keys():
			if mults8(key): dlatex['{}sigma8'.format(key)] = '{}\\sigma_{{8}}'.format(self.latex[key])
		dlatex['sigma8'] = '\\sigma_{{8}}'
		dsorted = pyspectrum.utils.sorted_parameters(dlatex.keys())
		params = {key:val for key,val in self.params.items()}
		params['scalecov'] = self.scale_parameter_covariance
		params['scalemockcov'] = self.scale_mock_parameter_covariance

		return dvalues,derrors,dlatex,dsorted,params

	@property
	def ndata(self):
		return 1

class BaseLikelihoodSpectrumMultipoles(BaseLikelihoodClustering):
	
	logger = logging.getLogger('BaseLikelihoodSpectrumMultipoles')

	@property
	def krange(self):
		return self.params['krange']	

	@property
	def xrange(self):
		return self.krange

	@property
	def k(self):
		return self.x

	@property
	def kdata(self):
		return self.xdata

	@property
	def estimator(self):
		return 'spectrum'

	@property
	def mode(self):
		return 'multipole'

	@property
	def recon(self):
		return self.params.get('recon',None)

	@recon.setter
	def recon(self,x):
		self.params['recon'] = x


class LikelihoodClusteringCombiner(BaseLikelihoodClustering):

	logger = logging.getLogger('LikelihoodClusteringCombiner')

	def combine(self,likelihoods,**kwargs):
		self.likelihoods = likelihoods
		for ilikelihood,likelihood in enumerate(self):
			for key in kwargs:
				setattr(likelihood,key,kwargs[key][ilikelihood])

	@utils.classparams
	def setup(self,path_covariance=None,invertcovariance=[]):
		for likelihood in self:
			likelihood.set_model()
			likelihood.set_ells()
		self.set_covariance(path_covariance)

	def init(self,path_data,path_covariance):
		self.data = []
		if not isinstance(path_data,list): path_data = [path_data]
		path_data = path_data + [None]*len(self.likelihoods)
		self.fitargs,self.sorted = {},[]
		for likelihood,path in zip(self.likelihoods,path_data):
			likelihood.init(path_data=path,path_covariance=path_covariance)
			self.data.append(likelihood.data)
			self.sorted += [par for par in likelihood.sorted if par not in self.sorted]
			self.fitargs.update(likelihood.fitargs)
		self.data = scipy.concatenate(self.data)
		self.id = '_'.join([likelihood.id for likelihood in self])
		if path_covariance is not None: self.set_covariance(path_covariance)
		else: self.logger.info('Using covariance {}.'.format(self.params['path_covariance']))
		if all(path is None for path in path_data):
			self.invcovariance *= self.nobs
			self.astddev /= scipy.sqrt(self.nobs)
			self.logger.info('Dividing covariance matrix by the number of observations: {:d}.'.format(self.nobs))
		self.set_lkl()

	def set_default(self):
		for likelihood in self: 
			likelihood.set_default()

	def __iter__(self):
		return self.likelihoods.__iter__()

	@property
	def ndata(self):
		return len(self.likelihoods)

	@utils.classparams	
	def set_covariance(self,path_covariance,invertcovariance=[],scalecovariance=None):
		
		self.logger.info('Setting covariance matrix: {}.'.format(path_covariance))
		#if self.remove_sn: self.logger.info('Removing shotnoise.')
		#else: self.logger.info('Keeping shotnoise.')
		#get_cov = utils.load_covariance(path_covariance,remove_sn=self.remove_sn)
		get_cov = utils.load_covariance(path_covariance,remove_sn=True)
		self.nobs = get_cov('nobs')
		xmask,self.xcovariance,self.mean,self.astddev = [],[],[],[]
		list_estimator,list_recon,list_mode,list_ells = [],[],[],[]
		for likelihood in self:
			xlabel = 'k' if 'spectrum' in likelihood.estimator else 's'
			for ell in likelihood.ells:
				x = get_cov('x',estimator=likelihood.estimator,recon=likelihood.recon,mode=likelihood.mode,ell=ell)
				self.logger.info('Setting {0}-range {1[0]:.4g} - {1[1]:.4g} of covariance for ell = {2}.'.format(xlabel,likelihood.xrange[ell],ell))
				xmask.append((x>=likelihood.xrange[ell][0]) & (x<=likelihood.xrange[ell][-1]))
				self.xcovariance.append(x[xmask[-1]])
				self.astddev.append(get_cov('stddev',estimator=likelihood.estimator,recon=likelihood.recon,mode=likelihood.mode,ell=ell)[xmask[-1]])
				self.mean.append(get_cov('mean',estimator=likelihood.estimator,recon=likelihood.recon,mode=likelihood.mode,ell=ell)[xmask[-1]])
				list_estimator.append(likelihood.estimator)
				list_recon.append(likelihood.recon)
				list_mode.append(likelihood.mode)
				list_ells.append(ell)

		self.nbins = sum(map(len,self.xcovariance))
		self.mean = scipy.concatenate(self.mean)
		self.logger.info('Covariance parameters: (nbins,nobs) = ({:d},{}).'.format(self.nbins,self.nobs))
		self.covariance = get_cov('covariance',estimator=list_estimator,recon=list_recon,mode=list_mode,ell=list_ells)

		self.set_invcovariance(xmask)

	def lnprior(self,**kwargs):
		return sum(likelihood.lnprior(**kwargs) for likelihood in self)
		
	def gradient(self,parameters,**kwargs):
		toret = {par:0. for par in parameters}
		ixstart = 0
		for likelihood in self:
			pars = [par for par in parameters if par in likelihood.parameters]
			ixend = ixstart
			if pars:
				gradient = likelihood.gradient(pars,**kwargs)
				ixend = ixstart + gradient.values()[0].shape[-1]
				for par in pars:
					toret[par] += scipy.pad(gradient[par],pad_width=(ixstart,self.nbins-ixend),mode='constant',constant_values=0.)
			ixstart = ixend
		return toret

	def model(self,**kwargs):
		self._model = scipy.concatenate([likelihood.model(**kwargs) for likelihood in self],axis=-1)
		return self._model

	def plot(self,*args,**kwargs):
		for likelihood in self:
			likelihood.plot(*args,**kwargs)

	def derived_parameters(self,values,errors={},latex={}):
		dvalues,derrors,dlatex,dsorted,dparams = {},{},{},[],{}
		for likelihood in self:
			likelihood.scale_parameter_covariance = self.scale_parameter_covariance
			likelihood.scale_mock_parameter_covariance = self.scale_mock_parameter_covariance
			values_,errors_,latex_,sorted_,dparams_ = likelihood.derived_parameters(values,errors=errors)
			dvalues.update(values_);derrors.update(errors_);dlatex.update(latex_);dparams.update(dparams_)
			dsorted += [par for par in sorted_ if par not in dsorted]
		dparams.update(self.params)
		return dvalues,derrors,dlatex,dsorted,dparams

	"""
	def getstate(self):
		state = super(LikelihoodClusteringCombiner,self).getstate()
		state['likelihoods'] = []
		for likelihood in self:
			state['likelihoods'].append({'__name__':likelihood.__class__.__name__,'__dict__':likelihood.getstate()})
		return state

	@utils.setstateclass
	def setstate(self,state):
		self = super(LikelihoodClusteringCombiner,self).setstate(state)
		for ilikelihood,likelihood in self.likelihoods:
			self.likelihoods[ilkelihood] = getattr(montelss.likelihoods,likelihood['__name__']).setstate(likelihood['__dict__'])
	"""

class LikelihoodClustering(BaseLikelihoodClustering):

	logger = logging.getLogger('LikelihoodClustering')
	
	@classmethod
	@utils.loadclass
	def load(self,state):
		cls = getattr(montelss.likelihoods,state.pop('__class__','BaseLikelihoodClustering'))
		self.setstate(state)
		new = object.__new__(cls)
		new.__dict__ = self.__dict__
		return new
