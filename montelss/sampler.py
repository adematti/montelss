import os
import sys
import functools
import copy
import logging
import scipy
from likelihood import Likelihood
import utils

def args_to_kwargs(func):
	@functools.wraps(func)
	def wrapper(self,*args,**kwargs):
		kwargs.update({key:val for key,val in zip(self.parameters,args)})
		return func(self,**kwargs)
	return wrapper

class Chain(object):

	logger = logging.getLogger('Chain')

	def __init__(self,values={},parameters=None,sorted=[],params={}):
	
		self.values = {}
		self.sorted = sorted
		if parameters is None: parameters = values.keys()
		for key in parameters:
			self.values[key] = scipy.asarray(values[key])
		self.params = params

	@classmethod
	def from_array(cls,array,parameters,sorted=[],params={}):
		return cls({par:col for par,col in zip(parameters,array)},sorted=sorted,params=params)
	
	@classmethod
	def empty(cls,parameters=[],shape=(0,),params={},**kwargs):
		return cls({par:scipy.empty(shape,**kwargs) for par in parameters},params=params)

	def indices(self):
		return scipy.arange(self.size)
	
	def getstate(self,parameters=None):
		return {'values':self.as_dict(parameters),'sorted':self.sorted,'params':self.params}

	@utils.setstateclass
	def setstate(self,state):
		pass

	@classmethod
	def loadstate(cls,state):
		self = cls()
		self.setstate(state)
		self.logger.info('Chain shape: (params,steps,walkers) = {}.'.format(self.shape))
		return self

	def copy(self):
		return self.__class__.loadstate(self.getstate())

	def deepcopy(self):
		import copy
		return copy.deepcopy(self)
	
	def slice(self,islice=0,nslices=1):
		size = len(self)
		return self[islice*size//nslices:(islice+1)*size//nslices]

	def __getitem__(self,name):
		if isinstance(name,(str,unicode)):
			if name in self.parameters:
				return self.values[name]
			else:
				raise KeyError('There is no parameter {} in the chain.'.format(name))
		else:
			return self.__class__({par:self.values[par][name] for par in self.parameters},params=self.params)

	def __setitem__(self,name,item):
		if isinstance(name,(str,unicode)):
			self.values[name] = item
		else:
			for key in self.parameters:
				self.values[key][name] = item

	def __delitem__(self,name):
		del self.values[name]

	def	__contains__(self,name):
		return name in self.values
	
	def __iter__(self):
		for par in self.values:
			yield par
	
	def __str__(self):
		return str(self.values)

	@property
	def parameters(self):
		return self.values.keys()

	@property
	def nparams(self):
		return len(self.parameters)

	@property
	def nsteps(self):
		return len(self[self.parameters[0]])

	@property
	def nwalkers(self):
		return self[self.parameters[0]].shape[-1]

	@property
	def shape(self):
		return (self.nparams,self.nsteps,self.nwalkers)

	def __len__(self):
		return self.nsteps

	@property
	def size(self):
		return len(self)

	def remove(self,name):
		del self.values[name]
	
	def __radd__(self,other):
		if other == 0: return self
		else: return self.__add__(other)
	
	def __add__(self,other):
		new = {}
		parameters = [par for par in self.parameters if par in other.parameters]
		for par in parameters:
			new[par] = scipy.concatenate([self[par],other[par]],axis=0)
		import copy
		params = copy.deepcopy(self.params)
		params.update(copy.deepcopy(other.params))
		return self.__class__(new,parameters=parameters,params=params)

	def get(self,par,walkers=None,flat=False):
		toret = self[par]
		if walkers is not None: toret = toret[:,walkers]
		if flat: toret = toret.flatten()
		return toret

	def as_dict(self,parameters=None,walkers=None,flat=False):
		if parameters is None: parameters = self.parameters
		return {par:self.get(par,walkers=walkers,flat=flat) for par in parameters}

	def as_array(self,parameters=[],walkers=None,flat=False):
		toret = self.as_dict(parameters=parameters,walkers=walkers,flat=flat)
		return scipy.array([toret[par] for par in parameters])

class BaseSampler(object):
	
	logger = logging.getLogger('BaseSampler')
	_ARGS = ['ids','chain','niterations','naccepted','fitted_values','fitted_errors','vary_covariance','fixed_values']
	
	def __init__(self,**params):
		self.params = params
		self.set_rng()
	
	@utils.classparams
	def set_rng(self,seed=None):
		self.logger.debug('Resetting random state with seed {}.'.format(seed))
		self.rng = scipy.random.RandomState(seed=seed)

	def set_likelihoods(self,likelihoods):
		if not isinstance(likelihoods,list): self.likelihoods = [likelihoods]
		else: self.likelihoods = likelihoods

	def setup(self,likelihoods):
		self.set_likelihoods(likelihoods)
		self.ids = []
		for like in self.likelihoods:
			like.prepare(self.likelihoods)
			self.ids.append(like.id)

		totallikelihood = sum(self.likelihoods)
		totallikelihood.set_default()
		for key in totallikelihood._ARGS + ['lnposterior','solve_update_analytic']:
			setattr(self,key,getattr(totallikelihood,key))

		self.latex['lnposterior'] = '\\ln{\\mathcal{L}}'
		self.latex['chi2'] = '\\chi^{2}'

	@utils.setstateclass
	def setstate(self,state):
		self.__init__(**state['params'])
		self.chain = Chain.loadstate(self.chain)
		self.rng = scipy.random.RandomState()
		self.rng.set_state(self.rstate)

	@classmethod
	@utils.loadstateclass	
	def loadstate(self,state):
		self.setstate(state)
		return self

	@utils.getstateclass
	def getstate(self,state):
		for key in Likelihood._ARGS + self._ARGS:
			if hasattr(self,key):
				state[key] = getattr(self,key)
				if key == 'chain': state[key] = state[key].getstate()
		state['sampler'] = self.__class__.__name__
		state['rstate'] = self.rng.get_state()
		return state

	@classmethod
	@utils.loadclass
	def load(self,state):
		self.setstate(state)
		return self

	@utils.saveclass
	def save(self):
		return self.getstate()

	def gaussian_diag(self):
		return [self.rng.normal(loc=self.fitted_values,scale=self.fitted_errors,size=self.nfitted) for i in range(self.nwalkers)]

	def gaussian_cov(self):
		return [self.rng.multivariate_normal(self.fitted_values,self.fitted_covariance) for i in range(self.nwalkers)]

	@property
	def nwalkers(self):
		return self.params['nwalkers']

	@property
	def nsteps(self):
		return self.params['nsteps']

	@property
	def nflush(self):
		return self.params['nflush']
	
	@utils.classparams
	def reset(self,nwalkers,pdf='gaussian_diag'):
		if pdf == 'gaussian_diag': self.first = self.gaussian_diag()
		elif pdf == 'gaussian_cov': self.first = self.gaussian_cov()
		else: raise ValueError('You should choose between gaussian_diag and gaussian_cov to initialise MCMC.')
		self.niterations = 0
		self.naccepted = scipy.zeros(self.nwalkers,dtype=scipy.float64)
		self.chain = 0

	@utils.classparams
	def init_from_bestfit(self,bestfit,nwalkers,pdf='gaussian_diag',seed=None):
		self.logger.info('Setting initial values and errors from migrad.')
		self.fitted_values = [bestfit['values'][v] for v in self.fitted]
		self.fitted_errors = []
		for v in self.fitted:
			if 'minos' in bestfit and v in bestfit['minos']['upper']:
				self.logger.info('Setting initial errors for {} from minos.'.format(v))
				self.fitted_errors.append((bestfit['minos']['upper'][v]-bestfit['minos']['lower'][v])/2.)
			else:
				self.fitted_errors.append(bestfit['errors'][v])
		self.fixed_values = {v:bestfit['values'][v] for v in self.fixed}
		self.reset()

	@utils.classparams
	def init_from_params(self,bestfit,nwalkers,pdf='gaussian_diag',seed=None):
		self.logger.info('Setting initial values and errors from parameters.')
		values = utils.get_values(self.fitargs)
		errors = utils.get_errors(self.fitargs)
		self.fitted_values = [values[v] for v in self.fitted]
		self.fitted_errors = [errors[v] for v in self.fitted]
		self.fixed_values = {v:values[v] for v in self.fixed}
		self.reset()

	def init_from_previous(self):
		self.logger.info('Setting initial values and errors from previous run.')
		self.first = scipy.asarray([self.chain[par][-1] for par in self.fitted]).T

	def lnposteriorargs(self,args,**fixed):
		kwargs = {}
		kwargs.update({key:val for key,val in zip(self.fitted,args)})
		kwargs.update(fixed)
		solved = self.solve_update_analytic(**kwargs)
		kwargs.update(solved)
		solved.update(fixed)
		return self.lnposterior(**kwargs),solved

	def get_dchain(self,chain):
		values = chain.as_dict()
		for like in self.likelihoods:
			values_,errors,scalecov,latex,sorted = like.derived_parameters(chain.as_dict())
			values.update(values_)
			self.latex.update(latex)
			chain.sorted += [par for par in sorted if par not in chain.sorted]
		chain.values = values
		chain.params['scalecov'] = scalecov
		chain['chi2'] = -2.*chain['lnposterior']
		return chain

	@property
	def acceptance_fraction(self):
		return self.naccepted/self.niterations

	def _run_(self):
		"""
		Placeholder to remind that this function needs to be defined for a new BaseSampler.
		Raises
		------
		NotImplementedError
		"""
		raise NotImplementedError('Must implement method _run_() in your {}.'.format(self.__class__.__name__))
		
	def get_pool(self,*args,**kwargs):
		return None

	@utils.classparams
	def run(self,nsteps,save,nflush=1000):
		sflush = int(scipy.ceil(nsteps*1./nflush))
		nrun = 0
		self.pool = self.get_pool()
		for iflush in range(sflush):
			self._run_(nsteps=min(nflush,nsteps-nrun))
			self.first = scipy.asarray([self.chain[par][-1] for par in self.fitted]).T
			self.solved = []
			nrun += nflush
			if nrun < nsteps: self.save()
		if self.pool is not None: self.pool.close()
		self.params['nsteps'] = nsteps

	def check_chain(self):
		if self.chain and (self.chain.nwalkers != self.nwalkers):
			raise ValueError('You cannot restart from a chain with {:d} != {:d} walkers.'.format(self.chain.nwalkers,self.nwalkers))

class Sampler(BaseSampler):

	@classmethod
 	@utils.loadstateclass   
	def loadstate(self,state):
 		new = object.__new__(globals()[state['sampler']])
		new.setstate(state)
		return new

	@classmethod
 	@utils.loadclass   
	def load(self,state):
 		new = object.__new__(globals()[state['sampler']])
		new.setstate(state)
		return new

class EmceeSampler(BaseSampler):

	def set_from_sampler(self,sampler):
		chain = Chain.from_array(scipy.transpose(sampler.chain,axes=(-1,1,0)),parameters=self.fitted,sorted=self.sorted,params=self.params)
		blobs = sampler.blobs
		if blobs:
			self.logger.info('Setting solved and fixed values.')
			solved = blobs[0][0].keys()
			for par in solved:
				chain[par] = scipy.array([[blobs[istep][iwalker][par] for iwalker in range(self.nwalkers)] for istep in range(self.nsteps)])
		chain['lnposterior'] = sampler.lnprobability.T
		chain = self.get_dchain(chain)
		self.chain += chain
		self.niterations += sampler.iterations
		self.naccepted += sampler.naccepted
		self.rng = sampler._random

	@utils.classparams
	def get_pool(self,mpi=False,nthreads=1):
		import emcee
		from emcee.utils import MPIPool
		from pathos.multiprocessing import ProcessingPool as PPool
		#from multiprocessing import Pool as PPool
		if mpi:
			pool = MPIPool(loadbalance=True)
			if not pool.is_master():
				pool.wait()
				sys.exit(0)
			self.logger.info('Creating MPI pool with {:d} workers.'.format(pool.size+1))
		elif nthreads > 1:
			pool = PPool(nthreads)
			self.logger.info('Creating multiprocessing pool with {:d} threads.'.format(nthreads))
		else:
			pool = None
		return pool

class EnsembleSampler(EmceeSampler):

	logger = logging.getLogger('EnsembleSampler')

	@utils.classparams
	def _run_(self,nwalkers,nsteps,mpi=False,nthreads=1):
		import emcee
		self.check_chain()
		sampler = emcee.EnsembleSampler(self.nwalkers,self.nfitted,self.lnposteriorargs,kwargs=self.fixed_values,pool=self.pool)
		self.logger.info('Running MCMC with {:d} walkers for {:d} steps...'.format(nwalkers,nsteps))
		sampler.run_mcmc(self.first,nsteps,rstate0=self.rng.get_state())
		self.logger.info('Done.')
		self.set_from_sampler(sampler)
		self.logger.info('Mean acceptance fraction of {:.4f} ({:.4f}/{:d}).'.format(scipy.mean(self.acceptance_fraction),scipy.mean(self.naccepted),self.niterations))

class MHSampler(EmceeSampler):

	logger = logging.getLogger('MHSampler')

	def __init__(self,**params):
		params['nwalkers'] = 1
		super(MHSampler,self).__init__(**params)

	@utils.classparams
	def _run_(self,nsteps,covariance=None,burnin=0):
		import emcee
		self.set_covariance(covariance)
		self.check_chain()
		sampler = emcee.MHSampler(self.fitted_covariance,self.nfitted,self.lnposteriorargs,kwargs=self.fixed_values)
		self.logger.info('Running MCMC for {:d} steps...'.format(nsteps))
		sampler.run_mcmc(self.first[0],nsteps,rstate0=self.rng.get_state())
		self.logger.info('Done.')
		sampler._chain = sampler.chain[None,...]
		sampler._lnprob = sampler._lnprob[None,...]
		self.set_from_sampler(sampler)
		self.logger.info('Mean acceptance fraction of {:.4f} ({:.4f}/{:d}).'.format(scipy.mean(self.acceptance_fraction),scipy.mean(self.naccepted),self.niterations))

	@utils.classparams
	def set_covariance(self,covariance=None,burnin=0):
		if covariance == 'new':
			self.logger.info('Setting new covariance.')
			self.fitted_covariance = scipy.diag(self.fitted_errors)
		elif covariance == 'previous':
			self.logger.info('Measuring covariance from previous run, which is then ignored.')
			if burnin >= self.chain.size:
				raise ValueError('Burn-in {:d} is larger than chain size {:d}.'.format(burnin,self.chain.size))
			chain = self.chain[burnin:].as_array(self.fitted,flat=True)
			self.fitted_covariance = scipy.cov(self.chain[burnin:].as_array(self.fitted,flat=True),ddof=1)
			self.set_rng()
			self.reset(pdf='gaussian_cov')
		elif hasattr(self,'vary_covariance'):
			self.logger.info('Taking same covariance as previous run.')
		else:
			raise ValueError('I need a covariance matrix! Try covariance == "new" or covariance == "previous" if starting from previous run.')
		self.params['covariance'] = None
