import logging
import os
import scipy
import utils

class Likelihood(object):

	logger = logging.getLogger('Likelihood')
	_ARGS = ['id','vary','nvary','fitted','nfitted','fixed','nfixed','nbins','sorted','fitargs','latex','parameters']
	
	def __init__(self,**params):
		self.name = self.__class__.__name__
		self.folder = os.path.abspath(os.path.join(utils.path['likelihoods'],self.name))
		self.params = params
		self.fitargs = {}
		self.sorted = []
		self.id = 'default'
		self.nbins = 0
	
	def lnlkl(self,**kwargs):
		"""
		Placeholder to remind that this function needs to be defined for a new likelihood.
		Raises
		------
		NotImplementedError
		"""
		raise NotImplementedError('Must implement method lnlkl() in your {}.'.format(self.__class__.__name__))
		

	def model(self,**kwargs):
		"""
		Placeholder to remind that this function needs to be defined for a new likelihood.
		Raises
		------
		NotImplementedError
		"""
		raise NotImplementedError('Must implement method model() in your {}.'.format(self.__class__.__name__))

	def gradient(self,parameters,**kwargs):
		if parameters: raise NotImplementedError('Must implement method gradient() in your {} for parameters {}.'.format(self.__class__.__name__,parameters))
		return {}

	def matrices_least_squares(self,parameters,**kwargs):
		solved = self.solved('lsq')
		gradient = self.gradient(solved,**kwargs)
		if solved: delta = self.data - self._model
		H = []
		for par in parameters:
			if par in solved: H.append(gradient[par])
			else: H.append(scipy.zeros(len(self.data),dtype='f8'))
		H = scipy.array(H)
		HV = H.dot(self.precision)
		P = HV.dot(H.T)
		Q = HV.dot(delta)
		return P,Q

	def solve_least_squares(self,**kwargs):
		solved = self.solved('lsq')
		if not solved: return {}
		P,Q = self.matrices_least_squares(solved,**kwargs)
		theta = scipy.linalg.inv(P).dot(Q)
		return {par:val for par,val in zip(solved,theta)}
	
	def update_least_squares(self,**kwargs):
		if self.solved('lsq'):
			self.model(**kwargs)
		
	def solve_update_analytic(self,**kwargs):
		for par in self.solved('lsq'): kwargs[par] = 0.
		self.model(**kwargs)
		toret = self.solve_least_squares(**kwargs)
		kwargs.update(toret)
		self.update_least_squares(**kwargs)
		return toret

	def lnprior(self,**kwargs):
		return 0.

	def lnposterior(self,**kwargs):
		return self.lnlkl(**kwargs) + self.lnprior(**kwargs)

	def chi2(self,**kwargs):
		return -2.*self.lnposterior(**kwargs)

	def init(self):
		pass

	def prepare(self,others):
		pass

	def derived_parameters(self,values,errors={}):
		return values,errors,1.
		
	@property
	def parameters(self):
		return utils.get_parameters(self.fitargs)
	
	@property
	def vary(self):
		toret = utils.get_vary_parameters(self.fitargs)
		toret = [key for key in self.sorted if key in toret] + [key for key in toret if key not in self.sorted]
		return toret

	@property
	def fitted(self):
		toret = utils.get_fitted_parameters(self.fitargs)
		toret = [key for key in self.sorted if key in toret] + [key for key in toret if key not in self.sorted]
		return toret

	def solved(self,method='all'):
		toret = utils.get_solved_parameters(self.fitargs,method=method)
		toret = [key for key in self.sorted if key in toret] + [key for key in toret if key not in self.sorted]
		return toret
	
	@property
	def fixed(self):
		toret = utils.get_fixed_parameters(self.fitargs)
		toret = [key for key in self.sorted if key in toret] + [key for key in toret if key not in self.sorted]
		return toret

	@property
	def latex(self):
		return utils.get_latex(self.fitargs)
	
	@property
	def nvary(self):
		return len(self.vary)
	
	@property
	def nfitted(self):
		return len(self.fitted)

	@property
	def nfixed(self):
		return len(self.fixed)
		
	def set_default(self):
		for par in self.parameters:
			if par not in self.latex:
				latex = utils.par_to_default_latex(par)
				self.logger.info('Adding default latex {} for parameter {}.'.format(latex,par))
				self.latex[par] = latex
			if par not in self.sorted:
				self.logger.info('Appending parameter {} to sorted.'.format(par))
				self.sorted.append(par)

	@utils.getstateclass
	def getstate(self,state):
		for key in self._ARGS:
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
	
	def copy(self):
		return self.__class__.loadstate(self.__dict__)
		
	def __radd__(self,other):
		if other == 0: return self.copy()
		return self.__add__(other)
		       
	def __add__(self,other):
		new = Likelihood()
		new.id = '{}_{}'.format(self.id,other.id)
		for inst in [self,other]:
			new.fitargs.update(inst.fitargs)
			new.sorted += [par for par in inst.sorted if par not in new.sorted]
		new.nbins = self.nbins + other.nbins
		def model(**kwargs):
			self.model(**kwargs)
			other.model(**kwargs)
		new.model = model
		def lnposterior(**kwargs):
			return self.lnposterior(**kwargs) + other.lnposterior(**kwargs)
		new.lnposterior = lnposterior
		def matrices_least_squares(*args,**kwargs):
			P1,Q1 = self.matrices_least_squares(*args,**kwargs)
			P2,Q2 = other.matrices_least_squares(*args,**kwargs)
			return P1+P2,Q1+Q2
		new.matrices_least_squares = matrices_least_squares
		def update_least_squares(**kwargs):
			self.update_least_squares(**kwargs)
			other.update_least_squares(**kwargs)
		new.update_least_squares = update_least_squares
		return new

