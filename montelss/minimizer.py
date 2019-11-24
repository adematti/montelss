import os
import functools
import copy
import logging
import scipy
from scipy import stats
from likelihood import Likelihood
import utils

def args_to_kwargs(func):
	@functools.wraps(func)
	def wrapper(self,*args,**kwargs):
		kwargs.update({key:val for key,val in zip(self.parameters,args)})
		return func(self,**kwargs)
	return wrapper

def nsigmas_to_quantiles_1d(nsigmas):
	return stats.norm.cdf(nsigmas,loc=0,scale=1) - stats.norm.cdf(-nsigmas,loc=0,scale=1)

def nsigmas_to_deltachi2(nsigmas,ndof=1):
	quantile = nsigmas_to_quantiles_1d(nsigmas)
	return stats.chi2.ppf(quantile,ndof)

class Minimizer(object):
	
	logger = logging.getLogger('Minimizer')
	_MINOS_ARGS = ['lower','upper','lower_valid','upper_valid','is_valid']
	
	def __init__(self,**params):

		self.params = params
		self.bestfit = {'values':{},'errors':{}}
		self.profile = {}
		self.contour = {}

	def set_likelihoods(self,likelihoods):
		if not isinstance(likelihoods,list): self.likelihoods = [likelihoods]
		else: self.likelihoods = likelihoods

	@utils.classparams
	def setup(self,likelihoods,minuit={}):
		
		import iminuit
		self.set_likelihoods(likelihoods)
		del self.params['likelihoods']
		self.ids = []
		for likelihood in self.likelihoods:
			likelihood.prepare(self.likelihoods)
			self.ids.append(likelihood.id)

		totallikelihood = sum(self.likelihoods)
		totallikelihood.set_default()
		for key in totallikelihood._ARGS + ['chi2','solve_update_analytic']:
			setattr(self,key,getattr(totallikelihood,key))

		self.minuitargs = {}
		self.minuitargs.update(utils.get_minuit_args(self.fitargs))
		self.minuitargs.update({'forced_parameters':self.parameters})
		self.minuitargs.update(self.params['minuit'])
		for key in self.bestfit['values']: #update with potential previous bestfit
			if key in self.vary:
				self.minuitargs[key] = self.bestfit['values'][key]
				self.minuitargs['error_{}'.format(key)] = self.bestfit['errors'][key]
			#if key in self.parameters: self.fitargs[key] = self.bestfit['values'][key]
		#print self.vary		
		#print [(key,self.fitargs[key]) for key in self.vary]
		#print self.minuitargs
		self.minuit = iminuit.Minuit(self.chi2args,**self.minuitargs)
	
	@args_to_kwargs
	def chi2args(self,**kwargs):
		self.solved = self.solve_update_analytic(**kwargs)
		kwargs.update(self.solved)
		return self.chi2(**kwargs)
	
	@utils.classparams
	def run_migrad(self,migrad={}):
		self.minuit.migrad(**self.params['migrad'])
		#self.minuit.hesse()
		self.set_bestfit(minos=False)
		self.set_dbestfit()
		
	@utils.classparams
	def run_minos(self,minos={}):
		minos = copy.deepcopy(minos)
		params = minos.pop('params',[])
		if params:
			for par in params: self.minuit.minos(par,**minos)
		else:
			self.minuit.minos(**minos)
		self.set_bestfit(minos=True)
		self.set_dbestfit()
			
	def set_bestfit(self,minos=False):
		for key in ['values','errors']:
			if hasattr(self.minuit,key): self.bestfit[key] = dict(getattr(self.minuit,key))
		self.bestfit['chi2'] = self.chi2args(*[self.bestfit['values'][key] for key in self.parameters])
		self.bestfit['values'].update(self.solved)
		self.bestfit['errors'].update({par:0. for par in self.solved}) # because too lazy to compute errors
		if hasattr(self.minuit,'fval'): self.bestfit['fval'] = getattr(self.minuit,'fval')
		if minos:
			minos = self.minuit.get_merrors()
			self.bestfit['minos'] = {}
			for par in minos:
				for key in self._MINOS_ARGS:
					if not key in self.bestfit['minos']: self.bestfit['minos'][key] = {}
					self.bestfit['minos'][key][par] = minos[par][key]
		self.bestfit['rchi2'] = self.bestfit['chi2']/(self.nbins-self.nvary)
		self.bestfit['covariance'] = self.minuit.covariance
		self.bestfit['sorted'] = self.sorted
		
	def set_dbestfit(self):
		self.dbestfit = {'values':{},'errors':{},'sorted':[]}
		if 'minos' in self.bestfit: self.dbestfit['minos'] = {'upper':{},'lower':{}}
		for likelihood in self.likelihoods:
			values,errors,scalecov,latex,sorted = likelihood.derived_parameters(self.bestfit['values'],errors=self.bestfit['errors'])
			self.dbestfit['values'].update(values)
			self.dbestfit['errors'].update(errors)
			self.dbestfit['sorted'] += [par for par in sorted if par not in self.dbestfit['sorted']]
			self.latex.update(latex)
			if 'minos' in self.dbestfit:
				for uplow in ['upper','lower']:
					self.dbestfit['minos'][uplow].update(likelihood.derived_parameters(self.bestfit['values'],errors=self.bestfit['minos'][uplow])[1])
		self.dbestfit['scalecov'] = scalecov
		for key in ['chi2','rchi2']: self.dbestfit[key] = self.bestfit[key]

	@utils.classparams
	def run_mnprofile(self,mnprofile={}):
		#self.profile = {}
		#self.contour = {}
		params = mnprofile.get('params',[])
		npoints = mnprofile.get('npoints',30)
		bounds = mnprofile.get('bounds',2)
		subtract_min = mnprofile.get('subtract_min',False)
		if scipy.isscalar(npoints): npoints = [npoints]*len(params)
		if scipy.isscalar(bounds) or isinstance(bounds,tuple): bounds = [bounds]*len(params)
		for par,npoint,bound in zip(params,npoints,bounds):
			kwargs = dict(bins=npoint,bound=bound,subtract_min=subtract_min)
			self.logger.info('Profiling {} with kwargs {}.'.format(par,kwargs))
			x,y,status = self.minuit.mnprofile(par,**kwargs)
			self.profile[par] = scipy.array([x,y])
			
	@utils.classparams
	def run_mncontour(self,mncontour={}):
		params = mncontour.get('params',[])
		npoints = mncontour.get('npoints',20)
		nsigmas = mncontour.get('nsigmas',1)
		ndof = mncontour.get('ndof',1)
		if scipy.isscalar(npoints): npoints = [npoints]*len(params)
		if scipy.isscalar(nsigmas) or len(nsigmas) != len(params): nsigmas = [nsigmas]*len(params) 
		nsigmas = [[nsigma] if scipy.isscalar(nsigma) else nsigma for nsigma in nsigmas]
		for par,npoint,nsigma in zip(params,npoints,nsigmas):
			for sigma in nsigma:
				kwargs = dict(numpoints=npoint,sigma=scipy.sqrt(nsigmas_to_deltachi2(sigma,ndof=ndof)))
				self.logger.info('Profiling {} with kwargs {}.'.format(par,kwargs))
				xy = self.minuit.mncontour(par[0],par[1],**kwargs)[-1]
				if par not in self.contour: self.contour[par] = {}
				self.contour[par][sigma] = scipy.array(xy).T
		
	@utils.getstateclass
	def getstate(self,state):
		for key in Likelihood._ARGS + ['ids','bestfit','dbestfit','profile','contour']:
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

	def export_to_latex(self,path,fmt='vertical'):

		output = ''
		output += '\\documentclass{report}'
		output += '\\usepackage[top=2.5cm, bottom=2.5cm, left=1cm, right=1cm]{geometry}\n'
		output += '\\begin{document}'
		output += '\\begin{center}\n'
		output += 'Fit results\n'
		output += '\\end{center}\n\n'
		
		output += '\\noindent Reduced $\chi^{{2}}$: ${0}/({1}-{2})={3}$\n'.format(utils.to_precision(self.bestfit['chi2'],precision=3),self.nbins,self.nvary,utils.to_precision(self.bestfit['rchi2'],precision=3))
			
		output += '\\medbreak\n'
		output += '\\noindent Best fit and errors from Minuit:\n'
		output += utils.fit_to_latex(self.bestfit['sorted'],self.bestfit['values'],errors=self.bestfit['errors'],uplow=self.bestfit.get('minos',None),latex=self.latex,fmt=fmt)

		output += '\\noindent Derived parameters:\n'
		output += utils.fit_to_latex(self.dbestfit['sorted'],self.dbestfit['values'],errors=self.dbestfit['errors'],uplow=self.dbestfit.get('minos',None),latex=self.latex,fmt=fmt)
		
		output += '\\end{document}\n'

		utils.save_latex(output,path)

