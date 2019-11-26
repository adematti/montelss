import logging
import scipy
from scipy import stats,optimize,constants
import functools
import utils
from montelss.minimizer import Minimizer,nsigmas_to_quantiles_1d,nsigmas_to_deltachi2

def vectorize(func):
	@functools.wraps(func)				
	def wrapper(self,pars,**kwargs):
		if scipy.isscalar(pars):
			return func(self,pars,**kwargs)
		toret = scipy.asarray([func(self,par,**kwargs) for par in pars])
		return toret
	return wrapper

def nsigmas_to_quantiles_1d_sym(nsigmas):
	total = nsigmas_to_quantiles_1d(nsigmas)
	out = (1.-total)/2.
	return out,1.-out

def nsigmas_to_quantiles_2d(nsigmas):
	return 1. - scipy.exp(-nsigmas**2/2.)

def sample_cic(grid,data):
	if not isinstance(grid,list): grid = [grid]
	pdf = scipy.zeros(map(len,grid),dtype=grid[0].dtype)
	findex = scipy.array([scipy.searchsorted(g,d) - 1 for g,d in zip(grid,data)]).T
	index = findex.astype(int)
	dindex = findex-index
	ishifts = scipy.array(scipy.meshgrid(*([[0,1]]*len(data)),indexing='ij')).reshape((len(data),-1)).T
	for ishift in ishifts:
		sindex = index + ishift
		sweight = scipy.prod((1-dindex) + ishift*(-1+2*dindex),axis=-1)
		scipy.add.at(pdf,tuple(sindex.T),sweight)
	return pdf

def density_1d(data,bins=20,nsigmas=2,method='gaussian_kde',bw_method='scott'):

	if scipy.isscalar(nsigmas): nsigmas = [nsigmas]

	if method == 'gaussian_kde':
		density = stats.gaussian_kde(data,bw_method='scott')
		x = scipy.linspace(data.min(),data.max(),bins)
		pdf = density(x)
		pdf /= pdf.sum()
	elif method == 'cic':
		x = scipy.linspace(data.min(),data.max(),bins)
		pdf = sample_cic(x,data)
		pdf /= pdf.sum()
	else:
		H, xedges = scipy.histogram(data,bins=bins,density=True)
		xbin = xedges[1:] - xedges[:-1]; x = (xedges[1:] + xedges[:-1])/2.
		pdf = H*xbin

	def find_confidence_interval(x, confidence_level):
		return pdf[pdf > x].sum() - confidence_level

	levels = [optimize.brentq(find_confidence_interval,0.,1.,args=(nsigmas_to_quantiles_1d(nsigma))) for nsigma in nsigmas]

	return x,pdf,levels

def density_2d(data,bins=[20,20],nsigmas=2,method='gaussian_kde',bw_method='scott',ndof=2):

	if scipy.isscalar(nsigmas): nsigmas = [nsigmas]
	if scipy.isscalar(bins): bins = [bins]*2

	if method == 'gaussian_kde':
		density = stats.gaussian_kde(data,bw_method=bw_method)
		x,y = (scipy.linspace(d.min(),d.max(),b) for d,b in zip(data,bins))
		xx,yy = scipy.meshgrid(x,y,indexing='ij')
		pdf = density([xx.ravel(),yy.ravel()]).reshape(xx.shape).T
		pdf /= pdf.sum()
	elif method == 'cic':
		x,y = (scipy.linspace(d.min(),d.max(),b) for d,b in zip(data,bins))
		pdf = sample_cic([x,y],data).T
		pdf /= pdf.sum()
	else:
		H, xedges, yedges = scipy.histogram2d(*data,bins=bins,density=True)
		xbin = xedges[1:] - xedges[:-1]; x = (xedges[1:] + xedges[:-1])/2.
		ybin = yedges[1:] - yedges[:-1]; y = (yedges[1:] + yedges[:-1])/2.
		pdf = (H*(xbin[:,None]*ybin[None,:])).T

	def find_confidence_interval(x, confidence_level):
		return pdf[pdf > x].sum() - confidence_level

	to_quantiles = nsigmas_to_quantiles_2d if ndof == 1 else nsigmas_to_quantiles_1d
	levels = [optimize.brentq(find_confidence_interval,0.,1.,args=(to_quantiles(nsigma))) for nsigma in nsigmas]

	return x,y,pdf,levels

class EnsembleValues(object):

	logger = logging.getLogger('EnsembleValues')
	_ARGS = ['values']
	_ATTRS = ['latex']

	def __init__(self,**params):
		self.params = params

	@property
	def parameters(self):
		return self.values.keys()
	
	def __len__(self):
		return len(self[self.parameters[0]])

	@property
	def size(self):
		return len(self)
		
	def __getitem__(self,name):
		if isinstance(name,list) and isinstance(name[0],(str,unicode)):
			return [self[name_] for name_ in name]
		if isinstance(name,(str,unicode)):
			if name in self.values:
				return self.values[name]
			else:
				raise KeyError('There is no parameter {}.'.format(name))
		else:
			new = self.deepcopy()
			for key in self._ARGS:
				if hasattr(self,key):
					tmp = self.get(key)
					setattr(new,key,{par:tmp[par][name] for par in tmp})
			return new
	
	def __setitem__(self,name,item):
		if isinstance(name,(str,unicode)):
			self.values[name] = item
		else:
			for key in self.parameters:
				self.values[key][name] = item

	def __delitem__(self,name):
		del self.values[name]
			
	def	__contains__(self,name):
		return name in self.parameters
	
	def __iter__(self):
		for par in self.parameters:
			yield par
	
	def __str__(self):
		return str(self.values)
	
	def slice(self,islice=0,nslices=1):
		size = len(self)
		return self[islice*size//nslices:(islice+1)*size//nslices]
	
	def covariance(self,pars,ddof=1,**kwargs):
		return scipy.cov([self[par] for par in pars],ddof=ddof,**kwargs)
		
	def correlation(self,pars,**kwargs):
		return scipy.corrcoef([self[par] for par in pars],**kwargs)
	
	def cov(self,par1,par2,**kwargs):
		return self.covariance([par1,par2],**kwargs)[0,1]
		
	def corr(self,par1,par2,**kwargs):
		return self.correlation([par1,par2],**kwargs)[0,1]
	
	def var(self,par,ddof=1,**kwargs):
		return scipy.var(self[par],ddof=ddof,axis=-1,**kwargs)
		
	def std(self,par,error='real',ddof=1,**kwargs):
		std = scipy.std(self[par],ddof=ddof,axis=-1,**kwargs)
		if error == 'mean': return std/scipy.sqrt(len(self))
		return std
		
	def mean(self,par):
		return scipy.mean(self[par],axis=-1)

	@vectorize
	def minimum(self,par,cost='chi2'):
		argmin = scipy.argmin(self[cost])
		return self[par][argmin]

	@vectorize
	def maximum(self,par,cost='lnposterior'):
		argmax = scipy.argmax(self[cost])
		return self[par][argmax]
	
	def percentile(self,par,q=[15.87,84.13]):
		return scipy.percentile(self[par],axis=-1,q=q)

	def quantile(self,par,q=[0.1587,0.8413]):
		return scipy.percentile(self[par],axis=-1,q=scipy.array(q)*100.)

	@vectorize
	def interval(self,par,nsigmas=1.,bins=100,method='gaussian_kde',bw_method='scott'):
		x,pdf,levels = density_1d(self[par],bins=bins,nsigmas=nsigmas,method=method,bw_method=bw_method)
		x = x[pdf>levels[0]]
		return x.min(),x.max()

	def par_to_latex(self,par):
		return self.latex.get(par,par)
	
	def par_to_label(self,par):
		if par in self.latex: return '${}$'.format(self.latex[par])
		return par

	def stats_to_latex(self,save,params=None,mean=scipy.mean,error=scipy.std,precision=2,fmt='vertical'):
		
		if params is None: params = self.parameters
		if isinstance(error(self[params[0]]),(tuple,list)):
			errors = None
			uplow = {'upper':{},'lower':{}}
			for par in params:
				uplow['upper'][par],uplow['lower'][par] = error(self[par])
		else:
			errors = {par:error(self[par]) for par in params}
			uplow = None

		output = ''
		output += '\\documentclass{report}'
		output += '\\usepackage[top=2.5cm, bottom=2.5cm, left=1cm, right=1cm]{geometry}\n'
		output += '\\begin{document}\n'
		output += '\\begin{center}\n'
		output += 'Statistics of {:d} points\n'.format(len(self))
		output += '\\end{center}\n\n'
		output += utils.fit_to_latex(params,{par:mean(self[par]) for par in params},errors=errors,uplow=uplow,rows=['Mean'],latex=self.latex,precision=precision,fmt=fmt)
		output += 'Correlation matrix\n'
		output += '\\begin{center}\n'
		latex = [self.par_to_label(par) for par in params]
		output += utils.array_to_latex(self.correlation(params),latex,latex,alignment='c',fmt='.3f')
		output += '\\end{center}\n\n'
		output += '\\end{document}\n'

		#logger.info('Saving log file to {}.'.format(pathlog))
		utils.save_latex(output,save)
		
	def points(self):
		for ipoint in range(len(self)):
			yield {par:self[par][ipoint] for par in self.parameters}

	def keep(self,parameters):
		self.latex = {self.latex[par] for par in parameters if par in self.latex}
		for key in self.__ARGS:
			tmp = self.get(key)
			tmp = {par:tmp[par] for par in parameters}
			setattr(self,key,tmp)
	
	def add(self,latex=None,**kwargs):
		for key in self._ARGS:
			self.get(key).update(kwargs.get(key,{}))
		if latex is None: latex = self.values.keys()
		self.latex.update(latex)

	def get(self,*args,**kwargs):
		return getattr(self,*args,**kwargs)
	
	def __radd__(self,other):
		if other == 0: return self
		else: return self.__add__(other)
	
	def __add__(self,other):
		new = self.deepcopy()
		for key in self._ARGS:
			tmpself,tmpother = self.get(key),other.get(key)
			tmp = {par:scipy.concatenate([tmpself[par],tmpother[par]],axis=0) for par in tmpself if par in tmpother}
			setattr(new,key,tmp)
		return new

	def __rsub__(self,other):
		if other == 0: return self
		else: return self.__sub__(other)

	def __sub__(self,other):
		if isinstance(other,self.__class__):
			new = self.deepcopy()
			for key in self._ARGS:
				tmp = new.get(key)
				tmpo = other.get(key)
				for par in tmp:
					if par in tmpo: tmp[par] -= tmpo[par]
			return new
		else:
			raise ValueError('Cannot subtract {} by {} object.'.format(self.__class__,type(other)))
		return new

	def __mul__(self,other):
		if isinstance(other,(scipy.floating,float)):
			new = self.deepcopy()
			for key in self._ARGS:
				tmp = new.get(key)
				for par in tmp:
					tmp[par] *= other
			return new
		else:
			raise ValueError('Cannot multiply {} by {} object.'.format(self.__class__,type(other)))
	
	def __div__(self,other):
		if isinstance(other,self.__class__):
			new = self.deepcopy()
			for key in self._ARGS:
				tmp = new.get(key)
				for par in tmp:
					tmp[par] /= other.values[par]
			return new
		elif isinstance(other,(scipy.floating,float)):
			new = self.deepcopy()
			for key in self._ARGS:
				tmp = new.get(key)
				for par in tmp:
					tmp[par] /= other
			return new
		else:
			raise ValueError('Cannot multiply {} by {} object.'.format(self.__class__,type(other)))

	def zeros(self,dtype=scipy.float64):
		return scipy.zeros(len(self),dtype=dtype)
	
	def ones(self,dtype=scipy.float64):
		return scipy.ones(len(self),dtype=dtype)
	
	def falses(self):
		return self.zeros(dtype=scipy.bool_)
	
	def trues(self):
		return self.ones(dtype=scipy.bool_)
	
	def nans(self):
		return self.ones()*scipy.nan

	@classmethod
	def combine(cls,ensembles,errors={},params=[]):
		assert (scipy.diff(map(len,ensembles)) == 0).all()
		self = ensembles[0].deepcopy()
		self.weights = {}
		for par in params:
			if errors.get(par,None) is None:
				errors[par] = [e.std(par) for e in ensembles]
			err = scipy.array(errors[par])
			corr = scipy.corrcoef([e[par] for e in ensembles])
			cov = corr * err[:,None].dot(err[None,:])
			invcov = scipy.linalg.inv(cov)
			self.weights[par] = scipy.sum(invcov,axis=-1)/scipy.sum(invcov)
			self.logger.info('Using for {} weights: {}.'.format(par,self.weights[par])) 
			for key in self._ARGS:
				tmp = getattr(self,key)
				tmp[par] = scipy.average([getattr(e,key)[par] for e in ensembles],weights=self.weights[par])
		return self

	@property
	def ndof(self):
		return self.nobs-self.nvary

	@utils.setstateclass
	def setstate(self,state):
		pass

	@classmethod
	@utils.loadstateclass	
	def loadstate(self,state):
		self.setstate(state)
		return self

	@utils.getstateclass
	def getstate(self,state):
		for key in self._ARGS + self._ATTRS:
			if hasattr(self,key): state[key] = self.get(key)
		return state

	@classmethod
	@utils.loadclass
	def load(self,state):
		self.setstate(state)
		return self

	@utils.saveclass
	def save(self):
		return self.getstate()

	def copy(self):
		return self.__class__.loadstate(self.getstate())

	def deepcopy(self):
		import copy
		return copy.deepcopy(self)

	def as_dict(self,params=None):
		if params is None: params = self.parameters
		return {par:self[par] for par in params}

	@classmethod
	def load_csv(cls,path,params,args=['values'],latex={},**kwargs):
		columns = scipy.loadtxt(path,unpack=True,**kwargs)
		self = cls()
		self.latex = latex
		params = params*len(columns)
		args = args*len(columns)
		for icol,col in enumerate(columns):
			if not hasattr(self,args[icol]): setattr(self,args[icol],{})
			getattr(self,args[icol])[params[icol]] = col
		return self

	def save_csv(self,path,params=None,stats='values',header='',fmt='%.8e',delimiter=' ',**kwargs):
		if params is None: params = self.parameters
		if header: header += '\n'
		if not isinstance(stats,list): stats = [stats]
		list_header,data = [],[]
		for key in stats:
			if key == 'values': template = '{}'
			else: template = '{}({{}})'.format(key)
			list_header += [template.format(self.latex[par]) for par in params]
			data += [self.get(key)[par] for par in params]
		header += delimiter.join(list_header)
		data = scipy.array(data).T
		self.logger.info('Saving {} to {}.'.format(self.__class__.__name__,path))
		scipy.savetxt(path,data,header=header,fmt=fmt,delimiter=delimiter,**kwargs)

	def save_eboss(self,path_mean,path_covariance,params=None,label_params=None,zeff=None,fmt='%.8e',delimiter=' '):
		if params is None: params = self.parameters
		if label_params is None: label_params = params
		output = ''
		for par,label in zip(params,label_params):
			val = fmt % self.mean(par)
			if zeff is not None: output += '{}{d}{}{d}{}\n'.format(zeff,val,label,d=delimiter)
			else: output += '{}{d}{}\n'.format(val,label,d=delimiter)
		self.logger.info('Saving mean to {}.'.format(path_mean))
		with open(path_mean,'w') as file:
			file.write(output)
		output = ''
		covariance = self.covariance(params,ddof=1)
		for line in covariance:
			for col in line[:-1]:
				output += (fmt % col) + delimiter
			output += (fmt % line[-1]) + '\n'
		self.logger.info('Saving covariance to {}.'.format(path_covariance))
		with open(path_covariance,'w') as file:
			file.write(output)

	def save_grid(self,path,params=None,stats='values',bins=30,method='cic',bw_method='scott',header='',fmt='%.8e',delimiter=' ',**kwargs):
		if params is None: params = self.parameters
		data = scipy.array([self.get(stats)[par] for par in params])
		points = []
		if not isinstance(bins,list): bins = [bins]*len(data)
		for d,b in zip(data,bins):
			tmp = scipy.linspace(d.min(),d.max(),b) if scipy.isscalar(b) else b
			points.append(tmp)
		mpoints = scipy.meshgrid(*points,indexing='ij')
		rpoints = [p.ravel() for p in mpoints]
		if method == 'cic':
			pdf = sample_cic(points,data).ravel()
		else:
			density = scipy.stats.gaussian_kde(data,bw_method=bw_method)
			pdf = density(rpoints)
		pdf /= pdf.max()
		rpoints.append(pdf)
		self.logger.info('Saving {} to {}.'.format(self.__class__.__name__,path))
		scipy.savetxt(path,scipy.array(rpoints).T,header=header,fmt=fmt,delimiter=delimiter,**kwargs)

def remove_ids(minimizer):
	
	def rm_id(txt):
		for mid in minimizer.ids: txt = txt.replace('_{}'.format(mid),'')
		return txt
	
	def rm_id_latex(txt):
		for mid in minimizer.ids: txt = txt.replace('^{{\\rm {}}}'.format(mid),'')
		return txt

	minimizer.latex = {rm_id(par):rm_id_latex(minimizer.latex[par]) for par in minimizer.latex}

	for key in ['values','errors']:
		minimizer.dbestfit[key] = {rm_id(par): minimizer.dbestfit[key][par] for par in minimizer.dbestfit[key]}
	if 'minos' in minimizer.dbestfit:
		for key in ['upper','lower']:
			minimizer.dbestfit['minos'][key] = {rm_id(par): minimizer.dbestfit['minos'][key][par] for par in minimizer.dbestfit['minos'][key]}
	minimizer.ids = []

class AnalyzeFits(EnsembleValues):

	logger = logging.getLogger('AnalyzeFits')
	_ARGS = ['values','errors','upper','lower']
	_ATTRS = ['latex','nvary','nfixed','nobs']

	@property
	def chi2(self):
		return self['chi2']

	@classmethod
	def load_bestfit(cls,paths,keep_params=None,ignore_ids=False):

		self = cls()
		self.logger.info('Loading {:d} files.'.format(len(paths)))
		minimizer = Minimizer.load(paths[0])
		if ignore_ids: remove_ids(minimizer)
		self.ids = minimizer.ids
		if keep_params is None: keep_params = minimizer.dbestfit['values']
		self.latex = {par:minimizer.latex[par] for par in minimizer.latex if par in keep_params}
		for key in ['values','errors']: setattr(self,key,{par:[] for par in minimizer.dbestfit[key] if par in keep_params})
		if 'minos' in minimizer.dbestfit:
			for key in ['upper','lower']: setattr(self,key,{par:[] for par in minimizer.dbestfit['minos'][key] if par in keep_params})
		else: self.upper,self.lower = {},{}
		chi2 = []
		for path in paths:
			minimizer = minimizer.load(path)
			if ignore_ids: remove_ids(minimizer)
			chi2.append(minimizer.bestfit['chi2'])
			bestfit = minimizer.dbestfit
			for par in self.parameters:
				for key in ['values','errors']: self.get(key)[par].append(bestfit[key].get(par,scipy.nan))
			if self.upper:
				for key in ['upper','lower']:
					for par in self.get(key):
						self.get(key)[par].append(bestfit.get('minos',{}).get(key,{}).get(par,scipy.nan))
		for key in ['values','errors']:
			for par in self.parameters: self.get(key)[par] = scipy.asarray(self.get(key)[par])
		if self.upper:
			for key in ['upper','lower']:
				for par in self.get(key):
					if par in keep_params: self.get(key)[par] = scipy.asarray(self.get(key)[par])

		self.values['chi2'] = scipy.asarray(chi2)
		self.latex['chi2'] = '\\chi^{2}'
		#self.sorted = minimizer.dbestfit['sorted']
		self.nvary = minimizer.nvary 
		self.nfixed = minimizer.nfixed
		self.nobs = minimizer.nbins
		
		return self
	
	def errorbar(self,par):
		if self.get('upper',{}) and self.get('lower',{}):
			return scipy.array([-self.lower[par],self.upper[par]])
		return scipy.array([self.errors[par]]*2)
	
	def residual(self,par,truth=None):
		if truth is None: truth = self.mean(par)
		diff = self[par] - truth
		mask = diff >= 0.
		errorbar = self.errorbar(par)
		diff[mask] /= errorbar[0][mask]
		diff[~mask] /= errorbar[1][~mask]
		return diff


from matplotlib import pyplot,cm,patches,gridspec,transforms
from matplotlib.ticker import MaxNLocator,AutoMinorLocator
from matplotlib.colors import Normalize

prop_cycle = pyplot.rcParams['axes.prop_cycle']
fontsize = 20
titlesize = 20
labelsize = 16
ticksize = 16
dpi = 200
		
def plot_corner_fits(fitlists,params=[],labels=[],truths=[],singles=[],singles_errors=[],means='real',errors=False,title='',colors=prop_cycle.by_key()['color'],scatter_kwargs={'marker':'o','markersize':3,'alpha':0.4,'markeredgecolor':'none','elinewidth':1},truths_kwargs={'linestyle':'--','linewidth':1,'color':'k'},means_points_kwargs={'elinewidth':1},means_lines_kwargs={'linestyle':'-','linewidth':1},singles_points_kwargs={'marker':'*','markersize':10,'elinewidth':1},singles_lines_kwargs={},singles_lines_errors_kwargs={'linestyle':'.','linewidth':1},bins=10,path='corner.png'):

	singles_lines = {'linestyle':':','linewidth':2,'color':'r','label':'data'}
	singles_lines.update(singles_lines_kwargs)
	singles_lines_kwargs = singles_lines

	if not isinstance(fitlists,list): fitlists = [fitlists]	
	xlims,xticks = [],{True:[],False:[]}
	truths = truths + [None]*len(params)
	singles = singles + [None]*len(params)
	singles_errors = singles_errors + [None]*len(params)

	for ipar,(sing,error) in enumerate(zip(singles,singles_errors)):
		if (sing is not None) and (error is not None):
			if scipy.isscalar(error): singles_errors[ipar] = [-error,error]

	handles = []
	if any([s is not None for s in singles]) and (singles_lines_kwargs.get('label',None) is not None):
		handles.append(patches.Patch(color=singles_lines_kwargs['color'],label=singles_lines_kwargs['label']))

	add_legend = bool(labels) or bool(handles)
	labels = labels + [None]*len(fitlists)
	
	ncols = nrows = len(params)
	fig = pyplot.figure(figsize=(10*ncols//3,8*nrows//3))
	if title: fig.suptitle(title,fontsize=titlesize)
	gs = gridspec.GridSpec(nrows,ncols,wspace=0.1,hspace=0.1)
	majorlocator = MaxNLocator(nbins=4,min_n_ticks=3,prune='both')
	minorlocator = AutoMinorLocator(2)
	
	for ipar1,par1 in enumerate(params):
		ax = pyplot.subplot(gs[ipar1,ipar1])
		for ifl,fl in enumerate(fitlists):
			if par1 in fl:
				ax.hist(fl[par1],bins=bins,color=colors[ifl],histtype='step',density=True)
				if means: ax.axvline(x=fl.mean(par1),ymin=0.,ymax=1.,color=colors[ifl],**means_lines_kwargs)
				if labels[ifl] is not None: handles.append(patches.Patch(color=colors[ifl],label=labels[ifl],alpha=1))
		if truths[ipar1] is not None: ax.axvline(x=truths[ipar1],ymin=0.,ymax=1.,**truths_kwargs)
		if singles[ipar1] is not None:
			ax.axvline(x=singles[ipar1],ymin=0.,ymax=1.,**singles_lines_kwargs)
			if singles_errors[ipar1] is not None:
				for error in singles_errors[ipar1]: ax.axvline(x=singles[ipar1]+error,ymin=0.,ymax=1.,color=singles_lines_kwargs.get('color',None),**singles_lines_errors_kwargs)
		if ipar1<len(params)-1: ax.get_xaxis().set_visible(False)
		else: ax.set_xlabel(fitlists[0].par_to_label(par1),fontsize=labelsize)
		ax.get_yaxis().set_visible(False)
		ax.tick_params(labelsize=ticksize)
		ax.xaxis.set_major_locator(majorlocator)
		ax.xaxis.set_minor_locator(minorlocator)
		xlims.append(ax.get_xlim())
		for minor in xticks: xticks[minor].append(ax.get_xticks(minor=minor))
		if add_legend and ipar1==0: ax.legend(**{'loc':'upper left','ncol':1,'fontsize':labelsize,'framealpha':0.5,'frameon':True,'bbox_to_anchor':(1.04,1.),'handles':handles})
	
	for ipar1,par1 in enumerate(params):
		for ipar2,par2 in enumerate(params):
			if nrows-1-ipar2 >= ncols-1-ipar1: continue
			ax = pyplot.subplot(gs[ipar2,ipar1])
			for ifl,fl in enumerate(fitlists):
				if par1 in fl and par2 in fl:
					if errors:
						xerr,yerr = fl.errorbar(par1),fl.errorbar(par2)
					else:
						xerr,yerr = None,None
					ax.errorbar(fl[par1],fl[par2],xerr=xerr,yerr=yerr,color=colors[ifl],linestyle='none',**scatter_kwargs)
					if means: ax.errorbar(fl.mean(par1),fl.mean(par2),xerr=fl.std(par1,error=means),yerr=fl.std(par2,error=means),color=colors[ifl],linestyle='none',**means_points_kwargs)
			if truths[ipar1] is not None: ax.axvline(x=truths[ipar1],ymin=0.,ymax=1.,**truths_kwargs)
			if truths[ipar2] is not None: ax.axhline(y=truths[ipar2],xmin=0.,xmax=1.,**truths_kwargs)
			if (singles[ipar1] is not None) and (singles[ipar2] is not None):
				ax.errorbar(x=singles[ipar1],y=singles[ipar2],xerr=singles_errors[ipar1],yerr=singles_errors[ipar2],color=singles_lines_kwargs.get('color',None),linestyle='none',**singles_points_kwargs)
			if ipar1>0: ax.get_yaxis().set_visible(False)
			else: ax.set_ylabel(fitlists[0].par_to_label(par2),fontsize=labelsize)
			if nrows-1-ipar2>0: ax.get_xaxis().set_visible(False)
			else: ax.set_xlabel(fitlists[0].par_to_label(par1),fontsize=labelsize)
			ax.tick_params(labelsize=ticksize)
			for minor in xticks:
				ax.set_xticks(xticks[minor][ipar1],minor=minor)
				ax.set_yticks(xticks[minor][ipar2],minor=minor)
			ax.set_xlim(xlims[ipar1])
			ax.set_ylim(xlims[ipar2])
	
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_corner_residuals(fitlists,params=[],labels=[],truths=[],singles=[],means='real',title='',colors=prop_cycle.by_key()['color'],scatter_kwargs={'s':3,'marker':'o','alpha':0.4,'edgecolors':'none'},truths_kwargs={'linestyle':'--','linewidth':1,'color':'k'},means_points_kwargs={'elinewidth':1},means_lines_kwargs={'linestyle':'-','linewidth':1},singles_points_kwargs={'marker':'*','s':10,'color':'r'},singles_lines_kwargs={},bins=10,path='corner.png'):

	singles_lines = {'linestyle':':','linewidth':2,'color':'r','label':'data'}
	singles_lines.update(singles_lines_kwargs)
	singles_lines_kwargs = singles_lines

	if not isinstance(fitlists,list): fitlists = [fitlists]
	xlims,xticks = [],{True:[],False:[]}
	truths = truths + [None]*len(params)
	singles = singles + [None]*len(params)
	handles = []
	if any([s is not None for s in singles]) and (singles_lines_kwargs.get('label',None) is not None):
		handles.append(patches.Patch(color=singles_lines_kwargs['color'],label=singles_lines_kwargs['label']))

	add_legend = bool(labels) or bool(handles)
	labels = labels + [None]*len(fitlists)
	
	fitlists = [fl.deepcopy() for fl in fitlists]
	for ifl,fl in enumerate(fitlists):
		for ipar1,par1 in enumerate(params):
			fl[par1] = fl.residual(par1,truths[ipar1])

	ncols = nrows = len(params)
	fig = pyplot.figure(figsize=(10*ncols//3,8*nrows//3))
	if title: fig.suptitle(title,fontsize=titlesize)
	gs = gridspec.GridSpec(nrows,ncols,wspace=0.1,hspace=0.1)
	majorlocator = MaxNLocator(nbins=4,min_n_ticks=3,prune='both')
	minorlocator = AutoMinorLocator(2)

	def par_to_residual(par):
		#return '$({0}-{0}^{{\mathrm{{truth}}}})/\sigma_{{{0}}}$'.format(fitlists[0].par_to_latex(par))
		return '$\Delta {0} / \sigma_{{{0}}}$'.format(fitlists[0].par_to_latex(par))
	
	for ipar1,par1 in enumerate(params):
		ax = pyplot.subplot(gs[ipar1,ipar1])
		for ifl,fl in enumerate(fitlists):
			if par1 not in fl: continue
			ax.hist(fl[par1],bins=bins,color=colors[ifl],label=labels[ifl],histtype='step',density=True)
			if means: ax.axvline(x=fl.mean(par1),ymin=0.,ymax=1.,color=colors[ifl],**means_lines_kwargs)
			if labels[ifl] is not None: handles.append(patches.Patch(color=colors[ifl],label=labels[ifl],alpha=1))
		if truths[ipar1] is not None: ax.axvline(x=0.,ymin=0.,ymax=1.,**truths_kwargs)
		if singles[ipar1] is not None: ax.axvline(x=singles[ipar1],ymin=0.,ymax=1.,**singles_lines_kwargs)
		if ipar1<len(params)-1: ax.get_xaxis().set_visible(False)
		else: ax.set_xlabel(par_to_residual(par1),fontsize=labelsize)
		ax.get_yaxis().set_visible(False)
		ax.tick_params(labelsize=ticksize)
		ax.xaxis.set_major_locator(majorlocator)
		ax.xaxis.set_minor_locator(minorlocator)
		xlims.append(ax.get_xlim())
		for minor in xticks: xticks[minor].append(ax.get_xticks(minor=minor))
		if add_legend and ipar1==0: ax.legend(**{'loc':'upper left','ncol':1,'fontsize':labelsize,'framealpha':0.5,'frameon':True,'bbox_to_anchor':(1.04,1.),'handles':handles})
	
	for ipar1,par1 in enumerate(params):
		for ipar2,par2 in enumerate(params):
			if nrows-1-ipar2 >= ncols-1-ipar1: continue
			ax = pyplot.subplot(gs[ipar2,ipar1])
			for ifl,fl in enumerate(fitlists):
				if par1 not in fl or par2 not in fl: continue
				ax.scatter(fl[par1],fl[par2],c=colors[ifl],**scatter_kwargs)
				if means: ax.errorbar(fl.mean(par1),fl.mean(par2),xerr=fl.std(par1,error=means),yerr=fl.std(par2,error=means),color=colors[ifl],linestyle='none',**means_points_kwargs)
			if truths[ipar1] is not None: ax.axvline(x=0.,ymin=0.,ymax=1.,**truths_kwargs)
			if truths[ipar2] is not None: ax.axhline(y=0.,xmin=0.,xmax=1.,**truths_kwargs)
			if (singles[ipar1] is not None) and (singles[ipar2] is not None): ax.scatter(x=singles[ipar1],y=singles[ipar2],**singles_points_kwargs) 
			if ipar1>0: ax.get_yaxis().set_visible(False)
			else: ax.set_ylabel(par_to_residual(par2),fontsize=labelsize)
			if nrows-1-ipar2>0: ax.get_xaxis().set_visible(False)
			else: ax.set_xlabel(par_to_residual(par1),fontsize=labelsize)
			ax.tick_params(labelsize=ticksize)
			for minor in xticks:
				ax.set_xticks(xticks[minor][ipar1],minor=minor)
				ax.set_yticks(xticks[minor][ipar2],minor=minor)
			ax.set_xlim(xlims[ipar1])
			ax.set_ylim(xlims[ipar2])
	
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)


def plot_histo_residuals(fitlists,nrows=1,params=[],labels=[],truths=[],singles=[],means=True,gaussian=True,title='',colors=prop_cycle.by_key()['color'],truths_kwargs={'linestyle':'--','linewidth':1,'color':'k'},means_kwargs={'linestyle':'-','linewidth':1},singles_lines_kwargs={},legend_kwargs={},gaussian_kwargs={'linestyle':'--','linewidth':2,'color':'k'},bins=10,path='chi2.png'):
	
	ticksize = 14
	labelsize = 16
	
	singles_lines = {'linestyle':':','linewidth':2,'color':'r','label':'data'}
	singles_lines.update(singles_lines_kwargs)
	singles_lines_kwargs = singles_lines

	legend = {'loc':'upper left','ncol':1,'fontsize':labelsize,'framealpha':0.5,'frameon':True,'bbox_to_anchor':(0., 1.4)}
	legend.update(legend_kwargs)
	legend_kwargs = legend

	truths = truths + [None]*len(params)
	singles = singles + [None]*len(params)
	if not isinstance(fitlists[0],list):
		fitlists = [fitlists]*len(params)
	
	add_legend = bool(labels) or (any([s is not None for s in singles]) and (singles_lines_kwargs.get('label',None) is not None))
	labels = labels + [None]*max(map(len,fitlists))

	fitlists = [[fl.deepcopy() for fl in fitlist] for fitlist in fitlists]
	for ipar1,par1 in enumerate(params):
		for ifl,fl in enumerate(fitlists[ipar1]):
			fl[par1] = fl.residual(par1,truths[ipar1])
	
	ncols = int(scipy.ceil(len(params)*1./nrows))
	fig = pyplot.figure(figsize=(10*ncols//3,10*nrows//3))
	if title: fig.suptitle(title,fontsize=titlesize)
	gs = gridspec.GridSpec(nrows,ncols,wspace=0.5,hspace=0.5)
	num = 100

	def normal(x):
		return 1./scipy.sqrt(2.*constants.pi)*scipy.exp(-x**2/2.)

	def par_to_residual(par):
		return '$\Delta {0} / \sigma_{{{0}}}$'.format(fitlists[0][0].par_to_latex(par))

	for ipar1,par1 in enumerate(params):
		ax = pyplot.subplot(gs[ipar1])
		for ifl,fl in enumerate(fitlists[ipar1]):
			if par1 not in fl: continue
			#ax.hist(fl[par1],bins=bins,color=colors[ifl],label=labels[ifl],histtype='step',density=True)
			values,edges = stats.binned_statistic(fl[par1],values=fl[par1],statistic='count',bins=bins)[:2]
			values /= len(fl)*scipy.diff(edges)
			mid = (edges[:-1]+edges[1:])/2.
			ax.plot(mid,values,color=colors[ifl],label=labels[ifl])
			if means: ax.axvline(x=fl.mean(par1),ymin=0.,ymax=1.,color=colors[ifl],**means_kwargs)
		if truths[ipar1] is not None: ax.axvline(x=0.,ymin=0.,ymax=1.,**truths_kwargs)
		if singles[ipar1] is not None: ax.axvline(x=singles[ipar1],ymin=0.,ymax=1.,**singles_lines_kwargs)
		x = scipy.linspace(*ax.get_xlim(),num=num)
		if gaussian: ax.plot(x,normal(x),**gaussian_kwargs)
		ax.set_xlabel(par_to_residual(par1),fontsize=labelsize)
		ax.tick_params(labelsize=ticksize)
		if add_legend and ipar1==0: ax.legend(**legend_kwargs)

	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)


def plot_histo_chi2(fitlists,labels=[],truths=None,singles=None,means=True,title='',colors=prop_cycle.by_key()['color'],truths_kwargs={'linestyle':'--','linewidth':1,'color':'k'},means_kwargs={'linestyle':'-','linewidth':1},singles_lines_kwargs={},bins=10,path='chi2.png'):

	singles_lines = {'linestyle':':','linewidth':2,'color':'r','label':'data'}
	singles_lines.update(singles_lines_kwargs)
	singles_lines_kwargs = singles_lines

	if not isinstance(fitlists,list): fitlists = [fitlists]
	if not isinstance(colors,list): colors = [colors]
	colors = colors*len(fitlists)
	
	add_legend = bool(labels) or (bool(singles) and (singles_lines_kwargs.get('label',None) is not None))
	if not add_legend: labels = []
	labels = labels + [None]*len(fitlists)

	fig = pyplot.figure(figsize=(8,6))
	if title: fig.suptitle(title,fontsize=titlesize)
	ax = fig.gca()
	
	for ifl,fl in enumerate(fitlists):
		ax.hist(fl.chi2,color=colors[ifl],label=labels[ifl],histtype='step',bins=bins)
		#print fl.chi2.mean()
		if means: ax.axvline(x=fl.chi2.mean(),ymin=0.,ymax=1.,color=colors[ifl],**means_kwargs)
	if singles is not None: ax.axvline(x=singles,ymin=0.,ymax=1.,**singles_lines_kwargs)
	if truths is not None:
		if truths == 'ndof': ax.axvline(x=fitlists[0].ndof,ymin=0.,ymax=1.,**truths_kwargs)
		else: ax.axvline(x=truths,ymin=0.,ymax=1.,**truths_kwargs)
		ax.tick_params(labelsize=ticksize)
		ax.set_xlabel('$\\chi^{2}$',fontsize=labelsize)
	
	if add_legend: ax.legend(**{'loc':'upper center','ncol':1,'fontsize':labelsize,'framealpha':0.5,'frameon':True,'bbox_to_anchor':(0.5, 1.27)})
	
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_comparison_fits(fitlists,params=[],ids=[],labels=[],truths=[],errors=True,means=False,limits=[],title='',ylabels=None,colors=prop_cycle.by_key()['color'],truths_kwargs={'linestyle':'--','linewidth':2,'color':'k'},limits_kwargs={'facecolor':'k','alpha':0.1,'linewidth':0,'linestyle':None},scatter_kwargs={},means_points_kwargs={'color':'k','elinewidth':1,'marker':'o','markersize':4},path='comparison.png'):

	scatter = {'marker':'o','markersize':4,'markeredgecolor':'none','elinewidth':1}
	scatter.update(scatter_kwargs)
	scatter_kwargs = scatter

	if not isinstance(fitlists,list): fitlists = [fitlists]
	truths = truths + [None]*len(params)
	limits = limits + [None]*len(params)
	add_legend = bool(labels)
	maxpoints = max(map(len,fitlists))
	labels = labels + [None]*maxpoints
	if not isinstance(errors,list): errors = [errors]
	errors = errors*maxpoints
	if not isinstance(colors,list): colors = [colors]
	colors = colors*maxpoints
	if ylabels is None: ylabels = map(fitlists[0].par_to_label,params)

	nrows = len(params)
	ncols = len(fitlists) if len(fitlists)>1 else maxpoints
	fig = pyplot.figure(figsize=(8*ncols//4,8*nrows//3))
	if title: fig.suptitle(title,fontsize=titlesize)
	gs = gridspec.GridSpec(nrows,1,wspace=0.1,hspace=0.1)
	minorlocator = AutoMinorLocator(2)
	
	xmain = scipy.arange(len(fitlists))
	xaux = scipy.linspace(-0.1,0.1,maxpoints)
	for ipar1,par1 in enumerate(params):
		ax = pyplot.subplot(gs[ipar1])
		for ifl,fl in enumerate(fitlists):
			if par1 not in fl: continue
			for ipoint,point in enumerate(fl[par1]):
				if errors[ipoint]: yerr = fl.errorbar(par1)[...,[ipoint]]
				else: yerr = None
				ax.errorbar(xmain[ifl]+xaux[ipoint],point,yerr=yerr,color=colors[ipoint],label=labels[ipoint] if ifl==0 else None,linestyle='none',**scatter_kwargs)
			if means: ax.errorbar(xmain[ifl],fl.mean(par1),yerr=fl.std(par1,error=means),linestyle='none',**means_points_kwargs)
		if truths[ipar1] is not None:
			ax.axhline(y=truths[ipar1],xmin=0.,xmax=1.,**truths_kwargs)
			if limits[ipar1] is not None:
				if limits[ipar1][-1] == 'abs': low,up = limits[ipar1][0],limits[ipar1][1]
				else: low,up = truths[ipar1]*(1-limits[ipar1][0]),truths[ipar1]*(1+limits[ipar1][1])
				#ax.axhline(y=[low,up],xmin=0.,xmax=1.,**limits_kwargs)
				ax.axhspan(low,up,**limits_kwargs)
		if (ipar1<nrows-1) or not ids: ax.get_xaxis().set_visible(False)
		else:
			ax.set_xticks(xmain)
			ax.set_xticklabels(ids,rotation=40,ha='right',fontsize=labelsize)
		ax.grid(True,axis='y')
		#ax.yaxis.set_minor_locator(minorlocator)
		ax.set_ylabel(ylabels[ipar1],fontsize=labelsize)
		ax.tick_params(labelsize=ticksize)
		if add_legend and ipar1==0: ax.legend(**{'loc':'upper left','ncol':1,'fontsize':labelsize,'framealpha':0.5,'frameon':True,'bbox_to_anchor':(1.04,1.)})
	
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_consensus(fitlists,params=[],labels=[],toplot=('values','values'),singles=[],regressions=False,diagonals=False,title='',colors=prop_cycle.by_key()['color'],scatter_kwargs={'s':3,'marker':'o','alpha':0.4,'edgecolors':'none'},singles_points_kwargs={},regressions_kwargs={'linestyle':'--','linewidth':2,'color':'k'},diagonals_kwargs={'linestyle':'--','linewidth':2,'color':'r'},path='consensus.png'):

	ticksize = 14
	labelsize = 16

	def form(tmp):
		if not isinstance(tmp,list):
			tmp = [tmp]
		return tmp

	def form_tuple(tmp):
		tmp = form(tmp)
		for it,t in enumerate(tmp):
			if not isinstance(t,tuple):
				tmp[it] = (t,t)
		return tmp

	fitlists = form_tuple(fitlists)
	params = form_tuple(params)
	toplot = form_tuple(toplot)
	singles = form_tuple(singles)
	regressions = form(regressions)
	diagonals = form(diagonals)

	fitlists = fitlists
	toplot = toplot*len(params)
	singles = singles + [None]*len(params)
	regressions = regressions + [regressions[-1]]*len(params)
	diagonals = diagonals + [diagonals[-1]]*len(params)
	labels = labels + [None]*len(params)

	singles_points = {'marker':'*','s':100,'color':'r','label':'data'}
	singles_points.update(singles_points_kwargs)
	singles_points_kwargs = singles_points
	regressions_ = {'linestyle':'--','linewidth':2,'color':'k','label':'regression'}
	regressions_.update(regressions_kwargs)
	regressions_kwargs = regressions_
	
	add_legend = bool(labels) or (any([s is not None for s in singles]) and (singles_points_kwargs.get('label',None) is not None)) or (any([s for s in regressions]) and (regressions_kwargs.get('label',None) is not None))
	label_regressions = regressions_kwargs.pop('label',None) 	

	nrows = 1 + len(params)//4
	ncols = int(scipy.ceil(len(params)*1./nrows))
	fig = pyplot.figure(figsize=(6*ncols,4*nrows))
	if title: fig.suptitle(title,fontsize=titlesize)
	gs = gridspec.GridSpec(nrows,ncols,wspace=0.3,hspace=0.3)

	def get_label_values(fl,par,tp):
		if tp == 'errors':
			label = '$\sigma_{{{0}}}$'.format(fl.par_to_latex(par))
			values = scipy.mean(fl.errorbar(par),axis=0)
		else:
			label = fl.par_to_label(par)
			values = fl[par]
		return label,values
	
	for ipar,(par1,par2) in enumerate(params):
		ax = pyplot.subplot(gs[ipar])
		plot_legend = ipar == 0
		for ifl,(fl1,fl2) in enumerate(fitlists):
			xlabel,values1 = get_label_values(fl1,par1,toplot[ipar][0])
			ylabel,values2 = get_label_values(fl2,par2,toplot[ipar][1])
			ax.scatter(values1,values2,color=colors[ifl],label=labels[ifl] if plot_legend else None,**scatter_kwargs)
		if singles[ipar] is not None: ax.scatter(x=singles[ipar][0],y=singles[ipar][1],**singles_points_kwargs)
		if (par1 == par2) and (toplot[ipar][0] == toplot[ipar][1]):
			xlim,ylim = ax.get_xlim(),ax.get_ylim()
			xylim = min(xlim[0],ylim[0]),max(xlim[1],ylim[1])
			#ax.set_xlim(xylim)
			#ax.set_ylim(xylim)
		xlim,ylim = [scipy.array(tmp) for tmp in [ax.get_xlim(),ax.get_ylim()]]
		if regressions[ipar]:
			a,b,r = stats.linregress(values1,values2)[:3]
			y = a*xlim + b
			if label_regressions is not None:
				label = '{} $\\rho = {:.4g}$'.format(label_regressions,r)
				plot_legend = True
			else: label = None
			ax.plot(xlim,y,label=label,**regressions_kwargs)
			ax.set_ylim(ylim)
		if diagonals[ipar]:
			ax.plot(xlim,xlim,**diagonals_kwargs)
			ax.set_ylim(ylim)
		ax.tick_params(labelsize=ticksize)
		ax.set_xlabel(xlabel,fontsize=labelsize)
		ax.set_ylabel(ylabel,fontsize=labelsize)
		if add_legend and plot_legend: ax.legend(**{'loc':'upper left','ncol':1,'fontsize':labelsize,'framealpha':0.5,'frameon':True})

	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_scatter_difference(fitlists,params=[],labels=[],truths=[],singles=[],singles_errors=[],means='real',errors=False,title='',colors=prop_cycle.by_key()['color'],scatter_kwargs={'s':3,'marker':'o','alpha':0.4,'edgecolors':'none','linestyle':'-'},means_points_kwargs={'linestyle':'-','elinewidth':1},singles_points_kwargs={},seed=42,path='difference.png'):

	singles_points = {'marker':'*','markersize':100,'color':'r','linestyle':'-','elinewidth':1,'label':'data'}
	singles_points.update(singles_points_kwargs)
	singles_points_kwargs = singles_points

	if not isinstance(fitlists,list): fitlists = [fitlists]
	truths = truths + [None]*len(params)
	singles = singles + [None]*len(params)
	singles_errors = singles_errors + [None]*len(params)
	for ipar,(sing,error) in enumerate(zip(singles,singles_errors)):
		if (sing is not None) and (error is not None):
			if scipy.isscalar(error): singles_errors[ipar] = [-error,error]
		
	add_legend = bool(labels) or (any([s is not None for s in singles]) and (singles_lines_kwargs.get('label',None) is not None))
	labels = labels + [None]*len(fitlists)
	
	fitlists = [fl.deepcopy() for fl in fitlists]
	for ifl,fl in enumerate(fitlists):
		for ipar1,par1 in enumerate(params):
			if truths[ipar1] is not None: fl[par1] -= truths[ipar1]

	fig = pyplot.figure(figsize=(2*len(params),4))
	if title: fig.suptitle(title,fontsize=titlesize)
	ax = fig.gca()

	xmain = scipy.arange(len(params))
	xaux = scipy.linspace(-1e-1,1e-1,len(fitlists)) if len(fitlists) > 1 else [0]
	rangex = (-5e-2,5e-2)
	offset_singles = 1e-2
	rng = scipy.random.RandomState(seed=seed)
	for ifl,fl in enumerate(fitlists):
		for ipar1,par1 in enumerate(params):
			x = ipar1 + xaux[ifl] + rng.uniform(*rangex,size=lf.size)
			ax.scatter(x,fl[par1],xerr=xerr,yerr=yerr,color=colors[ifl],label=labels[ifl],**scatter_kwargs)
			if means: ax.errorbar(ipar1+xaux[ifl],fl.mean(par1),xerr=None,yerr=fl.std(par1,error=means),color=colors[ifl],linestyle='none',**means_points_kwargs)
			if singles[ipar1] is not None:
				ax.errorbar(x=ipar1+xaux[ifl]+offset_singles,y=singles[ipar1],xerr=None,yerr=singles_errors[ipar1],linestyle='none',**singles_points_kwargs)

	ax.set_xticks(xmain)
	ax.set_xticklabels(map(fitlists[0].par_to_label,params),rotation=0,fontsize=labelsize)
	ax.grid(True,axis='y')
	ax.set_ylabel('$p - p_{\\mathrm{true}}$',fontsize=labelsize)
	ax.tick_params(labelsize=ticksize)
	
	if add_legend: ax.legend(**{'loc':'upper left','ncol':1,'fontsize':labelsize,'framealpha':0.5,'frameon':True,'bbox_to_anchor':(1.04,1.)})
	
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_gaussian_profile(ax,mean=0.,covariance=1.,minimum=0.,lim=None,**profile_kwargs):

	if lim is None: lim = ax.get_xlim()
	x = scipy.linspace(*lim,num=1000)
	y = (x-mean)**2/covariance + minimum
	plot = ax.plot(x,y,**profile_kwargs)

	return plot

def plot_profiles(profiles,nrows=1,params=[],labels=[],minima=[],latex={},truths=[],limits=[],nsigmas=[1,2],gaussian=None,title='',colors=prop_cycle.by_key()['color'],linestyles=['-'],truths_kwargs={'linestyle':'--','linewidth':2,'color':'k'},gaussian_profile_kwargs={},limits_kwargs={'facecolor':'k','alpha':0.1,'linewidth':0,'linestyle':None},sigmas_kwargs={'linestyle':'--','linewidth':2,'color':'k'},path='profiles.png'):

	profile = {'color':'r','label':'gaussian'}
	profile.update(gaussian_profile_kwargs)
	gaussian_profile_kwargs = profile

	if not isinstance(profiles,list): profiles = [profiles]
	minima = minima + [0.]*len(profiles)
	truths = truths + [None]*len(params)
	limits = limits + [None]*len(params)
	linestyles = linestyles*len(profiles)

	add_legend = bool(labels) or (gaussian is not None and gaussian_profile_kwargs.get('label',None) is not None)
	labels = labels + [None]*len(profiles)

	ncols = int(scipy.ceil(len(params)*1./nrows))
	fig = pyplot.figure(figsize=(12*ncols//2,12*nrows//2))
	if title: fig.suptitle(title,fontsize=titlesize)
	gs = gridspec.GridSpec(nrows,ncols,wspace=0.2,hspace=0.2)
	
	def data_to_axis(ax,y):
		axis_to_data = ax.transAxes + ax.transData.inverted()
		return axis_to_data.inverted().transform((0,y))[1]

	for ipar1,par1 in enumerate(params):
		ax = pyplot.subplot(gs[ipar1])
		for ipro,pro in enumerate(profiles):
			if par1 not in pro: continue
			ax.plot(pro[par1][0],pro[par1][1]-minima[ipro],color=colors[ipro],label=labels[ipro],linestyle=linestyles[ipro])
			#print pro[par1][1]-minima[ipro],minima[ipro]
		if gaussian is not None:
			plot_gaussian_profile(ax,gaussian['mean'][ipar1],gaussian['covariance'][ipar1],minimum=gaussian.get('minimum',0.),**gaussian_profile_kwargs)
		if truths[ipar1] is not None:
			ax.axvline(x=truths[ipar1],ymin=0.,ymax=1.,**truths_kwargs)
			if limits[ipar1] is not None:
				low,up = truths[ipar1]+limits[ipar1][0],truths[ipar1]+limits[ipar1][1]
				ax.axvspan(low,up,**limits_kwargs)
		for nsigma in nsigmas:
			y = nsigmas_to_deltachi2(nsigma,ndof=1)
			ax.axhline(y=y,xmin=0.,xmax=1.,**sigmas_kwargs)
			#ax.text(0.05,data_to_axis(ax,y),'${:d}\\sigma$'.format(nsigma),horizontalalignment='left',verticalalignment='bottom',transform=ax.transAxes,color='k',fontsize=labelsize)
			ax.text(0.05,y+0.1,'${:d}\\sigma$'.format(nsigma),horizontalalignment='left',verticalalignment='bottom',transform=transforms.blended_transform_factory(ax.transAxes,ax.transData),color='k',fontsize=labelsize)
		lim = ax.get_ylim()
		ax.set_ylim(0.,lim[-1]+2)
		ax.tick_params(labelsize=ticksize)
		ax.set_xlabel('${}$'.format(latex[par1]) if par1 in latex else par1,fontsize=labelsize)
		if ipar1==0: ax.set_ylabel('$\\Delta \\chi^{2}$',fontsize=labelsize)
		if add_legend and ipar1==0: ax.legend(**{'loc':'upper right','ncol':1,'fontsize':labelsize,'framealpha':0.5,'frameon':True})

	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)
