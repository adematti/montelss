import os
import logging
import scipy
from scipy import constants
from numpy import fft
import utils
import cosmology
from cosmology import Cosmology
from analyze_fits import EnsembleValues,Mesh,nsigmas_to_deltachi2,nsigmas_to_quantiles_1d,nsigmas_to_quantiles_1d_sym

class AnalyzeMCMC(EnsembleValues):
	
	logger = logging.getLogger('AnalyzeMCMC')
	STEP_VERBOSE = 10

	@classmethod
	def load_cosmomc(cls,path_chains,path_params):
		
		self = cls()
		params = []
		self.latex = {}
		
		self.logger.info('Loading parameters file: {}.'.format(path_params))
		with open(path_params) as file:
			for line in file:
				key,ltx = line.split('\t')
				params.append(key)
				self.latex[key] = ltx.replace('\n','')
		
		chains = []
		for path in path_chains:
			self.logger.info('Loading chain file: {}.'.format(path))
			chains.append(scipy.loadtxt(path,unpack=True))
		
		chains = scipy.concatenate(chains,axis=-1)[2:]
		self.values = {par:values for par,values in zip(params,chains)}
		
		return self

	@classmethod
	def load_montelss(cls,path,params=None,walkers=None):		
		self = cls()
		from montelss import Sampler
		sampler = Sampler.load(path)
		self.values = sampler.chain.as_dict(parameters=params,walkers=walkers,flat=True)
		self.latex = sampler.latex
		#self.latex['lnlkl'] = '\\ln{\\mathcal{L}}'
		#self.sorted = sampler.sorted
		return self

	@classmethod
	def load_csv(cls,path,params,latex={},**kwargs):
		columns = scipy.loadtxt(path,unpack=True,**kwargs)
		self = cls()
		self.latex = latex
		self.values = {}
		for icol,col in enumerate(columns):
			self[params[icol]] = col
		return self

	def cosmomc_to_class(self):
		self.values,self.latex = cosmology.cosmomc_to_class(self.values,self.latex)
		self.logger.info('Keeping Class parameters: {}.'.format(' '.join(self.parameters)))

	def run_class(self,cosmo,todo,todo_kwargs={}):
		extra = []
		step_verbose = max(len(self)*self.STEP_VERBOSE/100,1);
		for ipoint,params in enumerate(self.points()):
			if (scipy.mod(ipoint,step_verbose) == 0): self.logger.info('Computation done at {:d} percent ({:d}/{:d}).'.format(ipoint*self.STEP_VERBOSE/step_verbose,ipoint,len(self)))
			cosmoc = cosmo.copy()
			cosmoc.set_params(**params)
			extra.append(todo(cosmoc,**todo_kwargs))
			cosmoc.struct_cleanup()
			cosmoc.empty()
			#cosmo.set_params(**params)
			#extra.append(todo(cosmo,**todo_kwargs))
		for par in extra[0]:
			self[par] = scipy.asarray([ext[par] for ext in extra])
		return extra[0]
	
	def run_class_fs8_bao(self,cosmo=Cosmology.Planck2018(),z=0.):
		extra = self.run_class(cosmo,Cosmology.compute_fs8_bao,todo_kwargs={'z':z})
		for par in extra: self.latex[par] = cosmology.par_to_latex(par) + '(z={:.3f})'.format(z)

	def save_getdist(self,base,params=None,derived=None,ranges=None,lnposterior='lnposterior',weight='weight',ichain=None,fmt='%.8e',delimiter=' ',**kwargs):
		if params is None: params = self.parameters
		data = [self.values.get(weight,self.ones()),-self.values.get(lnposterior,self.zeros())] + self[params]
		data = scipy.array(data).T
		utils.mkdir(os.path.dirname(base))
		path_chain = '{}.txt'.format(base) if ichain is None else '{}_{:d}.txt'.format(base,ichain)
		self.logger.info('Saving chain to {}.'.format(path_chain))
		scipy.savetxt(path_chain,data,header='',fmt=fmt,delimiter=delimiter,**kwargs)
		if not isinstance(derived,list): derived = [derived]
		derived = derived + [False]*len(params)
		output = ''
		for par,der in zip(params,derived):
			if der: output += '{}* {}\n'.format(par,self.par_to_latex(par))
			else: output += '{} {}\n'.format(par,self.par_to_latex(par))
		path_params = '{}.paramnames'.format(base)
		self.logger.info('Saving parameter names to {}.'.format(path_params))
		with open(path_params,'w') as file:
			file.write(output)
		if ranges is not None:
			output = ''
			for par,ran in zip(params,ranges):
				if not isinstance(ran,tuple): ran = (ran,ran)
				ran = tuple('N' if r is None or r == scipy.inf else r for r in ran)
				output += '{} {} {}\n'.format(par,ran[0],ran[1])
			path_ranges = '{}.ranges'.format(base)
			self.logger.info('Saving parameter ranges to {}.'.format(path_ranges))
			with open(path_ranges,'w') as file:
				file.write(output)

from scipy import stats
from matplotlib import pyplot,gridspec,cm,patches
from matplotlib.ticker import MaxNLocator,AutoMinorLocator
from matplotlib.colors import Normalize

def lighten_color(color,amount=0.5):
	"""Lightens the given color by multiplying (1-luminosity) by the given amount.
	Input can be matplotlib color string, hex string, or RGB tuple.

	Examples:
	>> lighten_color('g', 0.3)
	>> lighten_color('#F034A3', 0.6)
	>> lighten_color((.3,.55,.1), 0.5)

	"""
	import matplotlib.colors as mc
	import colorsys
	try:
		c = mc.cnames[color]
	except:
		c = color
	c = colorsys.rgb_to_hls(*mc.to_rgb(c))
	lum = 1 - amount * (1 - c[1]) if amount > 0 else - amount * c[1]
	return colorsys.hls_to_rgb(c[0], lum, c[2])

def plot_density_contour(ax,x,y,pdf,bins=[20,20],nsigmas=2,ndof=2,color='blue',lighten=-0.5,**contour_kwargs):
	
	from scipy import stats,optimize	
	
	"""Create a density contour plot.

	Parameters
	----------
	x,y,pdf : 2d density 
	bins : bins
	ax : matplotlib.Axes
		plots the contour to this axis.
	contour_kwargs : dict
		kwargs to be passed to pyplot.contourf()

	"""
	if scipy.isscalar(nsigmas): nsigmas = 1 + scipy.arange(nsigmas)
	pdf = pdf/pdf.sum()

	def find_confidence_interval(x, confidence_level):
		return pdf[pdf > x].sum() - confidence_level

	to_quantiles = nsigmas_to_quantiles_2d if ndof == 1 else nsigmas_to_quantiles_1d
	levels = [optimize.brentq(find_confidence_interval,0.,1.,args=(to_quantiles(nsigma))) for nsigma in nsigmas]
	if isinstance(color,list):
		colors = color
	else:
		colors = [color]
		for icol in nsigmas[:-1]: colors.append(lighten_color(colors[-1],amount=lighten))

	contour = ax.contourf(x,y,pdf.T,levels=levels[::-1]+[pdf.max()+1.],colors=colors,**contour_kwargs)

	return contour

def plot_gaussian_contour(ax,mean=[0.,0.],covariance=[1.,1.,0.],nsigmas=2,ndof=2,color='blue',lighten=-0.5,**contour_kwargs):
	
	if scipy.isscalar(nsigmas): nsigmas = 1 + scipy.arange(nsigmas)
	if isinstance(color,list):
		colors = color
	else:
		colors = [color]
		for icol in nsigmas[:-1]: colors.append(lighten_color(colors[-1],amount=lighten))
		colors = colors[::-1]

	radii = scipy.sqrt(nsigmas_to_deltachi2(nsigmas,ndof=ndof))
	t = scipy.linspace(0.,2.*constants.pi,1000,endpoint=False)
	ct = scipy.cos(t)
	st = scipy.sin(t)
	sigx2,sigy2,sigxy = covariance

	for radius,color in zip(radii,colors):
		a = radius * scipy.sqrt(0.5 * (sigx2 + sigy2) + scipy.sqrt(0.25 * (sigx2 - sigy2)**2. + sigxy**2.))
		b = radius * scipy.sqrt(0.5 * (sigx2 + sigy2) - scipy.sqrt(0.25 * (sigx2 - sigy2)**2. + sigxy**2.))
		th = 0.5 * scipy.arctan2(2. * sigxy, sigx2 - sigy2)
		x = mean[0] + a * ct * scipy.cos(th) - b * st * scipy.sin(th)
		y = mean[1] + a * ct * scipy.sin(th) + b * st * scipy.cos(th)
		x,y = (scipy.concatenate([x_,x_[:1]],axis=0) for x_ in (x,y))
		ax.plot(x,y,color=color,**contour_kwargs)


def plot_density_profile(ax,x,pdf,normalize='max',**profile_kwargs):

	pdf = pdf/pdf.sum()

	if normalize == 'max': pdf /= pdf.max()
	plot = ax.plot(x,pdf,**profile_kwargs)
	ax.set_ylim(bottom=0)

	return plot

def plot_gaussian_profile(ax,mean=0.,covariance=1.,lim=None,normalize='max',**profile_kwargs):

	if lim is None: lim = ax.get_xlim()
	x = scipy.linspace(*lim,num=1000)
	y = scipy.exp(-(x-mean)**2/2./covariance)
	if normalize != 'max': y *= 1./(covariance*scipy.sqrt(2.*constants.pi))
	plot = ax.plot(x,y,**profile_kwargs)

	return plot

def plot_other_contour(ax,contour,nsigmas=2,color='blue',lighten=-0.5,**contour_kwargs):
	
	if scipy.isscalar(nsigmas): nsigmas = 1 + scipy.arange(nsigmas)
	if isinstance(color,list):
		colors = color
	else:
		colors = [color]
		for icol in nsigmas[:-1]: colors.append(lighten_color(colors[-1],amount=lighten))
		colors = colors[::-1]

	for nsigma,color in zip(nsigmas,colors):
		xy = contour[nsigma]
		x,y = (scipy.concatenate([xy_,xy_[:1]],axis=0) for xy_ in xy)
		ax.plot(x,y,color=color,**contour_kwargs)

def plot_other_profile(ax,xy,normalize=None,**profile_kwargs):

	if normalize == 'max': xy[1] /= xy[1].max()
	plot = ax.plot(xy[0],xy[1],**profile_kwargs)

	return plot

prop_cycle = pyplot.rcParams['axes.prop_cycle']
figsize = (10,8)
fontsize = 18
titlesize = 20
labelsize = 18
ticksize = 14
dpi = 200

def plot_corner_chains(chains,params=[],labels=[],truths=[],gaussian=None,other_profile=None,other_contour=None,title='',colors=prop_cycle.by_key()['color'],truths_kwargs={'linestyle':'--','linewidth':1,'color':'k'},profile_kwargs={},contour_kwargs={},gaussian_profile_kwargs={},gaussian_contour_kwargs={},other_profile_kwargs={},other_contour_kwargs={},path='corner.png'):


	keywords_plot = ['color','alpha','linestyle','normalize']	
	
	profile = {'bins':60,'method':'gaussian_kde','normalize':'max'}
	profile.update(profile_kwargs)
	profile_kwargs = profile
	contour = {'nsigmas':2,'bins':30,'alpha':0.8,'method':'gaussian_kde'}
	contour.update(contour_kwargs)
	contour_kwargs = contour

	profile = {'normalize':'max','color':'r','label':'gaussian'}
	profile.update(gaussian_profile_kwargs)
	gaussian_profile_kwargs = profile
	contour = {'nsigmas':2,'color':'r'}
	contour.update(gaussian_contour_kwargs)
	gaussian_contour_kwargs = contour

	profile = {'normalize':'','color':'r','label':None}
	profile.update(other_profile_kwargs)
	other_profile_kwargs = profile
	contour = {'color':'r'}
	contour.update(other_contour_kwargs)
	other_contour_kwargs = contour

	handles = []
	if (gaussian is not None) and (gaussian_profile_kwargs.get('label',None) is not None):
		handles.append(patches.Patch(color=gaussian_profile_kwargs['color'],label=gaussian_profile_kwargs['label']))
	if (other_profile is not None) and (other_profile_kwargs.get('label',None) is not None):
		handles.append(patches.Patch(color=other_profile_kwargs['color'],label=other_profile_kwargs['label']))
	
	if other_profile is None: other_profile = {par1:None for par1 in params}
	for par1 in params:
		if par1 not in other_profile: other_profile[par1] = None
	if other_contour is None: other_contour = {}
	for par1 in params:
		for par2 in params:
			if other_contour.get((par1,par2),None) is not None:
				other_contour[(par2,par1)] = other_contour[(par1,par2)]
			else: other_contour[(par2,par1)] = None
	
	add_legend = bool(labels) or bool(handles)
	labels = labels + [None]*len(chains)

	ncols = nrows = len(params)
	fig = pyplot.figure(figsize=(figsize[0]*nrows/3.,figsize[1]*ncols/3.))
	if title: fig.suptitle(title,fontsize=fontsize)
	gs = gridspec.GridSpec(nrows,ncols,wspace=0.1,hspace=0.1)
	minorlocator = AutoMinorLocator(2)
	ticklabel_kwargs = {'scilimits':(-3,3)}
	
	xlims,xticks = [],{True:[],False:[]}
	for ipar1,par1 in enumerate(params):
		ax = pyplot.subplot(gs[ipar1,ipar1])
		for ichain,chain in enumerate(chains):
			if not isinstance(chain,Mesh):
				mesh = chain.to_mesh(params=[par1],**{key:val for key,val in profile_kwargs.items() if key not in keywords_plot})
			else: mesh = chain
			x,pdf = mesh(params=[par1])
			plot_density_profile(ax,x,pdf,color=colors[ichain],label=labels[ichain],**{key:val for key,val in profile_kwargs.items() if key in keywords_plot})
			if labels[ichain] is not None: handles.append(patches.Patch(color=colors[ichain],label=labels[ichain],alpha=1))
		if gaussian is not None:
			plot_gaussian_profile(ax,gaussian['mean'][ipar1],gaussian['covariance'][ipar1][ipar1],**gaussian_profile_kwargs)
		if other_profile[par1] is not None:
			plot_other_profile(ax,other_profile[par1],**other_profile_kwargs)
		if truths: ax.axvline(x=truths[ipar1],ymin=0.,ymax=1.,**truths_kwargs)
		if ipar1<len(params)-1: ax.get_xaxis().set_visible(False)
		else: ax.set_xlabel(chains[0].par_to_label(par1),fontsize=fontsize)
		ax.get_yaxis().set_visible(False)
		ax.tick_params(labelsize=ticksize)
		xlims.append(ax.get_xlim())
		ax.xaxis.set_major_locator(MaxNLocator(nbins=3,min_n_ticks=2,prune='both'))
		ax.xaxis.set_minor_locator(minorlocator)
		for minor in xticks: xticks[minor].append(ax.get_xticks(minor=minor))
		ax.ticklabel_format(**ticklabel_kwargs)
		if add_legend and ipar1==0: ax.legend(**{'loc':'upper left','ncol':1,'fontsize':labelsize,'framealpha':0.5,'frameon':True,'bbox_to_anchor':(1.04,1.),'handles':handles})
	
	for ipar1,par1 in enumerate(params):
		for ipar2,par2 in enumerate(params):
			if nrows-1-ipar2 >= ncols-1-ipar1: continue
			ax = pyplot.subplot(gs[ipar2,ipar1])
			for ichain,chain in enumerate(chains):
				if not isinstance(chain,Mesh):
					mesh = chain.to_mesh(params=[par1,par2],**{key:val for key,val in profile_kwargs.items() if key not in keywords_plot})
				else: mesh = chain
				(x,y),pdf = mesh(params=[par1,par2])
				plot_density_contour(ax,x,y,pdf,color=colors[ichain],**contour_kwargs)
			if gaussian is not None:
				mean = [gaussian['mean'][ipar] for ipar in [ipar1,ipar2]]
				covariance = [gaussian['covariance'][ipar][ipar] for ipar in [ipar1,ipar2]] + [gaussian['covariance'][ipar1][ipar2]]
				plot_gaussian_contour(ax,mean,covariance,**gaussian_contour_kwargs)
			if other_contour[(par1,par2)] is not None:
				plot_other_contour(ax,other_contour[(par1,par2)],**other_contour_kwargs)
			if truths:
				ax.axvline(x=truths[ipar1],ymin=0.,ymax=1.,**truths_kwargs)
				ax.axhline(y=truths[ipar2],xmin=0.,xmax=1.,**truths_kwargs)
			if ipar1>0: ax.get_yaxis().set_visible(False)
			else: ax.set_ylabel(chains[0].par_to_label(par2),fontsize=fontsize)
			if nrows-1-ipar2>0: ax.get_xaxis().set_visible(False)
			else: ax.set_xlabel(chains[0].par_to_label(par1),fontsize=fontsize)
			ax.tick_params(labelsize=ticksize)
			for minor in xticks:
				ax.set_xticks(xticks[minor][ipar1],minor=minor)
				ax.set_yticks(xticks[minor][ipar2],minor=minor)
			ax.ticklabel_format(**ticklabel_kwargs)
			ax.set_xlim(xlims[ipar1])
			ax.set_ylim(xlims[ipar2])
	
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def corner(chain,params=None,truths=[],path='corner.png'):
	
	import corner
	if params is None: params = chains[0].parameters
	fig = corner.corner(scipy.asarray([chain[par] for par in params]).T,bins=30,labels=[chain.par_to_label(par) for par in params],truths=truths,show_titles=True,title_kwargs=dict(fontsize=fontsize))
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_chain(chain,params=None,title=None,path='chains.png'):
	
	if params is None: params = chains[0].parameters
	nparams = len(params)
	steps = 1+scipy.arange(len(chain))
	
	fig,lax = pyplot.subplots(nparams,sharex=True,sharey=False,figsize=(8,1.5*nparams),squeeze=True)
	if title is not None: fig.suptitle(title,fontsize=fontsize)
	
	for ax,par in zip(lax,params):
		ax.grid(True)
		ax.tick_params(labelsize=labelsize)
		ax.set_ylabel(chain.par_to_label(par),fontsize=fontsize)
		ax.plot(steps,chain[par],color='black')
		
	lax[-1].set_xlabel('step',fontsize=fontsize)
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def univariate_gelman_rubin(chains):
	"""
	http://www.stat.columbia.edu/~gelman/research/published/brooksgelman2.pdf
	dim 0: nchains
	dim 1: nsteps
	"""
	nchains = len(chains)
	mean = scipy.asarray([scipy.mean(chain,axis=0) for chain in chains])
	variance = scipy.asarray([scipy.var(chain,ddof=1,axis=0) for chain in chains])
	nsteps = scipy.asarray([len(chain) for chain in chains])
	Wn1 = scipy.mean(variance)
	Wn = scipy.mean((nsteps-1.)/nsteps*variance)
	B = scipy.var(mean,ddof=1)
	V = Wn + (nchains+1.)/nchains*B
	return scipy.sqrt(V/Wn1)

def multivariate_gelman_rubin(chains):
	"""
	http://www.stat.columbia.edu/~gelman/research/published/brooksgelman2.pdf
	dim 0: nchains
	dim 1: nsteps
	dim 2: ndim
	"""
	nchains = len(chains)
	mean = scipy.asarray([scipy.mean(chain,axis=0) for chain in chains])
	variance = scipy.asarray([scipy.cov(chain.T,ddof=1) for chain in chains])
	nsteps = scipy.asarray([len(chain) for chain in chains])
	Wn1 = scipy.mean(variance,axis=0)
	Wn = scipy.mean(((nsteps-1.)/nsteps)[:,None,None]*variance,axis=0)
	B = scipy.cov(mean.T,ddof=1)
	V = Wn + (nchains+1.)/nchains*B
	invWn1 = scipy.linalg.inv(Wn1)
	assert scipy.absolute(Wn1.dot(invWn1)-scipy.eye(Wn1.shape[0])).max() < 1e-5
	eigen = scipy.linalg.eigvalsh(invWn1.dot(V))
	return eigen.max()


def plot_gelman_rubin(chains,params=None,title=None,multivariate=False,threshold=1.1,threshold_kwargs={'linestyle':'--','linewidth':1,'color':'k'},path='gelman-rubin.png'):
	
	nsteps = scipy.amin([len(chain) for chain in chains])
	ends = scipy.arange(100,nsteps,500)
	if params is None: params = chains[0].parameters
	GRmulti = []
	GR = {par:[] for par in params}
	for end in ends:
		if multivariate: GRmulti.append(multivariate_gelman_rubin([scipy.asarray([chain[par][:end] for par in params]).T for chain in chains]))
		for par in GR: GR[par].append(univariate_gelman_rubin([chain[par][:end] for chain in chains]))
	for par in GR: GR[par] = scipy.asarray(GR[par])
	
	fig = pyplot.figure(figsize=figsize)
	if title is not None: fig.suptitle(title,fontsize=fontsize)
	ax = pyplot.gca()
	ax.grid(True)
	ax.tick_params(labelsize=labelsize)
	ax.set_xlabel('step',fontsize=fontsize)
	ax.set_ylabel('$\\hat{R}$',fontsize=fontsize)
	
	if multivariate: ax.plot(ends,GRmulti,label='multi',linestyle='-',linewidth=1,color='k')
	for par in params:
		ax.plot(ends,GR[par],label=chains[0].par_to_label(par),linestyle='--',linewidth=1)
	ax.axhline(y=threshold,xmin=0.,xmax=1.,**threshold_kwargs)
	ax.legend(**{'loc':1,'ncol':1,'fontsize':labelsize,'framealpha':0.5,'frameon':False})
	
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def autocorrelation_1d(x):
	"""Estimate the normalized autocorrelation function of a 1-D series

	Args:
		x: The series as a 1-D numpy array.

	Returns:
		array: The autocorrelation function of the time series.
	
	Taken from: https://github.com/dfm/emcee/blob/master/emcee/autocorr.py
	"""
	x = scipy.atleast_1d(x)
	if len(x.shape) != 1:
		raise ValueError("invalid dimensions for 1D autocorrelation function")

	def next_pow_two(n):
		"""Returns the next power of two greater than or equal to `n`"""
		i = 1
		while i < n:
			i = i << 1
		return i
	n = next_pow_two(len(x))

	# Compute the FFT and then (from that) the auto-correlation function
	f = fft.fft(x - scipy.mean(x), n=2*n)
	acf = fft.ifft(f * scipy.conjugate(f))[:len(x)].real
	
	acf /= acf[0]
	return acf

def integrated_autocorrelation_time(x, c=5, tol=50, quiet=False):
	"""Estimate the integrated autocorrelation time of a time series.

	This estimate uses the iterative procedure described on page 16 of
	`Sokal's notes <http://www.stat.unc.edu/faculty/cji/Sokal.pdf>`_ to
	determine a reasonable window size.

	Args:
		x: The time series. If multidimensional, set the time axis using the
			``axis`` keyword argument and the function will be computed for
			every other axis.
		c (Optional[float]): The step size for the window search. (default:
			``5``)
		tol (Optional[float]): The minimum number of autocorrelation times
			needed to trust the estimate. (default: ``50``)
		quiet (Optional[bool]): This argument controls the behavior when the
			chain is too short. If ``True``, give a warning instead of raising
			an :class:`AutocorrError`. (default: ``False``)

	Returns:
		float or array: An estimate of the integrated autocorrelation time of
			the time series ``x`` computed along the axis ``axis``.

	Raises
		ValueError: If the autocorrelation time can't be reliably estimated
			from the chain and ``quiet`` is ``False``. This normally means
			that the chain is too short.

	Taken from: https://github.com/dfm/emcee/blob/master/emcee/autocorr.py
	"""
	x = scipy.atleast_1d(x)
	if len(x.shape) == 1:
		x = x[:, scipy.newaxis, scipy.newaxis]
	if len(x.shape) == 2:
		x = x[:, :, scipy.newaxis]
	if len(x.shape) != 3:
		raise ValueError("Invalid dimensions")

	n_t, n_w, n_d = x.shape
	tau_est = scipy.empty(n_d)
	windows = scipy.empty(n_d, dtype=int)
	
	def auto_window(taus, c):
		m = scipy.arange(len(taus)) < c * taus
		if scipy.any(m):
			return scipy.argmin(m)
		return len(taus) - 1

	# Loop over parameters
	for d in range(n_d):
		f = scipy.zeros(n_t)
		for k in range(n_w):
			f += autocorrelation_1d(x[:, k, d])
		f /= n_w
		taus = 2.0*scipy.cumsum(f)-1.0
		windows[d] = auto_window(taus, c)
		tau_est[d] = taus[windows[d]]

	# Check convergence
	flag = tol * tau_est > n_t

	# Warn or raise in the case of non-convergence
	if scipy.any(flag):
		msg = (
			"The chain is shorter than {:d} times the integrated "
			"autocorrelation time for {:d} parameter(s). Use this estimate "
			"with caution and run a longer chain!\n"
		).format(tol, scipy.sum(flag))
		msg += "N/{:d} = {:.0f};\ntau: {}".format(tol,n_t*1./tol,tau_est)
		if not quiet: raise ValueError(msg)
		logger.warning(msg)

	return tau_est

def plot_autocorrelation_time(chains,params=None,title=None,threshold=50,threshold_kwargs={'linestyle':'--','linewidth':1,'color':'k'},path='autocorrelation-time.png'):
	
	nsteps = scipy.amin([len(chain) for chain in chains])
	ends = scipy.arange(100,nsteps,500)
	if params is None: params = chains[0].parameters

	autocorr = {par:[] for par in params}
	for end in ends:
		for par in autocorr:
			autocorr[par].append(integrated_autocorrelation_time(scipy.asarray([chain[par][:end] for chain in chains]).T,tol=0))
	for par in autocorr: autocorr[par] = scipy.asarray(autocorr[par])
	
	fig = pyplot.figure(figsize=figsize)
	if title is not None: fig.suptitle(title,fontsize=fontsize)
	ax = pyplot.gca()
	ax.grid(True)
	ax.tick_params(labelsize=labelsize)
	ax.set_xlabel('step $N$',fontsize=fontsize)
	ax.set_ylabel('$\\tau$',fontsize=fontsize)
	
	for par in params:
		ax.plot(ends,autocorr[par],label=chains[0].par_to_label(par),linestyle='--',linewidth=1)
	ax.plot(ends,ends*1./threshold,label='$N/{:d}$'.format(threshold),**threshold_kwargs)
	ax.legend(**{'loc':2,'ncol':1,'fontsize':labelsize,'framealpha':0.5,'frameon':False})
	
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)

