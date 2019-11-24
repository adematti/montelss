import os
import scipy
from scipy import integrate,constants
import logging
from matplotlib import pyplot
from matplotlib.ticker import AutoMinorLocator,LogLocator,FixedLocator,NullFormatter
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib import gridspec
from pyspectrum import *
from likelihood import LikelihoodClustering
import utils

logger = logging.getLogger('Plot')

fontsize = 20
labelsize = 16
figsize = (8,6)
dpi = 200
prop_cycle = pyplot.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
#linestyles = [(0,()),(0,(1,5)),(0,(5,5)),(0,(3,5,1,5)),(0,(3,5,1,5,1,5))]
linestyles = ['-','--',':','-.']
scalings = {}
scalings['rloglin'] = {'func':lambda x,y: (x,y),'xlabel':utils.plot_xlabel(estimator='rr'),'ylabel':'$\\xi_{{\\ell}}(s)$','xscale':'log','yscale':'linear','xlim':[1e-3,3000.]}
scalings['rlogr2lin'] = {'func':lambda x,y: (x,x**2*y),'xlabel':utils.plot_xlabel(estimator='rr'),'ylabel':'$s^{2}\\xi_{{\\ell}}(s) [(\\mathrm{Mpc} \ h^{-1})^{2}]$','xscale':'log','yscale':'linear','xlim':[1e-3,3000.]}
scalings['rlinr2lin'] = {'func':lambda x,y: (x,x**2*y),'xlabel':utils.plot_xlabel(estimator='rr'),'ylabel':'$s^{2}\\xi_{{\\ell}}(s) [(\\mathrm{Mpc} \ h^{-1})^{2}]$','xscale':'linear','yscale':'linear','xlim':[1e-3,3000.]}
scalings['slinlin'] = {'func':lambda x,y: (x,y),'xlabel':utils.plot_xlabel(estimator='rr'),'ylabel':'$\mathcal{W}_{{\\ell}}(s)$','xscale':'linear','yscale':'linear','xlim':[1e-3,200.]}
scalings['sloglin'] = {'func':lambda x,y: (x,y),'xlabel':utils.plot_xlabel(estimator='rr'),'ylabel':'$\mathcal{W}_{{\\ell}}(s)$','xscale':'log','yscale':'linear','xlim':[1e-3,3000.]}
scalings['slogslin'] = {'func':lambda x,y: (x,x*y),'xlabel':utils.plot_xlabel(estimator='rr'),'ylabel':'$s\mathcal{W}_{{\\ell}}(s)$','xscale':'log','yscale':'linear','xlim':[1e-3,3000.]}
scalings['slogs2lin'] = {'func':lambda x,y: (x,x**2*y),'xlabel':utils.plot_xlabel(estimator='rr'),'ylabel':'$s^{2}\mathcal{W}_{{\\ell}}(s)$','xscale':'log','yscale':'linear','xlim':[1e-3,3000.]}
scalings['slogs3lin'] = {'func':lambda x,y: (x,x**3*y),'xlabel':utils.plot_xlabel(estimator='rr'),'ylabel':'$s^{3}\mathcal{W}_{{\\ell}}(s)$','xscale':'log','yscale':'linear','xlim':[1e-3,3000.]}
scalings['sloglog'] = {'func':lambda x,y: (x,y),'xlabel':utils.plot_xlabel(estimator='rr'),'ylabel':'$\mathcal{W}_{{\\ell}}(s)$','xscale':'log','yscale':'log','xlim':[1e-3,3000.]}
scalings['slogs2log'] = {'func':lambda x,y: (x,x**2*y),'xlabel':utils.plot_xlabel(estimator='rr'),'ylabel':'$s^{2}\mathcal{W}_{{\\ell}}(s)$','xscale':'log','yscale':'log','xlim':[1e-3,3000.]}
scalings['klinlin'] = {'func':lambda x,y: (x,y),'xlabel':utils.plot_xlabel(estimator='spectrum'),'ylabel':'$P_{{\\ell}}(k)$ [$(\\mathrm{Mpc} \ h^{-1})^{3}$]','xscale':'linear','yscale':'linear','xlim':[0.,0.4]}
scalings['klinlog'] = {'func':lambda x,y: (x,y),'xlabel':utils.plot_xlabel(estimator='spectrum'),'ylabel':'$P_{{\\ell}}(k)$ [$(\\mathrm{Mpc} \ h^{-1})^{3}$]','xscale':'linear','yscale':'log','xlim':[0.,0.4]}
scalings['klinklin'] = {'func':lambda x,y: (x,x*y),'xlabel':utils.plot_xlabel(estimator='spectrum'),'ylabel':'$kP_{{\\ell}}(k)$ [$(\\mathrm{Mpc} \ h^{-1})^{2}$]','xscale':'linear','yscale':'linear','xlim':[0.,0.4]}
scalings['kloglog'] = {'func':lambda x,y: (x,y),'xlabel':utils.plot_xlabel(estimator='spectrum'),'ylabel':'$P_{{\\ell}}(k)$ [$(\\mathrm{Mpc} \ h^{-1})^{3}$]','xscale':'log','yscale':'log','xlim':[1e-3,1]}
scalings['klogk2log'] = {'func':lambda x,y: (x,x**2*y),'xlabel':utils.plot_xlabel(estimator='spectrum'),'ylabel':'$k^{2}P_{{\\ell}}(k)$ [$\\mathrm{Mpc} \ h^{-1}$]','xscale':'log','yscale':'log','xlim':[1e-3,1]}
scalings['kloglin'] = {'func':lambda x,y: (x,y),'xlabel':utils.plot_xlabel(estimator='spectrum'),'ylabel':'$P_{{\\ell}}(k)$ [$(\\mathrm{Mpc} \ h^{-1})^{3}$]','xscale':'log','yscale':'linear','xlim':[1e-3,1]}
scalings['klogklin'] = {'func':lambda x,y: (x,x*y),'xlabel':utils.plot_xlabel(estimator='spectrum'),'ylabel':'$kP_{{\\ell}}(k)$ [$(\\mathrm{Mpc} \ h^{-1})^{2}$]','xscale':'log','yscale':'linear','xlim':[1e-3,1]}
fiducial_params = dict(f=0.75,b1=2.,b2=1.,sigmav=4,FoG='lorentzian2')
#fiducial_params = dict(f=0.8,b1=1.4,b2=1.,sigmav=4,FoG='lorentzian2')

def sort_legend(ax,**kwargs):
	handles,labels = ax.get_legend_handles_labels()
	labels,handles = zip(*sorted(zip(labels,handles),key=lambda t: utils.str_to_int(t[0])[-1]))
	ax.legend(handles,labels,**kwargs)

def text_to_latex(txt):
	if txt in range(0,20,1): return '$\\ell = {:d}$'.format(txt)

def plot_baseline(scaling,title=''):
	fig = pyplot.figure(figsize=figsize)
	if title: fig.suptitle(title,fontsize=fontsize)
	ax = pyplot.gca()
	if scaling['xlim']: ax.set_xlim(scaling['xlim'])
	ax.set_xscale(scaling['xscale'])
	ax.set_yscale(scaling['yscale'])
	ax.grid(True)
	ax.tick_params(labelsize=labelsize)
	ax.set_xlabel(scaling['xlabel'],fontsize=fontsize)
	ax.set_ylabel(scaling['ylabel'],fontsize=fontsize)
	return ax

def plot_effect_ap_vary(parameters,scale='klinklin',title='Alcock-Paczynski',path='effect_ap_vary.png'):

	model = ModelTNS.load(parameters['ModelTNS']['save'])
	effect_ap = EffectAP.load(parameters['EffectAP']['save'])
	scaling = scalings[scale]
	ax = plot_baseline(scaling,title)
	effect_ap.set_input_model(model.spectrum_galaxy)
	kwargs = fiducial_params.copy()
	f = kwargs.pop('f')
	qgrid = [[1.,1.],[0.95,1.05],[1.05,0.95]]
	fgrid = [f,0.95*f,1.05*f]
	ax.set_xlim([0.,0.25])
	
	for ill,ell in enumerate(effect_ap.ells):
		params = utils.plot_params(mode='multipole',ell=(ill,ell))
		tmp = [effect_ap.spectrum_multipoles(f=fgrid[0],qpar=qpar,qper=qper,**kwargs)[0][ill] for qpar,qper in qgrid]
		#ax.plot(*scaling['func'](effect_ap.k,tmp[0]),label=utils.text_to_latex(ell))
		ax.fill_between(effect_ap.k,scaling['func'](effect_ap.k,tmp[1])[1],scaling['func'](effect_ap.k,tmp[2])[1],facecolor=params['color'],alpha=0.3,linewidth=0,label=params['label'])
		tmp = [effect_ap.spectrum_multipoles(f=f,qpar=qgrid[0][0],qper=qgrid[0][1],**kwargs)[0][ill] for f in fgrid[1:]]
		ax.fill_between(effect_ap.k,scaling['func'](effect_ap.k,tmp[0])[1],scaling['func'](effect_ap.k,tmp[1])[1],facecolor=params['color'],alpha=0.6,linewidth=0)

	ax.legend(**{'loc':1,'ncol':1,'fontsize':15,'framealpha':0.5,'frameon':False})
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_effect_ap_multipoles(parameters,scale='klinklin',title='Multipoles',path='effect_ap_multipoles.png'):

	model = ModelTNS.load(parameters['ModelTNS']['save'])
	effect_ap = EffectAP.load(parameters['EffectAP']['save'])
	scaling = scalings[scale]
	ax = plot_baseline(scaling,title)
	effect_ap.set_input_model(model.spectrum_galaxy)
	kwargs = dict(f=0.84,b1=1.2,b2=-0.4,sigmav=3.2,convolution=False,corrconstraint='none')
	spectrum = effect_ap.spectrum_multipoles(**kwargs)[0]
	ax.set_xlim([0.,0.4])
	
	for ill,ell in enumerate(effect_ap.ells):
		params = utils.plot_params(mode='multipole',ell=(ill,ell))
		ax.plot(*scaling['func'](effect_ap.k,spectrum[ill]),label=params['label'],color=params['color'],linestyle='-')
	
	ax.legend(**{'loc':1,'ncol':1,'fontsize':15,'framealpha':0.5,'frameon':False})
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_mu_function(parameters,scale='klinlin',title='$\\mu$ function',path='window_KMU.png'):

	window = MuFunction.load(parameters['MuFunction']['save'])
	k = window.k
	mu = window.mu
	#k = [0.,0.01]
	scaling = scalings[scale]
	scaling['xlim'] = None
	scaling['ylabel'] = '$\\mu$'
	xextend = 0.8
	ax = plot_baseline(scaling,title)
	#norm = Normalize(*[0.,1.])
	tmp = window.window
	norm = Normalize(tmp.min(),tmp.max())
	im = ax.pcolormesh(k,mu,window(k,mu).T,norm=norm,cmap=cm.jet_r)
	fig = pyplot.gcf()
	fig.subplots_adjust(right=xextend)
	cbar_ax = fig.add_axes([xextend+0.05,0.15,0.03,0.7])
	cbar_ax.tick_params(labelsize=labelsize) 
	cbar = fig.colorbar(im,cax=cbar_ax)

	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_window_function_1d(parameters,key,scale='sloglin',title='$\\mathcal{W}_{\ell}(s)$',path='window_{}.png',error=False):
	
	def plot(window,path):
		s = window.s[window.s>0]
		#s = scipy.logspace(-1,5,3000,base=10)
		#s = s[(s>1000.) & (s<2000.)]
		scaling = scalings[scale]
		ax = plot_baseline(scaling,title)
		ax.set_xlim(s[0],s[-1])
	
		for ill,ell in enumerate(window):
			if error and (ell==0): ax.errorbar(*scaling['func'](s,window(s,ell)),yerr=scaling['func'](s,window.poisson_error(s))[1],label=text_to_latex(ell),linestyle='-')
			else: ax.plot(*scaling['func'](s,window(s,ell)),label=text_to_latex(ell),linestyle='-')

		ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3))
		if losn % 2 != 1: sort_legend(ax,**{'loc':1,'ncol':1,'fontsize':15,'framealpha':0.5,'frameon':False})
		else: sort_legend(ax,**{'loc':2,'ncol':1,'fontsize':15,'framealpha':0.5,'frameon':False})
		#ax.legend(**{'loc':1,'ncol':1,'fontsize':15,'framealpha':0.5,'frameon':False})
		utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)
	
	if 0 in parameters['WindowFunction{}'.format(key)]:
		for losn in parameters['WindowFunction{}'.format(key)]:
			plot(WindowFunction.load(parameters['WindowFunction{}'.format(key)][losn]['save']),path.format('{}_losn{:d}'.format(key,losn)))
	else:
		plot(WindowFunction.load(parameters['WindowFunction{}'.format(key)]['save']),path.format(key))

def plot_window_function_2d(parameters,key,scale='slinlin',title='$\\mathcal{W}_{\ell}(s)$',path='w_{}.png'):
	
	def plot(window,path):
		s,d = window.s[0][window.s[0]>0.],window.s[1][window.s[1]>0.]
		#s,d = window.s[0][window.s[0]>0.],window.s[1][window.s[1]>1.e3]
		scaling = scalings[scale]
		s = s[(s>=scaling['xlim'][0]) & (s<scaling['xlim'][-1])]
		d = d[(d>=scaling['xlim'][0]) & (d<scaling['xlim'][-1])]
		figsize = 7; xextend = 0.8
		xlabel,ylabel = ('$s$ [$\\mathrm{Mpc} \ h^{-1}$]','$\\Delta$ [$\\mathrm{Mpc} \ h^{-1}$]')
		cmap = pyplot.get_cmap('Spectral')
		ells1,ells2 = window.ells
		#ells1,ells2 = ells1[:3],ells2[:3]
		ells1,ells2 = [2],[0]
		tmp = [window((s,d),(ell1,ell2)) for ell1 in ells1 for ell2 in ells2]
		norm = Normalize(scipy.amin(tmp),scipy.amax(tmp))
		#norm = Normalize(0,1)
		ncols = len(ells1); nrows = len(ells2)
		fig = pyplot.figure(figsize=(figsize/xextend,figsize))
		if title is not None: fig.suptitle(title,fontsize=fontsize)
		gs = gridspec.GridSpec(nrows,ncols,wspace=0.1,hspace=0.1)
		for ill1,ell1 in enumerate(ells1):
			for ill2,ell2 in enumerate(ells2):
				ax = pyplot.subplot(gs[nrows-1-ill2,ill1])
				im = ax.pcolormesh(s,d,window([s,d],(ell1,ell2)).T,norm=norm,cmap=cmap)
				#im = ax.pcolormesh(s,d,scipy.log(scipy.absolute(window([s,d],(ell1,ell2)).T)),norm=norm,cmap=cmap)
				ax.set_xscale(scaling['xscale'])
				ax.set_yscale(scaling['xscale'])
				if ill1>0: ax.get_yaxis().set_visible(False)
				elif 'log' in scale:
					ax.yaxis.set_major_locator(LogLocator(base=10.,subs=(1,),numticks=3))
					ax.yaxis.set_minor_locator(LogLocator(base=10.,subs=range(1,11),numticks=3))
					ax.yaxis.set_minor_formatter(NullFormatter())
				if ill2>0: ax.get_xaxis().set_visible(False)
				elif 'log' in scale:
					ax.xaxis.set_major_locator(LogLocator(base=10.,subs=(1,),numticks=3))
					ax.xaxis.set_minor_locator(LogLocator(base=10.,subs=range(1,11),numticks=3))
					ax.xaxis.set_minor_formatter(NullFormatter())
				ax.tick_params(labelsize=labelsize)
				text = '$\\mathcal{{W}}_{{{:d},{:d}}}$'.format(ell1,ell2)
				ax.text(0.05,0.95,text,horizontalalignment='left',verticalalignment='top',transform=ax.transAxes,color='black',fontsize=labelsize)
		utils.suplabel('x',xlabel,shift=-0.04,labelpad=7,size=labelsize)
		utils.suplabel('y',ylabel,shift=0,labelpad=8,size=labelsize)
		fig.subplots_adjust(right=xextend)
		cbar_ax = fig.add_axes([xextend+0.05,0.15,0.03,0.7])
		cbar_ax.tick_params(labelsize=labelsize)
		cbar = fig.colorbar(im,cax=cbar_ax,format='%.1e')

		utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)
		
	if 0 in parameters['WindowFunction{}'.format(key)]:
		for losn in parameters['WindowFunction{}'.format(key)]:
			plot(WindowFunction.load(parameters['WindowFunction{}'.format(key)][losn]['save']),path.format('{}_losn{:d}'.format(key,losn)))
	else:
		plot(WindowFunction.load(parameters['WindowFunction{}'.format(key)]['save']),path.format(key))

"""
def plot_window_function_comparison(parameters,scale='slogs2lin',title='Window function',path='window_comparison_{}.png'):
	
	for losn in parameters['WindowFunctionFFTRR']:
	
		window = WindowFunction.load(parameters['WindowFunctionFFTRR'][losn]['save'])
		s = window.s[window.s>0]
		scaling = scalings[scale]
		ax = plot_baseline(scaling,title)
		#ax.set_xlim(s[0]*0.9,s[-1])
		ax.set_xlim(100.,s[-1])
		colors = prop_cycle.by_key()['color']
		windowrr = WindowFunction.load(parameters['WindowFunctionRR'][losn]['save'])
		windowrrr = WindowFunction.load(parameters['WindowFunctionRRR0'][losn]['save'])
		d = windowrrr.s[0,windowrrr.s[0]>0.]

		for ill,ell in enumerate(window):
			ax.plot(*scaling['func'](s,window(s,ell)),label=text_to_latex(ell),linestyle='-',color=colors[ill])
			if (0,ell) in windowrrr.poles:
				proj = 4.*constants.pi*integrate.trapz(windowrrr([s,d],(0,ell))*s[:,None]**2,x=s,axis=0)
				ax.plot(*scaling['func'](d,proj),linestyle='--',color=colors[ill])
			if ell in windowrr.poles: ax.plot(*scaling['func'](s,windowrr(s,ell)),linestyle=':',color=colors[ill])

		ax.legend(**{'loc':1,'ncol':1,'fontsize':15,'framealpha':0.5,'frameon':False})
		utils.savefig(path.format('losn{:d}'.format(losn)),dpi=dpi,bbox_inches='tight',pad_inches=0.1)
"""
def plot_window_function_comparison(parameters,scale='sloglin',title='Window function',path='window_comparison_{}.png'):
	
	for losn in parameters['WindowFunctionRR']:
	
		window = WindowFunction.load(parameters['WindowFunctionRR'][losn]['save'])
		s = window.s[window.s>0]
		scaling = scalings[scale]
		ax = plot_baseline(scaling,title)
		ax.set_xlim(s[0]*0.9,s[-1])

		windowglo = WindowFunction.load(parameters['WindowFunctionRgloRglo'][losn]['save'])

		for ill,ell in enumerate(window):
			ax.plot(*scaling['func'](s,window(s,ell)),label=text_to_latex(ell),linestyle='-',color=colors[ill])
			ax.plot(*scaling['func'](s,windowglo(s,ell)),linestyle=':',color=colors[ill])

		ax.legend(**{'loc':1,'ncol':1,'fontsize':15,'framealpha':0.5,'frameon':False})
		utils.savefig(path.format('losn{:d}'.format(losn)),dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_window_function_shotnoise(parameters,key,scale='sloglin',title='$\\mathcal{W}_{\ell}(s)$',path='window_shotnoise_{}.png',error=False):
	
	window = WindowFunction.load(parameters['WindowFunction{}SN'.format(key)]['save'])
	s = window.s[window.s>1.]
	scaling = scalings[scale]
	ax = plot_baseline(scaling,title)
	ax.set_xlim(s[0]*0.9,s[-1])
	windowrrr = WindowFunction.load(parameters['WindowFunction{}'.format(key)][0]['save'])

	for ill,ell in enumerate(window):
		if error and (ell==0): ax.errorbar(*scaling['func'](s,window(s,ell)),yerr=scaling['func'](s,window.poisson_error(s))[1],label=text_to_latex(ell),linestyle='-',color=colors[ill])
		else: ax.plot(*scaling['func'](s,window(s,ell)),label=text_to_latex(ell),linestyle='-',color=colors[ill])
		ax.plot(*scaling['func'](s,windowrrr([s,0.],(ell,0))),linestyle='--',color=colors[ill])

	sort_legend(ax,**{'loc':1,'ncol':1,'fontsize':15,'framealpha':0.5,'frameon':False})
	#ax.legend(**{'loc':1,'ncol':1,'fontsize':15,'framealpha':0.5,'frameon':False})

	utils.savefig(path.format(key),dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_window_function_2d_projected(parameters,key,scale='sloglin',title='Window function',path='projected_window_{}.png'):

	divide = (key.count('R') == 3) or key in ['RRRradRglo','RRRangRglo','RRRradRang']
	
	for losn in parameters['WindowFunction{}'.format(key)]:
	
		window = WindowFunction.load(parameters['WindowFunction{}'.format(key)][losn]['save'])
		#window.window = scipy.transpose(window.window,axes=(0,2,1))
		s,d = window.s[0][window.s[0]>0.],window.s[1][window.s[1]>0.]
		#for i in range(len(window.window)): scipy.fill_diagonal(window.window[i],0.)
		#d = window.s[1][window.s[1]>1e3]
		scaling = scalings[scale]
		ax = plot_baseline(scaling,title)
		ax.set_xlim(d[0]*0.9,d[-1])
		windowrr = WindowFunction.load(parameters['WindowFunctionRR'][losn]['save'])

		for ill,ell in enumerate(window.ells[-1]):
			proj = 4.*constants.pi*integrate.trapz(window([s,d],(0,ell))*s[:,None]**2,x=s,axis=0)
			if divide: proj /= 2. # double los
			ax.plot(*scaling['func'](d,proj),label=text_to_latex(ell),linestyle='-',color=colors[ill])
			ax.plot(*scaling['func'](d,windowrr(d,ell)),linestyle='--',color=colors[ill])

		ax.legend(**{'loc':1,'ncol':1,'fontsize':15,'framealpha':0.5,'frameon':False})
		utils.savefig(path.format('{}_losn{:d}'.format(key,losn)),dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_window_convolution(parameters,scale='klinklin',title='Window function effect',toplot='geometry',path='window_convolution.png'):
	
	likelihood = LikelihoodClustering.load(parameters['Likelihood']['save'])
	likelihood.init(parameters['Likelihood']['path_data'])
	#likelihood.effect_ap.set_input_model(likelihood.model_tns.spectrum_galaxy,likelihood.model_tns.spectrum_galaxy_tree_real)
	geometry = likelihood.geometry
	#geometry.set_kout(geometry.k)
	mask = (geometry.k>1e-3) & (geometry.k<0.3)
	geometry.set_kout(geometry.k[mask])
	kwargs = fiducial_params.copy()
	scaling = scalings[scale]
	ax = plot_baseline(scaling,title)
	ax.set_xlim(geometry.kout[0].min(),geometry.kout[0].max())
	if toplot == 'geometry':
		updates,labels = [{},{'convolution':True}],['no window','with window']
	elif toplot == 'geometry-mu':
		updates,labels = [{},{'convolution':True},{'corrmu':True}],['no window','with window','with $k-\\mu$']
	elif toplot == 'fibercollisions':
		updates,labels = [{'convolution':True},{'convolution':True,'fibercollisions':'uncorrelated'},{'convolution':True,'fibercollisions':'all'}],['no fc','uncorrelated part','+ correlated part']
	else:
		updates,labels = toplot
	for update,label,linestyle in zip(updates,labels,linestyles):
		kwargs.update(update)
		spectrum = geometry.spectrum_multipoles(**kwargs)
		for ill,ell in enumerate(geometry.ellsout):
			params = utils.plot_params(mode='multipole',ell=(ill,ell))
			ax.plot(*scaling['func'](geometry.kout[ill],spectrum[ill]),label=label if ill==0 else None,color=params['color'],linestyle=linestyle)

	ax.legend(**{'loc':1,'ncol':1,'fontsize':15,'framealpha':0.5,'frameon':False})
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_window_contribution_multipole(parameters,scale='klinlin',title='Window function effect',path='window_contribution_multipole.png'):
	
	likelihood = LikelihoodClustering.load(parameters['Likelihood']['save'])
	likelihood.init(parameters['Likelihood']['path_data'])
	#likelihood.effect_ap.set_input_model(likelihood.model_tns.spectrum_galaxy,likelihood.model_tns.spectrum_galaxy_tree_real)
	geometry = likelihood.geometry
	convolution = geometry.convolution
	#geometry.set_kout(geometry.k)
	mask = (geometry.k>0.005) & (geometry.k<0.2)
	geometry.set_kout(geometry.k[mask])
	kwargs = fiducial_params.copy()
	scaling = scalings[scale]
	ax = plot_baseline(scaling,title)
	ax.set_xlim(geometry.kout[0].min(),geometry.kout[0].max())
	ax.set_ylabel('$P^{c,(\\ell)}(k)/P^{(\\ell)}(k)$')

	ellsin = scipy.unique([ell for ell,n in convolution.ellsin])
	nellsin = len(ellsin)
	for ellout,color in zip(geometry.ellsout,colors):
		index0 = convolution.indexout(ellout)
		ref = geometry.spectrum_multipoles(convolution=False,**kwargs)[index0]
		spectra = []
		for ellin in ellsin[::-1]:
			spectra.append(geometry.spectrum_multipoles(convolution=True,**kwargs)[index0])
			for (ell_,n_) in convolution.ellsin:
				if ell_ == ellin: convolution.multitomulti.conversion[index0,convolution.indexin((ell_,n_))][:] = 0.
		spectra = spectra[::-1]
		for ellin,spectrum,linestyle in zip(ellsin,spectra,linestyles):
			ax.plot(*scaling['func'](geometry.kout[0],spectrum/ref),label='$ + \, \\ell = {} $'.format(ellin) if ellout==0 else None,linestyle=linestyle,color=color)

	ax.legend(**{'loc':1,'ncol':1,'fontsize':15,'framealpha':0.5,'frameon':False})
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_convolved_integral_constraint(parameters,key,scale='kloglin',title='{} integral constraint',path='convolved_integral_constraint_{}.png'):

	likelihood = LikelihoodClustering.load(parameters['Likelihood']['save'])
	likelihood.init(parameters['Likelihood']['path_data'])
	geometry = likelihood.geometry
	kwargs = fiducial_params.copy()
	scaling = scalings[scale]
	ax = plot_baseline(scaling,title.format(key) if title else None)
	mask = (geometry.k>1e-4) & (geometry.k<3e-1)
	k = geometry.k[mask]
	ax.set_xlim(k.min(),k.max())

	Pl = geometry.input_model(**kwargs)
	ic = geometry.cic[key]
	geometry.toreal.integrate(*Pl,**kwargs)
	
	Xilc = geometry.convolution.convolve(geometry.toreal['convolution'])
	Plc = geometry.tofourier.integrate(Xilc)
	
	if key == 'global':
		windowRRR,windowRRRR,windowComb = 'WindowFunctionRRRglo','WindowFunctionRRRgloRglo','WindowFunctionGlo'
	if key == 'radial':
		windowRRR,windowRRRR,windowComb = 'WindowFunctionRRRrad','WindowFunctionRRRradRrad','WindowFunctionRad'
	if key == 'angular':
		windowRRR,windowRRRR,windowComb = 'WindowFunctionRRRang','WindowFunctionRRRangRang','WindowFunctionAng'

	for ill,ell in enumerate(geometry.ellsout):
		params = utils.plot_params(mode='multipole',ell=(ill,ell))
		ax.plot(*scaling['func'](k,Plc[ill][mask]),label='$P^{c}$' if ill==0 else None,color=params['color'],linestyle='-')
		max_ = Plc[ill][mask]
	
	for window,label,linestyle in zip([windowRRRR,windowRRR,windowComb],['$-IC^{{\\rm {0},\\rm {0}}}$'.format(key[:3]),'$-IC^{{\delta,\\rm {0}}}-IC^{{\\rm {0},\delta}}$'.format(key[:3]),'$-IC^{{\delta,\\rm {0}}}-IC^{{\\rm {0},\delta}}+IC^{{\\rm {0},\\rm {0}}}$'.format(key[:3])],linestyles[1:]):
		path_window = {n:parameters[window][n]['save'] for n in ic.ns}
		key_sn = '{}SN'.format(window)
		if key_sn in parameters:
			path_sn = parameters['{}SN'.format(window)]['save']
			if os.path.isfile(path_sn): path_window.update(SN=path_sn)
		ic.set_window(path_window=path_window)
		ic.set_grid()
		Plcic = geometry.tofourier.integrate(Xilc-ic.ic(geometry.toreal['cic_{}'.format(key)]))
		for ill,ell in enumerate(geometry.ellsout):
			params = utils.plot_params(mode='multipole',ell=(ill,ell))
			ax.plot(*scaling['func'](k,Plcic[ill][mask]),label=label if ill==0 else None,color=params['color'],linestyle=linestyle)
	
	ymin,ymax = ax.get_ylim()
	ax.set_ylim(ymin,ymax+(ymax-ymin)*0.15)
	ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3))
	
	ax.legend(**{'loc':2,'ncol':2,'fontsize':15,'framealpha':0.5,'frameon':False})
	utils.savefig(path.format(key),dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_convolved_integral_constraint_real(parameters,key,scale='rlogr2lin',title='{} integral constraint',path='convolved_integral_constraint_real_{}.png'):

	likelihood = LikelihoodClustering.load(parameters['Likelihood']['save'])
	likelihood.init(parameters['Likelihood']['path_data'])
	geometry = likelihood.geometry
	kwargs = fiducial_params.copy()
	scaling = scalings[scale]
	ax = plot_baseline(scaling,title.format(key))
	mask = (geometry.s>1.) & (geometry.s<1e3)
	s = geometry.s[mask]
	ax.set_xlim(s.min(),s.max())
	
	Pl = geometry.input_model(**kwargs)
	ic = geometry.cic[key]
	geometry.toreal.integrate(*Pl,**kwargs)
	
	Xilc = geometry.convolution.convolve(geometry.toreal['convolution'])
	
	if key == 'global':
		windowRRR,windowRRRR,windowComb = 'WindowFunctionRRRglo','WindowFunctionRRRgloRglo','WindowFunctionGlo'
	if key == 'radial':
		windowRRR,windowRRRR,windowComb = 'WindowFunctionRRRrad','WindowFunctionRRRradRrad','WindowFunctionRad'
	if key == 'angular':
		windowRRR,windowRRRR,windowComb = 'WindowFunctionRRRang','WindowFunctionRRRangRang','WindowFunctionAng'

	for ill,ell in enumerate(geometry.ellsout):
		params = utils.plot_params(mode='multipole',ell=(ill,ell))
		ax.plot(*scaling['func'](s,Xilc[ill][mask]),label='$\\xi^{c}$' if ill==0 else None,color=params['color'],linestyle='-')
	
	for window,label,linestyle in zip([windowRRR,windowRRRR,windowComb],['$-IC^{(4)}$','$-IC^{(3)}$','$-IC^{(3)}+IC^{(4)}$'],linestyles[1:]):
		path_window = {n:parameters[window][n]['save'] for n in ic.ns}
		path_window.update(SN=parameters['{}SN'.format(window)]['save'])
		ic.set_window(path_window=path_window)
		ic.set_grid()
		Xilcic = Xilc-ic.ic(geometry.toreal['cic_{}'.format(key)])
		for ill,ell in enumerate(geometry.ellsout):
			params = utils.plot_params(mode='multipole',ell=(ill,ell))
			ax.plot(*scaling['func'](s,Xilcic[ill][mask]),label=label if ill==0 else None,color=params['color'],linestyle=linestyle)

	ax.legend(**{'loc':2,'ncol':1,'fontsize':15,'framealpha':0.5,'frameon':False})
	utils.savefig(path.format(key),dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_convolved_integral_constraint_shotnoise(parameters,scale='kloglin',title='Integral constraint shotnoise',list_cic=['global','radial','angular'],path='convolved_integral_constraint_shotnoise_{}.png'):

	geometry = SurveyGeometry.load(parameters['SurveyGeometry']['save'])
	scaling = scalings[scale]
	ax = plot_baseline(scaling,title)
	mask = (geometry.k>1e-4) & (geometry.k<3e-1)
	k = geometry.k[mask]
	ax.set_xlim(k.min(),k.max())

	shotnoise = geometry.shotnoise*scipy.ones_like(k)
	for ill,ell in enumerate(geometry.ellsout):
		params = utils.plot_params(mode='multipole',ell=(ill,ell))
		ax.plot(*scaling['func'](k,shotnoise[ill]),label='Poisson' if ill==0 else None,color=params['color'],linestyle='-')
	
	tofourier = geometry.tofourier
	#tofourier = FFTlogBessel(**geometry.params['fftlog'])
	#tofourier.setup(s=geometry.s,k=geometry.k,ells=geometry.ellsout,bias=-2.5,direction='forward',lowringing=False)
	for cic,linestyle in zip(list_cic,linestyles[1:]):
		shotnoise = tofourier.integrate(geometry.cic[cic].real_shotnoise)
		for ill,ell in enumerate(geometry.ellsout):
			params = utils.plot_params(mode='multipole',ell=(ill,ell))
			ax.plot(*scaling['func'](k,shotnoise[ill][mask]),label='{}'.format(cic) if ill==0 else None,color=params['color'],linestyle=linestyle)
	
	ymin,ymax = ax.get_ylim()
	ax.set_ylim(ymin,ymax+(ymax-ymin)*0.3)

	ax.legend(**{'loc':1,'ncol':2 if len(list_cic) > 2 else 1,'fontsize':15,'framealpha':0.5,'frameon':False})
	utils.savefig(path.format('_'.join(list_cic)),dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_angular_convolved_integral_constraint_contributions(parameters,scale='rlogr2lin',title='Angular integral constraint',path='angular_convolved_integral_constraint_contributions_{}.png'):
	
	likelihood = LikelihoodClustering.load(parameters['Likelihood']['save'])
	likelihood.init(parameters['Likelihood']['path_data'])
	geometry = likelihood.geometry
	kwargs = fiducial_params.copy()
	scaling = scalings[scale]
	mask = (geometry.s>1.) & (geometry.s<1e3)
	s = geometry.s[mask]
	
	Pl = geometry.input_model(**kwargs)
	ic = geometry.cic['angular']
	geometry.toreal.integrate(*Pl,**kwargs)
	Xil = geometry.toreal['convolution']
	Xilc = geometry.convolution.convolve(Xil)
	ellsin = geometry.toreal.ells('cic_angular')[0]

	for ill,ell in enumerate(ic.ellsout):

		params = utils.plot_params(mode='multipole',ell=(ill,ell))
		ax = plot_baseline(scaling,'{} {}'.format(title,params['label']) if title else None)
		ax.set_xlim(s.min(),s.max())
		colors = prop_cycle.by_key()['color']
		ax.plot(*scaling['func'](s,Xilc[ill,mask]),label='$\\xi^{{({:d})}}(r)$'.format(ell),color='k',linestyle=':')
		for illin,ellin in enumerate(ellsin):
			#maskell = geometry.toreal.maskell(ellin)
			maskell = [ic.indexin((ellin,0))]
			xi = scipy.sum(Xil[:,ic.smask][None,maskell,None,:]*ic.W3s[ill,maskell,...],axis=(1,-1))[0]
			ax.plot(*scaling['func'](s,xi[mask]),label='$\\mathcal{{W}}_{{{:d},{:d}}}^{{({:d})}}$'.format(ell,ellin,3),color=colors[illin],linestyle='-')
			xi = scipy.sum(Xil[:,ic.smask][None,maskell,None,:]*ic.W4s[ill,maskell,...],axis=(1,-1))[0]
			ax.plot(*scaling['func'](s,xi[mask]),label='$\\mathcal{{W}}_{{{:d},{:d}}}^{{({:d})}}$'.format(ell,ellin,4),color=colors[illin],linestyle='--')

		ax.legend(**{'loc':2,'ncol':1,'fontsize':15,'framealpha':0.5,'frameon':False})
		utils.savefig(path.format(params['label']),dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_los_correction_convolved_integral_constraint(parameters,scale='kloglin',title='LOS correction',list_cic = ['radial'],path='los_correction_convolved_integral_constraint_{}.png'):

	likelihood = LikelihoodClustering.load(parameters['Likelihood']['save'])
	likelihood.init(parameters['Likelihood']['path_data'])
	geometry = likelihood.geometry
	kwargs = fiducial_params.copy()
	scaling = scalings[scale]
	kmin,kmax = 1e-3,1e-1
	mask = (geometry.k>kmin) & (geometry.k<kmax)
	k = geometry.k[mask]
	
	Pl = geometry.input_model(**kwargs)
	geometry.toreal.integrate(*Pl,**kwargs)

	conv = 'convolution'

	Xil = {}; ns = {}
	for key in [conv] + list_cic:
		key_toreal = key if key == conv else 'cic_{}'.format(key)
		ns_ = geometry.toreal.ns(key_toreal)[::-1]
		xin = {ns_[0]: geometry.toreal[key_toreal]}
		# remove contributions by decreasing order
		for in_,n in enumerate(ns_[1:]):
			xi = xin[ns_[in_]].copy()
			xi[geometry.toreal.mask(key_toreal,n=ns_[in_])] = 0.
			xin[n] = xi
		ns[key] = geometry.toreal.ns(key_toreal)
		Xil[key] = xin
	Pl = {key:{} for key in [conv] + list_cic}
	for n in Xil[conv]:
		Pl[conv][n] = geometry.tofourier.integrate(geometry.convolution.convolve(Xil[conv][n]-Xil[conv][ns[conv][0]]))
	for cic in list_cic:
		for n in ns[cic]:
			#Pl[cic][n] = Pl[conv][n]-geometry.tofourier.integrate(geometry.cic[cic].ic(Xil[cic][n]))
			Pl[cic][n] = -geometry.tofourier.integrate(geometry.cic[cic].ic(Xil[cic][n]))
	
	for ill,ell in enumerate(geometry.ellsout):
		params = utils.plot_params(mode='multipole',ell=(ill,ell))
		ax = plot_baseline(scaling,'{} {}'.format(title,params['label']) if title else None)
		ax.set_xlim([kmin,kmax])
		ax.set_ylabel('$P_{{{0:d}}}(k)/P_{{{0:d}}}^{{\mathrm{{c}}}}(k)$'.format(ell))
		Plref = geometry.tofourier.integrate(geometry.convolution.convolve(Xil[conv][ns[conv][0]]))
		for n in ns[conv][1:]:
			#ax.plot(*scaling['func'](k,Pl[conv][n][ill][mask]/Plref[ill][mask]),label='window, $n={}$'.format(','.join('{:d}'.format(i) for i in range(1,n+1))),color=colors[0],linestyle=linestyles[n])
			ax.plot(*scaling['func'](k,Pl[conv][n][ill][mask]/Plref[ill][mask]),label='window, $n={:d}$'.format(n),color=colors[0],linestyle=linestyles[n])
		for cic,color in zip(list_cic,colors[1:]):
			for n in ns[cic]:
				#ax.plot(*scaling['func'](k,Pl[cic][n][ill][mask]/Plref[ill][mask]),label='{} IC, $n={}$'.format(cic,','.join('{:d}'.format(i) for i in range(n+1))),color=color,linestyle=linestyles[n])
				ax.plot(*scaling['func'](k,Pl[cic][n][ill][mask]/Plref[ill][mask]),label='{} IC, $n={:d}$'.format(cic,n),color=color,linestyle=linestyles[n])
			#print scipy.absolute((Pl[cic][1][ill][mask]-Pl[cic][0][ill][mask])/Pl[cic][0][ill][mask]).max()
		ax.yaxis.set_minor_locator(AutoMinorLocator(2))
		ax.legend(**{'loc':4,'ncol':1,'fontsize':15,'framealpha':0.5,'frameon':False})
		utils.savefig(path.format(params['label']),dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_convolved_integral_constraint_effect(parameters,scale='kloglin',title='Integral constraint effect',list_cic=['global','radial','angular'],list_convolution=[False,True],path='convolved_integral_constraint_effect_{}.png'):
	
	likelihood = LikelihoodClustering.load(parameters['Likelihood']['save'])
	likelihood.init(parameters['Likelihood']['path_data'])
	#likelihood.effect_ap.set_input_model(likelihood.model_tns.spectrum_galaxy_1loop,likelihood.model_tns.spectrum_galaxy_tree_real)
	geometry = likelihood.geometry
	#geometry.set_kout(geometry.k)
	mask = (geometry.k>1e-4) & (geometry.k<3e-1)
	geometry.set_kout(geometry.k[mask])
	#geometry.set_kout(scipy.linspace(0.01,0.4,40))
	kwargs = fiducial_params.copy()
	kwargs.update(convolution=True)
	scaling = scalings[scale]
	ax = plot_baseline(scaling,title)
	ax.set_xlim(geometry.kout[0].min(),geometry.kout[0].max())
	
	for convolution,linestyle in zip(list_convolution,linestyles):
		kwargs.update(convolution=convolution)
		spectrum = geometry.spectrum_multipoles(**kwargs)
		for ill,ell in enumerate(geometry.ellsout):
			params = utils.plot_params(mode='multipole',ell=(ill,ell))
			label = 'with window' if convolution else 'no window'
			ax.plot(*scaling['func'](geometry.kout[ill],spectrum[ill]),label=label if ill==0 else None,color=params['color'],linestyle=linestyle)

	for cic,linestyle in zip(list_cic,linestyles[len(list_convolution):]):
		kwargs.update(cic=cic)
		spectrum = geometry.spectrum_multipoles(**kwargs)
		for ill,ell in enumerate(geometry.ellsout):
			params = utils.plot_params(mode='multipole',ell=(ill,ell))
			ax.plot(*scaling['func'](geometry.kout[ill],spectrum[ill]),label='with {} IC'.format(cic) if ill==0 else None,color=params['color'],linestyle=linestyle)

	if len(list_cic) >= 3:
		ymin,ymax = ax.get_ylim()
		ax.set_ylim(ymin,ymax+(ymax-ymin)*0.15)
	ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3))
	ax.legend(**{'loc':2,'ncol':1 if len(list_cic)<3 else 2,'fontsize':15,'framealpha':0.5,'frameon':False})
	utils.savefig(path.format('_'.join(list_cic)),dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_convolved_integral_constraint_effect_real(parameters,scale='rlogr2lin',title='Integral constraint effect',list_cic=['global','radial','angular'],list_convolution=[False,True],path='convolved_integral_constraint_effect_real_{}.png'):
	
	likelihood = LikelihoodClustering.load(parameters['Likelihood']['save'])
	likelihood.init(parameters['Likelihood']['path_data'])
	geometry = likelihood.geometry
	#geometry.set_kout(scipy.linspace(0.01,0.4,40))
	kwargs = fiducial_params.copy()
	kwargs.update(convolution=True)
	scaling = scalings[scale]
	ax = plot_baseline(scaling,title)
	smin,smax = 2e1,1e3
	mask = (geometry.s > smin*0.9) & (geometry.s < smax*1.1)
	s = geometry.s[mask]
	ax.set_xlim(smin,smax)
	
	for convolution,linestyle in zip(list_convolution,linestyles):
		kwargs.update(convolution=convolution)
		correlation = geometry.correlation_cic_multipoles(**kwargs)
		for ill,ell in enumerate(geometry.ellsout):
			params = utils.plot_params(mode='multipole',ell=(ill,ell))
			label = 'with window' if convolution else 'no window'
			ax.plot(*scaling['func'](s,correlation[ill][mask]),label=label if ill==0 else None,color=params['color'],linestyle=linestyle)
	
	for cic,linestyle in zip(list_cic,linestyles[len(list_convolution):]):
		kwargs.update(cic=cic)
		correlation = geometry.correlation_cic_multipoles(**kwargs)
		for ill,ell in enumerate(geometry.ellsout):
			params = utils.plot_params(mode='multipole',ell=(ill,ell))
			ax.plot(*scaling['func'](s,correlation[ill][mask]),label='with {} IC'.format(cic) if ill==0 else None,color=params['color'],linestyle=linestyle)

	ax.legend(**{'loc':1,'ncol':1,'fontsize':15,'framealpha':0.5,'frameon':False})
	utils.savefig(path.format('_'.join(list_cic)),dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_convolved_integral_constraint_window_effect_real(parameters,scale='rlogr2lin',title='Integral constraint window effect',list_cic=['global','radial','angular'],path='convolved_integral_constraint_window_effect_real_{}.png'):
	
	likelihood = LikelihoodClustering.load(parameters['Likelihood']['save'])
	likelihood.init(parameters['Likelihood']['path_data'])
	geometry = likelihood.geometry
	#geometry.set_kout(scipy.linspace(0.01,0.4,40))
	kwargs = fiducial_params.copy()
	kwargs.update(convolution=True)
	scaling = scalings[scale]
	ax = plot_baseline(scaling,title)
	smin,smax = 2e1,1e3
	mask = (geometry.s > smin*0.9) & (geometry.s < smax*1.1)
	s = geometry.s[mask]
	ax.set_xlim(smin,smax)
	
	baseline = geometry.correlation_cic_multipoles(**kwargs)
	
	for cic,linestyle in zip(list_cic,linestyles[1:]):
		kwargs.update(cicconv=cic)
		correlation = geometry.correlation_cic_multipoles(**kwargs)
		for ill,ell in enumerate(geometry.ellsout):
			params = utils.plot_params(mode='multipole',ell=(ill,ell))
			ax.plot(*scaling['func'](s,correlation[ill][mask]-baseline[ill][mask]),label='with {} IC'.format(cic) if ill==0 else None,color=params['color'],linestyle=linestyle)

	ax.legend(**{'loc':1,'ncol':1,'fontsize':15,'framealpha':0.5,'frameon':False})
	utils.savefig(path.format('_'.join(list_cic)),dpi=dpi,bbox_inches='tight',pad_inches=0.1)
	
def plot_convolved_integral_constraint_difference(parameters,scale='klinklin',title='Integral constraint effect',list_cic=['global','radial','angular'],path='convolved_integral_constraint_difference_{}.png'):
	
	likelihood = LikelihoodClustering.load(parameters['Likelihood']['save'])
	likelihood.init(parameters['Likelihood']['path_data'])
	geometry = likelihood.geometry
	geometry.set_kout(geometry.k)
	geometry.set_kout(scipy.linspace(0.01,0.4,40))
	kwargs = fiducial_params.copy()
	kwargs.update(convolution=True)
	scaling = scalings[scale]
	ax = plot_baseline(scaling,title)
	ax.set_xlim(geometry.kout[0].min(),geometry.kout[0].max())
	
	ref = geometry.spectrum_multipoles(**kwargs)
	kwargs.update(cic='global')
	spectrum = geometry.spectrum_multipoles(**kwargs)
	for ill,ell in enumerate(geometry.ellsout):
		params = utils.plot_params(mode='multipole',ell=(ill,ell))
		ax.plot(*scaling['func'](geometry.kout[ill],spectrum[ill]-ref[ill]),label='with global IC' if ill==0 else None,color=params['color'],linestyle='--')
		#print ell, scipy.argmax(scipy.absolute(spectrum[ill]-noint[ill])), scipy.amax(scipy.absolute(spectrum[ill]-noint[ill]))

	for cic,linestyle in zip(list_cic,linestyles[1:]):
		kwargs.update(cic=cic)
		spectrum = geometry.spectrum_multipoles(**kwargs)
		for ill,ell in enumerate(geometry.ellsout):
			params = utils.plot_params(mode='multipole',ell=(ill,ell))
			ax.plot(*scaling['func'](geometry.kout[ill],spectrum[ill]-ref[ill]),label='with {} IC'.format(cic) if ill==0 else None,color=params['color'],linestyle=linestyle)

	ax.legend(**{'loc':4,'ncol':1,'fontsize':15,'framealpha':0.5,'frameon':False})
	utils.savefig(path.format('_'.join(list_cic)),dpi=dpi,bbox_inches='tight',pad_inches=0.1)
	
	
def plot_best_fit(likelihood,values,scale='klinklin',title='Best fit',remove_sn=True,path='best_fit.png'):
	
	scaling = scalings[scale]
	nells = len(likelihood.ells)
	height_ratios = [max(nells,3)] + [1]*nells
	fsize = (10,2*sum(height_ratios))
	fig, lax = pyplot.subplots(len(height_ratios),sharex=True,sharey=False,gridspec_kw={'height_ratios':height_ratios},figsize=fsize,squeeze=False)
	lax = lax.flatten()
	fig.subplots_adjust(hspace=0)
	krange = [likelihood.krange[ell] for ell in likelihood.ells]
	xlim = [scipy.amin(krange),scipy.amax(krange)]

	def shotnoise(ell):
		if (ell == 0) and remove_sn and not likelihood.remove_sn:
			return likelihood.shotnoise
		return 0.

	for ax in lax:
		ax.set_xlim(xlim)
		ax.set_xscale(scaling['xscale'])
		ax.tick_params(labelsize=labelsize)
		ax.grid(which='both')

	kout = likelihood.geometry.kout
	k = likelihood.geometry.k
	krange = [[k_.min(),k_.max()] for k_ in likelihood.kdata]
	krange = [scipy.amin(krange),scipy.amax(krange)]
	k = k[(k>=krange[0]) & (k<=krange[1])]
	likelihood.geometry.set_kout(k)
	amodel = likelihood.amodel(**values)
	
	ax = lax[0]
	ax.set_yscale(scaling['yscale'])
	ax.set_ylabel(scaling['ylabel'],fontsize=fontsize)
	
	amodel = likelihood.amodel(**values)
	for ill,ell in enumerate(likelihood.ells):
		params = utils.plot_params(mode='multipole',ell=(ill,ell))
		params['marker'] = None
		params['label'] = None
		ax.plot(*scaling['func'](likelihood.geometry.kout[ill],amodel[ill]-shotnoise(ell)),**params)
	
	for ill,ell in enumerate(likelihood.ells):
		params = utils.plot_params(mode='multipole',ell=(ill,ell))
		params['linestyle'] = 'None'
		x,y = scaling['func'](likelihood.k[ill],likelihood.adata[ill]-shotnoise(ell))
		yerr = scaling['func'](likelihood.k[ill],likelihood.astddev[ill])[1]
		ax.plot(x,y,**params)
		ax.errorbar(x,y,yerr=yerr,fmt='None',color=params['color'],ecolor=params['color'])
	
	ymin,ymax = ax.get_ylim()
	ax.set_ylim(ymin,ymax+(ymax-ymin)*0.1)
	#ax.set_ylim(-400.,3000.)
	ax.legend(**{'loc':1,'ncol':1,'fontsize':15,'framealpha':0.5,'frameon':False})

	likelihood.geometry.kout = kout
	amodel = likelihood.amodel(**values)

	ylim = [-4.,4.]
	for ill,ax in enumerate(lax[1:]):
		ell = likelihood.ells[ill]
		ax.set_ylim(*ylim)
		ax.set_yticks(scipy.arange(ylim[0],ylim[-1]+0.1,1.),minor=True)
		ax.set_yticks(scipy.arange(-2.,2.1,2.))
		for offset in [-2.,2.]: ax.plot(scaling['xlim'],[offset]*2,color='k',linestyle='--')
		x,y = likelihood.kdata[ill],(likelihood.adata[ill]-amodel[ill])/likelihood.astddev[ill]
		params = utils.plot_params(mode='multipole',ell=(ill,ell))
		params['label'] = None
		params['linestyle'] = 'None'
		ax.plot(x,y,**params)
		ax.set_ylabel('$\\Delta P_{{{ell:d}}} / \\sigma_{{ P_{{{ell:d}}} }}$'.format(ell=ell),fontsize=fontsize)
		
	lax[-1].set_xlabel(scaling['xlabel'],fontsize=fontsize)
	
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)
