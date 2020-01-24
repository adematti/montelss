import os
import scipy
from scipy import integrate,constants
import logging
from matplotlib import pyplot
from matplotlib.ticker import AutoMinorLocator,LogLocator,FixedLocator,NullFormatter
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib import gridspec
import pyspectrum
from pyspectrum import *
from montelss.likelihoods.clustering import utils
from montelss.likelihoods.clustering.plot import *

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
#fiducial_params = dict(f=0.75,b1=2.,b2=1.,sigmav=4,FoG='lorentzian2')
fiducial_params = dict(f=0.8,b1=1.4,b2=1.,sigmav=4,FoG='lorentzian2')

def plot_model_tns_spectrum_nloop(parameters,nloop=2,scale='klinklin',title='RegPT outputs',path='spectrum_{:d}loop.png'):
	
	model = ModelTNS.load(parameters['ModelTNS']['save'])
	scaling = scalings[scale]
	ax = plot_baseline(scaling,title)

	for key,label in zip(['dd','dt','tt'],['pk_dd','pk_dt','pk_tt']):
		k = getattr(model,'spectrum_{:d}loop_{}'.format(nloop,key))['k']
		pk = getattr(model,'spectrum_{:d}loop_{}'.format(nloop,key)).pk()
		ax.plot(*scaling['func'](k,pk),label=pyspectrum.utils.text_to_latex(label),linestyle='-')
	ax.plot(*scaling['func'](model.spectrum_halofit['k'],model.spectrum_halofit.pk()),label='halofit',linestyle='-')

	ax.legend(**{'loc':1,'ncol':2,'fontsize':labelsize,'framealpha':0.5,'frameon':False})
	utils.savefig(path.format(nloop),dpi=dpi,bbox_inches='tight',pad_inches=0.1)
	
def plot_best_fit(likelihood,values,scale='klinklin',title='Best fit',remove_sn=True,path='best_fit.png',save=False):

	if save:
		toret = {}
		toret['poles'] = likelihood.ells
		toret['x'] = likelihood.kdata
		toret['y'] = likelihood.adata
		toret['ymodel'] = likelihood.amodel(**values)
		utils.save(path.replace('.png','_array.npy'),toret)
	
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
	likelihood.geometry.set_kout({ell:k[(k>=krange[0]) & (k<=krange[1])] for ell,krange in likelihood.krange.items()})
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

	likelihood.geometry.set_kout(kout)
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
