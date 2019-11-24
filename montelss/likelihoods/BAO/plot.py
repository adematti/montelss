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
from montelss.likelihoods.clustering import utils
from montelss.likelihoods.clustering.plot import *

fiducial_params = dict(qiso=1.,sigmanl=5.)

def plot_best_fit(likelihood,values,scale='klinklin',title='Best fit',plot_absolute=True,remove_sn=True,path='best_fit.png',save=True):

	def shotnoise(ell):
		if (ell == 0) and remove_sn and not likelihood.remove_sn:
			return likelihood.shotnoise
		return 0.

	if save:
		toret = {}
		toret['poles'] = likelihood.ells
		toret['x'] = likelihood.kdata
		toret['y'] = likelihood.adata
		toret['ymodel'] = likelihood.amodel(**values)
		for ill,ell in enumerate(likelihood.ells):
			toret['ymodel'][ill] = toret['ymodel'][ill] - shotnoise(ell)
		utils.save(path.replace('.png','_array.npy'),toret)
	
	scaling = scalings[scale]
	nells = len(likelihood.ells)
	height_ratios = ([max(nells,3)] if plot_absolute else []) + [1]*nells
	fsize = (10,3*sum(height_ratios))
	fig, lax = pyplot.subplots(len(height_ratios),sharex=True,sharey=False,gridspec_kw={'height_ratios':height_ratios},figsize=fsize,squeeze=False)
	lax = lax.flatten()
	fig.subplots_adjust(hspace=0)
	krange = [likelihood.krange[ell] for ell in likelihood.ells]
	xlim = [scipy.amin(krange),scipy.amax(krange)]

	start_ratio  = int(plot_absolute)
	for ax in lax:
		ax.set_xlim(xlim)
		ax.set_xscale(scaling['xscale'])
		ax.tick_params(labelsize=labelsize)
		ax.grid(which='both')
	
	kout = likelihood.geometry.kout
	k = likelihood.geometry.k
	likelihood.geometry.set_kout({ell:k[(k>=krange[0]) & (k<=krange[1])] for ell,krange in likelihood.krange.items()})
	amodel = likelihood.amodel(**values)
	asmooth = likelihood.amodel_smooth(**values)

	if plot_absolute:

		ax = lax[0]
		ax.set_yscale(scaling['yscale'])
		ax.set_ylabel(scaling['ylabel'],fontsize=fontsize)
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

		#ymin,ymax = ax.get_ylim()
		#ax.set_ylim(ymin,ymax+(ymax-ymin)*0.2)
		ax.legend(**{'loc':1,'ncol':1,'fontsize':15,'framealpha':0.5,'frameon':False})

	for ax in lax[start_ratio:]:
		ax.set_yscale('linear')
		ax.set_ylabel('$P_{{{ell:d}}} / P_{{\\mathrm{{sm}}}}$'.format(ell=ell),fontsize=fontsize)

	for ill,ell in enumerate(likelihood.ells):
		params = utils.plot_params(mode='multipole',ell=(ill,ell))
		params['marker'] = None
		params['label'] = None
		lax[ill+start_ratio].plot(likelihood.geometry.kout[ill],amodel[ill]/asmooth[ill],**params)

	likelihood.geometry.kout = kout
	asmooth = likelihood.amodel_smooth(**values)

	for ill,ell in enumerate(likelihood.ells):
		params = utils.plot_params(mode='multipole',ell=(ill,ell))
		params['linestyle'] = 'None'
		x,y = likelihood.k[ill],likelihood.adata[ill]/asmooth[ill]
		yerr = likelihood.astddev[ill]/asmooth[ill]
		lax[ill+start_ratio].plot(x,y,**params)
		lax[ill+start_ratio].errorbar(x,y,yerr=yerr,fmt='None',color=params['color'],ecolor=params['color'])

	lax[-1].set_xlabel(scaling['xlabel'],fontsize=fontsize)
	
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)
