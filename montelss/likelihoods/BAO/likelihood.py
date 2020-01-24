import logging
import re
import copy
import scipy
from montelss import utils
from montelss.likelihoods.clustering import BaseLikelihoodSpectrumMultipoles
import pyspectrum
import plot	

class LikelihoodBAOSpectrum(BaseLikelihoodSpectrumMultipoles):

	logger = logging.getLogger('LikelihoodBAOSpectrum')

	def gradient(self,parameters,**kwargs):
		kwargs = self.to_model_kwargs(**kwargs)
		kwargs['Ns'] = 0.
		if self.isiso and not self.issmooth: wiggles = self.model_sp.wiggles_damped_iso(self.geometry.k/kwargs.get('qiso',1.),sigmanl=kwargs.get('sigmanl',0.))
		#if self.isiso and not self.issmooth: wiggles = self.model_sp.wiggles_damped_iso(self.geometry.k/kwargs['qiso'],sigmanl=kwargs['sigmanl'])
		else: wiggles = 1.
		model_origin = self.geometry.input_model
		model_multipoles = scipy.zeros((len(self.geometry.ellsin),len(self.geometry.k)),dtype='f8')
		toret = {}
		for par in parameters:
			toret[par] = 0.
			mpar = self.to_model_par(par)
			for illin,ellin in enumerate(self.geometry.ellsin):
				if re.match('a[p,m][0-9]$',mpar) or re.match('a[p,m][0-9]_l{:d}'.format(ellin),mpar):
					def poly(*args,**kwargs):
						tmp = 0.*model_multipoles
						tmp[illin] = self.model_sp.polynomial(self.geometry.k,**{mpar.replace('_l{:d}'.format(ellin),''):1.})*wiggles
						return [tmp] + [getattr(self.model_sp,model)(*args,**kwargs) for model in self.params['models'][1:]]
					self.geometry.set_input_model(poly)
					toret[par] += scipy.concatenate(self.geometry.spectrum_multipoles(**kwargs))

		self.geometry.set_input_model(model_origin)
		return toret
	
	def broadband_coeffs(self,ell=0,**kwargs):
		#if self.isiso: return {key:val for key,val in kwargs.items() if re.match('a[p,m][0-9]',key)}
		#return {key.replace('_l{:d}'.format(ell),''):val for key,val in kwargs.items() if re.match('a[p,m][0-9]_l{:d}'.format(ell),key)}
		return {key.replace('_l{:d}'.format(ell),''):val for key,val in kwargs.items() if re.match('a[p,m][0-9]',key)}

	@property
	def isiso(self):
		return 'aniso' not in self.params['models'][0]

	@property
	def issmooth(self):
		return 'smooth' in self.params['models'][0]

	def set_ells(self):
		super(LikelihoodBAOSpectrum,self).set_ells()
		if self.isiso:
			assert self.ells == [0], 'You cannot fit ell > 0 multipoles with an isotropic BAO template.'

	@utils.classparams
	def set_model(self,models=['spectrum_galaxy_iso','spectrum_galaxy_tree_real']):
		self.model_sp = pyspectrum.ModelBAO.load(self.params['path_model_bao'])
		self.geometry = pyspectrum.SurveyGeometry.load(self.params['path_survey_geometry'])
		if self.isiso:
			self.set_model_iso(models)
		else:
			self.set_model_aniso(models)
		self._amodel_ = self.geometry.spectrum_multipoles
		
	def set_model_iso(self,models):
		issmooth = 'smooth' in models[0]
		models = [getattr(self.model_sp,model) for model in models]
		def spectrum_multipoles(*args,**kwargs):
			toret = [model(self.geometry.k,*args,**kwargs) for model in models]
			if not issmooth: wiggles = self.model_sp.wiggles_damped_iso(self.geometry.k/kwargs.get('qiso',1.),sigmanl=kwargs.get('sigmanl',0.))
			else: wiggles = 1.
			toret[0] += self.model_sp.polynomial(self.geometry.k,**self.broadband_coeffs(ell=0,**kwargs))*wiggles
			return toret
		self.geometry.set_input_model(spectrum_multipoles)

	def set_model_aniso(self,models):
		models = [getattr(self.model_sp,model) for model in models]
		self.effect_ap = pyspectrum.EffectAP.load(self.params['path_effect_ap'])
		self.effect_ap.set_input_model(*models)
		def spectrum_multipoles(*args,**kwargs):
			toret = self.effect_ap.spectrum_multipoles(*args,**kwargs)
			for ill,ell in enumerate(self.effect_ap.ells):
				toret[0][ill] += self.model_sp.polynomial(self.effect_ap.k,**self.broadband_coeffs(ell=ell,**kwargs))
			return toret
		self.geometry.set_input_model(spectrum_multipoles)

	def amodel_smooth(self,**kwargs):
		model_origin = self.geometry.input_model
		if self.isiso:
			models = ['spectrum_smooth_iso'] + self.params['models'][1:]
			self.set_model_iso(models)
		else:
			models = ['spectrum_smooth_aniso'] + self.params['models'][1:]
			self.set_model_aniso(models)
		toret = self.amodel(**kwargs)
		self.geometry.set_input_model(model_origin)
		return toret

	def plot(self,*args,**kwargs):
		kwargs['path'] = kwargs.pop('path','best_fit_{}.png').format(self.id)
		plot.plot_best_fit(self,*args,**kwargs)
