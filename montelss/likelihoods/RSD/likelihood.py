import logging
import copy
import scipy
from montelss import utils
from montelss.likelihoods.clustering import BaseLikelihoodSpectrumMultipoles
import pyspectrum as pysp
import plot

class LikelihoodRSDSpectrum(BaseLikelihoodSpectrumMultipoles):

	logger = logging.getLogger('LikelihoodRSDSpectrum')

	def gradient(self,parameters,**kwargs):
		toret = {}
		for par in parameters:
			toret[par] = 0.
			mpar = self.to_model_par(par)
			if hasattr(self.geometry,'systematics'):
				tmp = self.geometry.systematics.gradient([mpar])[mpar]
				toret[par] += scipy.concatenate([tmp[ill] for ill in self.ills])
		return toret

	@utils.classparams
	def set_model(self,models=['spectrum_galaxy','spectrum_galaxy_tree_real']):
		self.model_sp = pysp.ModelTNS.load(self.params['path_model_tns'])
		self.effect_ap = pysp.EffectAP.load(self.params['path_effect_ap'])
		self.effect_ap.set_input_model(*[getattr(self.model_sp,model) for model in models])
		self.geometry = pysp.SurveyGeometry.load(self.params['path_survey_geometry'])
		self.geometry.set_input_model(self.effect_ap.spectrum_multipoles)
		self._amodel_ = self.geometry.spectrum_multipoles

	def amodel(self,**kwargs):
		kwargs = self.to_model_kwargs(**kwargs)
		if kwargs == self._kwargs: return self._amodel
		for new,old in zip(['Ng','Ns'],['Ag','As']):
			if old in kwargs: kwargs[new] = kwargs.pop(old)*self.shotnoise
		kwargs['normwindow'] = self.norm
		spectrum = self.geometry.spectrum_multipoles(**kwargs)
		self._kwargs = kwargs
		self._amodel = [spectrum[ill] for ill in self.ills]
		return self._amodel

	def plot(self,*args,**kwargs):
		kwargs['path'] = kwargs.pop('path','best_fit_{}.png').format(self.id)
		plot.plot_best_fit(self,*args,**kwargs)
