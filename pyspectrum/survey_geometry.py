import logging
import scipy
from window_function import WindowFunction1D,WindowFunction2D,TemplateSystematics
from fiber_collisions import FiberCollisions
from integral_constraint import ConvolvedIntegralConstraint,ConvolvedIntegralConstraintConvolution
from window_convolution import ConvolutionMultipole,LOSCorrections
from FFTlog import FFTlogBessel
import utils

class SurveyGeometry(object):

	TYPE_FLOAT = scipy.float64
	logger = logging.getLogger('SurveyGeometry')

	def __init__(self,**params):
		self.params = params
	
	@utils.classparams
	def setup(self,kin,s,d=[0.,scipy.inf],ellmax=4,ellsout=[0,2],path_convolution={},path_cic={},path_cicconv={},path_fiber_collisions={},path_systematics={}):
		
		self.logger.info('Setting windows.')
		self.logger.info('Setting los corrections.')
		assert max(ellsout) <= ellmax
		self.toreal = LOSCorrections(**self.params)
		self.toreal.add('baseline',ellmax=ellmax,nmax=0,los='midpoint')
		self.toreal.setup(s=s,k=kin)
		self.s = self.toreal.s; self.k = self.toreal.k
		self.logger.info('Setting real-to-fourier transform.')
		self.tofourier = FFTlogBessel(**self.params['fftlog'])
		self.tofourier.setup(s=self.s,k=self.k,ells=self.ellsout,direction='forward',lowringing=False)
		if path_convolution:
			self.logger.info('Setting window convolution.')
			self.convolution = ConvolutionMultipole(**self.params)
			los,ns = self.convolution.set_window(path_window=path_convolution)
			ells = self.toreal.add('convolution',ellmax=ellmax,nmax=max(ns),los=los)
			ellsconv = self.toreal.flatellsn('convolution',0)
			self.convolution.set_grid(s=self.s,ellsin=ells)
			self.normwindow = self.convolution.normwindow
		self.cic = {}
		for cic in path_cic:
			self.logger.info('Setting {} convolved integral constraint.'.format(cic))
			self.cic[cic] = ConvolvedIntegralConstraint(**self.params)
			los,ns = self.cic[cic].set_window(path_window=path_cic[cic])
			ells = self.toreal.add('cic_{}'.format(cic),ellmax=ellmax,nmax=max(ns),los=los)
			self.cic[cic].set_grid(s=self.s,k=self.k,ellsin=ells)
		self.cicconv = {}
		for cicconv in path_cicconv:
			self.logger.info('Setting {} convolved integral constraint correction to convolution.'.format(cicconv))
			self.cicconv[cicconv] = ConvolvedIntegralConstraintConvolution(**self.params)
			los,ns = self.cicconv[cicconv].set_window(path_window=path_cicconv[cicconv])
			ells = self.toreal.add('cicconv_{}'.format(cicconv),ellmax=ellmax,nmax=max(ns),los=los)
			self.cicconv[cicconv].set_grid(s=self.s,k=self.k,ellsin=ells,ellsconv=ellsconv)
		if path_fiber_collisions:
			self.logger.info('Setting fiber collisions.')
			self.fiber_collisions = FiberCollisions.load(path_fiber_collisions)
			self.fiber_collisions.setup(k=self.k,ellsin=self.toreal.ellsref,ellsout=self.toreal.ellsref)
		if path_systematics:
			self.logger.info('Setting systematics.')
			self.systematics = TemplateSystematics.load(path_systematics)
		self.toreal.setup(s=self.s,k=self.k,fftlog={'lowringing':False})
		self.ellsin = self.toreal.ellsref
		self.set_kout(self.k)
		self.set_sout(self.s)
		self.shotnoise = scipy.zeros((len(self.ellsout),1),dtype=self.TYPE_FLOAT)
		self.shotnoise[self.indexout(0),0] = 1.
	
	@utils.classparams		
	def set_kout(self,kout=None):
		if kout is None:
			self.kout = [self.k.copy() for ell in self.ellsout]
		elif isinstance(kout,dict):
			self.kout = [scipy.asarray(kout.get(ell,[]),dtype=self.TYPE_FLOAT) for ell in self.ellsout]
		elif isinstance(kout,list):
			self.kout = kout
		else:
			self.kout = [scipy.array(kout,dtype=self.TYPE_FLOAT) for ell in self.ellsout]
		for ill,ell in enumerate(self.ellsout):
			if len(self.kout[ill])==0: continue #in case kout is empty
			if self.k[0]>self.kout[ill][0]: self.logger.warning('First k = {:.5g} is outside integration range for ell = {:d}: the power spectrum will be constant below first k = {:.5g}.'.format(self.kout[ill][0],ell,self.k[0]))
			if self.k[-1]<self.kout[ill][-1]: self.logger.warning('Last k = {:.5g} is outside integration range for ell = {:d}: the power spectrum will be constant beyond last k = {:.5g}.'.format(self.kout[ill][-1],ell,self.k[-1]))
		if hasattr(self,'systematics'):
			self.systematics.setup(kout=self.kout,ellsout=self.ellsout)
		
	@utils.classparams		
	def set_sout(self,sout=None):
		if sout is None:
			self.sout = [self.s.copy() for ell in self.ellsout]
		elif isinstance(sout,dict):
			self.sout = [scipy.asarray(sout.get(ell,[]),dtype=self.TYPE_FLOAT) for ell in self.ellsout]
		elif isinstance(sout,list):
			self.sout = sout
		else:
			self.sout = [scipy.array(sout,dtype=self.TYPE_FLOAT) for ell in self.ellsout]
		for ill,ell in enumerate(self.ellsout):
			if len(self.sout[ill])==0: continue #in case sout is empty
			if self.s[0]>self.sout[ill][0]: self.logger.warning('First s = {:.5g} is outside integration range for ell = {:d}: the correlation function will be constant below first s = {:.5g}.'.format(self.sout[ill][0],ell,self.s[0]))
			if self.s[-1]<self.sout[ill][-1]: self.logger.warning('Last s = {:.5g} is outside integration range for ell = {:d}: the correlation function will be constant beyond last s = {:.5g}.'.format(self.sout[ill][-1],ell,self.s[-1]))
	
	@property
	def ellsout(self):
		return self.params['ellsout']
		
	def indexin(self,ell):
		return self.ellsin.index(ell)

	def indexout(self,ell):
		return self.ellsout.index(ell)
	
	def norm(self,kwargs,corrconvolution):
		norm = kwargs.get('normwindow',None)
		if (norm is not None) and corrconvolution: return norm/self.normwindow
		return 1.
	
	def apply_fiber_collisions(self,Pl,corrfibercol='all'):
		if corrfibercol == 'all':
			Pl -= self.fiber_collisions(Pl)
		elif corrfibercol == 'correlated':
			Pl -= self.fiber_collisions.correlated(Pl)
		elif corrfibercol == 'uncorrelated':
			Pl -= self.fiber_collisions.uncorrelated()
			
	def correlation_multipoles(self,*args,**kwargs):

		corrfibercol = kwargs.get('fibercollisions',False)
		Pl = list(self.input_model(*args,**kwargs))
		self.apply_fiber_collisions(Pl[0],corrfibercol=corrfibercol)
		Xil = self.toreal.integrate(*Pl,**kwargs)
		
		return [scipy.interp(sout,self.s,xil) for sout,pl in zip(self.sout,Xil)]
	
	def correlation_cic_multipoles(self,*args,**kwargs):

		corrfibercol = kwargs.get('fibercollisions',False)
		corrconvolution = kwargs.get('convolution',False)
		corrcic = kwargs.get('cic',False)
		corrcicconv = kwargs.get('cicconv',False)
		norm = self.norm(kwargs,corrconvolution)
		Pl = list(self.input_model(*args,**kwargs))
		self.toreal.integrate(*Pl,**kwargs)
		Ns = kwargs.get('Ns',0.)*norm
		self.apply_fiber_collisions(Pl[0],corrfibercol=corrfibercol)
		if corrconvolution:
			Xil = self.convolution.convolve(self.toreal['convolution'])
		else:
			Xil = self.toreal['baseline']
		if corrcic:
			Xil -= self.cic[corrcic].ic(self.toreal['cic_{}'.format(corrcic)]) + self.cic[corrcic].real_shotnoise * Ns
		if corrcicconv:
			Xiln0 = self.toreal['convolution'][self.toreal.mask('convolution',n=0)]
			Xil -= self.cicconv[corrcicconv].convolution(Xiln0,self.toreal['cicconv_{}'.format(corrcicconv)]) + self.cicconv[corrcicconv].real_shotnoise(self.toreal['convolution']) * Ns

		return [scipy.interp(sout,self.s,xil)/norm for sout,xil in zip(self.sout,Xil)]
	
	def spectrum_multipoles(self,*args,**kwargs):
	
		corrfibercol = kwargs.get('fibercollisions',False)
		corrconvolution = kwargs.get('convolution',False)
		corrcic = kwargs.get('cic',False)
		corrcicconv = kwargs.get('cicconv',False)
		norm = self.norm(kwargs,corrconvolution)
		#norm = 0.994660854012
		Pl = list(self.input_model(*args,**kwargs))
		Ns = kwargs.get('Ns',0.)*norm
		self.apply_fiber_collisions(Pl[0],corrfibercol=corrfibercol)
		if corrconvolution:
			self.toreal.integrate(*Pl,**kwargs)
			Xil = self.convolution.convolve(self.toreal['convolution'])
		else:
			return [scipy.interp(kout,self.k,pl+sn*Ns)/norm for kout,pl,sn in zip(self.kout,Pl[0],self.shotnoise)]
		if corrcic:
			Xil -= self.cic[corrcic].ic(self.toreal['cic_{}'.format(corrcic)]) + self.cic[corrcic].real_shotnoise*Ns
		if corrcicconv:
			Xiln0 = self.toreal['convolution'][self.toreal.mask('convolution',n=0)]
			Xil -= self.cicconv[corrcicconv].convolution(Xiln0,self.toreal['cicconv_{}'.format(corrcicconv)]) + self.cicconv[corrcicconv].real_shotnoise(self.toreal['convolution']) * Ns
		Plcic = self.tofourier.integrate(Xil) + self.shotnoise * Ns	# add Poisson shotnoise

		toret = [scipy.interp(kout,self.k,pl)/norm for kout,pl in zip(self.kout,Plcic)]
		if hasattr(self,'systematics'):
			model = self.systematics.model(**kwargs)
			for illout,mout in enumerate(model):
				toret[illout] += mout

		return toret

	def set_input_model(self,func):
		self.input_model = func

	@utils.getstateclass
	def getstate(self,state):
		for key in ['s','k','sout','kout','ellsin','normwindow','shotnoise']:
			if hasattr(self,key): state[key] = getattr(self,key)
		for key in ['toreal','tofourier','convolution','fiber_collisions','systematics']:
			if hasattr(self,key): state[key] = getattr(self,key).getstate()
		for key in ['cic','cicconv']:
			if hasattr(self,key):
				state[key] = {key:val.getstate() for key,val in getattr(self,key).items()}
		return state

	@utils.setstateclass
	def setstate(self,state):
		if 'toreal' in state:
			self.toreal = LOSCorrections.loadstate(self.toreal)
		if 'tofourier' in state:
			self.tofourier = FFTlogBessel.loadstate(self.tofourier)
		if 'convolution' in state:
			self.convolution = ConvolutionMultipole.loadstate(self.convolution)
		if 'cic' in state:
			self.cic = {key: ConvolvedIntegralConstraint.loadstate(val) for key,val in state['cic'].items()}
		if 'cicconv' in state:
			self.cicconv = {key: ConvolvedIntegralConstraintConvolution.loadstate(val) for key,val in state['cicconv'].items()}
		if 'fiber_collisions' in state:
			self.fiber_collisions = FiberCollisions.loadstate(self.fiber_collisions)
		if 'systematics' in state:
			self.systematics = TemplateSystematics.loadstate(self.systematics)
		#print self.params['path_cic']

	@classmethod
	@utils.loadclass
	def load(self,state):
		self.setstate(state)
		return self

	@utils.saveclass
	def save(self):
		return self.getstate()
