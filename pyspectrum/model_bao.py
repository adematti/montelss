import logging
import functools
import scipy
from scipy import special,constants,integrate
from model_base import BasePowerSpectrumModel,damping
import utils

class ModelBAO(BasePowerSpectrumModel):

	TERMS = ['spectrum_lin','spectrum_nowiggle','spectrum_lin_smooth']
	logger = logging.getLogger('ModelBAO')

	def setup(self):
		self.set_cosmology()
		self.set_spectrum_lin()
		self.set_spectrum_smooth()

	def set_spectrum_smooth(self,kfit=scipy.linspace(0.005,0.5,100)):
		from scipy import optimize
		def model(k,ap0,ap1,ap2,ap3,ap4):
			return self.spectrum_lin.pk_interp(k,Dgrowth=1.,left=0.,right=0.)/(self.spectrum_nowiggle.pk_interp(k,Dgrowth=1.,left=0.,right=0.) * (ap0 + ap1*k + ap2*k**2 + ap3*k**3 + ap4*k**4))
		popt, pcov = optimize.curve_fit(model,kfit,scipy.ones_like(kfit),p0=[1.]+[0.]*4,maxfev=100000)
		k = self.spectrum_lin.k
		wiggles = model(self.spectrum_lin.k,*popt)
		kmin,kmax = kfit[0],kfit[-1]
		ones = scipy.ones_like(k)
		mask = k>kmax
		ones[mask] *= scipy.exp(-1e3*(k[mask]/kmax-1)**2)
		mask = k<kmin
		ones[mask] *= scipy.exp(-(kmin/k[mask]-1)**2)
		wiggles = (wiggles-1.)*ones + 1.
		self.spectrum_lin_smooth = self.spectrum_lin.deepcopy()
		self.spectrum_lin_smooth['pk'] /= wiggles

	def wiggles_eh(self,k):
		return self.spectrum_lin.pk_interp(k,Dgrowth=1.,left=None,right=None)/self.spectrum_nowiggle.pk_interp(k,Dgrowth=1.,left=None,right=None)

	def wiggles_smooth(self,k):
		return self.spectrum_lin.pk_interp(k,Dgrowth=1.,left=None,right=None)/self.spectrum_lin_smooth.pk_interp(k,Dgrowth=1.,left=None,right=None)

	def wiggles_damped_iso(self,k,sigmanl=0.):
		return 1. + (self.wiggles_smooth(k) - 1.) * scipy.exp(-1./2.*(sigmanl*k)**2)

	def wiggles_damped_aniso(self,k,mu,sigmapar=0.,sigmaper=0.):
		return 1. + (self.wiggles_smooth(k) - 1.) * scipy.exp(-1./2.*((sigmapar*k*mu)**2 + (sigmaper*k)**2*(1.-mu**2)))

	def _polynomial_(self,k,ap0=0.,ap1=0.,ap2=0.,ap3=0.,am1=0.,am2=0.,am3=0.,am4=0.,**kwargs):
		return ap0 + ap1*k + ap2*k**2 + ap3*k**3 + am1/k + am2/k**2 + am3/k**3 + am4/k**4

	@damping(kmin=5e-3,kmax=1.)
	def polynomial(self,k,**kwargs):
		return self._polynomial_(k,**kwargs)

	@damping(kmin=5e-3,kmax=1.)
	def spectrum_smooth_iso(self,k,b=1.,sigmas=0.,FoG='lorentzian2',rsigma8=1.,**kwargs):
		DFoG = self.DFoG(k,mu=1.,sigmav=abs(sigmas),FoG=FoG)
		if sigmas < 0: DFoG = 1./DFoG
		#return 1./qiso**3 * b**2 * self.spectrum_lin_smooth.pk_interp(k/qiso,Dgrowth=rsigma8,left=0.,right=0.) * DFoG
		return b**2 * self.spectrum_lin_smooth.pk_interp(k,Dgrowth=rsigma8,left=0.,right=0.) * DFoG

	def spectrum_galaxy_iso(self,kobs,qiso=1.,sigmanl=0.,decoupled=True,**kwargs):
		k = kobs if decoupled else kobs/qiso
		Psm = self.spectrum_smooth_iso(k,**kwargs)
		return Psm * self.wiggles_damped_iso(kobs/qiso,sigmanl=sigmanl)
	
	@damping(kmin=5e-3,kmax=1.)
	def spectrum_smooth_aniso(self,k,mu,b=1.,beta=0.,sigmapar=0.,sigmaper=0.,sigmas=0.,sigmasm=0.,FoG='lorentzian2',rsigma8=1.,recon=False,kobs=None,decoupled=True,**kwargs):
		if decoupled: k = kobs
		if recon: r = 1. - scipy.exp(-1./2*(k*sigmasm)**2)
		else: r = 1.
		DFoG = self.DFoG(k,mu,sigmav=abs(sigmas),FoG=FoG)
		if sigmas < 0: DFoG = 1./DFoG
		return b**2 * (1. + beta*mu**2*r)**2 * self.spectrum_lin_smooth.pk_interp(k,Dgrowth=rsigma8,left=0.,right=0.) * DFoG

	def spectrum_galaxy_aniso(self,k,mu,sigmapar=0.,sigmaper=0.,**kwargs):
		Psm = self.spectrum_smooth_aniso(k,mu,sigmapar=sigmapar,sigmaper=sigmaper,**kwargs)
		return Psm * self.wiggles_damped_aniso(k,mu,sigmapar=sigmapar,sigmaper=sigmaper)
