import logging
import functools
import scipy
from scipy import special,constants,integrate
from model_base import BasePowerSpectrumModel,damping
import pyregpt
from pyregpt import *
import utils

class ModelTNS(BasePowerSpectrumModel):

	TERMS = ['spectrum_lin','spectrum_nowiggle','spectrum_halofit','spectrum_2loop_dd','spectrum_2loop_dt','spectrum_2loop_tt','spectrum_1loop_dd','spectrum_1loop_dt','spectrum_1loop_tt','bias_1loop','A_1loop','A_2loop','B_1loop','B_2loop']
	logger = logging.getLogger('ModelTNS')

	def setup(self,nloop=2):
		self.set_cosmology()
		self.set_spectrum_lin()
		self.set_spectrum_nonlin()
		if nloop >= 1:
			self.set_spectrum_1loop()
			self.set_bias_1loop()
			self.set_A_1loop()
			self.set_B_1loop()
		if nloop >= 2:
			self.set_spectrum_2loop()
			self.set_A_2loop()
			self.set_B_2loop()
	
	@utils.classparams
	def set_spectrum_1loop(self,kspectrum,kspectrumout=None,precision=[],nthreads=8):
		pyregpt = Spectrum1Loop()
		pyregpt.set_spectrum_lin(self.spectrum_lin)
		for prec in precision: pyregpt.set_precision(**prec)
		for a,b in [('delta','delta'),('delta','theta'),('theta','theta')]:
		#for a,b in [('theta','theta')]:
			pyregpt.set_terms(kspectrum)
			pyregpt.run_terms(a,b,nthreads=nthreads)
			pyregpt.nan_to_zero()
			if kspectrumout is not None: pyregpt.pad_k(k=kspectrumout)
			setattr(self,'spectrum_1loop_{}{}'.format(a[0],b[0]),pyregpt.deepcopy())
		pyregpt.clear()

	@utils.classparams
	def set_spectrum_2loop(self,kspectrum,kspectrumout=None,precision=[],nthreads=8):
		pyregpt = Spectrum2Loop()
		for prec in precision: pyregpt.set_precision(**prec)
		for a,b in [('delta','delta'),('delta','theta'),('theta','theta')]:
		#for a,b in [('delta','delta')]:
			pyregpt.set_spectrum_lin(self.spectrum_lin)
			pyregpt.set_terms(kspectrum)
			pyregpt.run_terms(a,b,nthreads=nthreads)
			pyregpt.nan_to_zero()
			if kspectrumout is not None: pyregpt.pad_k(k=kspectrumout)
			setattr(self,'spectrum_2loop_{}{}'.format(a[0],b[0]),pyregpt.deepcopy())
		pyregpt.clear()
	
	@utils.classparams
	def set_bias_1loop(self,kbias,precision=[],nthreads=8):
		pyregpt = Bias1Loop()
		pyregpt.set_spectrum_lin(self.spectrum_lin)
		pyregpt.set_terms(kbias)
		for prec in precision: pyregpt.set_precision(**prec)
		pyregpt.run_terms(nthreads=nthreads)
		self.bias_1loop = pyregpt
		pyregpt.clear()
		
	@utils.classparams
	def set_A_1loop(self,kA,precision=[],nthreads=8):
		pyregpt = A1Loop()
		pyregpt.set_spectrum_lin(self.spectrum_lin)
		pyregpt.set_terms(kA)
		for prec in precision: pyregpt.set_precision(**prec)
		pyregpt.run_terms(nthreads=nthreads)
		self.A_1loop = pyregpt
		pyregpt.clear()
		
	@utils.classparams
	def set_A_2loop(self,kA,precision=[],nthreads=8):
		pyregpt = A2Loop()
		pyregpt.set_spectrum_lin(self.spectrum_lin)
		pyregpt.set_terms(kA)
		for prec in precision: pyregpt.set_precision(**prec)
		pyregpt.run_terms(nthreads=nthreads)
		self.A_2loop = pyregpt
		pyregpt.clear()
		
	@utils.classparams
	def set_B_1loop(self,kB,precision=[],nthreads=8):
		pyregpt = B1Loop()
		pyregpt.set_spectrum_lin(self.spectrum_lin)
		pyregpt.set_terms(kB)
		for prec in precision: pyregpt.set_precision(**prec)
		pyregpt.run_terms(nthreads=nthreads)
		self.B_1loop = pyregpt
		pyregpt.clear()
	
	@utils.classparams
	def set_B_2loop(self,kB,precision=[],nthreads=8):
		pyregpt = B2Loop()
		pyregpt.set_spectrum_lin(self.spectrum_lin)
		pyregpt.set_terms(kB)
		for prec in precision: pyregpt.set_precision(**prec)
		pyregpt.run_terms(nthreads=nthreads)
		self.B_2loop = pyregpt
		pyregpt.clear()

	def bs2(self,b1):
		return -4./7.*(b1-1.)

	def b3nl(self,b1):
		return 32./315.*(b1-1.)
	
	@damping(kmin=1e-5,kmax=1.)
	def spectrum_galaxy(self,k,mu,f=0.8,b1=1.3,b2=0.7,sigmav=None,bs2=None,b3nl=None,Ng=0.,rsigma8=1.,FoG='gaussian',sigmad2=None,uvcutoff=None,wFoG=None,avir=0.,sigmaerr=0.,**kwargs):
		# Beutler 2016 (arXiv: 1607.03150v1) eq. 23-25

		if bs2 is None: bs2 = self.bs2(b1)
		if b3nl is None: b3nl = self.b3nl(b1)
		if uvcutoff is not None:
			sigmad2_2loop = self.pyregpt.calc_running_sigmad2(self.spectrum_2loop_dd.k,uvcutoff=uvcutoff)
			sigmad2_bias = self.pyregpt.calc_running_sigmad2(self.bias_1loop.k,uvcutoff=uvcutoff)
		else:
			sigmad2_2loop = sigmad2_bias = sigmad2
		#testing.assert_allclose(sigmad2_2loop,self.spectrum_2loop_dd['sigmad2'],rtol=1e-6,atol=1e-7)
		#testing.assert_allclose(sigmad2_bias,self.bias_1loop['sigmad2'],rtol=1e-6,atol=1e-7)

		"""
		Pgdd = b1**2*self.spectrum_2loop_dd.pk_interp(k,Dgrowth=rsigma8,sigmad2=sigmad2_2loop,left=0.,right=0.) \
			+ scipy.interp(k,self.bias_1loop.k,2*b2*b1*self.bias_1loop.pk_b2d(rsigma8)
			+ 2*bs2*b1*self.bias_1loop.pk_bs2d(rsigma8)
			+ 2*b3nl*b1*self.bias_1loop.pk_sigma3sq(rsigma8)
			+ b2**2*self.bias_1loop.pk_b22_damp(rsigma8,sigmad2=sigmad2_bias)
			+ 2*b2*bs2*self.bias_1loop.pk_b2s2_damp(rsigma8,sigmad2=sigmad2_bias)
			+ bs2**2*self.bias_1loop.pk_bs22_damp(rsigma8,sigmad2=sigmad2_bias),
			,left=0.,right=0.) + Ng
		
		"""
		Pgdd = b1**2*self.spectrum_2loop_dd.pk_interp(k,Dgrowth=rsigma8,sigmad2=sigmad2_2loop,left=0.,right=0.) \
			+ scipy.interp(k,self.bias_1loop.k,2*b2*b1*self.bias_1loop.pk_b2d(rsigma8)
			+ 2*bs2*b1*self.bias_1loop.pk_bs2d(rsigma8)
			+ 2*b3nl*b1*self.bias_1loop.pk_sigma3sq(rsigma8)
			+ b2**2*self.bias_1loop.pk_b22(rsigma8)
			+ 2*b2*bs2*self.bias_1loop.pk_b2s2(rsigma8)
			+ bs2**2*self.bias_1loop.pk_bs22(rsigma8)
			,left=0.,right=0.) + Ng

		Pgdt = b1*self.spectrum_2loop_dt.pk_interp(k,Dgrowth=rsigma8,sigmad2=sigmad2_2loop,left=0.,right=0.) \
			+ scipy.interp(k,self.bias_1loop.k,b2*self.bias_1loop.pk_b2t(rsigma8)
			+ bs2*self.bias_1loop.pk_bs2t(rsigma8)
			+ b3nl*self.bias_1loop.pk_sigma3sq(rsigma8)
			,left=0.,right=0.)

		Ptt = self.spectrum_2loop_tt.pk_interp(k,Dgrowth=rsigma8,sigmad2=sigmad2_2loop,left=0.,right=0.)

		mu2 = mu**2
		fmu2 = f*mu2
		A = self.A_2loop.pk_interp(k,mu2,Dgrowth=rsigma8,beta=f/b1,left=0.,right=0.)
		B = self.B_2loop.pk_interp(k,mu2,Dgrowth=rsigma8,beta=f/b1,left=0.,right=0.)

		if sigmav is None: sigmav = self.spectrum_lin.sigmav(Dgrowth=rsigma8)
		DFoG = self.sum_FoG(k,mu,FoG=FoG,sigmav=sigmav,wFoG=wFoG,avir=avir,sigmaerr=sigmaerr,f=f)

		return DFoG*(Pgdd+2*fmu2*Pgdt+fmu2**2*Ptt+b1**3*A+b1**4*B)
	
	@damping(kmin=1e-5,kmax=1.)
	def spectrum_galaxy_1loop(self,k,mu,f=0.8,b1=1.3,b2=0.7,sigmav=4.,bs2=None,b3nl=None,Ng=0.,rsigma8=1.,FoG='gaussian',sigmad2=None,uvcutoff=None,wFoG=None,infall=None,**kwargs):
		# Beutler 2016 (arXiv: 1607.03150v1) eq. 23-25

		if bs2 is None: bs2 = self.bs2(b1)
		if b3nl is None: b3nl = self.b3nl(b1)
		if uvcutoff is not None:
			sigmad2_1loop = self.pyregpt.calc_running_sigmad2(self.spectrum_1loop_dd.k,uvcutoff=uvcutoff)
			sigmad2_bias = self.pyregpt.calc_running_sigmad2(self.bias_1loop.k,uvcutoff=uvcutoff)
		else:
			sigmad2_1loop = sigmad2_bias = sigmad2
		
		Pgdd = b1**2*self.spectrum_1loop_dd.pk_interp(k,Dgrowth=rsigma8,sigmad2=sigmad2_1loop,left=0.,right=0.) \
			+ scipy.interp(k,self.bias_1loop.k,2*b2*b1*self.bias_1loop.pk_b2d(rsigma8)
			+ 2*bs2*b1*self.bias_1loop.pk_bs2d(rsigma8)
			+ 2*b3nl*b1*self.bias_1loop.pk_sigma3sq(rsigma8)
			+ b2**2*self.bias_1loop.pk_b22(rsigma8)
			+ 2*b2*bs2*self.bias_1loop.pk_b2s2(rsigma8)
			+ bs2**2*self.bias_1loop.pk_bs22(rsigma8)
			,left=0.,right=0.) + Ng
	
		Pgdt = b1*self.spectrum_1loop_dt.pk_interp(k,Dgrowth=rsigma8,sigmad2=sigmad2_1loop,left=0.,right=0.) \
			+ scipy.interp(k,self.bias_1loop.k,b2*self.bias_1loop.pk_b2t(rsigma8)
			+ bs2*self.bias_1loop.pk_bs2t(rsigma8)
			+ b3nl*self.bias_1loop.pk_sigma3sq(rsigma8)
			,left=0.,right=0.)

		Ptt = self.spectrum_1loop_tt.pk_interp(k,Dgrowth=rsigma8,sigmad2=sigmad2_1loop,left=0.,right=0.)

		mu2 = mu**2
		fmu2 = f*mu2
		A = self.A_1loop.pk_interp(k,mu2,Dgrowth=rsigma8,beta=f/b1,left=0.,right=0.)
		B = self.B_1loop.pk_interp(k,mu2,Dgrowth=rsigma8,beta=f/b1,left=0.,right=0.)

		DFoG = self.sum_FoG(k,mu,FoG=FoG,sigmav=sigmav,wFoG=wFoG,avir=avir,f=f)

		return DFoG*(Pgdd+2*fmu2*Pgdt+fmu2**2*Ptt+b1**3*A+b1**4*B)

	@damping(kmin=1e-5,kmax=1.)
	def spectrum_galaxy_lognormal(self,k,mu,f=0.8,b1=1.3,Ng=0.,rsigma8=1.,sigmav=4.,FoG='lorentzian',**kwargs):

		Plin = self.spectrum_lin.pk_interp(k,Dgrowth=rsigma8,left=0.,right=0.)

		Pgdd = b1**2*Plin + Ng
		Pgdt = b1*Plin
		Ptt = Plin

		mu2 = mu**2
		fmu2 = f*mu2
		A = self.A_1loop.pk_interp(k,mu2,Dgrowth=rsigma8,beta=f/b1,left=0.,right=0.)
		B = self.B_1loop.pk_interp(k,mu2,Dgrowth=rsigma8,beta=f/b1,left=0.,right=0.)
		#A,B = 0.,0.

		return self.DFoG(k,mu,f=f,sigmav=sigmav,FoG=FoG)*(Pgdd+2*fmu2*Pgdt+fmu2**2*Ptt+b1**3*A+b1**4*B)
	
	def spectrum_matter_2loop(self,k,mu,f=0.8,sigmav=4.,rsigma8=1.,FoG='gaussian',sigmad2=None,uvcutoff=None,**kwargs):
	
		if uvcutoff is not None:
			sigmad2_2loop = self.pyregpt.calc_running_sigmad2(self.spectrum_2loop_dd.k,uvcutoff=uvcutoff)
		else:
			sigmad2_2loop = sigmad2
		
		Pdd = self.spectrum_2loop_dd.pk_interp(k,Dgrowth=rsigma8,sigmad2=sigmad2_2loop,left=0.,right=0.)
		Pdt = self.spectrum_2loop_dt.pk_interp(k,Dgrowth=rsigma8,sigmad2=sigmad2_2loop,left=0.,right=0.)
		Ptt = self.spectrum_2loop_tt.pk_interp(k,Dgrowth=rsigma8,sigmad2=sigmad2_2loop,left=0.,right=0.)
		mu2 = mu**2
		fmu2 = f*mu2
		A = self.A_2loop.pk_interp(k,mu2,Dgrowth=rsigma8,beta=f,left=0.,right=0.)
		B = self.B_2loop.pk_interp(k,mu2,Dgrowth=rsigma8,beta=f,left=0.,right=0.)
		#A = 0.; B = 0.; Pdt = Pdd; Ptt = Pdd

		return self.DFoG(k,mu,f=f,sigmav=sigmav,FoG=FoG)*(Pdd+2*fmu2*Pdt+fmu2**2*Ptt+A+B)
	
	def spectrum_matter_1loop(self,k,mu,f=0.8,sigmav=4.,rsigma8=1.,FoG='gaussian',sigmad2=None,uvcutoff=None,**kwargs):
	
		if uvcutoff is not None:
			sigmad2_1loop = self.pyregpt.calc_running_sigmad2(self.spectrum_1loop_dd.k,uvcutoff=uvcutoff)
		else:
			sigmad2_1loop = sigmad2
		
		Pdd = self.spectrum_1loop_dd.pk_interp(k,Dgrowth=rsigma8,sigmad2=sigmad2_1loop,left=0.,right=0.)
		Pdt = self.spectrum_1loop_dt.pk_interp(k,Dgrowth=rsigma8,sigmad2=sigmad2_1loop,left=0.,right=0.)
		Ptt = self.spectrum_1loop_tt.pk_interp(k,Dgrowth=rsigma8,sigmad2=sigmad2_1loop,left=0.,right=0.)
		mu2 = mu**2
		fmu2 = f*mu2
		A = self.A_1loop.pk_interp(k,mu2,Dgrowth=rsigma8,beta=f,left=0.,right=0.)
		B = self.B_1loop.pk_interp(k,mu2,Dgrowth=rsigma8,beta=f,left=0.,right=0.)

		#return self.DFoG(k,mu,f=f,sigmav=sigmav,FoG=FoG)*(Pdd+2*fmu2*Pdt+fmu2**2*Ptt)
		return self.DFoG(k,mu,f=f,sigmav=sigmav,FoG=FoG)*(Pdd+2*fmu2*Pdt+fmu2**2*Ptt+A+B)

	def spectrum_A_B_2loop(self,k,mu,f=0.8,b1=1.3,sigmav=4.,FoG='gaussian',rsigma8=1.,**kwargs):
		mu2 = mu**2
		A = self.A_2loop.pk_interp(k,mu2,Dgrowth=rsigma8,beta=f/b1,left=0.,right=0.)
		B = self.B_2loop.pk_interp(k,mu2,Dgrowth=rsigma8,beta=f/b1,left=0.,right=0.)
		return scipy.asarray([self.DFoG(k,mu,f=f,sigmav=sigmav,FoG=FoG)*b1**3*A,self.DFoG(k,mu,f=f,sigmav=sigmav,FoG=FoG)*b1**4*B])
	
	def spectrum_A_B_1loop(self,k,mu,f=0.8,b1=1.3,sigmav=4.,FoG='gaussian',rsigma8=1.,**kwargs):
		mu2 = mu**2
		A = self.A_1loop.pk_interp(k,mu2,Dgrowth=rsigma8,beta=f/b1,left=0.,right=0.)
		B = self.B_1loop.pk_interp(k,mu2,Dgrowth=rsigma8,beta=f/b1,left=0.,right=0.)
		return scipy.asarray([self.DFoG(k,mu,f=f,sigmav=sigmav,FoG=FoG)*b1**3*A,self.DFoG(k,mu,f=f,sigmav=sigmav,FoG=FoG)*b1**4*B])
