import logging
import scipy
from scipy import constants
from classy import Class,CosmoSevereError

COSMOMC_TO_CLASS = {'omegabh2':'omega_b','omegach2':'omega_cdm','ns':'n_s','logA':'ln10^{10}A_s','tau':'tau_reio','H0*':'H0'}
LATEX = {'sigma8':'\\sigma_{8}','fsigma8':'f\\sigma_{8}','alpha_par':'\\alpha_{\\parallel}','alpha_per':'\\alpha_{\\perp}',
		'DA':'D_{A}','DV':'D_{V}','FAP':'F_{\\rm AP}',
		'Hrd':'Hr_{d}','DA/rd':'D_{A}/r_{d}','DV/rd':'D_{V}/r_{d}','DM/rd':'D_{M}/r_{d}',
		'Hrd/rdfid':'Hr_{d}/r_{d}^{\\rm fid}','DArdfid/rd':'D_{A}r_{d}^{\\rm fid}/r_{d}','DVrdfid/rd':'D_{V}r_{d}^{\\rm fid}/r_{d}','DMrdfid/rd':'D_{M}r_{d}^{\\rm fid}/r_{d}'}
celerity = constants.c/1e3

def par_to_latex(par):
	return LATEX.get(par,par)

def cosmomc_to_class(cosmomc,clatex={}):
	
	Class = {}
	Clatex = {}
	
	for cpar in cosmomc.keys():
		if cpar in COSMOMC_TO_CLASS:
			Cpar = COSMOMC_TO_CLASS[cpar]
			Class[Cpar] = cosmomc[cpar]
			if cpar in clatex: Clatex[Cpar] = clatex[cpar]
	
	return Class,Clatex

class Cosmology(Class):

	logger = logging.getLogger('Cosmology')

	def __init__(self,*args,**kwargs):
		super(Cosmology,self).__init__(*args,**kwargs)
		self.derived = {}
		self.set_params(**kwargs)
	
	def set_params(self,format='',**params):
		new = {}
		new.update(self.pars)
		if format == 'cosmomc':
			for par in COSMOMC_TO_CLASS:
				new[COSMOMC_TO_CLASS[par]] = params[par]
		else:
			new.update(params)
			if 'H0' in new:
				self.derived.update({'H0':new['H0']})
				new['h'] = new.pop('H0')/100.
			if 'Omega_b' in new:
				self.derived.update({'Omega_b':new['Omega_b']})
				new['omega_b'] = new.pop('Omega_b')*new['h']**2
			if 'Omega_m' in new:
				self.derived.update({'Omega_m':new['Omega_m']})
				new['omega_m'] = new.pop('Omega_m')*new['h']**2
			if 'omega_m' in new:
				self.derived.update({'omega_m':new['omega_m']})
				new['omega_cdm'] = new.pop('omega_m')-new['omega_b']-new.get('m_ncdm',0.)/93.14
		self.set(**new)
	
	@property
	def params(self):
		return {k:v for d in (self.pars,self.derived) for k,v in d.items()}
	
	def compute_background(self):
		self.compute(level=['background'])
	
	def set_neutrinos(self,N_ur=2.0328,N_ncdm=1,m_ncdm=0.06):
		self.set_params(N_ur=N_ur,N_ncdm=N_ncdm,m_ncdm=m_ncdm)

	def compute_fs8(self,z,cb=False):
		self.set_params(**{'output':'tCl,pCl,lCl,mPk','lensing':'yes','z_pk':z})
		self.compute()
		sigma8 = self.compute_sigma8(z,cb=cb)
		return sigma8*self.scale_independent_growth_factor_f(z)

	def compute_fs8_bao(self,z,cb=False):
		"""H0 in km/s/Mpc, distances in Mpc"""
		self.set_params(**{'output':'tCl,pCl,lCl,mPk','lensing':'yes','z_pk':z})
		self.compute()
		sigma8 = self.compute_sigma8(z,cb=cb)
		f = self.scale_independent_growth_factor_f(z)
		fsigma8 = f*sigma8
		H = self.Hubble(z) #Hubble in 1/Mpc
		DA = self.angular_distance(z) #DA in Mpc
		DM = (1+z)*DA
		FAP = (1+z)*DA*H
		DV = (((1+z)*DA)**2*z/H)**(1./3.)
		rd = self.rs_drag()
		Hrd = H*rd
		DArd = DA/rd
		DMrd = DM/rd
		DVrd = DV/rd
		return {'f':f,'sigma8':sigma8,'fsigma8':fsigma8,'H':H*celerity,'DA':DA,'DM':DM,'DV':DV,'FAP':FAP,'Hrd':Hrd*celerity,'DA/rd':DArd,'DM/rd':DMrd,'DV/rd':DVrd,'rd':rd}

	def compute_pk(self,k,z=None,cb=False,nonlinear=''):
		"""k in h/Mpc, P(k) in (Mpc/h)^3"""
		k = scipy.atleast_1d(k)
		if cb and self.params.get('N_ncdm',0) == 0:
			self.logger.warning('No neutrinos, switching to compute_pk_lin...')
			cb = False
		if z is not None: self.set_params(z_pk=z)
		self.set_params(**{'output':'tCl,pCl,lCl,mPk','lensing':'yes','P_k_max_h/Mpc':k.max(),'non linear':nonlinear})
		self.compute()
		h = self.h()
		if cb:
			self.logger.info('Computing power spectrum of cold dark matter and baryons.')
			if nonlinear: fun = self.pk_cb
			else: fun = self.pk_cb_lin
		else:
			self.logger.info('Computing matter power spectrum.')
			if nonlinear: fun = self.pk
			else: fun = self.pk_lin
		
		return h**3*scipy.asarray([fun(k_*h,self.params['z_pk']) for k_ in k])

	def compare_ap(self,other,z=0.,z_other=None):
		if z_other is None: z_other = z
		alpha_par = self.Hubble(z)*self.rs_drag()/(other.Hubble(z_other)*other.rs_drag())
		alpha_per = other.angular_distance(z_other)*self.rs_drag()/(self.angular_distance(z)*other.rs_drag())
		alpha_iso = (alpha_per**2*alpha_par)**(1./3.)
		epsilon = (alpha_par/alpha_per)**(1./3.)-1.
		return {'alpha_par':alpha_par,'alpha_per':alpha_per,'alpha_iso':alpha_iso,'epsilon':epsilon}

	def compute_bao_from_ap(self,alpha_par=1.,alpha_per=1.,alpha_iso=None,epsilon=0.,z=0.,remove_rdfid=True,rdfid=None):
		"""H0 in km/s/Mpc, distances in Mpc"""
		if alpha_iso is not None:
			alpha_par = (1.+epsilon)**2*alpha_iso
			alpha_per = 1./(1.+epsilon)*alpha_iso
		if remove_rdfid:
			rd = self.rs_drag()
			self.logger.info('Removing rdfid = {:.5g}.'.format(rd))
			suffix_H = 'rd'
			suffix_D = '/rd'
		else:
			rd = 1.
			if rdfid is not None:
				rd = self.rs_drag()/rdfid
				self.logger.info('Renormalising rdfid by {:.8g}.'.format(rd))
			suffix_H = 'rd/rdfid'
			suffix_D = 'rdfid/rd'
		Hrd = (self.Hubble(z)*rd)/alpha_par
		DArd = (self.angular_distance(z)/rd)*alpha_per
		DMrd = (1+z)*DArd
		FAP = DMrd*Hrd
		DVrd = (DMrd**2*z/Hrd)**(1./3.)
		return {'H'+suffix_H:Hrd*celerity,'DA'+suffix_D:DArd,'DM'+suffix_D:DMrd,'DV'+suffix_D:DVrd,'FAP':FAP}

	def compute_sigma8(self,z=None,cb=False):
		if z is not None: self.set_params(z_pk=z)
		if cb and self.params.get('N_ncdm',0) == 0:
			self.logger.warning('No neutrinos, switching to matter...')
			cb = False
		if cb: fun = self.sigma_cb
		else: fun = self.sigma
		return fun(8./self.h(),self.params['z_pk'])

	def Omega0_m(self):
		return self.params.get('Omega_m',self.Omega_m())

	def efunc(self,z):
		return self.Hubble(z)/self.Hubble(0.)
	
	def comoving_distance(self,z):
		if scipy.isscalar(z): return self.z_of_r([z])[0]
		return self.z_of_r(z)
		
	def angular_distance(self,z):
		if scipy.isscalar(z): return super(Cosmology,self).angular_distance(z)
		return map(super(Cosmology,self).angular_distance,z)

	def growth_factor(self,z):
		"""Approximation for the matter growth factor.
		Uses a Pade approximation.
		Parameters
		----------
		z: array_like
			Redshift to calculate at.
		Returns
		-------
		growth_factor : array_like
		Notes
		-----
		See _[1].
		.. [1] http://arxiv.org/abs/1012.2671
		"""
		x = ((1.0 / self.Omega0_m()) - 1.0) / (1.0 + z)**3
	
		num = 1.0 + 1.175*x + 0.3064*x**2 + 0.005355*x**3
		den = 1.0 + 1.857*x + 1.021 *x**2 + 0.1530  *x**3
		d = (1.0 + x)**0.5 / (1.0 + z) * num / den

		return d

	def growth_rate_pade(self,z):
		"""Approximation for the matter growth rate.
		From explicit differentiation of the Pade approximation for the growth factor.
		Parameters
		----------
		z: array_like
			Redshift to calculate at.
		Returns
		-------
		growth_factor : array_like
		Notes
		-----
		See _[1].
		.. [1] http://arxiv.org/abs/1012.2671
		"""

		x = ((1.0 / self.Omega0_m()) - 1.0) / (1.0 + z)**3
		dnum = 3.0*x*(1.175 + 0.6127*x + 0.01607*x**2)
		dden = 3.0*x*(1.857 + 2.042 *x + 0.4590 *x**2)

		num = 1.0 + 1.175*x + 0.3064*x**2 + 0.005355*x**3
		den = 1.0 + 1.857*x + 1.021 *x**2 + 0.1530  *x**3
		f = 1.0 + 1.5 * x / (1.0 + x) + dnum / num - dden / den

		return f

	"""
	def growth_rate_linder(self,z):
		#Linder 2005: :astro-ph/0507263v2 eq. 17
		back = self.get_background()
		Gamma = 0.55 + 0.05*(1+self.w(z))
		return self.Omega_m(z)**Gamma
	"""

	def derived_parameters(self,names):
		if not isinstance(names,list):
			return self.get_current_derived_parameters([names])[names]
		return self.get_current_derived_parameters(names)
			

	def copy(self):
		new = self.__class__()
		new.set_params(**self.params)
		return new

	def getstate(self):
		return self.params
	
	def setstate(self,state):
		self.set_params(**state)
	
	@classmethod
	def loadstate(cls,state):
		self = cls()
		self.setstate(**state)
		return self

	def clear(self):
		self.struct_cleanup()

	@classmethod
	def Planck2018(cls):
		self = cls()
		self.set_params(omega_b=0.02237,omega_cdm=0.1200,H0=67.36,tau_reio=0.0544,n_s=0.9649,N_ur=2.0328,N_ncdm=1,m_ncdm=0.06)
		self.set_params(**{'ln10^{10}A_s':3.044})
		return self
	
	@classmethod
	def BOSS(cls):
		self = cls()
		self.set_params(h=0.676,Omega_m=0.31,omega_b=0.022,sigma8=0.8,n_s=0.97,N_ur=2.0328,N_ncdm=1,m_ncdm=0.06)
		return self

	@classmethod
	def MultiDark(cls):
		self = cls()
		# ns = 0.9611 and not 0.96 because Cheng's email
		self.set_params(h=0.6777,Omega_m=0.307115,Omega_b=0.048206,sigma8=0.8288,n_s=0.9611)
		return self

	@classmethod
	def EZ(cls):
		#Patchy, EZ...
		self = cls()
		self.set_params(h=0.6777,Omega_m=0.307115,Omega_b=0.048206,sigma8=0.8225,n_s=0.9611)
		return self

	@classmethod
	def NSeries(cls):
		self = cls()
		self.set_params(h=0.70,Omega_m=0.286,omega_b=0.02303,sigma8=0.82,n_s=0.96)
		return self

	@classmethod
	def OuterRim(cls):
		self = cls()
		self.set_params(h=0.71,omega_cdm=0.1109,omega_b=0.02258,sigma8=0.8,n_s=0.963)
		return self
