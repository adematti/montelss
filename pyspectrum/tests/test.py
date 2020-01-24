import os
import scipy
from matplotlib import pyplot
from pyspectrum import *
from cosmology import Cosmology

setup_logging()
dirtest = os.path.join(os.getenv('BAO'),'pyspectrum','tests')
dirplot = os.path.join(dirtest,'plots')
utils.mkdir(dirplot)

parameters = {'ModelTNS':{},'ModelBAO':{},'EffectAP':{},'SurveyGeometry':{}}

parameters['ModelTNS']['kin'] = scipy.logspace(-5,4.1,4500,base=10) #do not take wider (-6,5) as Class takes years to compute
#parameters['ModelTNS']['kin'] = scipy.logspace(scipy.log10(5e-4),1,4500,base=10)
#parameters['ModelTNS']['redshift'] = 0.61
parameters['ModelTNS']['redshift'] = 0.5
parameters['ModelTNS']['cosmology'] = 'BOSS'
parameters['ModelTNS']['Class'] = {'input_verbose':1}
parameters['ModelTNS']['precision'] = [] #RegPT values
#parameters['ModelTNS']['precision'] = [{'calculation':'all_q','n':400}]
#parameters['ModelTNS']['precision'] = [{'calculation':'bias_1loop_q','n':2000,'min':5e-4,'max':1e1},{'calculation':'bias_1loop_mu','n':400},{'calculation':'A_1loop_q','min':5e-4,'max':1e1}]
#parameters['ModelTNS']['precision'] = [{'calculation':'bias_1loop_q','n':2000,'min':5e-4,'max':1e1},{'calculation':'bias_1loop_mu','n':400},{'calculation':'A_1loop_q','n':2000},{'calculation':'A_1loop_mu','n':200}]
#parameters['ModelTNS']['precision'] = [{'calculation':'spectrum_1loop_q','n':2000}]
#parameters['ModelTNS']['precision'] = [{'calculation':'gamma1_1loop_q','n':2000}]
#parameters['ModelTNS']['precision'] = [{'calculation':'gamma2_tree_q','n':2000}]
parameters['ModelTNS']['precision'] = [{'calculation':'gamma1_1loop_q','n':2000},{'calculation':'gamma1_2loop_q','n':2000},{'calculation':'gamma2_tree_q','n':2000},{'calculation':'gamma2_1loop_q','n':200},{'calculation':'bias_1loop_q','n':2000,'min':5e-4,'max':1e1},{'calculation':'A_2loop_I_q','n':2000}]
parameters['ModelTNS']['kspectrum'] = scipy.logspace(-2,1,1500,base=10)
#parameters['ModelTNS']['kspectrum'] = scipy.logspace(-2,1,5,base=10)
#parameters['ModelTNS']['kspectrum'] = scipy.logspace(-2,2,1500,base=10)
#parameters['ModelTNS']['kspectrum'] = scipy.logspace(0.9,1.1,100,base=10)
#parameters['ModelTNS']['kspectrum'] = scipy.logspace(-5,4,4500,base=10)
parameters['ModelTNS']['kspectrumout'] = scipy.logspace(-5,4,4500,base=10)
parameters['ModelTNS']['kbias'] = scipy.logspace(-5,4,4500,base=10)
#parameters['ModelTNS']['kbias'] = scipy.logspace(0.9,1.1,100,base=10)
#parameters['ModelTNS']['kbias'] = [9.,9.4,9.8,10.]
#parameters['ModelTNS']['kA'] = scipy.logspace(-3,2,2500,base=10)
#parameters['ModelTNS']['kA'] = scipy.logspace(-2.5,1.9,2200,base=10)
#parameters['ModelTNS']['kA'] = scipy.logspace(-5,4,4500,base=10)
#parameters['ModelTNS']['kA'] = scipy.logspace(0.9,1.1,100,base=10)
parameters['ModelTNS']['kA'] = scipy.logspace(-2.5,1.9,1000,base=10)
parameters['ModelTNS']['kB'] = scipy.logspace(-3,2,2500,base=10)
parameters['ModelTNS']['nthreads'] = 8
parameters['ModelTNS']['save'] = os.path.join(dirtest,'model_tns.npy')

parameters['ModelBAO']['kin'] = scipy.logspace(-5,4.1,4500,base=10) # do not take wider (-6,5) as Class takes years to compute
parameters['ModelBAO']['redshift'] = 0.874
#parameters['ModelBAO']['cosmology'] = 'BOSS'
parameters['ModelBAO']['cosmology'] = 'OuterRim'
#parameters['ModelBAO']['cosmology'] = 'Cutsky'
parameters['ModelBAO']['Class'] = {'input_verbose':1}
parameters['ModelBAO']['save'] = os.path.join(dirtest,'model_bao.npy')

parameters['EffectAP']['mu'] = scipy.linspace(0.,1.,200)
parameters['EffectAP']['save'] = os.path.join(dirtest,'effect_ap.npy')

parameters['SurveyGeometry']['ellsin'] = {0:[0,2,4]}
parameters['SurveyGeometry']['ellsout'] = [0,2,4]
parameters['SurveyGeometry']['save'] = os.path.join(dirtest,'geometry.npy')
parameters['SurveyGeometry']['fftlog'] = {'bias':-1.,'lowringing':False}
#parameters['SurveyGeometry']['s'] = scipy.logspace(-4,3,2000,base=10)
#parameters['SurveyGeometry']['s'] = scipy.logspace(-4,5,2000,base=10)
#parameters['SurveyGeometry']['s'] = scipy.logspace(-4,6,2000,base=10)
parameters['SurveyGeometry']['s'] = scipy.logspace(-4,5,1500,base=10)
#parameters['SurveyGeometry']['s'] = scipy.logspace(-3,6,2000,base=10)
#parameters['SurveyGeometry']['s'] = scipy.logspace(-4,5,500,base=10)
#parameters['SurveyGeometry']['s'] = scipy.logspace(scipy.log10(1.9*10**(-4)),scipy.log10(1.9*10**3),2000,base=10)
parameters['SurveyGeometry']['kin'] = 1./parameters['SurveyGeometry']['s'][::-1]
parameters['SurveyGeometry']['kout'] = scipy.linspace(0.,0.3,100)

fontsize = 20
labelsize = 16
figsize = (8,6)
dpi = 200
prop_cycle = pyplot.rcParams['axes.prop_cycle']
scalings = {}
scalings['rlinr2lin'] = {'func':lambda x,y: (x,x**2*y),'xlabel':'$r$ [$\\mathrm{Mpc} \ h^{-1}$]','ylabel':'$r^{{2}}\\xi^{{({\\ell})}}(r)$ [$(\\mathrm{{Mpc}} \ h^{{-1}})^{{2}}$]','xscale':'linear','yscale':'linear','xlim':[0.,500.],'ylim':None}
scalings['rlogr2lin'] = {'func':lambda x,y: (x,x**2*y),'xlabel':'$r$ [$\\mathrm{Mpc} \ h^{-1}$]','ylabel':'$r^{{2}}\\xi^{{({\\ell})}}(r)$ [$(\\mathrm{{Mpc}} \ h^{{-1}})^{{2}}$]','xscale':'log','yscale':'linear','xlim':[1e-3,1e3],'ylim':None}
scalings['rloglin'] = {'func':lambda x,y: (x,y),'xlabel':'$r$ [$\\mathrm{Mpc} \ h^{-1}$]','ylabel':'$\\xi^{{({\\ell})}}(r)$','xscale':'log','yscale':'linear','xlim':[1e-3,1e3],'ylim':None}
scalings['rloglog'] = {'func':lambda x,y: (x,scipy.absolute(y)),'xlabel':'$r$ [$\\mathrm{Mpc} \ h^{-1}$]','ylabel':'$\\xi^{{({\\ell})}}(r)$','xscale':'log','yscale':'log','xlim':[1e-3,1e3],'ylim':None}
scalings['klinklin'] = {'func':lambda x,y: (x,x*y),'xlabel':'$k$ [$h \ \\mathrm{Mpc}^{-1}$]','ylabel':'$kP(k)$ [$(\\mathrm{Mpc} \ h^{-1})^{2}$]','xscale':'linear','yscale':'linear','xlim':[0.01,0.3],'ylim':None}
scalings['klinlin'] = {'func':lambda x,y: (x,y),'xlabel':'$k$ [$h \ \\mathrm{Mpc}^{-1}$]','ylabel':'$P(k)$ [$(\\mathrm{Mpc} \ h^{-1})^{3}$]','xscale':'linear','yscale':'linear','xlim':[0.01,0.3],'ylim':None}
scalings['kloglog'] = {'func':lambda x,y: (x,y),'xlabel':'$k$ [$h \ \\mathrm{Mpc}^{-1}$]','ylabel':'$P(k)$ [$(\\mathrm{Mpc} \ h^{-1})^{3}$]','xscale':'log','yscale':'log','xlim':[1e-3,10.],'ylim':[1e-10,1e5]}
scalings['kloglin'] = {'func':lambda x,y: (x,y),'xlabel':'$k$ [$h \ \\mathrm{Mpc}^{-1}$]','ylabel':'$P(k)$ [$(\\mathrm{Mpc} \ h^{-1})^{3}$]','xscale':'log','yscale':'linear','xlim':[1e-3,10.],'ylim':None}
scalings['klogklin'] = {'func':lambda x,y: (x,x*y),'xlabel':'$k$ [$h \ \\mathrm{Mpc}^{-1}$]','ylabel':'$kP(k)$ [$(\\mathrm{Mpc} \ h^{-1})^{2}$]','xscale':'log','yscale':'linear','xlim':[1e-3,10.],'ylim':None}
scalings['klinlog'] = {'func':lambda x,y: (x,y),'xlabel':'$k$ [$h \ \\mathrm{Mpc}^{-1}$]','ylabel':'$P(k)$ [$(\\mathrm{Mpc} \ h^{-1})^{3}$]','xscale':'linear','yscale':'log','xlim':[0.,2.],'ylim':[1e-3,1e5]}

BOSS = Cosmology.BOSS()

def setup_model_tns():

	try:
		model = ModelTNS.load(parameters['ModelTNS']['save'],**parameters['ModelTNS'])
	except IOError:
		model = ModelTNS(**parameters['ModelTNS'])
	
	model.set_cosmology()
	model.set_spectrum_lin()
	model.set_spectrum_nonlin()
	#model.set_spectrum_2loop()
	#model.set_spectrum_1loop()
	#model.set_bias_1loop()
	#model.set_A_1loop()
	#model.set_A_2loop()
	#model.set_B_1loop()
	#model.set_B_2loop()
	#model.setup()
	"""
	for key in ['spectrum_2loop_dd','spectrum_2loop_dt','spectrum_2loop_tt']:
		print key, len(getattr(model,key))
		setattr(model,key,getattr(model,key)[::5])
	"""
	model.save()

def setup_model_bao():
	try:
		model = ModelBAO.load(parameters['ModelBAO']['save'],**parameters['ModelBAO'])
	except IOError:
		model = ModelBAO(**parameters['ModelBAO'])
	#model.set_cosmology()
	#model.set_spectrum_lin()
	model.set_spectrum_smooth()
	#model.setup()
	model.save()

def setup_effect_ap():
	
	geometry = SurveyGeometry.load(parameters['SurveyGeometry']['save']) 
	effect_ap = EffectAP(**parameters['EffectAP'])
	effect_ap.setup(ells=geometry.params['ellsin'][0],k=geometry.k)
	effect_ap.save()

def setup_survey_geometry():
	
	geometry = SurveyGeometry(**parameters['SurveyGeometry'])
	geometry.setup()
	geometry.save()

def plot_baseline(scaling,title=''):
	fig = pyplot.figure(figsize=figsize)
	if title: fig.suptitle(title,fontsize=fontsize)
	ax = pyplot.gca()
	ax.set_xlim(scaling['xlim'])
	if scaling['ylim'] is not None: ax.set_ylim(scaling['ylim'])
	ax.set_xscale(scaling['xscale'])
	ax.set_yscale(scaling['yscale'])
	ax.grid(True)
	ax.tick_params(labelsize=labelsize)
	ax.set_xlabel(scaling['xlabel'],fontsize=fontsize)
	ax.set_ylabel(scaling['ylabel'],fontsize=fontsize)
	return ax

def plot_difference_cb_m(title='',path=os.path.join(dirplot,'difference_cb_m.png')):
	
	fontsize = 16
	
	z = 0.5
	model = Cosmology.BOSS()
	#model = Cosmology.MultiDark()
	neutrinos = {0.0:{},0.06:{'N_ur':2.0328,'N_ncdm':1,'m_ncdm':0.06},0.2:{'N_ur':2.0328,'N_ncdm':1,'m_ncdm':0.2}}
	k = scipy.logspace(-3,1,100,base=10)
	Pk = {}
	for key in neutrinos:
		model.set_params(**neutrinos[key])
		print model.params
		Pk[key] = {}
		Pk[key]['cb'] = model.compute_pk_cb_lin(k,z=z)
		Pk[key]['m'] = model.compute_pk_lin(k,z=z)
	model.clear()
	fig, ax = pyplot.subplots(2,sharex=True,sharey=False,gridspec_kw={'height_ratios':[3,1]},figsize=(8,6),squeeze=True)
	fig.subplots_adjust(hspace=0)
	if title: fig.suptitle(title,fontsize=fontsize)
	colors = prop_cycle.by_key()['color']
	
	for key,color in zip(neutrinos,colors):
		ax[0].plot(k,Pk[key]['cb'],label='{} {}'.format('cb',key),color=color,linestyle='--')
		ax[0].plot(k,Pk[key]['m'],label='{} {}'.format('m',key),linestyle='-')
		mask = k<0.15; print key, scipy.absolute(Pk[key]['cb'][mask]/Pk[key]['m'][mask]-1.).max()
		ax[1].plot(k,Pk[key]['cb']/Pk[key]['m']-1.,linestyle='-')
		
	ax[0].set_ylabel('$P(k)$ [$(\\mathrm{Mpc} \ h^{-1})^{3}$]',fontsize=fontsize)
	ax[0].grid(True)
	ax[0].set_yscale('log')
	ax[1].set_ylabel('$\\Delta P_{\\rm cb-m} / P$',fontsize=fontsize)
	ax[1].grid(True)
	ax[1].set_xscale('log')
		
	ax[0].legend(**{'loc':1,'ncol':2,'fontsize':labelsize,'framealpha':0.5,'frameon':False})
	ax[-1].set_xlabel('$k$ [$h \ \\mathrm{Mpc}^{-1}$]',fontsize=fontsize)
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_difference_class_camb(title='',path=os.path.join(dirplot,'difference_class_camb.png')):
	
	fontsize = 16
	z = 0.
	Pk = {}
	k,Pk['Camb'] = scipy.loadtxt(os.path.join(dirtest,'Challenge_matterpower.dat'),unpack=True)
	model = Cosmology.BOSS()
	#model = Cosmology.MultiDark()
	Pk['Class'] = model.compute_pk_lin(k,z=z)
	
	fig, ax = pyplot.subplots(2,sharex=True,sharey=False,gridspec_kw={'height_ratios':[3,1]},figsize=(8,6),squeeze=True)
	fig.subplots_adjust(hspace=0)
	if title: fig.suptitle(title,fontsize=fontsize)
	colors = prop_cycle.by_key()['color']
	
	ax[0].set_ylabel('$P(k)$ [$(\\mathrm{Mpc} \ h^{-1})^{3}$]',fontsize=fontsize)
	ax[0].grid(True)
	ax[0].set_yscale('log')
	for key in Pk: ax[0].plot(k,Pk[key],label=key,linestyle='-')
	ax[1].plot(k,Pk['Class']/Pk['Camb']-1.,color='k',linestyle='-')
	ax[1].set_ylabel('$\\Delta P_{\\rm Class-Camb} / P_{\\rm Camb}$',fontsize=fontsize)
	ax[1].grid(True)
	ax[1].set_xscale('log')
		
	ax[0].legend(**{'loc':1,'ncol':2,'fontsize':labelsize,'framealpha':0.5,'frameon':False})
	ax[-1].set_xlabel('$k$ [$h \ \\mathrm{Mpc}^{-1}$]',fontsize=fontsize)
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_model_tns_spectrum_lin(scale='kloglog',title='Class outputs',path=os.path.join(dirplot,'spectrum_lin.png')):

	model = ModelTNS.load(parameters['ModelTNS']['save'])
	scaling = scalings[scale]
	ax = plot_baseline(scaling,title)
	ax.set_xlim(model.spectrum_lin.k[0],model.spectrum_lin.k[-1])

	for key,label in zip(['spectrum_lin'],['pk_lin']):
		k = getattr(model,key)['k']
		pk = getattr(model,key).pk()
		ax.plot(*scaling['func'](k,pk),label=utils.text_to_latex(label),linestyle='-')

	ax.legend(**{'loc':1,'ncol':2,'fontsize':labelsize,'framealpha':0.5,'frameon':False})
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_model_tns_spectrum_nloop(nloop=2,scale='klinklin',title='RegPT outputs',path=os.path.join(dirplot,'spectrum_{:d}loop.png')):
	
	model = ModelTNS.load(parameters['ModelTNS']['save'])
	scaling = scalings[scale]
	ax = plot_baseline(scaling,title)

	for key,label in zip(['dd','dt','tt'],['pk_dd','pk_dt','pk_tt']):
		k = getattr(model,'spectrum_{:d}loop_{}'.format(nloop,key))['k']
		pk = getattr(model,'spectrum_{:d}loop_{}'.format(nloop,key)).pk()
		ax.plot(*scaling['func'](k,pk),label=utils.text_to_latex(label),linestyle='-')
	ax.plot(*scaling['func'](model.spectrum_halofit['k'],model.spectrum_halofit.pk()),label='halofit',linestyle='-')

	ax.legend(**{'loc':1,'ncol':2,'fontsize':labelsize,'framealpha':0.5,'frameon':False})
	utils.savefig(path.format(nloop),dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_model_tns_terms_spectrum_2loop(scale='kloglog',title='RegPT outputs',path=os.path.join(dirplot,'terms_spectrum_2loop.png')):
	
	model = ModelTNS.load(parameters['ModelTNS']['save'])
	scaling = scalings[scale]
	ax = plot_baseline(scaling,title)
	ax.set_xlim(model.spectrum_2loop_dd.k[0],model.spectrum_2loop_dd.k[-1])
	ax.set_ylim(1e-2,1e9)
	
	#ax.plot(*scaling['func'](model.kin,model.Pin),label='$P_{in}$',linestyle='-')
	for key in model.spectrum_2loop_dd:
		if key not in ['k','pk_lin','sigmav2']: ax.plot(*scaling['func'](model.spectrum_2loop_dd.k,scipy.absolute(model.spectrum_2loop_dd[key])),label=key,linestyle='-')

	ax.legend(**{'loc':1,'ncol':2,'fontsize':labelsize,'framealpha':0.5,'frameon':False})
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_model_tns_damping(scale='kloglin',title='RegPT outputs',path=os.path.join(dirplot,'regpt_damping.png')):
	
	model = ModelTNS.load(parameters['ModelTNS']['save'])
	scaling = scalings[scale]
	ax = plot_baseline(scaling,title)
	ax.set_xlim(1e-1,1e1)

	factor = 0.5 * model.spectrum_2loop_dd.k**2 * model.spectrum_2loop_dd['sigmad2']
	ax.plot(*scaling['func'](model.spectrum_2loop_dd.k,scipy.exp(-2*factor)) * (1. + factor)**2,label='factor',linestyle='-')
	
	ax.legend(**{'loc':1,'ncol':2,'fontsize':labelsize,'framealpha':0.5,'frameon':False})
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)
	
def plot_model_tns_terms_bias_1loop(scale='kloglog',title='Bias terms',path=os.path.join(dirplot,'terms_bias_1loop.png')):
	
	model = ModelTNS.load(parameters['ModelTNS']['save'])
	scaling = scalings[scale]
	ax = plot_baseline(scaling,title)
	ax.set_xlim(model.bias_1loop.k[0],model.bias_1loop.k[-1])
	#ax.set_xlim(8.,13.)
	
	for key in ['pk_b2d','pk_bs2d','pk_b2t','pk_bs2t','pk_b22','pk_b2s2','pk_bs22','pk_sigma3sq']:
		spectrum = scipy.absolute(getattr(model.bias_1loop,key)())
		ax.plot(*scaling['func'](model.bias_1loop.k,spectrum),label=utils.text_to_latex(key),linestyle='-')
		#print key,spectrum
	ax.plot(*scaling['func'](model.bias_1loop.k,model_tns.damping_kernel(model.bias_1loop.k)),label='damping',linestyle='-')
	
	ax.legend(**{'loc':3,'ncol':2,'fontsize':labelsize,'framealpha':0.5,'frameon':False})
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)
	
def plot_model_tns_terms_A_B_2loop(scale='kloglin',title='A and B terms',path=os.path.join(dirplot,'terms_A_B_2loop.png')):
	
	model = ModelTNS.load(parameters['ModelTNS']['save'])
	effect_ap = EffectAP.load(parameters['EffectAP']['save'])
	scaling = scalings[scale]
	ax = plot_baseline(scaling,title)
	ax.set_ylabel('$P(k)/P_{\\rm nowiggle}(k)$')
	effect_ap.set_input_model(model.spectrum_A_B)
	kwargs = dict(f=BOSS.growth_rate_pade(z=1.),b1=1.,sigmav=model.spectrum_lin.sigmav())
	#kwargs = dict(f=0.8737070302,b1=1.,sigmav=3.81078379941)
	A,B = effect_ap.spectrum_multipoles_no_AP(**kwargs)
	effect_ap.set_input_model(model.spectrum_no_wiggle)
	kwargs.update({'sigmav':0.})
	PNoWiggle = effect_ap.spectrum_multipoles_no_AP(**kwargs)
	xlim = [model.B_2loop.k[0],model.B_2loop.k[-1]]
	#xlim = scaling['xlim']
	ax.set_xlim(xlim)
	mask = (effect_ap.k>=xlim[0]) & (effect_ap.k<xlim[1])

	for ill,ell in enumerate(effect_ap.params['ells'][:2]):
		#ax.plot(*scaling['func'](effect_ap.k[mask],A[ill][mask]/PNoWiggle[ill][mask]),label='A {}'.format(utils.text_to_latex(ell)),linestyle='-')
		#ax.plot(*scaling['func'](effect_ap.k[mask],B[ill][mask]/PNoWiggle[ill][mask]),label='B {}'.format(utils.text_to_latex(ell)),linestyle='-')
		ax.plot(*scaling['func'](effect_ap.k[mask],scipy.absolute(A[ill][mask]/PNoWiggle[ill][mask])),label='A {}'.format(utils.text_to_latex(ell)),linestyle='-')
		ax.plot(*scaling['func'](effect_ap.k[mask],scipy.absolute(B[ill][mask]/PNoWiggle[ill][mask])),label='B {}'.format(utils.text_to_latex(ell)),linestyle='-')
	
	ax.legend(**{'loc':2,'ncol':2,'fontsize':labelsize,'framealpha':0.5,'frameon':False})
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)


def plot_model_bao_wiggles(scale='klinlin',title='BAO template',path=os.path.join(dirplot,'spectrum_bao_wiggles.png')):

	model = ModelBAO.load(parameters['ModelBAO']['save'])
	scaling = scalings[scale]
	ax = plot_baseline(scaling,title)
	k = model.spectrum_lin.k
	k = k[k<0.8]
	ax.set_xlim(k[0],k[-1])

	kwargs = dict(sigmanl=5.)
	pk = model.wiggles_eh(k)
	ax.plot(*scaling['func'](k,pk),label='original wiggles',linestyle='-')
	pk = model.wiggles_smooth(k)
	ax.plot(*scaling['func'](k,pk),label='corrected wiggles',linestyle='-')
	pk = model.wiggles_damped_iso(k,**kwargs)
	ax.plot(*scaling['func'](k,pk),label='damped wiggles',linestyle='-')
	ax.set_ylabel('')

	ax.legend(**{'loc':1,'ncol':1,'fontsize':labelsize,'framealpha':0.5,'frameon':False})
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_model_bao_iso(scale='klinklin',title='BAO template',path=os.path.join(dirplot,'spectrum_bao_iso.png')):

	model = ModelBAO.load(parameters['ModelBAO']['save'])
	scaling = scalings[scale]
	ax = plot_baseline(scaling,title)
	k = model.spectrum_lin.k

	kwargs = dict(qiso=1.,am1=0.,sigmanl=0.)
	ax.plot(*scaling['func'](k,model.spectrum_lin.pk_interp(k)),label=utils.text_to_latex('pk_lin'),linestyle='-')
	pk = model.spectrum_smooth_iso(k,**kwargs)
	ax.plot(*scaling['func'](k,pk),label='no wiggle',linestyle='-')
	pk = model.spectrum_galaxy_iso(k,**kwargs)
	ax.plot(*scaling['func'](k,pk),label='wiggle',linestyle='-')
	ax.set_ylabel('')

	ax.legend(**{'loc':1,'ncol':1,'fontsize':labelsize,'framealpha':0.5,'frameon':False})
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_model_bao_aniso(scale='klinklin',title='BAO template',path=os.path.join(dirplot,'spectrum_bao_aniso.png')):

	model = ModelBAO.load(parameters['ModelBAO']['save'])
	effect_ap = EffectAP.load(parameters['EffectAP']['save'])
	scaling = scalings[scale]
	ax = plot_baseline(scaling,title)
	effect_ap.set_input_model(model.spectrum_galaxy_aniso)
	
	kwargs = dict(b1=1.3,beta=0.5,sigmas=model.spectrum_lin.sigmav())
	qgrid = [[1.,1.],[0.95,1.05],[1.05,0.95]]
	ax.set_xlim(0.,0.2)
	colors = ['b','r','g']
	
	for ill,(ell,color) in enumerate(zip(effect_ap.ells,colors)):
		tmp = [effect_ap.spectrum_multipoles(qpar=qpar,qper=qper,**kwargs)[0][ill] for qpar,qper in qgrid]
		ax.plot(*scaling['func'](effect_ap.k,tmp[0]),label=utils.text_to_latex(ell))
		ax.fill_between(effect_ap.k,scaling['func'](effect_ap.k,tmp[1])[1],scaling['func'](effect_ap.k,tmp[2])[1],facecolor=color,alpha=0.3,linewidth=0)

	ax.legend(**{'loc':1,'ncol':1,'fontsize':labelsize,'framealpha':0.5,'frameon':False})
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_effect_ap_vary(scale='klinklin',title='Alcock-Paczynski',path=os.path.join(dirplot,'effect_ap_vary.png')):

	model = ModelTNS.load(parameters['ModelTNS']['save'])
	effect_ap = EffectAP.load(parameters['EffectAP']['save'])
	scaling = scalings[scale]
	ax = plot_baseline(scaling,title)
	effect_ap.set_input_model(model.spectrum_galaxy)
	kwargs = dict(b1=1.3,b2=0.4,sigmav=model.spectrum_lin.sigmav())
	qgrid = [[1.,1.],[0.95,1.05],[1.05,0.95]]
	fgrid = [0.84,0.95*0.84,1.05*0.84]
	ax.set_xlim(0.,0.2)
	colors = ['b','r','g']
	
	for ill,(ell,color) in enumerate(zip(effect_ap.ells,colors)):
		tmp = [effect_ap.spectrum_multipoles(f=fgrid[0],qpar=qpar,qper=qper,**kwargs)[0][ill] for qpar,qper in qgrid]
		#ax.plot(*scaling['func'](effect_ap.k,tmp[0]),label=utils.text_to_latex(ell))
		ax.fill_between(effect_ap.k,scaling['func'](effect_ap.k,tmp[1])[1],scaling['func'](effect_ap.k,tmp[2])[1],facecolor=color,alpha=0.3,linewidth=0,label=utils.text_to_latex(ell))
		tmp = [effect_ap.spectrum_multipoles(f=f,qpar=qgrid[0][0],qper=qgrid[0][1],**kwargs)[0][ill] for f in fgrid[1:]]
		ax.fill_between(effect_ap.k,scaling['func'](effect_ap.k,tmp[0])[1],scaling['func'](effect_ap.k,tmp[1])[1],facecolor=color,alpha=0.6,linewidth=0)

	ax.legend(**{'loc':1,'ncol':1,'fontsize':labelsize,'framealpha':0.5,'frameon':False})
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_effect_ap_multipoles(scale='kloglin',title='Multipoles',path=os.path.join(dirplot,'effect_ap_multipoles.png')):

	model = ModelTNS.load(parameters['ModelTNS']['save'])
	effect_ap = EffectAP.load(parameters['EffectAP']['save'])
	scaling = scalings[scale]
	ax = plot_baseline(scaling,title)
	ax.set_xlim(effect_ap.k[0],effect_ap.k[-1])
	effect_ap.set_input_model(model.spectrum_galaxy,model.spectrum_galaxy_tree_real)
	colors = ['b','r','g']
	
	kwargs = dict(f=BOSS.growth_rate_pade(z=1.),qpar=0.9,qper=1.1,b1=1.3,b2=0.4,sigmav=4.)
	Pap = effect_ap.spectrum_multipoles(**kwargs)[0]
	for ill,(ell,color) in enumerate(zip(effect_ap.ells,colors)):
		ax.plot(*scaling['func'](effect_ap.k,Pap[ill]),label=utils.text_to_latex(ell),color=color,linestyle='-')
	
	kwargs.update(Ns=3000)
	Pap = effect_ap.spectrum_multipoles(**kwargs)[0]
	for ill,(ell,color) in enumerate(zip(effect_ap.ells,colors)):
		ax.plot(*scaling['func'](effect_ap.k,Pap[ill]),label=utils.text_to_latex(ell),color=color,linestyle='--')
	
	ax.legend(**{'loc':1,'ncol':1,'fontsize':labelsize,'framealpha':0.5,'frameon':False})
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)


def plot_effect_ap_wedges(scale='kloglog',title='Wedges',path=os.path.join(dirplot,'effect_ap_wedges.png')):

	model = ModelTNS.load(parameters['ModelTNS']['save'])
	effect_ap = EffectAP.load(parameters['EffectAP']['save'])
	scaling = scalings[scale]
	ax = plot_baseline(scaling,title)
	effect_ap.set_input_model(model.spectrum_galaxy)
	kwargs = dict(f=BOSS.growth_rate_pade(z=1.),b1=1.3,b2=0.4,sigmav=4.)
	#kwargs = dict(qpar=1.386926,qper=1.330642,f=2.989074,b1=1.144080,b2=-5.347969,sigmav=9.296017)
	Pap = effect_ap.spectrum_multipoles(**kwargs)[0]
	ax.set_xlim(effect_ap.k[0],effect_ap.k[-1])
	
	ax.plot(*scaling['func'](effect_ap.k,model.spectrum_galaxy(effect_ap.k,0.,**kwargs)),label='$\\mu = 0.$',linestyle='-')
	ax.plot(*scaling['func'](effect_ap.k,model.spectrum_galaxy(effect_ap.k,1.,**kwargs)),label='$\\mu = 1.$',linestyle='-')
	
	ax.legend(**{'loc':1,'ncol':1,'fontsize':labelsize,'framealpha':0.5,'frameon':False})
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)
	
def plot_survey_geometry_accuracy(scale='kloglin',title='Bessel transform accuracy',path=os.path.join(dirplot,'fourier_to_fourier_accuracy.png')):

	model = ModelTNS.load(parameters['ModelTNS']['save'])
	effect_ap = EffectAP.load(parameters['EffectAP']['save'])
	geometry = SurveyGeometry.load(parameters['SurveyGeometry']['save'])
	scaling = scalings[scale]
	ax = plot_baseline(scaling,title)
	ax.set_ylabel('$P_{\\rm ftof}(k)/P_{\\rm ref}(k)$')
	effect_ap.set_input_model(model.spectrum_galaxy)
	kwargs = dict(f=BOSS.growth_rate_pade(z=1.),b1=1.3,b2=0.4,sigmav=model.spectrum_lin.sigmav(),uvcutoff=0.6,Ns=0.)
	Pref = effect_ap.spectrum_multipoles(**kwargs)[0]
	geometry.set_input_model(effect_ap.spectrum_multipoles)
	geometry.set_kout(geometry.k)
	Pftof = geometry.spectrum_multipoles(corrconstraint=False,**kwargs)
	mask = (geometry.kout[0]>0) & (geometry.kout[0]<0.5)
	ax.set_xlim(geometry.kout[0][mask][0],geometry.kout[0][mask][-1])
	
	for ill,ell in enumerate(geometry.ellsout):
		ref = scipy.interp(geometry.kout[ill],effect_ap.k,Pref[ill])
		ax.plot(*scaling['func'](geometry.kout[ill][mask],Pftof[ill][mask]/ref[mask]-1.),label=utils.text_to_latex(ell),linestyle='-')
	
	delta = 1e-5
	ax.set_ylim(-delta,delta)
	ax.legend(**{'loc':1,'ncol':1,'fontsize':labelsize,'framealpha':0.5,'frameon':False})
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)
	
def plot_survey_geometry_spectrum(scale='kloglog',title='Power spectrum',path=os.path.join(dirplot,'power_spectrum.png')):

	model = ModelTNS.load(parameters['ModelTNS']['save'])
	effect_ap = EffectAP.load(parameters['EffectAP']['save'])
	geometry = SurveyGeometry.load(parameters['SurveyGeometry']['save'])
	scaling = scalings[scale]
	ax = plot_baseline(scaling,title)
	effect_ap.set_input_model(model.spectrum_galaxy)
	kwargs = dict(f=BOSS.growth_rate_pade(z=1.),b1=1.3,b2=0.4,sigmav=model.spectrum_lin.sigmav())
	geometry.set_input_model(effect_ap.spectrum_multipoles)
	Pftof = geometry.spectrum_multipoles(**kwargs)
	ax.set_xlim(geometry.kout[0][0],geometry.kout[0][-1])
	
	for ill,ell in enumerate(geometry.ellsout):
		ax.plot(*scaling['func'](geometry.kout[ill],Pftof[ill]),label=utils.text_to_latex(ell),linestyle='-')

	ax.legend(**{'loc':1,'ncol':1,'fontsize':labelsize,'framealpha':0.5,'frameon':False})
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def plot_survey_geometry_correlation(scale='rlogr2lin',title='Correlation function',path=os.path.join(dirplot,'correlation_function.png')):

	model = ModelTNS.load(parameters['ModelTNS']['save'])
	effect_ap = EffectAP.load(parameters['EffectAP']['save'])
	geometry = SurveyGeometry.load(parameters['SurveyGeometry']['save'])
	scaling = scalings[scale]
	#scaling['xlim'] = [geometry.s[0],geometry.s[-1]]
	scaling['xlim'] = [1e-3,1e2]
	#scaling['xlim'] = [1e2,1e4]
	ax = plot_baseline(scaling,title)
	effect_ap.set_input_model(model.spectrum_galaxy)
	geometry.set_input_model(effect_ap.spectrum_multipoles)
	mask = (geometry.s >= scaling['xlim'][0]) & (geometry.s < scaling['xlim'][-1])
	colors = ['b','r','g']
	
	#kwargs = dict(f=BOSS.growth_rate_pade(z=1.),b1=1.3,b2=0.4,sigmav=model.spectrum_lin.sigmav())
	kwargs = dict(qpar=1.,qper=1.,f=0.7,b1=1.2,b2=-0.4,sigmav=3.2,Ns=0.)
	corr = geometry.correlation_multipoles(**kwargs)
	for ill,(ell,color) in enumerate(zip(geometry.ellsout,colors)):
		#print ell, corr[ill][mask].min(), corr[ill][mask].max() 
		ax.plot(*scaling['func'](geometry.s[mask],corr[ill][mask]),label=utils.text_to_latex(ell),color=color,linestyle='-')
	
	kwargs.update(Ns=3000.)
	corr = geometry.correlation_multipoles(**kwargs)
	for ill,(ell,color) in enumerate(zip(geometry.ellsout,colors)):
		#print ell, corr[ill][mask].min(), corr[ill][mask].max() 
		ax.plot(*scaling['func'](geometry.s[mask],corr[ill][mask]),label=utils.text_to_latex(ell),color=color,linestyle='--')
	
	ax.legend(**{'loc':1,'ncol':1,'fontsize':labelsize,'framealpha':0.5,'frameon':False})
	utils.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def test_to_redshift():
	model = ModelTNS.load(parameters['ModelTNS']['save'])
	tmp = model.spectrum_2loop_dd.pk_lin[0]
	new = model.deepcopy().to_redshift(0.)
	assert model.spectrum_2loop_dd.pk_lin[0]==tmp
	#assert new.growth_factor==1.
	#assert new.spectrum_2loop_dd.pk_lin[0]==tmp*(new.growth_factor/model.growth_factor)**2

def setup():
	#setup_model_tns()
	setup_survey_geometry()
	setup_effect_ap()

def plot_model_tns():
	plot_model_tns_spectrum_lin()
	plot_model_tns_spectrum_2loop()
	plot_model_tns_terms_spectrum_2loop()
	plot_model_tns_terms_bias_1loop()
	plot_model_tns_terms_A_B_2loop()

def plot_effect_ap():
	#plot_effect_ap_vary()
	plot_effect_ap_multipoles()
	plot_effect_ap_wedges()
	
def plot_survey_geometry():
	plot_survey_geometry_accuracy()
	plot_survey_geometry_spectrum()
	plot_survey_geometry_correlation()
	
def check_ref():

	pathref = '/home/gpfs/manip/mnt0607/bao/adematti/elg_model/tests/model_tns.npy'
	ref = scipy.load(pathref)[()]
	test = ModelTNS.load(parameters['ModelTNS']['save'])
	
	for key in ModelTNS.PBIAS:
		if key=='Psigma3sq': print key,scipy.interp(ref['k'],test.terms_bias.k,getattr(test.terms_bias,key)())/ref['sigma3SqPlin']
		else: print key,scipy.interp(ref['k'],test.terms_bias.k,getattr(test.terms_bias,key)())/ref[key]

	for key in ModelTNS.PAB:
		shape = test.terms_A_B.SHAPE[key[1:]]
		for i in range(shape[0]):
			for j in range(shape[-1]):
				print key,i,j,scipy.interp(ref['k'],test.terms_A_B.k,getattr(test.terms_A_B,key)()[i,j])/ref[key[1:]][i,j]
	
	print ref['k'].min(), ref['k'].max()

setup_model_tns()
#setup_model_bao()
#plot_model_bao_wiggles()
#plot_model_bao_iso()
#plot_model_bao_aniso()
#plot_model_tns_spectrum_lin()
#plot_model_tns()
#plot_effect_ap()
#plot_survey_geometry()
plot_model_tns_spectrum_nloop()
#plot_model_tns_terms_spectrum_2loop()
#plot_model_tns_spectrum_2loop()
#plot_model_tns_terms_A_B_2loop()
#plot_model_tns_terms_bias_1loop()
#plot_model_tns_damping()
#plot_effect_ap_multipoles()
#plot_effect_ap_wedges()
#plot_survey_geometry_accuracy()
#plot_survey_geometry_correlation()
#plot_effect_ap_vary()
#check_ref()
#test_to_redshift()
#plot_difference_cb_m()
#plot_difference_class_camb()
