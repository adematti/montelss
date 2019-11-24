import scipy
from montelss import *

setup_logging()

class LikelihoodGaussian(Likelihood):

	def __init__(self,**params):
		super(LikelihoodGaussian,self).__init__(**params)
		self.fitargs = dict(a=1.,error_a=1.,b=1.,error_b=1.,c=1.,error_c=1.)
		
	def lnlkl(self,a=1.,b=1.,c=1.):
		x = scipy.array([a,b,c])
		return -0.5*scipy.sum(x**2)

def test_minimizer():
	lkl = LikelihoodGaussian()
	minimizer = Minimizer()
	minimizer.setup(lkl,minuit={'errordef':1.})
	minimizer.run_migrad()
	minimizer.run_minos()
	minimizer.save(save='test_bestfit.npy')

def test_ensemble():
	lkl = LikelihoodGaussian()
	minimizer = Minimizer.load('test_bestfit.npy')
	sampler = EnsembleSampler(nsteps=1000,nwalkers=10,nthreads=2,seed=42)
	sampler.setup(lkl)
	sampler.init_from_bestfit(minimizer.bestfit)
	sampler.run()
	sampler.save(save='test_ensemble.npy')

def test_metropolis():
	lkl = LikelihoodGaussian()
	for ichain in range(4):
		sampler = MHSampler.load('test_ensemble.npy',nsteps=10000,seed=ichain)
		sampler.setup(lkl)
		sampler.init_from_previous()
		sampler.run(covariance='previous',burnin=100)
		sampler.save(save='test_metropolis_{:d}.npy'.format(ichain))


def plot_chain(path_mcmc=['test_metropolis_{:d}.npy'.format(ichain) for ichain in range(4)]):

	if isinstance(path_mcmc,list): chains = [AnalyzeMCMC.load_montelss(path) for path in path_mcmc]
	else: chains = [AnalyzeMCMC.load_montelss(path_mcmc)]
	chain = sum(chains)
	params = ['a','b','c']
	chain.stats_to_latex('log.tex',params=params,mean=scipy.mean,error=scipy.std,precision=2,fmt='vertical')

	gaussian = None
	gaussian = {'mean':chain.mean(params),'covariance':chain.covariance(params,ddof=1)}	

	analyze_mcmc.plot_corner_chains([chain],params=params,truths=[0.,0.,0.],gaussian=gaussian,contour_kwargs={'nsigmas':3},gaussian_contour_kwargs={'nsigmas':3},path='corner.png')
	analyze_mcmc.plot_chain(chain,params=params,path='chain.png')
	analyze_mcmc.plot_gelman_rubin(chains,params=params,path='gelman-rubin.png')
	analyze_mcmc.plot_autocorrelation_time(chains,params=chain.parameters,path='autocorrelation-time.png')

#test_minimizer()
#test_ensemble()
#test_metropolis()
plot_chain()
