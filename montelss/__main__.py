import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import logging
import imp
import inspect
import argparse
import scipy
from montelss import Minimizer,EnsembleSampler,MHSampler,utils,setup_logging

logger = logging.getLogger('MonteLSS')

def load_cmdline():
	parser = argparse.ArgumentParser(description='Run Markov-Chain on LSS')
	parser.add_argument('-config',type=str,help='file path to config file',required=True)
	parser.add_argument('-data',nargs='*',type=str,default=None,help='file path to data',required=True)
	parser.add_argument('-cov',nargs='*',type=str,default=None,help='file path to covariance',required=False)
	parser.add_argument('-todo',nargs='*',type=str,help='todo',required=True)
	parser.add_argument('-load',type=str,help='file path to input',default=None,required=False)
	parser.add_argument('-save',type=str,help='file path to output',required=True)
	parser.add_argument('-seed',type=int,help='random seed',default=-1,required=False)
	parser.add_argument('-grid',type=str,help='grid',default=None,required=False)
	cmdline = parser.parse_args()

	parameters = imp.load_source('config',cmdline.config).parameters
	if not cmdline.data: cmdline.data = [None]*len(cmdline.cov)
	cmdline.data = [None if data in ['None','none'] else data for data in cmdline.data]
	if not cmdline.cov: cmdline.cov = [None]*len(cmdline.data)
	cmdline.cov = [None if cov in ['None','none'] else cov for cov in cmdline.cov]
	
	return cmdline,parameters

def load_likelihoods(cmdline,parameters):
	logger.info('Loading likelihoods...')
	likelihoods = []
	idata = 0
	for likeargs,path_covariance in zip(parameters['likelihoods'],cmdline.cov):
		likelihood = imp.load_source('likelihood',likeargs['path']).get_likelihood(*likeargs.get('args',[]),**likeargs.get('kwargs',{}))
		path_data = cmdline.data[idata] if likelihood.ndata == 1 else cmdline.data[idata:idata+likelihood.ndata]
		idata += likelihood.ndata
		likelihood.init(path_data=path_data,path_covariance=path_covariance)
		likelihood.set_default()
		likelihoods.append(likelihood)
	return likelihoods

if __name__ == '__main__':
	
	setup_logging()
	cmdline,parameters = load_cmdline()
	likelihoods = load_likelihoods(cmdline,parameters)
	
	if any([key in cmdline.todo for key in ['migrad','minos','mnprofile','mncontour','grid','info']]):
		if cmdline.load is not None:
			try:
				minimizer = Minimizer.load(cmdline.load,**parameters['minimizer'])
			except IOError:
				logger.info('Minimizer {} does not exist. I am creating one.'.format(cmdline.load))
				minimizer = Minimizer(**parameters['minimizer'])
		else:
			minimizer = Minimizer(**parameters['minimizer'])
		minimizer.setup(likelihoods)

	if 'migrad' in cmdline.todo:
		minimizer.run_migrad()
		minimizer.save(save=cmdline.save)

	if 'minos' in cmdline.todo:
		#minimizer.run_migrad()
		minimizer.run_minos()
		minimizer.save(save=cmdline.save)

	if 'mnprofile' in cmdline.todo:
		minimizer.run_mnprofile()
		minimizer.save(save=cmdline.save)

	if 'mncontour' in cmdline.todo:
		minimizer.run_mncontour()
		minimizer.save(save=cmdline.save)
	
	if 'grid' in cmdline.todo:
		grid = minimizer.params.get('grid',{})
		if cmdline.grid is not None:
			logger.info('Loading grid {}.'.format(cmdline.grid))
			grid['points'] = scipy.loadtxt(cmdline.grid)
		minimizer.run_grid(grid=grid)
		minimizer.save(save=cmdline.save)

	if 'info' in cmdline.todo:
		minimizer.set_bestfit()
		minimizer.set_dbestfit()
		for fmt in ['vertical']:
			minimizer.export_to_latex(cmdline.save.replace('.npy','_{}.tex'.format(fmt)),fmt=fmt)
		for likelihood in likelihoods:
			values = minimizer.bestfit['values']
			likelihood.plot(values=values,path=cmdline.save.replace('.npy','_{}.png'))
		#minimizer.save(save=cmdline.save)
	
	init_sampler = False
	if 'ensemble' in cmdline.todo:
		Sampler = EnsembleSampler
		sampler_params = parameters['sampler']['ensemble']
		init_sampler = True
	if 'metropolis' in cmdline.todo:
		Sampler = MHSampler
		sampler_params = parameters['sampler']['metropolis']
		init_sampler = True
	
	if init_sampler:
		if cmdline.load is not None:
			state = utils.load(cmdline.load,allow_pickle=True)
			if not (cmdline.seed < 0): sampler_params['seed'] = cmdline.seed
			if 'sampler' in state:
				logger.info('Loading {}: {}.'.format(Sampler.__name__,cmdline.load))
				sampler = Sampler.loadstate(state,**sampler_params)
				sampler.setup(likelihoods)
				sampler.init_from_previous()
			else:
				logger.info('Loading Minimizer: {}.'.format(cmdline.load))
				bestfit = Minimizer.loadstate(state,**parameters['minimizer']).bestfit
				sampler = Sampler(save=cmdline.save,**sampler_params)
				sampler.setup(likelihoods)
				sampler.init_from_bestfit(bestfit)
		else:
			sampler = Sampler(**sampler_params)
			sampler.set_likelihoods(likelihoods)
			sampler.setup(likelihoods)
			sampler.init_from_params()
		sampler.run()
		sampler.save(save=cmdline.save)
