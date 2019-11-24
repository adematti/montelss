import logging
import os
import re
import functools
import time
import copy
import subprocess
import numpy
import scipy
import matplotlib
from utilities.precision import to_precision,to_error

path = {}
path['monteless'] = os.path.abspath(os.path.dirname(__file__))
path['likelihoods'] = os.path.join(path['monteless'],'likelihoods')

MINUIT_PREFIX = ['error_','limit_','fix_','latex_','solve_']
OTHER_PREFIX = ['latex_','solve_']

def classparams(func):
	@functools.wraps(func)
	def wrapper(self,*args,**kwargs):
		varnames = list(func.__code__.co_varnames)[1:func.__code__.co_argcount]
		defaults = [] if func.__defaults__ is None else list(func.__defaults__)
		funcargs = {key:val for key,val in zip(varnames,args)}
		funcdefaults = {key:val for key,val in zip(varnames[-len(defaults):],defaults)}
		for key in funcargs:
			self.params[key] = funcargs[key]
		for key in kwargs:
			self.params[key] = kwargs[key]
		for key in funcdefaults:
			if key not in self.params: self.params[key] = funcdefaults[key]
		for key in varnames:
			if key not in self.params:
				raise ValueError('Argument {} not provided to {} instance.'.format(key,self.__class__.__name__))
		for key in self.params:
			if key in varnames: funcargs[key] = self.params[key]
		return func(self,**funcargs)
	return wrapper

def setstateclass(func):
	@functools.wraps(func)
	def wrapper(self,state):
		self.__dict__.update(state)
		return func(self,state)
	return wrapper

def getstateclass(func):
	@functools.wraps(func)
	def wrapper(self):
		state = {}
		if hasattr(self,'params'): state['params'] = self.params
		return func(self,state)
	return wrapper

def loadstateclass(func):
	@functools.wraps(func)
	def wrapper(cls,state,**kwargs):
		self = object.__new__(cls)
		if kwargs:
			if 'params' in state: state['params'].update(kwargs)
			else: raise ValueError('Too many arguments for a class that does not contain params.')
		return func(self,state)
	return wrapper

def loadclass(func):
	@functools.wraps(func)
	def wrapper(cls,path,**kwargs):
		state = {}
		try:
			state = scipy.load(path,allow_pickle=True)[()]
		except IOError:
			raise IOError('Invalid path: {}.'.format(path))
		cls.logger.info('Loading {}: {}.'.format(cls.__name__,path))
		return loadstateclass(func)(cls,state,**kwargs)
	return wrapper

def saveclass(func):
	@functools.wraps(func)
	@classparams
	def wrapper(self,save=None,**kwargs):
		state = func(self)
		pathdir = os.path.dirname(save)
		mkdir(pathdir)
		self.logger.info('Saving {} to {}.'.format(self.__class__.__name__,save))
		scipy.save(save,state)
	return wrapper

def par_to_default_latex(par):
	latex = ''
	for p in par:
		latex += p
		if p == '_':
			latex += '{'
			par += '}'
	return latex

def save(path,*args,**kwargs):
	mkdir(os.path.dirname(path))
	logger.info('Saving to {}.'.format(path))
	scipy.save(path,*args,**kwargs)

def load(path,allow_pickle=True,**kwargs):
	logger.info('Loading {}.'.format(path))
	try:
		return scipy.load(path,allow_pickle=allow_pickle,**kwargs)[()]
	except IOError:
		raise IOError('Invalid path: {}.'.format(path))

def blockinv(blocks,inv=numpy.linalg.inv):
	A = blocks[0][0]
	if (len(blocks),len(blocks[0])) == (1,1):
		return inv(A)
	B = numpy.bmat(blocks[0][1:]).A
	C = numpy.bmat([b[0].T for b in blocks[1:]]).A.T
	invD = blockinv([b[1:] for b in blocks[1:]],inv=inv)
	def dot(*args):
		return numpy.linalg.multi_dot(args)
	invShur = inv(A - dot(B,invD,C))
	return numpy.bmat([[invShur,-dot(invShur,B,invD)],
						[-dot(invD,C,invShur), invD + dot(invD,C,invShur,B,invD)]]).A

def merge_dict(dic1,dic2):
	new = copy.deepcopy(dic1)
	new.update(dic2)
	return new

def edges_to_mid(edges):
	return (edges[1:]+edges[:-1])/2.

def get_parameters(args):
	toret = []
	for key in args:
		if not any([key.startswith(prefix) for prefix in MINUIT_PREFIX + OTHER_PREFIX]): toret += [key]
	return toret

def get_values(kwargs):
	parameters = get_parameters(kwargs.keys())
	return {par:kwargs[par] for par in parameters}

def get_errors(kwargs):
	parameters = get_parameters(kwargs.keys())
	return {par:kwargs['error_{}'.format(par)] for par in parameters}

def get_vary_parameters(args):
	parameters = get_parameters(args)
	toret = []
	for par in parameters:
		if not args.get('fix_{}'.format(par),False): toret += [par]
	return toret

def get_fixed_parameters(args):
	parameters = get_parameters(args)
	toret = []
	for par in parameters:
		if args.get('fix_{}'.format(par),False): toret += [par]
	return toret

def get_fitted_parameters(args):
	parameters = get_parameters(args)
	toret = []
	for par in parameters:
		if not args.get('fix_{}'.format(par),False) and not args.get('solve_{}'.format(par),False): toret += [par]
	return toret

def get_solved_parameters(args,method='lsq'):
	parameters = get_parameters(args)
	toret = []
	for par in parameters:
		if not args.get('fix_{}'.format(par),False):
			if ((method == 'all') and args.get('solve_{}'.format(par),False)) or (args.get('solve_{}'.format(par),False) == method):
				toret += [par]
	return toret

def get_latex(args):
	parameters = get_parameters(args)
	toret = {}
	for par in parameters:
		tmp = args.get('latex_{}'.format(par),False)
		if tmp: toret[par] = tmp
	return toret

def get_minuit_args(args):
	toret = {}
	for key,val in args.items():
		if not any([key.startswith(prefix) for prefix in OTHER_PREFIX]):
			toret[key] = val
	for key in get_vary_parameters(args):
		if args.get('solve_{}'.format(key),False):
			toret['fix_{}'.format(key)] = True
	return toret

def str_to_int(txt):
	li = []
	for c in re.split('(\d+)',txt):
		if c.isdigit(): li.append(int(c))
	return li

def array_to_latex(data,columns,rows,alignment='c',fmt=None):
	numcolumns = len(columns)
	numrows = len(rows)
	output = ''
	fmtcolumns = '{}|{}'.format(alignment,alignment*numcolumns)
	#Write header
	output += '\\begin{{tabular}}{{{}}}\n'.format(fmtcolumns)
	labelcolumns = ['{}'.format(label) for label in columns]
	output += '& {}\\\\\n\\hline\n'.format(' & '.join(labelcolumns))
	#Write data lines
	for i in range(numrows):
		if fmt is not None: strrows = [format(val,fmt) for val in data[i]]
		else: strrows = [format(val) for val in data[i]]
		output += '{} & {}\\\\\n'.format(rows[i], ' & '.join(strrows))
	#Write footer
	output += '\\end{tabular}\n'
	return output

def fit_to_latex(params,values,errors=None,uplow=None,rows=['best fit'],latex={},precision=2,fmt='vertical'):

	def prec(val):
		return to_precision(val,precision=precision)
		
	def err(val,u=0.1,d=None):
		return to_error(val,u=u,v=d,precision=precision)

	output = ''		
	data = [[]]
	if uplow is None:
		for par in params:
			data[-1] += ['${0}\pm{1}$'.format(*err(values[par],u=errors[par]))]
	else:
		for par in params:
			try: data[-1] += ['${{{0}}}^{{{1}}}_{{{2}}}$'.format(*err(values[par],u=uplow['upper'][par],d=uplow['lower'][par]))]
			except KeyError: data[-1] += ['${0}\pm{1}$'.format(*err(values[par],u=errors[par]))]
		if 'upper_valid' in uplow:
			data += [[]]
			rows += ['valid']
			for par in params:
				try: data[-1] += ['${{}}^{{{0}}}_{{{1}}}$'.format(uplow['upper_valid'][par],uplow['lower_valid'][par])]
				except KeyError: data[-1] += ['']
	
	output += '\\begin{center}\n'
	columns = ['${}$'.format(latex[par]) if par in latex else par for par in params]

	if fmt == 'horizontal':
		output += array_to_latex(data=scipy.asarray(data),columns=columns,rows=rows[:len(data)],alignment='l',fmt=None)
	else:
		output += array_to_latex(data=scipy.asarray(data).T,columns=rows[:len(data)],rows=columns,alignment='l',fmt=None)
	
	output += '\n\\end{center}\n'
	
	return output

def save_latex(s,path):
	pathdir = os.path.dirname(path)
	if not pathdir: pathdir = '.'
	mkdir(pathdir)
	with open(path,'w') as file:
		file.write(s)
		file.close()
	logger.info('Saving dvi file to {}.'.format(path.replace('.tex','.dvi')))
	subprocess.Popen(['latex',path],cwd=pathdir,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
	#subprocess.Popen(['dvipdfm',path.replace('.tex',.dvi)],cwd=pathdir)
	subprocess.Popen(['dvips',path.replace('.tex','.dvi')],cwd=pathdir,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
	#subprocess.Popen(['ps2pdf',path.replace('.tex','.ps')],cwd=pathdir)

def plot_xlabel(estimator='spectrum'):
	if estimator == 'spectrum':
		return '$k$ [$h \ \\mathrm{Mpc}^{-1}$]'
	if estimator == 'rmu':
		return '$r$ [$\\mathrm{Mpc} \ h^{-1}$]'
	if 'angular' in estimator:
		return '$\\theta$ [deg]'
	if  estimator == 'rppi':
		return '$r_{p}$ [$\\mathrm{Mpc} \ h^{-1}$]'
	if estimator == 'rr':
		return '$s$ [$\\mathrm{Mpc} \ h^{-1}$]'

def suplabel(axis,label,shift=0,labelpad=5,ha='center',va='center',**kwargs):
	"""Add super ylabel or xlabel to the figure. Similar to matplotlib.suptitle.
	Taken from https://stackoverflow.com/questions/6963035/pyplot-axes-labels-for-subplots.

	Parameters
	----------
	axis : str
		'x' or 'y'.
	label : str
		label.
	shift : float, optional
		shift.
	labelpad : float, optional
		padding from the axis.
	ha : str, optional
		horizontal alignment.
	va : str, optional
		vertical alignment.
	kwargs : dict
		kwargs for matplotlib.pyplot.text

	"""
	fig = matplotlib.pyplot.gcf()
	xmin = []
	ymin = []
	for ax in fig.axes:
		xmin.append(ax.get_position().xmin)
		ymin.append(ax.get_position().ymin)
	xmin,ymin = min(xmin),min(ymin)
	dpi = fig.dpi
	if axis.lower() == 'y':
		rotation = 90.
		x = xmin - float(labelpad)/dpi
		y = 0.5 + shift
	elif axis.lower() == 'x':
		rotation = 0.
		x = 0.5 + shift
		y = ymin - float(labelpad)/dpi
	else:
		raise Exception('Unexpected axis: x or y')
	matplotlib.pyplot.text(x,y,label,rotation=rotation,transform=fig.transFigure,ha=ha,va=va,**kwargs)

def mkdir(path):
	path = os.path.abspath(path)
	if not os.path.isdir(path): os.makedirs(path)

def savefig(path,*args,**kwargs):
	mkdir(os.path.dirname(path))
	logger.info('Saving figure to {}.'.format(path))
	matplotlib.pyplot.savefig(path,*args,**kwargs)
	matplotlib.pyplot.close(matplotlib.pyplot.gcf())

_logging_handler = None

def setup_logging(log_level="info"):
	"""
	Turn on logging, with the specified level.

	Parameters
	----------
	log_level : 'info', 'debug', 'warning'
		the logging level to set; logging below this level is ignored
	"""

	# This gives:
	#
	# [ 000000.43 ]   0: 06-28 14:49  measurestats	INFO	 Nproc = [2, 1, 1]
	# [ 000000.43 ]   0: 06-28 14:49  measurestats	INFO	 Rmax = 120

	levels = {
			"info" : logging.INFO,
			"debug" : logging.DEBUG,
			"warning" : logging.WARNING,
			}

	logger = logging.getLogger();
	t0 = time.time()


	class Formatter(logging.Formatter):
		def format(self, record):
			s1 = ('[ %09.2f ]: ' % (time.time() - t0))
			return s1 + logging.Formatter.format(self, record)

	fmt = Formatter(fmt='%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
					datefmt='%m-%d %H:%M ')

	global _logging_handler
	if _logging_handler is None:
		_logging_handler = logging.StreamHandler()
		logger.addHandler(_logging_handler)

	_logging_handler.setFormatter(fmt)
	logger.setLevel(levels[log_level])

#setup_logging()
logger = logging.getLogger('Utils')
