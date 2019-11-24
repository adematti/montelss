import scipy
from pyspectrum import *

setup_logging()

def save_catalogue(nrand=1000,rmax=10.):
	rmin = -rmax
	position = []
	for j in range(3): position.append(scipy.random.uniform(rmin,rmax,nrand))
	position = scipy.asarray(position).T
	weight = scipy.random.uniform(0.5,1.,nrand)
	cat = Catalogue({'Position':position,'Weight':weight})
	cat.attrs['nrand'] = nrand
	cat.save('cat.npy')
	
def load_catalogue():
	return Catalogue.load('cat.npy')
	
def test_catalogue():
	cat = load_catalogue()
	cat.box()
	assert cat.boxsize()==scipy.sqrt(((cat['Position'].max(axis=0)-cat['Position'].min(axis=0))**2).sum())
	size = cat.size
	cat2 = cat.slice(0,10)
	assert cat2.size==size//10
	assert cat.size==size
	assert 'Position' in cat2
	assert 'nrand' in cat2.attrs
	
	
def test_nbodykit():
	cat = load_catalogue()
	ncat = cat.to_nbodykit()
	print ncat.columns
	print ncat['Position']
	
#save_catalogue()
test_catalogue()
#test_nbodykit()
