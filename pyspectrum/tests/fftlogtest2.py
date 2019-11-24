
# coding: utf-8

# # FFTlog
# 
# This notebook is a translation of `fftlogtest.f` from the Fortran package `FFTlog`, which was presented in Appendix B of Hamilton, 2000, and published at <http://casa.colorado.edu/~ajsh/FFTlog>. It serves as an example for the python package `fftlog` (which is a `f2py`-wrapper around `FFTlog`), in the same manner as the original file `fftlogtest.f` serves as an example for Fortran package `FFTlog`.
# 
# 
# ### Reference
# Hamilton, A. J. S., 2000, Uncorrelated modes of the non-linear power spectrum: Monthly Notices of the Royal Astronomical Society, 312, pages 257-284; DOI: <http://dx.doi.org/10.1046/j.1365-8711.2000.03071.x>.
# 
# ---

# ## This is fftlogtest.f
# 
# This is a simple test program to illustrate how `FFTlog` works.
# The test transform is:
# 
# $$
# \int^\infty_0 r^{\mu+1} \exp\left(-\frac{r^2}{2} \right)\
# J_\mu(k, r)\ k\ {\rm d}r = k^{\mu+1} \exp\left(-\frac{k^2}{2}
# \right) $$
# 
# 
# **Disclaimer:**  
# `FFTlog` does NOT claim to provide the most accurate possible
# solution of the continuous transform (which is the stated aim
# of some other codes).  Rather, `FFTlog` claims to solve the exact
# discrete transform of a logarithmically-spaced periodic sequence.
# If the periodic interval is wide enough, the resolution high
# enough, and the function well enough behaved outside the periodic
# interval, then `FFTlog` may yield a satisfactory approximation
# to the continuous transform.
# 
# Observe:
# 1.  How the result improves as the periodic interval is enlarged.
#     With the normal FFT, one is not used to ranges orders of
#     magnitude wide, but this is how `FFTlog` prefers it.
# 2.  How the result improves as the resolution is increased.
#     Because the function is rather smooth, modest resolution
#     actually works quite well here.
# 3.  That the central part of the transform is more reliable
#     than the outer parts.  Experience suggests that a good general
#     strategy is to double the periodic interval over which the
#     input function is defined, and then to discard the outer
#     half of the transform.
# 4.  That the best bias exponent seems to be $q = 0$.
# 5.  That for the critical index $\mu = -1$, the result seems to be
#     offset by a constant from the 'correct' answer.
# 6.  That the result grows progressively worse as mu decreases
#     below -1.
# 
# The analytic integral above fails for $\mu \le -1$, but `FFTlog`
# still returns answers.  Namely, `FFTlog` returns the analytic
# continuation of the discrete transform.  Because of ambiguity
# in the path of integration around poles, this analytic continuation
# is liable to differ, for $\mu \le -1$, by a constant from the 'correct'
# continuation given by the above equation.
# 
# `FFTlog` begins to have serious difficulties with aliasing as
# $\mu$ decreases below $-1$, because then $r^{\mu+1} \exp(-r^2/2)$ is
# far from resembling a periodic function.
# You might have thought that it would help to introduce a bias
# exponent $q = \mu$, or perhaps $q = \mu+1$, or more, to make the
# function $a(r) = A(r) r^{-q}$ input to `fhtq` more nearly periodic.
# In practice a nonzero $q$ makes things worse.
# 
# A symmetry argument lends support to the notion that the best
# exponent here should be $q = 0,$ as empirically appears to be true.
# The symmetry argument is that the function $r^{\mu+1} \exp(-r^2/2)$
# happens to be the same as its transform $k^{\mu+1} \exp(-k^2/2)$.
# If the best bias exponent were q in the forward transform, then
# the best exponent would be $-q$ that in the backward transform;
# but the two transforms happen to be the same in this case,
# suggesting $q = -q$, hence $q = 0$.
# 
# This example illustrates that you cannot always tell just by
# looking at a function what the best bias exponent $q$ should be.
# You also have to look at its transform.  The best exponent $q$ is,
# in a sense, the one that makes both the function and its transform
# look most nearly periodic.
# 
# ---

# ## Test-Integral:  $\int_0^\infty r^{\mu+1}\ \exp\left(-\frac{r^2}{2}\right)\ J_\mu(k,r)\ k\ {\rm d}r = k^{\mu+1} \exp\left(-\frac{k^2}{2}\right)$

# ### Import `fftlog` as well as `numpy` and `matplotlib`; some plot settings


import os
import scipy
from scipy import constants
import matplotlib.pyplot as pyplot
import fftlog
from pyspectrum import ModelTNS

pathModel = os.path.join(os.getenv('BAOELGMOD'),'tests','model_tns.npy')
model = ModelTNS.load(pathModel)

# Range of periodic interval
logrmin = -1.
logrmax = 3.

# Number of points (Max 4096)
n = 4000

# Order mu of Bessel function
mu = 1./2.

# Bias exponent: q = 0 is unbiased
q = 0

# Sensible approximate choice of k_c r_c
kr = 1

# Tell fhti to change kr to low-ringing value
# WARNING: kropt = 3 will fail, as interaction is not supported
kropt = 1

# Forward transform (changed from dir to tdir, as dir is a python fct)
tdir = 1


# ### Calculation related to the logarithmic spacing

# Central point log10(r_c) of periodic interval
logrc = (logrmin + logrmax)/2

print('Central point of periodic interval at log10(r_c) = ', logrc)

# Central index (1/2 integral if n is even)
nc = (n + 1)/2.0

# Log-spacing of points
dlogr = (logrmax - logrmin)*1./(n-1.)
dlnr = dlogr*scipy.log(10.0)


# ### Calculate input function: $r^{\mu+1}\exp\left(-\frac{r^2}{2}\right)$

r = 10**(logrc + (scipy.arange(1, n+1) - nc)*dlogr)
ar = r**(mu + 1)*scipy.exp(-r**2/2.0)


# ### Initialize FFTlog transform - note fhti resets `kr`

# In[5]:

kr, wsave, ok = fftlog.fhti(n, mu, dlnr, q, kr, kropt)
print('fftlog.fhti: ok =', bool(ok), '; New kr = ', kr)


# ### Call `fftlog.fht` (or `fftlog.fhtl`)

# In[6]:

logkc = scipy.log10(kr) - logrc
print('Central point in k-space at log10(k_c) = ', logkc)

# rk = r_c/k_c
rk = 10**(logrc - logkc)

# Transform
k = 10**(logkc + (scipy.arange(1, n+1) - nc)*dlogr)
Pk = scipy.interp(k,model.k,model.Plin)*k**(3./2.)

print r.min(), r.max(), len(r)
print k.min(), k.max(), len(k)
Xi = fftlog.fht(Pk.copy(), wsave, tdir)/(2.*constants.pi*r)**(3./2)

# ### Plot result

mask = (r > 1) & (r < 200)
pyplot.figure(figsize=(8,6))
pyplot.plot(r[mask],r[mask]**2*Xi[mask])
pyplot.yscale('linear')
pyplot.show()

