montelss - Monte-Carlo for Large Scale Structures
=================================================

Introduction
------------

**montelss** is a package to perform cosmological inference from large-scale structures.
For the moment, it contains:

montelss
--------

A flexible framework to perform minimization, profiling (based on iminuit) and MCMC (based on emcee) of likelihoods.
**montelss** also integrates a least-square solver.
Likelihoods for RSD and BAO are in likelihoods/RSD and likelihoods/BAO and use the **pyspectrum** package (see below).
analyze_fits.py and analyze_mcmc.py contain a lot of routines for post-processing.

pyspectrum
----------

A code to compute theory power spectrum:
- RSD model (same as in https://arxiv.org/abs/1904.08851v3), using pyregpt
- BAO templates (isotropic and anisotropic)

Apply survey geometry effects:
- window functions, following Wilson et al. 2015: https://arxiv.org/abs/1511.07799, using pycute
- integral constraint corrections, following de Mattia et al. 2019: https://arxiv.org/abs/1904.08851v3, using pycute
- wide-angle contributions, following Beutler et al. 2019: https://arxiv.org/abs/1810.05051
- fiber collisions, following Hahn et al. 2016: https://arxiv.org/abs/1609.01714

**montelss** and **pyspectrum** will eventually be two separate, independent packages.

Requirements
------------

- scipy
- fftlog
- iminuit
- emcee
- pathos
- class
- pyregpt
- pycute
