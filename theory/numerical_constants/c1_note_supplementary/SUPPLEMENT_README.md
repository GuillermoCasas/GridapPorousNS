# Supplementary material: computation of the sharp constants

Companion to "A note on the dimension and shape dependence of the
viscous stabilization constant c1 for quadratic simplicial elements in
ASGS/VMS methods".

Files:
- compute_constants.py : full sweep (scalar-gradient, vector-divergence,
  div(sym grad) unrestricted and divergence-free constants) for the five
  simplex shapes, P1 and P2; exact rational Gram assembly, scipy
  generalized eigensolves, Monte-Carlo Rayleigh validation. Writes
  constants.json.
- verify2.py : exact rational eigenvalue verification of the 2D right
  triangle values (48 and 84) and 50-digit (mpmath) verification of the
  3D values against the closed forms 440/3 and 200+10*sqrt(2).
- verify3.py : 50-digit verification of all remaining table entries.
- threshold.py : exact elementwise coercivity threshold c1*(K) with the
  symmetric-gradient (non-deviatoric) projector, for comparison.
- threshold_dev.py : the same with the paper's deviatoric-symmetric
  projector; establishes Call2 = Cdf2 (collapse) and c1*(K) = 2*chat^2(K)
  for all five shapes, and prints the 3D/2D ratios.
- constants.json : raw output of the sweep.

Requirements: python3 with sympy, numpy, scipy, mpmath.
Run order: compute_constants.py, then verify2.py, verify3.py,
threshold_dev.py. Each run is a few minutes.
