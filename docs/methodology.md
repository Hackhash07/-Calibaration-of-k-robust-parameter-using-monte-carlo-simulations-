# Methodology

Mathematical and empirical framework for the calibration of the robust $k$ (kappa) parameter using Monte Carlo simulations.

## 1. Uncertainty Sets & Robust Optimization
Robust portfolio optimization models parameter uncertainty explicitly. For return vector $\mu$:
$$\mu \in \mathcal{U}(k) = \{ \bar{\mu} + \Omega^{1/2} z \mid \|z\|_2 \le k \}$$

## 2. Monte Carlo Calibration
We sample scenario paths inside the uncertainty set to calibrate $k$.
