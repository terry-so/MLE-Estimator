# Distribution Fitting Library

This Python library provides functions for performing **Maximum Likelihood Estimation (MLE)** and **Chi-squared Goodness-of-Fit** tests for the following probability distributions:

- Bernoulli
- Geometric
- Normal
- Exponential
- Gamma
- Weibull


The source code is located in **"...\mle_estimator\mle_estimator\estimator.py"**


## Installation

Please make sure the following packages are installed:
- numpy
- scipy

To use the library, create a Jupyter notebook or Python script in the root folder (where mle_gof_demo.ipynb is located)
Import the functions as follow:

```python
from mle_estimator.estimator import mle_normal, mle_exponential, mle_gamma, mle_weibull, mle_bernoulli, mle_geometric, gof_test
```
## Function Overview

```python
mle_bernoulli(data)   → probability (p)
mle_geometric(data)   → probability (p)
mle_normal(data)      → mean, variance
mle_exponential(data) → rate (λ)
mle_gamma(data)       → shape (α), scale (β)
mle_weibull(data)     → shape (α), scale (β)
```

```python
gof_test(data, dist_name, alpha)  → chi2_stat, decision
```

## Usage
**For the full demo please go to mle_gof_demo.ipynb**
```python
import numpy as np
from mle_estimator.estimator import mle_gamma, gof_test

data = np.random.gamma(shape=2, scale=1.5, size=1000)
alpha_hat, beta_hat = mle_gamma(data)
chi2_stat, decision = gof_test(data, "gamma", alpha=0.05)

print("Estimated α:", alpha_hat)
print("Estimated β:", beta_hat)
print("GoF Test:", chi2_stat, decision)
```


