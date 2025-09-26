import numpy as np
from scipy import optimize
from scipy.special import psi
from scipy.stats import norm, expon, gamma, weibull_min, geom, bernoulli, chi2

def mle_normal(data):
    data = np.array(data)
    mu = np.mean(data)
    var = np.mean((data - mu)**2)
    return mu, var

def mle_exponential(data):
    data = np.array(data)
    return (1 / np.mean(data),)

#https://colab.research.google.com/drive/1NYditcln3OTiOF7qpY1kMbunKx26291H (course note)
def mle_gamma(data):
    data = np.array(data)
    mean_log_data = np.mean(np.log(data))
    mean = np.mean(data)
    def alpha_equation(alpha):
        return np.log(mean/alpha) + psi(alpha) - mean_log_data
    sol = optimize.root(alpha_equation, [0.5])
    alpha_hat = sol.x[0]
    beta_hat = mean / alpha_hat
    return alpha_hat, beta_hat


def mle_weibull(data):
    data = np.array(data)
    
    def alpha_equation(alpha):
        return (np.sum(data**alpha * np.log(data)) / np.sum(data**alpha)) - (1/alpha) - np.mean(np.log(data))
    sol = optimize.root(alpha_equation, [0.5])
    alpha_hat = sol.x[0]
    beta_hat = (np.mean(data**alpha_hat))**(1/alpha_hat)
    return alpha_hat, beta_hat

def mle_geometric(data):
    data = np.array(data)
    return (1 / np.mean(data),)

def mle_bernoulli(data):
    data = np.array(data)
    return (np.mean(data),)


def gof_test(data, dist_name, alpha):
    data = np.array(data)
    n = len(data)
    
    if dist_name == "bernoulli":
        
        edges = np.array([-0.5, 0.5, 1.5])  
        counts, _ = np.histogram(data, bins=edges)

    elif dist_name == "geometric":
        max_val = int(np.max(data))
        
        edges = np.arange(0.5, max_val + 1.5)  
        counts, _ = np.histogram(data, bins=edges)
    else:
        num_bins = int(n/5)
        counts, edges = np.histogram(data, bins=num_bins)


    dist_mle_dict = {
        "normal": (mle_normal, lambda edge, param: norm.cdf(edge, loc=param[0], scale=np.sqrt(param[1]))),
        "exponential": (mle_exponential, lambda edge, param: expon.cdf(edge, scale=1/param[0])),
        "gamma": (mle_gamma, lambda edge, param: gamma.cdf(edge, a=param[0], scale=param[1])),
        "weibull": (mle_weibull, lambda edge, param: weibull_min.cdf(edge, c=param[0], scale=param[1])),
        "geometric": (mle_geometric, lambda edge, param: geom.cdf(edge, p=param[0])),
        "bernoulli": (mle_bernoulli, lambda edge, param: bernoulli.cdf(edge, p=param[0]))
    }

    mle_func, cdf_func = dist_mle_dict[dist_name]
    params = mle_func(data)

    expected_counts = []
    for i in range(len(edges) - 1):
        prob = cdf_func(edges[i + 1], params) - cdf_func(edges[i], params)
        expected_counts.append(prob * n)

    expected_counts = np.array(expected_counts)

    chi2_stat = np.sum((counts - expected_counts) ** 2 / expected_counts)
    df = len(counts) - 1 - len(params)
    conf_lvl = 1-alpha
    

    if df <= 0:
        return chi2_stat, df

    crit = chi2.ppf(conf_lvl, df)
    
    if chi2_stat > crit:
        result = "reject"
    elif chi2_stat <= crit:
        result = "fail to reject"  

    return (chi2_stat, result)