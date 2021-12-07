# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
def log_gaussian(x, mu, sigma):
    # The np.sum() is for compatibility with sample_MH
    return - 0.5 * np.sum((x - mu) ** 2) / sigma ** 2 \
           - np.log(np.sqrt(2 * np.pi * sigma ** 2))

class GaussianMixture(object):
    
    def __init__(self, mu1, mu2, sigma1, sigma2, w1, w2):
        self.mu1, self.mu2 = mu1, mu2
        self.sigma1, self.sigma2 = sigma1, sigma2
        self.w1, self.w2 = w1, w2
        
    def log_prob(self, x):
        return np.logaddexp(np.log(self.w1) + log_gaussian(x, self.mu1, self.sigma1),
                            np.log(self.w2) + log_gaussian(x, self.mu2, self.sigma2))
    
    def log_p_x_k(self, x, k):
        # logarithm of p(x|k)
        mu = (self.mu1, self.mu2)[k]
        sigma = (self.sigma1, self.sigma2)[k]
    
        return log_gaussian(x, mu, sigma)
    
    def p_k_x(self, k, x):
        # p(k|x) using Bayes' theorem
        mu = (self.mu1, self.mu2)[k]
        sigma = (self.sigma1, self.sigma2)[k]
        weight = (self.w1, self.w2)[k]
        log_normalization = self.log_prob(x)

        return np.exp(log_gaussian(x, mu, sigma) + np.log(weight) - log_normalization)

# %%
mix_params = dict(mu1=-1.5, mu2=2.0, sigma1=0.5, sigma2=0.2, w1=0.3, w2=0.7)
mixture = GaussianMixture(**mix_params)
temperatures = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# %%
from scipy.integrate import quad

def plot_tempered_distributions(log_prob, temperatures, axes, xlim=(-4, 4)):
    xspace = np.linspace(*xlim, 1000)
    for i, (temp, ax) in enumerate(zip(temperatures, axes)):
        pdf = lambda x: np.exp(temp * log_prob(x))
        Z = quad(pdf, -1000, 1000)[0]
        ax.plot(xspace, np.array(list(map(pdf, xspace))) / Z)
        ax.text(0.8, 0.3, r'$\beta={}$'.format(temp), transform=ax.transAxes)
        ax.text(0.05, 0.3, 'replica {}'.format(len(temperatures) - i - 1), 
                transform=ax.transAxes)
        ax.set_yticks(())
    plt.show()
    
fig, axes = plt.subplots(
                len(temperatures), 
                1, 
                sharex=True, 
                sharey=True,
                figsize=(8, 7)
            )
plot_tempered_distributions(mixture.log_prob, temperatures, axes)
plt.show()
# %%
fig, axes = plt.subplots(
                len([0.1, 1.0]), 
                1, 
                sharex=True, 
                sharey=True,
                figsize=(8, 7)
            )
plot_tempered_distributions(mixture.log_prob, [0.1, 1.0], axes)
plt.show()
# %%
