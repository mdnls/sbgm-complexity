import numpy as np
import scipy
import scipy.linalg
import matplotlib.pyplot as plt

class GMM_SBGM():
    def __init__(self, means, variance=1, weights=None):
        '''
        Initialize the SBGM corresponding to an equally weighted mixture of Gaussians
        :param means: a K by D matrix of K GMM means, each D dimensional
        :param variance: scalar variance of each Gaussian in the mixture
        :param weights: None, or a length K array of floats summing to one. Represents the weights of each Gaussian in the
        mixture. If None, the weights are assumed to be uniform.
        '''
        self.means = means
        self.n_means, self.dim = means.shape
        self.variance = variance

        if(weights is None):
            weights = np.ones((self.n_means,)) / self.n_means

        weights = np.array(weights)
        assert weights.shape == (self.n_means,), "Weights must be a length k numpy array"
        assert np.isclose(np.sum(weights), 1), "Input weights must sum to 1"
        self.weights = weights

    def score(self, x):
        '''
        :param x: a P by D matrix of P input samples each D dimensional
        :return: a P by D matrix of scores for each point
        '''
        densities_per_mixture = self.pdf(x, per_mixture=True)
        #                           P by K by 1                               P by 1 by D             1 by K by D
        grad_per_mixture = - densities_per_mixture[:, :, None] * (x[:, None, :] - self.means[None, :, :]) / self.variance
        #                                                             |--------------------\/--------------------|
        #                                                              broadcast to P by K by D, scale by densities
        return np.sum(self.weights.reshape((1, -1, 1)) * grad_per_mixture, axis=1) / np.sum(self.weights[None, :] * densities_per_mixture, axis=1).reshape((-1, 1))

    def pdf(self, x, per_mixture=False):
        '''
        Return the probability density of x under this GMM. If per_mixture=True, returns a length K array which is the
        (unweighted) probability densit of x under each Gaussian in the mixture.
        :param x: a P by D matrix of P input samples each D dimensional
        :param per_mixture: if True, return the (unweighted) density per mixture of this point, a P by K matrix
        :return: if per_mixture is true, return a P by K matrix of densities, otherwise return a P, matrix of densities
        '''
        densities_per_mixture = np.exp(- np.sum((x[:, None, :] - self.means[None, :, :])**2, axis=2) / (2*self.variance) ) / ((2 * np.pi * self.variance)**(self.dim/2))
        if(per_mixture):
            return densities_per_mixture
        else:
            return np.sum(self.weights[None, :] * densities_per_mixture, axis=1)

    def sample(self, n_samples):
        indices = np.random.choice(self.n_means, size=n_samples, p=self.weights)
        means_per_sample = self.means[indices]
        deviations_per_sample = np.sqrt(self.variance) * np.random.normal(size=(n_samples, self.dim))
        return means_per_sample + deviations_per_sample

    def iterate_langevin(self, initialization, iters=1000, step_size=0.01, playback=False):
        '''
        Iterate langevin dynamics according to this SBGM.

        :param initialization: a P by D matrix of initialization points
        :param iters: number of langevin iterations
        :param step_size:
        :param playback: if True, return a T by P by D array of samples at each timestep. If False, return a P
            by D array of samples at the final timestep.
        :return:
        '''
        n_samples, dim = initialization.shape
        assert dim == self.dim, "Initialization must have the same dimension as means."
        playback_samples = None
        samples = initialization

        if(playback):
            playback_samples = np.zeros((iters, n_samples, dim))
            playback_samples[0] = initialization

        for itr in range(1, iters):
            samples = samples + step_size * self.score(samples) + np.sqrt(2 * step_size) * np.random.normal(size=samples.shape)
            if(playback):
                playback_samples[itr] = samples

        if(playback):
            return playback_samples
        else:
            return samples

    def init_random(dim, scale=1, n_means=2, variance=1, weights=None):
        '''
        Randomly initialize a GMM_SBGM whose means are sampled from a centered Gaussian with covariance (scale**2)*I
            :param dim: dimension of the means
        :param scale: the standard deviation of coordinates of the random means
        :param n_means:
        :param variances:
        :param weights:
        :return:
        '''
        means = np.random.normal(size=(n_means, dim), scale=scale)
        return GMM_SBGM(means=means, variance=variance, weights=weights)

'''
MD 08/26/2022: below is my first draft implementation which is over engineered. Specifically it allows arbitrary 
    covariances, which we will probably not ever need, and which are also quite expensive to invert if we want to compute
    the corresponding pdf or score. I'm leaving it around just in case, but probably useless. 
    
class GMM_SBGM_Aniso():
    def __init__(self, means, variances=1, weights=None):
        ''
        Initialize the SBGM corresponding to an equally weighted mixture of Gaussians
        :param means: a K by D matrix of K GMM means, each D dimensional
        :param variances: a K by D by D matrix of GMM covariances. This argument broadcasts,
            so input can be 1-dimensional (homogeneous isotropic mixture),
            K by 1-dimensional (inhomogeneous isotropic mixture), and so on
        :param weights: None, or a length K array of floats summing to one. Represents the weights of each Gaussian in the
            mixture. If None, the weights are assumed to be uniform.
        ''
        self.means = means
        self.n_means, self.dim = means.shape
        self.variances = np.stack(self.n_means * [np.eye(self.dim)])

        variances = np.array(variances)
        if(len(variances.shape) == 3):
            # K by D by D
            self.variances = variances
        elif(len(variances.shape) == 2):
            # K by D
            self.variances = np.stack([np.diag(x) for x in variances])
        elif(len(variances.shape) == 1):
            self.variances = self.variances * variances.reshape((-1, 1, 1))
        elif(len(variances.shape) == 0):
            self.variances = variances * self.variances
        else:
            raise ValueError("Invalid input variance")

        self.sqrt_variances = np.stack([scipy.linalg.sqrtm(R) for R in self.variances])
        self.inv_variances = np.stack([np.linalg.pinv(R) for R in self.variances])

        if(weights is None):
            weights = np.ones((self.n_means,)) / self.n_means

        weights = np.array(weights)
        assert weights.shape == (self.n_means,), "Weights must be a length k numpy array"
        assert np.isclose(np.sum(weights), 1), "Input weights must sum to 1"
        self.weights = weights

    def score(self, x):
        pass

    def sample(self, n_samples):
        indices = np.random.choice(self.n_means, size=n_samples, p=self.weights)
        means_per_sample = self.means[indices]
        deviations_per_sample = self.sqrt_variances[indices] @ np.random.normal(size=(n_samples, self.dim, 1))
        return means_per_sample + deviations_per_sample.reshape((n_samples, self.dim))

    def init_random(dim, scale=1, n_means=2, variances=1, weights=None):
        ''
        Randomly initialize a GMM_SBGM whose means are sampled from a centered Gaussian with covariance (scale**2)*I
        :param dim: dimension of the means
        :param scale: the standard deviation of coordinates of the random means
        :param n_means:
        :param variances:
        :param weights:
        :return:
        ''
        means = np.random.normal(size=(n_means, dim), scale=scale)
        return GMM_SBGM_Aniso(means=means, variances=variances, weights=weights)
'''

if __name__ == "__main__":
    sbgm = GMM_SBGM(means=np.eye(2), variance=0.01, weights=[0.5, 0.5])
    samples = sbgm.sample(n_samples=200)

    init = np.random.normal(size=(200, 2))
    langevin_samples = sbgm.iterate_langevin(init, iters=5000, step_size=0.001)
    plt.scatter(*samples.T, color="red", alpha=0.5)
    plt.scatter(*langevin_samples.T, color="blue", alpha=0.5)
    plt.show()