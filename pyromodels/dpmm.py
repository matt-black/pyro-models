
import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist

from tqdm.auto import tqdm


class DirichletProcessGaussianMixtureModel(pyro.nn.PyroModule):
    def __init__(self, max_clusters, alpha, dim):
        super(DirichletProcessGaussianMixtureModel, self).__init__()
        self._T = max_clusters
        self._alpha = alpha
        self._dim = dim
        
    def model_unitdiagcov(self, data):
        N = data.shape[0]
        with pyro.plate("beta_plate", self._T-1):
            beta = pyro.sample("beta", dist.Beta(1, self._alpha))
        with pyro.plate("mu_plate", T):
            mu = pyro.sample(
                "mu", dist.MultivariateNormal(torch.zeros(self._dim),
                                              5*torch.eye(self._dim))
            )
        with pyro.plate("data", N):
            z = pyro.sample("z", dist.Categorical(_mix_weights(beta)))
            pyro.sample(
                "obs", dist.MultivariateNormal(mu[z], torch.eye(self._dim)),
                obs=data
            )

    def guide_unitdiagcov(self, data):
        N = data.shape[0]
        kappa = pyro.param(
            "kappa", lambda : dist.Uniform(0, 2).sample([self._T-1]),
            constraint=constraints.positive
        )
        tau = pyro.param(
            "tau", lambda : dist.MultivariateNormal(
                torch.zeros(2), 3*torch.eye(2)).sample([self._T])
        )
        phi = pyro.param(
            "phi", lambda : dist.Dirichlet(1/T * torch.ones(T)).sample([N]),
            constraints=constraint.simplex
        )
        with pyro.plate("beta_plate", self._T-1):
            q_beta = pyro.sample(
                "beta", dist.Beta(torch.ones(self._T-1), kappa)
            )
        with pyro.plate("mu_plate", self._T):
            q_mu = pyro.sample(
                "mu", dist.MultivariateNormal(tau, torch.eye(self._dim))
            )
        with pyro.plate("data", N):
            z = pyro.sample("z", dist.Categorical(phi))

    def _truncate(self, centers, weights):
        thresh = self._alpha**-1 / 100.
        filt = weights > thresh
        new_cent = centers[filt]
        new_wgt  = weights[filt] / torch.sum(weights[filt])
        return new_cent, new_wgt

    def train(self, x, n_iter, opt_pars={'lr' : 0.05}):
        pyro.clear_param_store()
        adam = pyro.optim.Adam(opt_pars)
        svi = pyro.infer.SVI(self.model, self,guide, adam,
                             loss=pyro.infer.Trace_ELBO())
        losses = []
        with tqdm(total=n_iter) as pbar:
            for n in range(n_iter):
                loss = svi.step(x)
                losses.append(loss)
                if n % 10:
                    pbar.set_postfix({'elbo' : loss})
        

def _mix_weights(beta):
    beta1m_cumprod = (1 - beta).cumprod(-1)
    return F.pad(beta, (0,1), value=1) * F.pad(beta1m_cumprod, (1,0), value=1)


        
