import pyro
import pyro.distributions as dist
import torch
from torch.distributions import constraints

from tqdm.auto import tqdm


class ProbabilisticPCA(pyro.nn.PyroModule):
    def __init__(self, d, k, loc_prior=1.0):
        super(ProbabilisticPCA, self).__init__()
        self._d = d
        self._k = k
        self.W = torch.nn.Linear(k, d, bias=False)
        self.register_buffer("z_mu", torch.zeros(k))
        self.register_buffer("loc_prior", torch.tensor(loc_prior))

    def model_fixsigma(self, x):
        pyro.module("W", self.W)
        with pyro.plate("data", x.shape[0]):
            z = pyro.sample(
                "z", dist.Normal(self.z_mu, 1.0).to_event(1)
            )
            pyro.sample(
                "obs", dist.Normal(self.W(z), 1.0).to_event(1),
                obs=x
            )
        
    def model(self, x):
        pyro.module("W", self.W)
        loc = pyro.param("loc", self.loc_prior,
                         constraint=constraints.positive)
        with pyro.plate("data", x.shape[0]):
            # sample z ~ N(0, I)
            z = pyro.sample(
                "z", dist.Normal(self.z_mu, 1.0).to_event(1)
            )
            pyro.sample(
                "obs",
                dist.Normal(self.W(z), loc).to_event(1),
                obs=x
            )
    
    def forward(self, x):
        pass

    def sample(self, n_samp):
        z = dist.Normal(self.z_mu, 1.0).to_event(1).sample([n_samp])
        return self.W(z).squeeze(1).detach()
        

    def fit_map(self, x, n_iter, opt_pars={'lr' : 0.005}, fix_sigma=False):
        if fix_sigma:
            model = self.model_fixsigma
        else:
            model = self.model
        guide = pyro.infer.autoguide.AutoDelta(model)
        pyro.clear_param_store()
        adam = pyro.optim.Adam(opt_pars)
        svi = pyro.infer.SVI(model, guide, adam, loss=pyro.infer.Trace_ELBO())
        losses = []
        with tqdm(total=n_iter) as pbar:
            for _ in range(n_iter):
                loss = svi.step(x)
                pbar.update(1)
                if n_iter % 10:
                    pbar.set_postfix({'elbo' : loss})
                losses.append(loss)
        return torch.tensor(losses), guide


class BayesianPCA(pyro.nn.PyroModule):
    def __init__(self, dim, a=0.1, b=0.1, c=0.1, d=0.1, beta=0.01):
        super(BayesianPCA, self).__init__()
        self._dim = dim  # data dimension
        # parameters for gamma priors on alpha (a,b), tau (c,d)
        self.register_buffer("a", torch.tensor(a))
        self.register_buffer("b", torch.tensor(b))
        self.register_buffer("c", torch.tensor(c))
        self.register_buffer("d", torch.tensor(d))
        # cov. matrix for prior on mu
        self.register_buffer("ibetaI", torch.eye(dim)*1/beta)
        # zero-vectors for location of normal dists.
        self.register_buffer("dzero", torch.zeros(dim))
        self.register_buffer("lzero", torch.zeros(dim-1))
        # single value
        self.register_buffer("one", torch.tensor(1))
            
    def model(self, data):
        # sample from priors
        mu = pyro.sample(
            "mu", dist.MultivariateNormal(self.dzero, self.ibetaI)
        )
        tau = pyro.sample("tau", dist.Gamma(self.c, self.d))
        with pyro.plate("q", self._dim-1):
            alpha = pyro.sample("alpha", dist.Gamma(self.a, self.b))
        with pyro.plate("d", self._dim):
            W = pyro.sample(
                "W", dist.Normal(0., 1./alpha).to_event(1)
            )
        with pyro.plate("data", data.shape[0]):
            x = pyro.sample(
                "x", dist.Normal(self.lzero, 1.0).to_event(1)
            )
            pyro.sample(
                "obs", dist.Normal(x.matmul(W.t()) + mu, 1/tau).to_event(1),
                obs=data
            )

    def guide(self, data):
        """implements the variational distribution proposed
        in "Variational Principal Components" (Bishop, 1999)
        """
        eye = torch.ones(self._dim).to(data.device)
        # Q(mu) ~ N(m_mu, sigma_mu)
        mu_loc = pyro.param("mu_loc", lambda : self.dzero)
        mu_scl = pyro.param("mu_scl", lambda : eye,
                            constraint=constraints.positive)
        mu = pyro.sample("mu", dist.Normal(mu_loc, mu_scl).to_event(1))
        # Q(tau) ~ Gamma(a_tau, b_tau)
        a_tau = pyro.param("a_tau", lambda : self.one*0.01,
                           constraint=constraints.positive)
        b_tau = pyro.param("b_tau", lambda : self.one*0.01,
                           constraint=constraints.positive)
        tau = pyro.sample("tau", dist.Gamma(a_tau, b_tau))
        # Q(alpha) ~ PI Gamma(a_alpha, b_alpha,i)
        with pyro.plate("q", self._dim-1):
            a_alpha = pyro.param("a_alpha", lambda : self.one*0.01,
                                 constraint=constraints.positive)
            b_alpha = pyro.param("b_alpha", lambda : self.one*0.01,
                                 constraint=constraints.positive)
            alpha = pyro.sample("alpha", dist.Gamma(a_alpha, b_alpha))
        # Q(W) ~ PI Normal(m_wk, sigma_w)
        with pyro.plate("d", self._dim):
            mu_w = pyro.param("mu_w", lambda : self.lzero)
            loc_w = pyro.param("loc_w", lambda : self.one,
                               constraint=constraints.positive)
            W = pyro.sample("W", dist.Normal(mu_w, loc_w).to_event(1))
        # Q(X)
        with pyro.plate("data", data.shape[0]):
            mu_x = pyro.param("mu_x", lambda : self.lzero)
            loc_x = pyro.param("loc_x", lambda : self.one,
                               constraint=constraints.positive)
            x = pyro.sample("x", dist.Normal(mu_x, loc_x).to_event(1))
        
    def fit_map(self, data, n_iter, opt_pars={'lr' : 0.005}):
        guide = pyro.infer.autoguide.AutoDelta(self.model)
        pyro.clear_param_store()
        adam = pyro.optim.Adam(opt_pars)
        svi = pyro.infer.SVI(self.model, guide, adam,
                             loss=pyro.infer.Trace_ELBO())
        losses = []
        with tqdm(total=n_iter) as pbar:
            for n in range(n_iter):
                loss = svi.step(data)
                pbar.update(1)
                if n % 50:
                    pbar.set_postfix({'elbo' : loss})
                losses.append(loss)
        return guide, torch.tensor(losses)
