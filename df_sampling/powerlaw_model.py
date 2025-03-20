from .core_imports import np,dataclass,pareto, gammaf,os

@dataclass
class Params:
    """
    Class to represent the physical parameters for the power-law model.
    """
    # Model Params
    phi0: float             # Gravitational potential scale
    gamma: float            # Potential power-law slope
    alpha: float            # Tracer density power-law slope
    beta: float             # Velocity anisotropy parameter
    nsim: int               # Number of observations to sample

    stan_model_path: str = 'Stanmodels/model-vrvt.stan'
    params_list: list = None
    param_labels: list = None
    bounds: list = None
    prior_params: list = None
    test_dir: str = 'test_dir'  

    # Physical constants
    rmin: float = 1e-4      # Minimum radius (prevent r=0)
    rmax: float = 1e5       # Maximum radius (avoid unreasonable r)
    H0: float = 0.678e-3    # Hubble constant 
    overdens: float = 200.  # Overdensity for virial properties
    G: float = 4.302e-6     # Gravitational Constant in kpc,km/s,Msun
    saving: bool = True

    def __post_init__(self):
        """Post-initialization for default lists and derived attributes."""
        # Set default lists inside __post_init__
        if self.params_list is None:
            self.params_list = ['p_phi0', 'p_gamma', 'p_beta', 'M200']
        if self.param_labels is None:
            self.param_labels = [r'$\Phi_0$', r'$\gamma$',  r'$\beta$', r'$M_{200}$']
        if self.bounds is None:
            self.bounds = [[0, 120], [0, 1], [-0.5, 0.9], [0.2, 1.8]]
        if self.prior_params is None:
            self.prior_params = ['prior_phi0', 'prior_gamma', 'prior_beta', 'M200_prior']
        self.prior_dict = {
            'pg': {'means': np.array([self.phi0, self.gamma]), 
                   'cov': np.array([[40, 0.23], [0.23, 0.01]])},
            'alpha': {'mean': self.alpha, 'sigma': 0.4, 'min': 3, 'max': np.inf},
            'beta': {'mean': self.beta, 'sigma': 0.21, 'min': -np.inf, 'max': 1.0},
        }

        # Derived parameters
        self.stan_alpha = self.alpha
        self.output_dir = os.path.join(
            self.test_dir,
            f"deason_truea{self.alpha}/nsim{self.nsim}/stana{self.stan_alpha}_b{self.beta}"
        )

        # Ensure output directory exists
        if self.saving: os.makedirs(self.output_dir, exist_ok=True)

        # Compute critical density
        self.pcrit = 3 * (self.H0**2) / (8 * np.pi * self.G)

        # Compute virial quantities
        self.rvir = self.calculate_rvir()
        self.Mvir = self.calculate_Mvir()

    def Mr(self, r):
        return 2.325e-3 * self.gamma * self.phi0 * r**(1 - self.gamma)
    
    def phi(self, r):        
        return self.phi0 / r**self.gamma
    
    def relE(self,y):
        vr, vt, r = y
        e = np.abs(self.phi(r)) - (vr**2 + vt**2)/2
        if e <= 0:return -np.inf    
        return e
    
    def df(self, y):
        vr, vt, r = y  # Radial velocity, tangential velocity, and distance
        v2 = vr**2 + vt**2
        # e = self.phi0 / r**self.gamma - v2 / 2.
        e = self.relE(y)
        if e <= 0:
            return -np.inf    
        L = r * vt
        n1 = L**(-2 * self.beta)
        n2 = e**((self.beta * (self.gamma - 2)) / self.gamma + self.alpha / self.gamma - 3 / 2)
        n3 = gammaf(self.alpha / self.gamma - 2 * self.beta / self.gamma + 1)
        d1 = np.sqrt(8 * np.pi**3 * 2**(-2 * self.beta))
        d2 = self.phi0**(-2 * self.beta / self.gamma + self.alpha / self.gamma)
        d3 = gammaf((self.beta * (self.gamma - 2) / self.gamma) + self.alpha / self.gamma - 1 / 2)
        d4 = gammaf(1 - self.beta)
        return (n1 * n2 * n3) / (d1 * d2 * d3 * d4)    
    
    def logdf(self, y):
        return np.log(self.df(y))
    
    def calculate_rvir(self):
        return (self.gamma * self.phi0 / (100 * self.H0**2))**(1 / (self.gamma + 2))
    
    def calculate_Mvir(self):
        rvir = self.calculate_rvir()
        return 2.325e-3 * self.gamma * self.phi0 * rvir**(1 - self.gamma)
    
    def draw_r(self, n=1, rmax=1e4):
        eta = self.alpha - 3
        rs = []
        while len(rs) < n:
            r = pareto.rvs(eta, scale=self.rmin)
            if self.rmin <= r <= self.rmax:
                rs.append(r)  
        return rs
    
    def vcirc(self, r):
        return np.sqrt(r * self.phi(r))
    
    def vbound(self, r):
        return np.sqrt(2 * self.phi(r))
    
    @property
    def pars(self):
        return [self.phi0, self.gamma, self.alpha, self.beta, self.Mvir]
