
from .core_imports import np,dataclass,interp1d,uniform, gammaf,hyp2f1,os

@dataclass
class ParamsHernquist:
    """
    Class to represent Hernquist potential parameters in physical units.
    """
    Mtot: float             # Total mass [in 1e12 Msun]
    a: float                # Scale radius [in kpc]
    beta: float             # Velocity anisotropy parameter (constant)
    nsim: int               # Number of observations 
    
    stan_model_path: str = 'Stanmodels/hernq-vrvt.stan' # Stan file path
    params_list: list = None                            # List of parameters
    params_true: list = None
    param_labels: list = None                           # Plotting labels
    bounds: list = None                                 # Parameter bounds
    test_dir: str = 'test_dir'                          # Output directory for results
    
    # Physical constants
    rmin: float = 1e-4      # Minimum radius (prevent r=0)
    rmax: float = 1e5       # Maximum radius (avoid unreasonable r)
    H0: float = 0.678e-3    # Hubble constant 
    overdens: float = 200.  # Overdensity for virial properties
    G: float = 4.302e-6     # Gravitational Constant in kpc, km/s, Msun
    saving: bool = True

    def __post_init__(self):
        """Post-initialization for derived attributes.""" 
        if self.params_list is None:
            self.params_list = ['p_Mtot', 'p_a', 'p_beta']
        if self.params_true is None:
            self.params_true = [self.Mtot,self.a,self.beta]
        if self.param_labels is None:
            self.param_labels = [r'$M_{tot}$', r'$a$', r'$\beta$']
        if self.bounds is None:
            # self.bounds = [[1e-4, 4], [1, 50], [-0.7, 0.7]]
            self.bounds = [[1e-4, 2000], [1, 50], [-0.7, 0.7]]
        self.output_dir = os.path.join(self.test_dir, f"M{self.Mtot:.2e}_a{self.a:.2f}_n{self.nsim}_b{self.beta:.3f}")
        
        if self.saving: os.makedirs(self.output_dir, exist_ok=True)
        self.pcrit = 3 * (self.H0**2) / (8 * np.pi * self.G)
        self.prior_dict = {
            'p_Mtot': {'mean': self.Mtot, 'sigma': 100, 'min': 1e-4},
            'p_arad': {'mean': self.a, 'sigma': 4, 'min': 3, 'max': np.inf},
            'p_beta': {'mean': self.beta, 'sigma': 0.3, 'min': -np.inf, 'max': 1.0},
        }

    def phi(self, r):      
        """ Hernquist potential. """
        return -self.Mtot / (r + self.a) 
    
    def Mr(self, r):
        """ Mass enclosed within radius r. """
        return self.Mtot * r**2 / (r + self.a)**2
    
    def rM(self, M):
        """ Inverse mass function to solve for r given M. """
        return (2 * self.a * M + np.sqrt((2 * self.a * M)**2 + 4 * (self.Mtot - M) * M * self.a**2))\
              / (2 * (self.Mtot - M))
    
    def draw_r(self, n=1, kind='analytic'):
        """ Sample radii from the Hernquist profile. """
        if kind == 'numeric':
            r_vals = np.logspace(np.log(self.rmin), np.log(self.rmax), 1050)
            M_vals = self.Mr(r_vals) / self.Mr(self.rmax)
            inv_M = interp1d(M_vals, r_vals, kind='linear', fill_value="extrapolate")
            return inv_M(uniform.rvs(0, 1, n))
        elif kind == 'analytic':
            M_max = self.Mr(self.rmax)
            return np.array([self.rM(M) for M in uniform.rvs(0, M_max, n)])
        else:
            raise ValueError("Invalid sampling kind. Choose 'numeric' or 'analytic'.")
    
    def psi(self,r):
        return np.abs(self.phi(r))
    def vcirc(self, r):
        """ Circular velocity at radius r. """
        return np.sqrt(r * self.psi(r))  

    def vbound(self, r):
        """ Escape velocity at radius r. """
        return np.sqrt(2 * self.psi(r))
    
    def relE(self,y):
        vr, vt, r = y
        e = self.psi(r) - (vr**2 + vt**2)/2
        if e <= 0:return -np.inf    
        return e
    
    def tilde_E(self,y):
        relE = self.relE(y)
        return relE/(self.Mtot/self.a)
    
    def df_iso_hernq(self,y):
        """ Isotropic Hernquist potential (meant for checking main DF to ensure converge to isotropic case)"""
        tilde_E = self.tilde_E(y)
        # tilde_E = self.relE(y)
        t1 = ((-self.phi(0))*self.phi(self.a)/np.sqrt(2.)/(2*np.pi)**3/((self.M*self.a)**1.5))
        t2 = (np.sqrt(tilde_E)/(1-tilde_E)**2.)
        t3 = ((1.-2.*tilde_E)*(8.*tilde_E**2.-8.*tilde_E-3.)+\
              ((3.*np.arcsin(np.sqrt(tilde_E)))\
               /np.sqrt(tilde_E*(1.-tilde_E))))
        fE = t1*t2*t3
        return fE
    
    def df(self, y):
        """ Constant anisotropy distribution function for Hernquist model. """
        vr, vt, r = y
        E = self.tilde_E(y)
        beta = self.beta
        if E <= 0:
            return -np.inf  
        L = r/self.a * vt/np.sqrt(self.Mtot/self.a)
        t1 = 2**beta / (2*np.pi)**(5/2) * E**(5/2-beta)
        t2 = gammaf(5-2*beta)/(gammaf(1-beta)*(gammaf(7/2-beta)))
        t3 = hyp2f1(5 - 2 * beta, 1 - 2 * beta, 7/2 - beta, E)
        fE = t1*t2*t3
        # checked with Hernq paper, Bovy Galaxies book, Baes paper, 
        #       Cuddeford paper, Eddington paper, Binney Tremaine 
        return fE * L**(-2*beta)
     
    def logdf(self, y):
        """ Logarithm of the distribution function self.df """
        return np.log(self.df(y))
    
    # def sigma_r2(self,r):
    #     """ Radial velocity dispersion"""
    #     beta = self.beta
    #     a = self.a
    #     # if beta == 1/2:
    #     #     return 1/4*1/(1+r)
    #     if beta == 0.0:
    #         t0 = self.Mtot/(12*a)
    #         t1 = 12*r*(r+a)**3/a**4 * np.log((r+a)/r)
    #         t2 = -r/(r+a)*(25 + 52*r/a+42*(r/a)**2+12*(r/a)**3)
    #         return t0*(t1 + t2)
    #     else:
    #         t1 = self.Mtot*(r/a)**(1-2*beta) * (1+(r/a))**3
    #         t2 = betainc(5-2*beta,2*beta+1e-10,1/(1+(r/a)))
    #         return t1*t2

    # def rho(self, r):
    #     """Density function."""
    #     return self.Mtot * self.a / (2 * np.pi * r * (r + self.a)**3)

    # def dlnrho_dr(self, r):
    #     """ Derivative of log(rho) with respect to r """
    #     return -(3 * r + self.a) / (r + self.a)
    
    # def dphi_dr(self, r):
    #     """First derivative of the potential with respect to radius."""
    #     return  self.Mtot / (r + self.a)**2
    
    @property
    def pars(self):
        return [self.Mtot, self.a, self.beta]
