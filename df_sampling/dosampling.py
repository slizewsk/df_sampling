from .core_imports import np,pd,plt,sns,time,dataclass,interp1d,uniform,pareto, gammaf,hyp2f1,os
# from .draw_rs import draw_r
# from .peak_like import get_thresh
# from .acc_rej import sample_ar
def objective_function(y, r, params, verbose=False):
    logdf = params.logdf
    vr, vt = y
    if verbose: print(f'rmin={params.rmin},total vel={vr**2 + vt**2},vbound={params.vbound(r)**2}')
    if vr**2 + vt**2 > params.vbound(r)**2: return -np.inf
    if r < params.rmin:  return -np.inf
    if vt < 0.0: return -np.inf  # Large penalt y for invalid values
    lvt = np.log(vt)  # Logarithm of tangential velocity
    ldf = logdf(np.array([vr, vt, r]))  # Log of DF
    if verbose: print(lvt,ldf)
    return -(lvt + ldf)  

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
        os.makedirs(self.output_dir, exist_ok=True)

        # Compute critical density
        self.pcrit = 3 * (self.H0**2) / (8 * np.pi * self.G)

        # Compute virial quantities
        self.rvir = self.calculate_rvir()
        self.Mvir = self.calculate_Mvir()

    # def calculate_rvir(self):
    #     """Calculate virial radius (stub function, replace with actual calculation)."""
    #     return (self.G * self.Mvir / self.H0**2) ** (1/3)

    # def calculate_Mvir(self):
    #     """Calculate virial mass (stub function, replace with actual calculation)."""
    #     return (4/3) * np.pi * self.rvir**3 * self.pcrit * self.overdens

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
    
    def __post_init__(self):
        """Post-initialization for derived attributes."""
        # self.Mtot /= 2.325e-3

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
        os.makedirs(self.output_dir, exist_ok=True)
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

    def rho(self, r):
        """Density function."""
        return self.Mtot * self.a / (2 * np.pi * r * (r + self.a)**3)

    def dlnrho_dr(self, r):
        """ Derivative of log(rho) with respect to r """
        return -(3 * r + self.a) / (r + self.a)
    
    def dphi_dr(self, r):
        """First derivative of the potential with respect to radius."""
        return  self.Mtot / (r + self.a)**2
    
    @property
    def pars(self):
        return [self.Mtot, self.a, self.beta]


class DataSampler:
    """
    A class to store and implement draws from a phase-space distribution function,
    f(E,L), mainly relying on the rejection sampling method.

    This class has methods to draw positions, calculate thresholds, sample velocities, 
    and generate a DataFrame of observed data based on the sampling process.

    Attributes:
        params (Params): A `Params` object containing the parameters for the distribution function and other calculations.
        pars (list): A list of parameters extracted from the `params` object, used in various calculations.
        obsdf (pandas.DataFrame): A DataFrame to store the sampled positions and velocities (r, vr, vt, vtheta, vphi, v).
        calc_beta (float or None): The calculated value of velocity anisotropy (beta), or None if not computed yet.

    Args:
        params (Params): A `Params` object that contains the necessary parameters for all the functions and methods.

    Methods:
        draw_positions(n=None, verbose=False): Draw random positions based on the specified or default parameters.
        calculate_thresholds(rvals, verbose=False, max_retries=5): Calculate thresholds for the sampled positions.
        sample_velocities(rvals, threshs, verbose=False): Sample velocities based on the drawn positions and thresholds.
        create_dataframe(obs): Generate a pandas DataFrame from the sampled data.
        compute_beta(): Compute the velocity anisotropy (beta) from the observed data.
        run_sampling(n=None, verbose=False): Execute full sampling pipeline and produce observed data.
        plot_rvcurve(): Plot the galactocentric distance vs total speed from the sampled data.
        plot_stats(): Plot various statistical analyses, such as histograms and KDEs, of the sampled data.
    """
    def __init__(self, params: Params):
        self.params = params
        self.obsdf = None
        self.calc_beta = None
        self.thresh_interp = None
        self.eq_obs = None
    def relE(self,y):
        vr, vt, r = y
        e = np.abs(self.params.phi(r)) - (vr**2 + vt**2)/2
        if e <= 0:return -np.inf    
        return e
    
    def draw_positions(self, n=None, verbose=False):
        """
        Generate n sampled radial positions according to the model's draw_r method

        Args: 
            n (int): How many rs to sample.
            verbose (Boolean): Print progress or not. was using this for tracking how long things took 

        Returns: 
            ndarray of length n containing r values (floats)
        """
        if verbose:
            print("Starting to draw positions...")
        if n is None:
            n = int(self.params.nsim)
        start_time = time.time()
        rvals = self.params.draw_r(n=n)
        dur = time.time() - start_time
        if verbose:
            print(f"Time to draw positions: {dur:.5f} seconds")
        return rvals

    def get_interp_thresh(self, plot=True, verbose=False):
        """
        Generates and calculates a function of the threshold as a function of r using grid search.
        """
        if verbose:
            print("Getting interpolated threshold function...")
        eps = 1e-8
        r_vals = np.logspace(np.log(self.params.rmin+eps), np.log(self.params.rmax-eps), 70)
        threshs = []
        for idx, rtest in enumerate(r_vals):
            vmax = self.params.vbound(rtest)  
            vr_vals = np.linspace(-vmax+eps, vmax-eps, 500)
            vt_vals = np.linspace(eps, vmax-eps, 500)
            vr_mesh, vt_mesh = np.meshgrid(vr_vals, vt_vals, indexing='ij')
            
            logdf_vals = np.array([[self.params.logdf([vr, vt, rtest]) if vt > 0 else -np.inf for vt in vt_vals] for vr in vr_vals])
            weighted_logdf_vals = np.log(vt_mesh) + logdf_vals  # Compute log(v_t * DF)
            # thresh,thresh_vels = get_thresh(rtest, params)  
            # print(np.nanmax(weighted_logdf_vals))
            threshs.append(np.nanmax(weighted_logdf_vals))
        # Interpolate the threshold function
        thresh_interp = interp1d(r_vals, threshs, kind='linear', fill_value="extrapolate")
        if plot:
            plt.plot(r_vals, threshs, label='Thresholds')
            plt.plot(r_vals, thresh_interp(r_vals), label='Interpolated Fit', linestyle='--')
            plt.xlabel('r')
            plt.ylabel('Threshold')
            plt.legend()
            plt.show()

        self.thresh_interp = thresh_interp
        return thresh_interp

    def calculate_thresholds(self,rvals,verbose=False):
            if self.thresh_interp is None:
                fcn = self.get_interp_thresh(plot=verbose,verbose=verbose)
            else:
                fcn = self.thresh_interp
            return fcn(rvals)
    
    def sample_ar(self, r, thresh, verbose=False,testing=False):
        """
        Sample velocities using the acceptance-rejection method with uniform proposals.

        Args:
            r (float): The radial position for the sample.
            thresh (float): The threshold value for the rejection criterion.
            verb (bool): Whether to print debug information.
            testing (bool): Whether to return all attempts 

        Returns:
            list: The accepted sample of [r, vr, vt, vtheta, vphi], or None if no sample is accepted.
        """
        samples = None
        ghs = []
        fvs = []
        yvs = []
        objs = []
        attempts = 0
        while samples is None and attempts < 10000:
            attempts += 1
            eps = 1e-8
            vmax = self.params.vbound(r) 
            vr_gen = uniform.rvs(-vmax + eps, 2*vmax - eps)
            vt_max = np.sqrt(2*np.abs(self.params.phi(r)) - vr_gen**2)
            # vt_max = np.sqrt(np.clip(2 * self.params.phi(r) - vr_gen**2, eps, None))
            vt_gen = uniform.rvs(loc=eps, scale=vt_max)

            v_gen = np.sqrt(vr_gen**2 + vt_gen**2)
            eta = np.arctan(vt_gen/vr_gen)
            coseta = np.cos(eta) if vr_gen > 0 else -np.cos(eta)
            psia = uniform.rvs(0, 2 * np.pi)
            v_theta = vt_gen * np.cos(psia)
            v_phi = vt_gen * np.sin(psia)

            gen_height = np.log(uniform.rvs()) + thresh
            y = [vr_gen, vt_gen, r]
        
            func_value = self.params.logdf(y) + np.log(vt_gen)

            ghs.append(gen_height)
            fvs.append(func_value)
            yvs.append(y)
            
            if verbose:
                print(f'vr is {vr_gen}')
                print(f'gen height {gen_height}')
                print(f'fcn height {func_value}')        

            if gen_height <= func_value:
                samples = [r, vr_gen, vt_gen, v_theta, v_phi]
                break
        if testing:
            return samples, ghs, fvs, yvs, objs
        else:
            return samples

    def sample_velocities(self, rvals, threshs, verbose=False):
        """
        For a given distance and corresponding max(vt*df), or an array of rvals and threshs, sample velocities with accept-reject algorithm sample_ar. 

        Args: 
            rvals: Positions drawn from draw_positions() or draw_r()
            threshs: Max objective fcn corresponding to rvals calculated with calculate_thresholds() or get_thresh()
            
        Returns: 
            obs: Array of (r,vr,vt) for all len(rvals)
        """

        if verbose:
            print("Starting velocity sampling...")
        start_time = time.time()
        obs = []

        for i in range(len(rvals)):
            if verbose and i % 20 == 0:  # Log every 20th iteration
                print(f"Sampling velocity for r={rvals[i]:.2f} and thresh={threshs[i]:.2f}...")
            try:
                # Sample velocity using sample_ar function
                sampled_velocities = self.sample_ar(rvals[i], threshs[i],verbose)
                obs.append(sampled_velocities)
            except Exception as e:
                # Catch any error from sample_ar and print it
                print(f"Error sampling velocities at index {i} (r={rvals[i]}): {e}")
                obs.append([np.nan, np.nan, np.nan, np.nan, np.nan])  # Fallback values

        obs = np.array(obs)
        dur = time.time() - start_time
        if verbose:
            print(f"Time to sample velocities: {dur:.5f} seconds")
        return obs

    def create_dataframe(self, obs, verbose=False, error_perc=None):
        """
        Reworks sampled observations (r, vr, vt) into a pandas dataframe for easy calling. 
        Updates self attribute to save this obsdf.
        
        Args:
            obs: Array of (r, vr, vt, vtheta, vphi) from the sampling process
            error_pct: Dictionary containing percentage error values for 'r', 'vr', 'vt'
        """
        
        # Default percentage error if none are provided
        if error_perc is None:
            error_perc = {
                'r_err_pct': 0.01,  # Default 5% error for r
                'vr_err_pct': 0.01,  # Default 5% error for vr
                'vt_err_pct': 0.01   # Default 5% error for vt
            }
            
        if verbose:
            print("Creating DataFrame...")
            
        # Create the dataframe from the observations
        self.obsdf = pd.DataFrame({
            'r': obs[:, 0],
            'vr': obs[:, 1],
            'vt': obs[:, 2],
            'vtheta': obs[:, 3],
            'vphi': obs[:, 4]
        })
        if self.params.tilde_E is not None:
            self.obsdf["tilde_E"] = self.obsdf.apply(lambda row: self.params.tilde_E([row.vr, row.vt, row.r]), axis=1)

        # Calculate total velocity v
        self.obsdf['v'] = np.sqrt(self.obsdf.vr**2 + self.obsdf.vt**2)
        
        # Calculate errors as percentage of r, vr, and vt
        self.obsdf['r_err'] = self.obsdf['r'] * error_perc['r_err_pct']
        self.obsdf['vr_err'] = np.abs(self.obsdf['vr']) * error_perc['vr_err_pct']
        self.obsdf['vt_err'] = self.obsdf['vt'] * error_perc['vt_err_pct']

        return self.obsdf

    def compute_beta(self, verbose=False):
        """
        Calculate velocity anisotropy from a set of observations stored in self.obsdf. 
        Must be used after sampling velocities and creating dataframe of observations. 
        """

        if verbose:
            print("Computing beta...")
        computed_beta = 1 - ((np.var(self.obsdf.vtheta, ddof=1) + np.var(self.obsdf.vphi, ddof=1)) /
                            (2 * np.var(self.obsdf.vr, ddof=1)))
        if verbose:
            print(f'true/calc beta: {self.params.beta:.2f} / {computed_beta:.2f}')
        self.calc_beta = computed_beta

    def run_sampling(self, n=None, verbose=False):
        """
        Implements full pipeline of generating samples from the DF. 
        Starts with drawing positions, then getting the thresholds, sampling velocities, and converting to a DF. 
        Args: 
            n (int): How many rs to sample.
            verbose (Boolean): Print progress or not 
            
        Returns: 
            ndarray of length n containing r values (floats)
        """

        if verbose:
            print("Running the full sampling process...")
        rvals = self.draw_positions(n, verbose)
        threshs = self.calculate_thresholds(rvals)
        obs = self.sample_velocities(rvals, threshs, verbose)
        self.create_dataframe(obs, verbose)
        self.compute_beta(verbose)
        return self.obsdf
    
    def make_obs(self):
        if self.obsdf is None:
            print('Need to generate samples first, call .run_sampling()')
        eq_obs = mockobs(self.obsdf,self)
        self.eq_obs = eq_obs
        return eq_obs
    
    def plot_rvcurve(self):
        """
        Convenience function to scatter plot the samples' positions and velocities, compared to the escape curve of the model. 
        Meant for easy check to confirm only bound samples. 
        """

        if self.obsdf is None:
            print("Error: Sampling not completed. Cannot plot data.")
            return
        if self.obsdf is not None:
            print("Plotting radial-velocity curve...")
        fig, axs = plt.subplots(figsize=(6, 3))
        rs = np.logspace(-3, 3.5, 300)
        axs.plot(rs, self.vbound(rs), label=None, color='orange', ls='--')
        axs.scatter(self.obsdf.r, self.obsdf.v, marker='o', s=5, alpha=0.4, label='Sampled Points', rasterized=True)
        axs.set_xscale('log')
        axs.set_xlabel('Galactocentric Distance [kpc]')
        axs.set_ylabel('Total Speed [100 km/s]')
        plt.show()

    def plot_stats(self):
        """
        Convenience function to look at each marginal distribution of sampled velocities and distances. 
        Plots kde for rs, then histograms for the veloctieis. 
        """

        if self.obsdf is None:
            print("Error: Sampling not completed. Cannot plot data.")
            return
        if self.obsdf is not None:
            print("Plotting statistics...")
        fig, axs = plt.subplots(1, 2, figsize=(7, 3))
        sns.kdeplot(x=self.obsdf.r, ax=axs[0])
        sns.histplot(x=self.obsdf.vr, ax=axs[1], label=r'$vr$')
        sns.histplot(x=self.obsdf.vtheta, ax=axs[1], label=r'$v_\theta$')
        sns.histplot(x=self.obsdf.vphi, ax=axs[1], label=r'$v_\phi$')
        axs[0].set_xlim(1e-5, 5e3)
        axs[0].set_xlabel('Galactocentric Distance [kpc]')
        axs[1].set_xlabel('Speed [100 km/s]')
        plt.legend()
        plt.show()
