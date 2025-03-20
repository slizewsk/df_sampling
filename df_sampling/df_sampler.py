from .core_imports import np,pd,plt,sns,time,interp1d,uniform

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
    def __init__(self, params):
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
