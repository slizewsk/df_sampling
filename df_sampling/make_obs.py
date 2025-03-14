from .core_imports import np,pd,uniform
from .misc_fcns import calc_EL,quad_form_diag,is_positive_definite
from .coord_transforms import sph2cart_pos, gc2eq_pos, gc2eq_vel
# from .draw_rs import draw_r
# from .acc_rej import sample_ar
# from .peak_like import get_thresh

def mockobs(samps_df,params,angles=None):
    tab = pd.DataFrame()
    for i in samps_df.index:
        vr = samps_df.vr[i]
        vt = samps_df.vt[i]
        r = samps_df.r[i]

        tab.loc[i,'r'] = r
        tab.loc[i,'vr'] = vr
        tab.loc[i,'vt'] = vt
        tab.loc[i,'v'] = np.sqrt((vr)**2+(vt)**2)
    
        if angles is None:
            up = uniform.rvs(0,1)
            theta = np.arccos(1-2*up)
            phi = uniform.rvs(0,2*np.pi)
        else:
            theta,phi = samps_df.theta,samps_df.phi

        rtp = np.array([r,theta,phi])
        vel = np.array([vr, vt*np.cos(phi), vt*np.sin(phi)])
        tab.loc[i,'vrgc'] = vel[0]
        tab.loc[i,'vtgc'] = vel[1]
        tab.loc[i,'vthgc'] = vel[2]
        
        tab.loc[i,'theta_gc'] = theta
        tab.loc[i,'phi_gc'] = phi
        
        xyz_gc = sph2cart_pos(rtp)
        tab.loc[i,'xgc'] = xyz_gc[0]
        tab.loc[i,'ygc'] = xyz_gc[1]
        tab.loc[i,'zgc'] = xyz_gc[2]

        pos_eq = gc2eq_pos(xyz_gc)
        tab.loc[i,'plx'] = (1/(pos_eq[0]))
        tab.loc[i,'distance'] = pos_eq[0]
        tab.loc[i,'ra'] = pos_eq[1]
        tab.loc[i,'dec'] = pos_eq[2]

        vel_eq = gc2eq_vel(vel_gc=vel,pos_gc=rtp)
        tab.loc[i,'vlos'] = vel_eq[0]
        tab.loc[i,'pmra'] = vel_eq[1]
        tab.loc[i,'pmdec'] = vel_eq[2]
        
        tab.loc[i,'E'] = params.relE(np.array([vr,vt,r]))
        tab.loc[i,'L'] = r*vt

        vtan_eqs = (4.740470463533349 * np.sqrt(vel_eq[1]**2 + vel_eq[2]**2)* pos_eq[0])
        tab.loc[i,'vtan_eq'] = vtan_eqs
    return tab


# def sample_and_create_df(nsim, alpha, rmin, rmax, pars, save=False, fname='test-fname'):
    # Sample radial positions
    drawn_rs = draw_r(alpha=alpha, rmin=rmin, rmax=rmax, n=nsim)
    # Sample the corresponding observations for each radial position
    observations = np.array([sample_ar(r, get_thresh(r, pars, verb=False), pars, verb=False) for r in drawn_rs])
    # Create DataFrame from the sampled observations
    df = pd.DataFrame({'r': observations[:, 0], 'vr': observations[:, 1], 'vt': observations[:, 2]})
    df['v'] = np.sqrt(df['vr'] ** 2 + df['vt'] ** 2)
    # Compute rel energy (E) and angular momentum (L) for each sample
    df[['E', 'L']] = df.apply(lambda row: calc_EL(np.array([row['vr'], row['vt'], row['r']]), pars), axis=1, result_type='expand')
    # start_time = time.time()
    # Optionally save the DataFrame to CSV
    if save:
        df.to_csv('Data/rawdraws-' + fname)
    return df



def construct_cov_matrices(obs):
    """Construct covariance matrices for positional and proper motion uncertainties."""

    # Define 2x2 covariance matrix construction
    def create_cov_matrix(corr, err1, err2):
        return quad_form_diag(np.array([[1, corr], [corr, 1]]), [err1, err2])

    # Compute covariance matrices efficiently using vectorized operations
    pm_cov_mats = np.array([
        create_cov_matrix(c, e1, e2) for c, e1, e2 in zip(obs['pmra_pmdec_corr'], obs['pmra_error'], obs['pmdec_error'])
    ])
    
    pos_cov_mats = np.array([
        create_cov_matrix(c, e1, e2) for c, e1, e2 in zip(obs['ra_dec_corr'], obs['ra_error'], obs['dec_error'])
    ])
    
    # 4x4 covariance matrices
    def construct_4x4_cov(row):
        if row['pos_obs'] and row['pm_obs']:
            corr_matrix = np.array([
                [1, row['pos_corr'], row['ra_pmra_corr'], row['ra_pmdec_corr']],
                [row['pos_corr'], 1, row['dec_pmra_corr'], row['dec_pmdec_corr']],
                [row['ra_pmra_corr'], row['dec_pmra_corr'], 1, row['pm_corr']],
                [row['ra_pmdec_corr'], row['dec_pmdec_corr'], row['pm_corr'], 1]
            ])
            error_vector = [row['ra_error'], row['dec_error'], row['pmra_error'], row['pmdec_error']]
            return quad_form_diag(corr_matrix, error_vector)
        return np.eye(4)

    pos_pm_cov_mats = np.stack(obs.apply(construct_4x4_cov, axis=1))

    # Identify bad covariance matrices (not positive definite)
    bad_idx = np.where([not is_positive_definite(cov) for cov in pos_pm_cov_mats])[0].tolist()

    return pm_cov_mats, pos_cov_mats, pos_pm_cov_mats, bad_idx


def generate_random_values(obs, results, cols):
    """Generate random values for observational uncertainties using Gaussian noise."""
    for col in cols:
        valid_bins = obs['r_gc_bin'].isin(results.index)
        means = obs['r_gc_bin'].map(results[f'mean_{col}'])
        variances = obs['r_gc_bin'].map(results[f'var_{col}'])

        noise = np.random.normal(loc=means[valid_bins], scale=np.sqrt(variances[valid_bins]))
        obs.loc[valid_bins, col] = noise

        obs.loc[~valid_bins, col] = np.nan  # Handle missing bins
    return obs


def bin_and_aggregate_data(moredat, bin_edges):
    """Bin data radially and compute means/variances of errors."""
    labels = np.arange(1, len(bin_edges))
    moredat['r_gc_bin'] = pd.cut(moredat['rgc'], bins=bin_edges, labels=labels)

    agg_funcs = {col: ['mean', 'var'] for col in [
        'ra_error', 'dec_error', 'parallax_error', 'pmra_error', 'pmdec_error',
        'ra_dec_corr', 'ra_pmra_corr', 'ra_pmdec_corr', 'dec_pmra_corr',
        'dec_pmdec_corr', 'pmra_pmdec_corr', 'radial_velocity_error'
    ]}
    
    results = moredat.groupby('r_gc_bin').agg(agg_funcs)
    # results.columns = ['_'.join(col).strip() for col in results.columns]  # Flatten MultiIndex
    results.columns = [f"{stat}_{col}" for col, stat in results.columns]
    return results


def finalize_observations(obs, bad_idx, save=True, fname='test-fname'):
    """Filter and finalize the selected observations."""
    obs.drop(index=bad_idx, inplace=True)
    obs.dropna(inplace=True)
    print(f'{len(bad_idx)} bad indices removed, {len(obs)} observations remaining.')
    # selected = obs[obs['r'] > RCUT].sample(n=ndat, replace=False).copy()
    selected = obs.copy()
    selected.reset_index(drop=True, inplace=True)

    selected[['ra', 'dec']] = np.rad2deg(selected[['ra', 'dec']])

    if save:
        selected.to_csv(f'./Data/inc_errors_{fname}', index=False)
        print(f'Saving mock observations to ./Data/inc_errors_{fname}')
    
    return selected


# i believe these are old for when i was doing things within astropy, 
# would be called from core imports if needed
# import astropy.coordinates as Coords
# from astropy.coordinates import SkyCoord
# import astropy.units as u
