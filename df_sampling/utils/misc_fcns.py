
from ..core_imports import np, plt, sns, linregress

def plot_success(all_results_df,true_alpha,param_labels,num_simulations,ntracers,save=True,show=False,output_dir=''):
    colors = {"p_phi0": "deepskyblue", "p_gamma": "orangered", "p_beta": "orange", "M200": "darkviolet"}
    linestyles = {"p_phi0": "-", "p_gamma": "-.", "p_beta": ":", "M200": "--"}
    line_widths = {"p_phi0": 1.8, "p_gamma": 1.7, "p_beta": 1.75, "M200": 2}  # Adjust widths for visibility

    param_mapping = {
        "p_phi0": r"$\phi_0$",
        "p_gamma": r"$\gamma$",
        "p_beta": r"$\beta$",
        "M200": r"$M_{200}$"}
    plt.figure(figsize=(6, 3))
    for param in all_results_df['param'].unique():
        subset = all_results_df[all_results_df['param'] == param]
        sns.lineplot(
            data=subset,
            x="stan_alpha",
            y="success_percentage",
            label=param_mapping.get(param, param),  # Use mapping for labels
            color=colors.get(param, "black"),  # Assign color
            linestyle=linestyles.get(param, "-"),  # Assign linestyle
            lw=line_widths.get(param, 1.5)  # Assign line width
        )

    plt.axvline(x=true_alpha, color="gray", linestyle="--", alpha=0.7, linewidth=1)
    plt.xlabel(r"Model fixed $\alpha$")
    plt.ylabel("Success Percentage")
    plt.title(r" True $\alpha$ = %.2f Model Performance" % (true_alpha))
    plt.legend(title="Parameter", loc="center", bbox_to_anchor=(1.11, 0.5))
    plt.tight_layout()
    if show:
        plt.show()
    if save:
        plt.savefig(output_dir+'success_rate_plot.png', dpi=300)
    plt.close()

def plot_biases(all_biases_df,true_alpha,param_labels,num_simulations,ntracers,save=True,show=False,output_dir=''):
    colors = {"p_phi0": "deepskyblue", "p_gamma": "orangered", "p_beta": "orange", "M200": "darkviolet"}
    linestyles = {"p_phi0": "-", "p_gamma": "-.", "p_beta": ":", "M200": "--"}
    line_widths = {"p_phi0": 1.8, "p_gamma": 1.7, "p_beta": 1.75, "M200": 2}  # Adjust widths for visibility

    param_mapping = {
        "p_phi0": r"$\phi_0$",
        "p_gamma": r"$\gamma$",
        "p_beta": r"$\beta$",
        "M200": r"$M_{200}$"}
    plt.figure(figsize=(6, 3))    
    for param in all_biases_df['param'].unique():
        subset = all_biases_df[all_biases_df['param'] == param]
        sns.lineplot(
            data=subset,
            x="stan_alpha",
            y="bias",
            label=param_mapping.get(param, param),  # Use mapping for labels
            color=colors.get(param, "black"),  # Assign color
            linestyle=linestyles.get(param, "-"),  # Assign linestyle
            lw=line_widths.get(param, 1.5)  # Assign line width
        )

    plt.axvline(x=true_alpha, color="gray", linestyle="--", alpha=0.7, linewidth=1)
    plt.axhline(0, color="gray", linestyle="--", alpha=0.7, linewidth=1)
    plt.xlabel(r"Model fixed $\alpha$")
    plt.ylabel("Bias")
    plt.title(r" True $\alpha$ = %.2f Model Bias Analysis" % (true_alpha))
    plt.legend(title="Parameter", loc="center", bbox_to_anchor=(1.11, 0.5))
    plt.tight_layout()
    if show:
        plt.show()
    if save:
        plt.savefig(output_dir+'biases_plot.png', dpi=300)
    plt.close()
def plot_hdi_results(all_summaries, params, pars, param_labels, N_runs, RCUT, confidence_level=95,
                     modelname='Non-observational, FIXED-ALPHA',fname=None,save=False,show=False):
    """
    Function to plot HDI results for multiple parameters with dynamic credible intervals.

    Parameters:
    - all_summaries: pd.DataFrame with summary statistics (mean, HDI bounds).
    - params: List of parameter names (strings) to plot.
    - pars: List of true parameter values corresponding to the params.
    - param_labels: List of labels for parameters for display (in LaTeX format).
    - N_runs: Number of simulation runs.
    - RCUT: A cut-off value for the radial distance (used in the title).
    - confidence_level: The desired HDI confidence level (65 or 95).
    """
    
    # Adjusting the number of subplots dynamically based on the number of parameters
    fig, axes = plt.subplots(1, len(params), figsize=(4 * len(params), 4), tight_layout=True)
    axes = axes.flatten()

    # Define the HDI bounds based on the confidence level
    if confidence_level == 95:
        lower_bound_col = 'hdi_2.5%'  # 95% lower bound
        upper_bound_col = 'hdi_97.5%'  # 95% upper bound
    else:  # Default to 65% credible interval
        lower_bound_col = 'hdi_17.5%'  # 65% lower bound
        upper_bound_col = 'hdi_82.5%'  # 65% upper bound

    for i, (param, ax) in enumerate(zip(params, axes)):  
        # Extract mean and intervals for the specific parameter across all runs
        means = all_summaries.loc[:, 'mean'].xs(param, level='Parameter')
        lower_bounds = all_summaries.loc[:, lower_bound_col].xs(param, level='Parameter')
        upper_bounds = all_summaries.loc[:, upper_bound_col].xs(param, level='Parameter')

        # Call the function to get the success percentage and formatted success text
        success_percentage, success_text = calculate_hdi_success(pars[i], lower_bounds, upper_bounds, N_runs)

        # Plot each run's mean and credible interval, color-coded based on if the true parameter is within the HDI
        for run in range(N_runs):
            # Check if the true parameter is within the HDI for this run
            if pars[i] >= lower_bounds[run] and pars[i] <= upper_bounds[run]:
                # Run where true parameter is within the HDI
                ax.plot([lower_bounds[run], upper_bounds[run]], [run, run], color='teal', label=f'{confidence_level}% C.I.' if run == 0 else "", alpha=0.5)
                ax.scatter(means[run], run, color='green', label='Mean' if run == 0 else "")
            else:
                # Run where true parameter is NOT within the HDI
                ax.plot([lower_bounds[run], upper_bounds[run]], [run, run], color='orange', label=f'{confidence_level}% C.I. (Not within HDI)' if run == 0 else "", alpha=0.5)
                ax.scatter(means[run], run, marker='^',color='red', label='Mean (Not within HDI)' if run == 0 else "")

        # Add combined mean and CI for all runs
        combined_mean = means.mean()
        combined_lower = lower_bounds.min()
        combined_upper = upper_bounds.max()

        ax.plot([combined_lower, combined_upper], [N_runs, N_runs], color='purple', label=f'Combined {confidence_level}% C.I.', alpha=0.5)
        ax.scatter(combined_mean, N_runs, color='maroon', label='Combined Mean')

        ax.set_yticks(range(N_runs + 1))
        ax.set_yticklabels([f' {j+1}' for j in range(N_runs)] + ['Combined'], rotation=0, fontsize=11)

        # Set x-axis label using the LaTeX expression from the param_labels array
        ax.set_xlabel(param_labels[i])

        # Set the title with the LaTeX expression for the parameter name and the success count
        ax.set_title(f'True {param_labels[i]} = {pars[i]:.2f}\n{success_text}', fontsize=12)
        # print(param_labels[i],success_percentage)
        ax.axvline(pars[i], color='red', linestyle='--')
        if i >= 0: ax.set_yticklabels([])  

    # Adjust legend and layout
    fig.suptitle(f'{modelname} with r>{RCUT} kpc \n {confidence_level}% HDI of parameters across {N_runs} runs', fontsize=14)
    if save: fig.savefig(f'{fname}.pdf',dpi=300)
    if show:plt.show()
    plt.close()

def calculate_hdi_success(true_param, lower_bounds, upper_bounds, N_runs,verbose=False,shorttext=False):
    """
    Helper function to calculate the success percentage of the true parameter being within the HDI.

    Parameters:
    - true_param: The true value of the parameter being evaluated.
    - lower_bounds: The lower bounds of the HDI for each simulation run.
    - upper_bounds: The upper bounds of the HDI for each simulation run.
    - N_runs: The number of simulation runs.

    Returns:
    - success_percentage: The percentage of runs where the true parameter is within the HDI.
    - success_text: A formatted string with the success count and percentage.
    """
    # Count how many times the true parameter falls within the HDI
    
    count_within_hdi = sum((true_param >= lower_bounds) & (true_param <= upper_bounds))
    success_percentage = (count_within_hdi / N_runs) * 100
    if verbose:     
        print(f"Success Percentage: {success_percentage:.0f}%")
    if shorttext:
        success_text = f'{success_percentage:.0f}%'
    else:
        success_text = f'{count_within_hdi}/{N_runs}'

    return success_percentage, success_text

def plot_kde(params_samples, prior_samples, param_labels, bounds, true_values, colors=('plum', 'lightcoral', 'limegreen'), fname='testfname',figsize=(7, 4), show=False,save=True,title=None):
    num_params = len(params_samples)
    fig, axs = plt.subplots(1, num_params, figsize=figsize)
    axs = axs.flatten()

    for i in range(num_params):
        # Plot KDEs for prior and posterior samples
        sns.kdeplot(prior_samples[i], ax=axs[i], color=colors[0], label='Prior')
        sns.kdeplot(params_samples[i], ax=axs[i], color=colors[1], label='Posterior')

        # Calculate and plot vertical lines for medians
        prior_median = np.nanmedian(prior_samples[i])
        posterior_median = np.nanmedian(params_samples[i])
        true_value = true_values[i]

        axs[i].axvline(prior_median, ls='--', color=colors[0], alpha=0.9)
        axs[i].axvline(posterior_median, ls='--', color=colors[1], alpha=0.9)
        axs[i].axvline(true_value, color=colors[2], alpha=0.9,label='Truth')

        # Add text annotations for the medians and true value
        # axs[i].text(0.95, 0.9, f'P: {posterior_median:.2f}', color=colors[1], ha='right', va='center', transform=axs[i].transAxes)
        # axs[i].text(0.95, 0.8, f'T: {true_value:.2f}', color=colors[2], ha='right', va='center', transform=axs[i].transAxes)
        axs[i].set_title( f'P: {posterior_median:.2f}, T: {true_value:.2f}')
        # Set axis labels and bounds
        axs[i].set_xlabel(param_labels[i])
        axs[i].set_xlim(*bounds[i])
        axs[i].set_ylabel('')
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=False, fancybox=False)
    for j in range(num_params, len(axs)):
        fig.delaxes(axs[j])

    if title:
        fig.suptitle(title, y=1.06)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if show: 
        plt.show()
    if save: fig.savefig(f'{fname}.pdf',dpi=300)
    plt.close()


def plot_single_param(param_samples, prior_samples, label, bounds, true_value, colors=('plum', 'lightcoral', 'limegreen'),prior_only=False):
    fig, ax = plt.subplots(figsize=(4, 3))

    # Calculate medians
    prior_median = np.nanmedian(prior_samples)
    posterior_median = np.nanmedian(param_samples)

    # Plot KDEs for prior and posterior samples
    sns.kdeplot(prior_samples, ax=ax, color=colors[0], label='Prior')
    if not prior_only: 
        sns.kdeplot(param_samples, ax=ax, color=colors[1], label='Posterior')
        ax.axvline(posterior_median, ls='--', color=colors[1], alpha=0.8, label=f' Post: {posterior_median:.2f}')

    # Plot vertical lines for medians and true value
    ax.axvline(prior_median, ls='--', color=colors[0], alpha=0.8,label=f' Prior: {prior_median:.2f}')
    ax.axvline(true_value, color=colors[2], alpha=0.9, label=f'Truth: {true_value:.2f}')

    # Set labels and bounds
    ax.set_xlabel(label)
    ax.set_xlim(*bounds)
    ax.set_ylabel('Density')

    # Add legend and show plot
    ax.legend()
    plt.tight_layout()
    plt.show()

    # make sure y is velocity first then radius

def estimate_alpha(r_values,true_alpha,rmin=1e-5,plot=False):
    bins = np.logspace(np.log(np.nanmin(r_values)), np.log(np.nanmax(r_values)), num=25)
    counts, edges = np.histogram(r_values, bins=bins)
    # Calculate midpoints and volumes of each shell
    bin_midpoints = (edges[:-1] + edges[1:]) / 2
    bin_volumes = (4/3) * np.pi * (edges[1:]**3 - edges[:-1]**3)
    # Estimate density in each shell
    density_estimates = counts / bin_volumes
    # Filter out zero counts to avoid log issues
    valid_indices = density_estimates > 0
    bin_midpoints = bin_midpoints[valid_indices]
    density_estimates = density_estimates[valid_indices]
    log_r = np.log(bin_midpoints)
    log_density = np.log(density_estimates)
    slope, intercept, r_value, p_value, std_err = linregress(log_r, log_density)
    alpha_fit = -slope
    perc_corrs = ((alpha_fit-true_alpha)/true_alpha)
    alphavals = (alpha_fit)
    if plot: 
        fig, ax= plt.subplots(2, 1, figsize=(4, 4), sharex=True, sharey=False, tight_layout=True)
        sns.scatterplot(x=log_r, y=log_density,c='grey',alpha=0.5,ax=ax[0])    
        sns.lineplot(x=log_r, y=slope * log_r + intercept, label=r'Fit $\alpha$ = %.2f'%(alpha_fit),ax=ax[0])
        ax[0].set_xlim(np.log(rmin),4)
        sns.kdeplot(np.log(r_values),alpha=0.5,ax=ax[1])
        sns.lineplot(x=log_r, y=-true_alpha * log_r + intercept, c='green',ls='--', label=r'True $\alpha$ = %.2f'%true_alpha,ax=ax[0])
        ax[1].set_xlabel('log(r)')
        ax[0].set_ylabel('log(Density)')
        ax[1].set_ylabel('Density of r')
        ax[0].set_title('Log-Log Plot of Density vs. r')
        print(f'Average alpha: {np.mean(alphavals)}')
        print(f'Averge error {np.nanmean(perc_corrs):.2f}%')
        plt.show()
        plt.close()
    return alpha_fit

def quad_form_diag(correlation_matrix, errors):
    # Convert correlation matrix to covariance matrix
    diag_errors = np.diag(errors)
    covariance_matrix = diag_errors @ correlation_matrix @ diag_errors
    return covariance_matrix
def is_positive_definite(matrix):
    try:
        # Perform Cholesky decomposition
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False
