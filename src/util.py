from matplotlib import gridspec

# seaborn helpers used in original
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy.stats import beta as beta_dist

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Tuple, Union
import matplotlib as mpl
from seaborn import blend_palette, set_hls_values
import anndata

COLORS_LARGE_PALLETE = np.array([
    '#0F4A9C', '#3F84AA', '#C9EBFB', '#8DB5CE', '#C594BF', '#DFCDE4',
    '#B51D8D', '#6f347a', '#683612', '#B3793B', '#357A6F', '#989898',
    '#CE778D', '#7F6874', '#E09D37', '#FACB12', '#2B6823', '#A0CC47',
    '#77783C', '#EF4E22', '#AF1F26'
])

random_seed = 42

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def transpose_adata(adata):
    return anndata.AnnData(
        obs=adata.var,
        var=adata.obs,
        X=adata.X.T,
        uns=adata.uns,
    )

def gini(array, eps=1e-7):
    """
    Calculate the Gini coefficient of a numpy array.
    from: https://github.com/oliviaguest/gini
    """
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array = array + eps
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return (np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))


def initialize_dataframe_beta(n_individuals=10000, n_cpgs=1000, var=10, random_seed=None, beta_kwargs=None):
    """Generate population-level methylation probabilities using Beta distribution."""
    logger.info(f"Initializing dataframe with {n_individuals} individuals, {n_cpgs} CpGs, concentration={var}")
    if random_seed is not None:
        logger.debug(f"Setting random seed to {random_seed}")
        np.random.seed(random_seed)

    # Generate random means for each CpG from uniform distribution [0,1]
    mus = np.random.uniform(0, 1, n_cpgs)

    # Initialize the matrix
    matrix = np.zeros((n_individuals, n_cpgs))

    # Fill each column with Beta distribution
    for col in range(n_cpgs):
        mu = mus[col]

        # Mathematical constraint of this paramerization
        var_safe = min(var, (mu * (1 - mu))**2 - np.finfo(dtype=np.float32).eps)
        # var_safe = min(var, mu * (1 - mu) - 0.0001)

        val = beta_mean_std(mu, var_safe, size=n_individuals)

        matrix[:, col] = val

    row_names = [f"individual_{n}" for n in range(n_individuals)]
    col_names = [f"CpG_{n}" for n in range(n_cpgs)]

    df = pd.DataFrame(matrix, index=row_names, columns=col_names)
    logger.info(f"Generated dataframe with shape {df.shape}, probability range: [{df.min().min():.3f}, {df.max().max():.3f}]")

    return df, mus



def orthogonalize_power_term(
    t_linear: np.ndarray, t_power: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Orthogonalize a power age term against the linear age term.

    Removes the component of t_power that lies along t_linear, yielding a
    residual term that captures purely super-linear (non-linear) clonal
    expansion signal uncorrelated with the linear age component.

    Parameters
    ----------
    t_linear : np.ndarray
        1D array of linearly normalized age values (t_linear = (age - a) / (l - a)).
        Should be the same population-level vector used to build the design matrix.
    t_power : np.ndarray
        1D array of power-transformed age values (t_power = t_linear ** n).
        Must have the same length as t_linear.

    Returns
    -------
    t_power_orth : np.ndarray
        1D array of orthogonalized power term: t_power - proj * t_linear.
        This residual is orthogonal to t_linear and can be used as a design
        matrix column alongside t_linear without collinearity.
    proj_coef : float
        Projection coefficient: dot(t_power, t_linear) / dot(t_linear, t_linear).
        Represents how much of t_power lies along t_linear direction.

    Notes
    -----
    The orthogonalization is computed by projecting t_power onto t_linear and
    subtracting that projection:
        proj = dot(t_power, t_linear) / dot(t_linear, t_linear)
        t_power_orth = t_power - proj * t_linear

    At t_linear = 1.0 (biological lifespan boundary), t_power = 1.0 and
    proj * t_linear = 1.0, so t_power_orth = 0 at that point by construction.

    Examples
    --------
    >>> t_lin = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    >>> t_pow = t_lin ** 2.0
    >>> t_orth, proj = orthogonalize_power_term(t_lin, t_pow)
    >>> np.isclose(np.dot(t_orth, t_lin), 0.0)  # should be orthogonal
    True
    """
    proj_coef = np.dot(t_power, t_linear) / np.dot(t_linear, t_linear)
    t_power_orth = t_power - proj_coef * t_linear
    return t_power_orth, proj_coef

def transform_age(
    age: Union[float, np.ndarray, pd.Series],
    n: float = 2.0,
    a: float = 13.5,
    l: float = 122.5
) -> Union[float, np.ndarray, pd.Series]:
    """
    Transform chronological age using power-law scaling.

    Parameters
    ----------
    age : float, np.ndarray, or pd.Series
        Chronological age(s) in years
    n : float, default=2.0
        Exponent for power-law transformation
        - n=1: Linear scaling
        - n=2: Quadratic scaling (default, matches age² performance)
        - n>2: Super-exponential scaling
    a : float, default=13.5
        Age of sexual maturity (lower bound)
        When HSC pool stops rapid expansion
    l : float, default=122.5
        Maximum human lifespan (upper bound)
        Convergence point for uniclonality

    Returns
    -------
    float, np.ndarray, or pd.Series
        Transformed age values in [0, 1] range
        Returns same type as input

    Examples
    --------
    >>> transform_age(13.5, n=2, a=13.5, l=122.5)
    0.0
    >>> transform_age(122.5, n=2, a=13.5, l=122.5)
    1.0
    >>> transform_age(68.0, n=2, a=13.5, l=122.5)
    0.25

    Notes
    -----
    The transformation accounts for:
    1. Sexual maturity (a): HSC pool stabilization
    2. Maximum lifespan (l): Theoretical convergence to uniclonality
    3. Exponent (n): Rate of clonal expansion

    For n=2, matches observed quadratic relationship between age and
    variant allele frequency (VAF) in clonal hematopoiesis.
    """
    # Store input type for return
    input_type = type(age)

    # Convert to numpy for calculation
    age_array = np.asarray(age)

    # Apply transformation: ((age - a) / (l - a))^n
    age_normalized = (age_array - a) / (l - a)
    age_transformed = np.power(age_normalized, n)

    # Return in original type
    if input_type == pd.Series:
        return pd.Series(age_transformed, index=age.index)
    elif input_type == float or (isinstance(age, np.ndarray) and age.ndim == 0):
        return float(age_transformed)
    else:
        return age_transformed


def normal_clipped(size, loc, scale):
    """Sample from clipped normal distribution"""
    arr = np.random.normal(loc=loc, scale=scale, size=size)
    # return arr[np.logical_and(arr <= 1, arr >= 0)] # This does filtering, we want clipping (?)
    return np.clip(arr, 0, 1)


def beta_shift(omega=0.5, kappa=10, sigma=0.5, mu=-0.2) -> float:
    """Re-parameterized beta distribution with omega (mode) and kappa (concentration)"""
    a = omega*(kappa-2)+1
    b = (1-omega) * (kappa-2) + 1
    return sigma*beta_dist(a, b).rvs() + mu


def beta_mean_std(mu, s, size=1):
    var = s**2
    v = (mu*(1-mu) / var) - 1
    a = mu * v
    b = (1-mu) * v
    return np.random.beta(a, b, size=size)

def beta_mode_conc(omega, kappa, shift=0, size=1):
    a = omega*(kappa - 2) + 1
    b = (1-omega) * (kappa-2) + 1
    if size == 1:
        return (beta_dist(a, b).rvs(size) - shift)[0]
    else:
        return beta_dist(a, b).rvs(size) - shift


def wright_F_statistic(beta, F, **params):
    """
    Calculate genotype frequencies using Wright's F-statistic.

    Parameters:
    -----------
    beta : float or array-like
        Average number of alleles (0 to 1), where p = beta
    F : float or array-like
        Inbreeding coefficient (scalar or same shape as beta)

    Returns:
    --------
    tuple : (prob_UU, prob_MU, prob_MM)
        Frequencies of genotypes with 0, 1, and 2 alleles
        Returns arrays if beta is array-like
    """
    # Convert to arrays
    beta = np.asarray(beta)
    F = np.asarray(F)

    # Validate beta
    if np.any((beta < 0) | (beta > 1)):
        raise ValueError(f"All beta values must be between 0 and 1, got min={beta.min()}, max={beta.max()}")

    p = beta
    q = 1 - p

    # Calculate valid F range for each p value
    # At fixation (p=0 or p=1), F range is [0, 0]
    # For p <= 0.5: F_min = -p/(1-p), F_max = 1
    # For p > 0.5: F_min = -(1-p)/p, F_max = 1

    F_min = np.where(p == 0, 0.0,
                     np.where(p == 1, 0.0,
                              np.where(p <= 0.5, -p / (1 - p), -(1 - p) / p)))
    F_max = np.where((p == 0) | (p == 1), 0.0, 1.0)

    # Clip F to valid range
    F_clipped = np.clip(F, F_min, F_max)

    # Warn if clipping occurred (optional)
    if np.any(F != F_clipped):
        n_clipped = np.sum(F != F_clipped)
        if beta.shape == ():  # scalar case
            print(f"Warning: F clipped to valid range [{F_min:.4f}, {F_max:.4f}] for p={p:.4f}")
        else:
            # print(f"Warning: {n_clipped} F values were clipped to valid ranges")
            pass

    F = F_clipped

    # Calculate genotype frequencies (vectorized)
    prob_UU = q ** 2 * (1 - F) + q * F
    prob_MU = 2 * p * q * (1 - F)
    prob_MM = p ** 2 * (1 - F) + p * F

    # Sanity checks (vectorized)
    prob_sum = prob_UU + prob_MU + prob_MM
    assert np.allclose(prob_sum, 1.0, atol=1e-10), \
        f"Probabilities don't sum to 1 (range: [{prob_sum.min()}, {prob_sum.max()}])"

    assert np.all(prob_UU >= -1e-10) and np.all(prob_MU >= -1e-10) and np.all(prob_MM >= -1e-10), \
        f"Negative probability detected: UU min: {prob_UU.min()}, MU min: {prob_MU.min()}, MM min: {prob_MM.min()}"

    return prob_UU, prob_MU, prob_MM


def hardy_weinberg_probabilities(beta, **params):
    """
    Default Hardy-Weinberg probability function.

    Parameters:
    - beta: methylation level (0-1), can be scalar or array
    - params: ignored for HW (no additional parameters)

    Returns:
    - tuple: (prob_UU, prob_MU, prob_MM) where outcomes are 0.0, 0.5, 1.0 respectively
    """
    prob_UU = (1 - beta) ** 2
    prob_MU = 2 * beta * (1 - beta)
    prob_MM = beta ** 2
    return prob_UU, prob_MU, prob_MM

def plot_methylation_histogram(sim_object, timepoints, gene_mask=None, n_vafs=3,
                               title="Distribution of Population-Averaged Methylation",
                               bins=50, figsize=(10, 6), ncols=None, savepath=None):
    if sim_object.initial_methylation_matrix is None:
        print("Simulation was not initialized with an methylation matrix. Cannot plot.")
        return

    if not sim_object.time_history or len(sim_object.time_history) <= 1:
        print("Simulation has not been run long enough to generate history.")
        return

    num_timepoints = len(timepoints)

    # Calculate grid dimensions
    if ncols is None:
        # Default: try to make a roughly square grid, but favor wider layouts
        ncols = min(4, max(2, int(np.ceil(np.sqrt(num_timepoints * 1.5)))))

    nrows = int(np.ceil(num_timepoints / ncols))

    # Adjust figure size for grid layout
    adjusted_figsize = (figsize[0] * ncols / 2, figsize[1] * nrows / 2)

    fig, axes = plt.subplots(nrows, ncols, figsize=adjusted_figsize, sharex=True, sharey=True)

    # Handle the case where we have a single subplot
    if num_timepoints == 1:
        axes = np.array([axes])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    sim_times = np.array(sim_object.time_history)
    all_avg_values = []

    for i, t in enumerate(timepoints):
        # Calculate row and column indices
        row = i // ncols
        col = i % ncols

        # Get the appropriate axis
        if nrows == 1 and ncols == 1:
            ax = axes[0]
        elif nrows == 1:
            ax = axes[col]
        elif ncols == 1:
            ax = axes[row]
        else:
            ax = axes[row, col]

        # Find the closest index in history for the requested timepoint
        t_idx = np.argmin(np.abs(sim_times - t))
        actual_time = sim_times[t_idx]

        # Get the state at that time
        counts = sim_object.counts_history[t_idx]
        methylation_matrix = sim_object.get_methylation_matrix_at_time(actual_time)

        # Ensure matrix and counts have matching numbers of lineages
        num_lineages_at_t = len(counts)
        methylation_matrix = methylation_matrix[:num_lineages_at_t, :]

        # Apply gene mask
        selected_methylation = methylation_matrix[:, gene_mask] if gene_mask is not None else methylation_matrix

        if selected_methylation.shape[1] == 0:
            print(f"Timepoint {t}: No genes selected by the mask. Skipping.")
            continue

        if np.sum(counts) > 0:
            avg_methylation_per_gene = np.average(
                selected_methylation, axis=0, weights=counts
            )
        else:
            # Handle case where all counts are zero to avoid division error
            avg_methylation_per_gene = np.mean(selected_methylation, axis=0)

        all_avg_values.append(avg_methylation_per_gene)

        sns.histplot(avg_methylation_per_gene, bins=bins, kde=True, ax=ax)

        # Calculate statistics on the per-gene averages
        mean_of_averages = np.mean(avg_methylation_per_gene)
        ax.axvline(mean_of_averages, color='red', linestyle='--', label=f'Mean: {mean_of_averages:.3f}')

        # plot_subtitle = (f"at Year {actual_time:.1f} ({num_lineages_at_t} lineages, "
        #                  f"{len(avg_methylation_per_gene)} CpGs)")
        all_vafs = sim_object.get_all_vafs_at_timepoint(t)[0]
        ind = np.argpartition(all_vafs, -n_vafs)[-n_vafs:]
        print(all_vafs[ind])
        max_vafs = all_vafs[ind].round(3)
        plot_subtitle = (f"Year {actual_time:.1f}: Max VAF(s): {max_vafs}")
        ax.set_title(plot_subtitle)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(num_timepoints, nrows * ncols):
        row = i // ncols
        col = i % ncols
        if nrows == 1 and ncols == 1:
            continue
        elif nrows == 1:
            axes[col].set_visible(False)
        elif ncols == 1:
            axes[row].set_visible(False)
        else:
            axes[row, col].set_visible(False)

    # Set labels only on edge subplots
    for i in range(nrows):
        for j in range(ncols):
            if nrows == 1 and ncols == 1:
                ax = axes[0]
            elif nrows == 1:
                ax = axes[j]
            elif ncols == 1:
                ax = axes[i]
            else:
                ax = axes[i, j]

            # Y-label on leftmost column
            if j == 0:
                ax.set_ylabel("Frequency")

            # X-label on bottom row
            if i == nrows - 1:
                ax.set_xlabel("Average Methylation State (Methylation β)")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(title, fontsize=16, y=0.99)

    # Set x-axis limits for all subplots
    for ax in axes.flat:
        if ax.get_visible():
            ax.set_xlim(0, 1)
    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    else:
        plt.show()

    return all_avg_values

def verify_hardy_weinberg(methylation_probs_df, methylation_dict, timepoint=0, sample_size=1000):
    """
    Verify that the generated states follow Hardy-Weinberg equilibrium.

    Parameters:
    - methylation_probs_df: original probability DataFrame
    - methylation_dict: dictionary of methylation tensors by timepoint
    - timepoint: which timepoint to verify (default: 0)
    - sample_size: number of random (individual, CpG) pairs to test
    """
    n_individuals, n_cpgs = methylation_probs_df.shape
    methylation_tensor = methylation_dict[timepoint]
    n_cells = methylation_tensor.shape[0]

    # Randomly sample individual-CpG pairs
    np.random.seed(42)  # For reproducible verification
    test_individuals = np.random.randint(0, n_individuals, sample_size)
    test_cpgs = np.random.randint(0, n_cpgs, sample_size)

    deviations = []

    for i in range(sample_size):
        ind_idx = test_individuals[i]
        cpg_idx = test_cpgs[i]

        # Expected probabilities
        p = methylation_probs_df.iloc[ind_idx, cpg_idx]
        expected_UU = (1 - p) ** 2
        expected_MU = 2 * p * (1 - p)
        expected_MM = p ** 2

        # Observed frequencies
        states = methylation_tensor[:, ind_idx, cpg_idx]
        observed_UU = np.mean(states == 0.0)
        observed_MU = np.mean(states == 0.5)
        observed_MM = np.mean(states == 1.0)

        # Calculate deviation from expected
        deviation = abs(observed_UU - expected_UU) + abs(observed_MU - expected_MU) + abs(observed_MM - expected_MM)
        deviations.append(deviation)

    mean_deviation = np.mean(deviations)
    print(f"\nHardy-Weinberg verification:")
    print(f"Mean absolute deviation from expected frequencies: {mean_deviation:.6f}")
    print(f"Standard deviation of deviations: {np.std(deviations):.6f}")

    # Show some examples
    print("\nExample comparisons (first 5):")
    for i in range(5):
        ind_idx = test_individuals[i]
        cpg_idx = test_cpgs[i]
        p = methylation_probs_df.iloc[ind_idx, cpg_idx]
        states = methylation_tensor[:, ind_idx, cpg_idx]

        expected = [(1 - p) ** 2, 2 * p * (1 - p), p ** 2]
        observed = [np.mean(states == 0.0), np.mean(states == 0.5), np.mean(states == 1.0)]

        print(f"Individual {ind_idx}, CpG {cpg_idx} (p={p:.3f}):")
        print(f"  Expected: UU={expected[0]:.3f}, MU={expected[1]:.3f}, MM={expected[2]:.3f}")
        print(f"  Observed: UU={observed[0]:.3f}, MU={observed[1]:.3f}, MM={observed[2]:.3f}")


def plot_methylation_analysis(methylation_probs_df, methylation_dict, timepoint=0, individual_idx=None):
    """
    Create visualization comparing population probabilities to cellular states.

    Parameters:
    - methylation_probs_df: DataFrame with population methylation probabilities
    - methylation_dict: dictionary of methylation tensors by timepoint
    - timepoint: which timepoint to visualize (default: 0)
    - individual_idx: specific individual to analyze (default: None for population-level analysis)
    """
    analysis_type = f"Individual {individual_idx}" if individual_idx is not None else "Population"

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Distribution of methylation probabilities
    if individual_idx is not None:
        # Focus on specific individual's probabilities
        individual_probs = methylation_probs_df.iloc[individual_idx, :].values
        axes[0, 0].hist(individual_probs, bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title(f'Individual {individual_idx}: Methylation Probabilities')
    else:
        # Population-level probabilities
        axes[0, 0].hist(methylation_probs_df.values.flatten(), bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title('Population Methylation Probabilities')

    axes[0, 0].set_xlabel('Methylation Probability')
    axes[0, 0].set_ylabel('Frequency')

    # Plot 2: Distribution of cellular states
    methylation_tensor = methylation_dict[timepoint]

    if individual_idx is not None:
        # Focus on specific individual's cellular states
        individual_states = methylation_tensor[:, individual_idx, :].flatten()
        unique_states, counts = np.unique(individual_states, return_counts=True)
        axes[0, 1].bar(unique_states, counts, alpha=0.7, color=['red', 'orange', 'green'])
        axes[0, 1].set_title(f'Individual {individual_idx}: Cellular Methylation States')
    else:
        # Population-level cellular states
        all_states = methylation_tensor.flatten()
        unique_states, counts = np.unique(all_states, return_counts=True)
        axes[0, 1].bar(unique_states, counts, alpha=0.7, color=['red', 'orange', 'green'])
        axes[0, 1].set_title('Population: Cellular Methylation States')

    axes[0, 1].set_xlabel('State (0.0=UU, 0.5=MU, 1.0=MM)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_xticks([0.0, 0.5, 1.0])

    # Plot 3: Mean cellular methylation vs population probability
    # Calculate mean methylation across cells for each individual-CpG pair
    mean_cellular = np.mean(methylation_tensor, axis=0)  # Average across cells

    if individual_idx is not None:
        # Focus on specific individual
        pop_probs_flat = methylation_probs_df.iloc[individual_idx, :].values
        mean_cellular_flat = mean_cellular[individual_idx, :]

        # Sample for plotting if too many CpGs
        if len(pop_probs_flat) > 5000:
            sample_indices = np.random.choice(len(pop_probs_flat), 5000, replace=False)
            pop_probs_sample = pop_probs_flat[sample_indices]
            mean_cellular_sample = mean_cellular_flat[sample_indices]
        else:
            pop_probs_sample = pop_probs_flat
            mean_cellular_sample = mean_cellular_flat

        axes[1, 0].scatter(pop_probs_sample, mean_cellular_sample, alpha=0.6, s=2)
        axes[1, 0].set_title(f'Individual {individual_idx}: Population Prob vs Mean Cellular State')
    else:
        # Population-level analysis
        pop_probs_flat = methylation_probs_df.values.flatten()
        mean_cellular_flat = mean_cellular.flatten()

        # Sample for plotting (too many points otherwise)
        sample_indices = np.random.choice(len(pop_probs_flat), 5000, replace=False)
        axes[1, 0].scatter(pop_probs_flat[sample_indices], mean_cellular_flat[sample_indices],
                           alpha=0.5, s=1)
        axes[1, 0].set_title('Population: Prob vs Mean Cellular State')

    axes[1, 0].plot([0, 1], [0, 1], 'r--', alpha=0.8)
    axes[1, 0].set_xlabel('Population Methylation Probability')
    axes[1, 0].set_ylabel('Mean Cellular State')

    # Plot 4: Example individual across CpGs
    display_individual = individual_idx if individual_idx is not None else 0
    cpg_range = range(min(50, methylation_probs_df.shape[1]))

    pop_probs = methylation_probs_df.iloc[display_individual, cpg_range]
    mean_states = np.mean(methylation_tensor[:, display_individual, cpg_range], axis=0)

    axes[1, 1].scatter(cpg_range, pop_probs, alpha=0.7, label='Population Prob', s=20)
    axes[1, 1].scatter(cpg_range, mean_states, alpha=0.7, label='Mean Cell State', s=20)
    axes[1, 1].set_title(f'Individual {display_individual}: First 50 CpGs')
    axes[1, 1].set_xlabel('CpG Index')
    axes[1, 1].set_ylabel('Methylation Level')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


def plot_beta_histogram(methylation_dict, cpg_mask, title="Beta Values Distribution",
                        bins=30, figsize=(10, 6), timepoints=None, individual_idx=None):
    """
    Plot histogram of beta values (average methylation) for selected CpG sites.
    If multiple timepoints exist, plots them as subplots in one column.

    Parameters:
    - methylation_dict: dictionary where keys are timepoints and values are 3D arrays (cells x individuals x CpGs)
    - cpg_mask: 1D boolean array indicating which CpG sites to include
    - title: title for the histogram
    - bins: number of bins for histogram
    - figsize: figure size tuple (will be adjusted for multiple timepoints)
    - timepoints: list of timepoints to plot (default: all available timepoints)
    - individual_idx: specific individual to analyze (default: None for population-level analysis)

    Returns:
    - beta_values: array or list of arrays of calculated beta values for the selected CpGs
    """

    # Determine timepoints to plot
    if timepoints is None:
        timepoints = sorted(methylation_dict.keys())
    num_timepoints = len(timepoints)

    # Adjust figure size for multiple subplots
    if num_timepoints > 1:
        adjusted_figsize = (figsize[0], figsize[1] * num_timepoints * 0.8)
    else:
        adjusted_figsize = figsize

    # Create subplots
    fig, axes = plt.subplots(num_timepoints, 1, figsize=adjusted_figsize,
                             sharex=True, sharey=True)

    # If only one timepoint, make axes a list for consistent handling
    if num_timepoints == 1:
        axes = [axes]

    beta_values_all = []

    for i, t in enumerate(timepoints):
        # Get data for this timepoint
        methylation_data = methylation_dict[t]  # shape: (cells, individuals, CpGs)

        # Select CpG sites based on mask
        selected_data = methylation_data[:, :, cpg_mask]  # shape: (cells, individuals, selected_CpGs)

        # Calculate beta values (mean across cells) for each individual-CpG pair
        beta_values = np.mean(selected_data, axis=0)  # shape: (individuals, selected_CpGs)

        # Focus on specific individual if requested
        if individual_idx is not None:
            if individual_idx >= beta_values.shape[0]:
                raise IndexError(f"Individual index {individual_idx} out of range")
            beta_values_flat = beta_values[individual_idx, :]  # shape: (selected_CpGs,)
            analysis_label = f"Individual {individual_idx}"
        else:
            # Population-level analysis (flatten all individuals)
            beta_values_flat = beta_values.flatten()
            analysis_label = f"Population ({beta_values.shape[0]} individuals)"

        beta_values_all.append(beta_values_flat)

        # Create histogram for this timepoint
        ax = axes[i]
        sns.histplot(beta_values_flat, bins=bins, kde=True, alpha=0.7, ax=ax)

        # Add summary statistics
        mean_beta = np.mean(beta_values_flat)
        std_beta = np.std(beta_values_flat)
        ax.axvline(mean_beta, color='red', linestyle='--', alpha=0.8,
                   label=f'Mean: {mean_beta:.3f}')

        # Set labels and title for this subplot
        ax.set_ylabel('Frequency')
        ax.set_title(
            f'{title} - Timepoint {t}\n{analysis_label} - ({np.sum(cpg_mask)} CpG sites, {len(beta_values_flat)} data points)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Print statistics for this timepoint
        print(f"Timepoint {t} - Beta value statistics:")
        print(f"  Mean: {mean_beta:.4f}")
        print(f"  Std:  {std_beta:.4f}")
        print(f"  Min:  {np.min(beta_values_flat):.4f}")
        print(f"  Max:  {np.max(beta_values_flat):.4f}")
        print()

    # Set x-label only on the bottom subplot
    axes[-1].set_xlabel('Beta Value')

    if num_timepoints > 1:
        fig.suptitle(f'{title} - Time Series Comparison', fontsize=14, y=0.98)

    plt.tight_layout()
    plt.xlim((0, 1))
    plt.show()

    # Return single array if one timepoint, list if multiple
    if num_timepoints == 1:
        return beta_values_all[0]
    else:
        return beta_values_all


def generate_diploid_methylation_states(methylation_probs_df,
                                        distribution_func=hardy_weinberg_probabilities,
                                        n_cells=100, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    n_individuals, n_cpgs = methylation_probs_df.shape
    beta = methylation_probs_df  # (n_individuals, n_cpgs)

    prob_UU, prob_MU, prob_MM = distribution_func(beta)  # each (n_individuals, n_cpgs)

    # Random samples: (n_cells, n_individuals, n_cpgs)
    random_vals = np.random.random(size=(n_cells, n_individuals, n_cpgs))

    # Broadcast prob arrays to (1, n_individuals, n_cpgs)
    prob_UU = prob_UU[np.newaxis, :, :]
    prob_MU = prob_MU[np.newaxis, :, :]

    # Assign states
    methylation_tensor = np.zeros((n_cells, n_individuals, n_cpgs))
    methylation_tensor[random_vals >= prob_UU] = 0.5
    methylation_tensor[random_vals >= (prob_UU + prob_MU)] = 1.0

    return methylation_tensor


def color_nonempty_edges(hb, color, linewidth=0.5):
    counts = hb.get_array()

    face_colors = hb.get_facecolors()
    # If matplotlib collapsed to a single color, broadcast it
    if len(face_colors) == 1:
        face_colors = np.tile(face_colors, (len(counts), 1))
    else:
        face_colors = face_colors.copy()

    edge_colors = []
    alphas = np.ones_like(counts)
    for i, c in enumerate(counts):
        if c > 0:
            edge_colors.append(mpl.colors.to_rgba(color))
        else:
            edge_colors.append((0, 0, 0, 0))
            face_colors[i, :] = [0]  # set face alpha to 0
            alphas[i] = 0

    hb.set_alpha(alphas)
    hb.set_linewidths(linewidth)
    hb.set_edgecolors(np.array(edge_colors))

def color_histbars_from_hexbin(ax_hist, hb, min_age=0.0, max_age=np.inf):
    cmap = hb.cmap
    offsets = hb.get_offsets()
    counts = hb.get_array()
    hex_x = offsets[:, 0]
    hex_y = offsets[:, 1]

    # Restrict to the relevant age band globally
    age_mask = (hex_y >= min_age) & (hex_y <= max_age)
    total_in_band = counts[age_mask].sum()

    # First pass: for each bar, sum counts in [x_left, x_right] within age band
    bar_counts = []
    for patch in ax_hist.patches:
        x_left = patch.get_x()
        x_right = x_left + patch.get_width()
        mask = age_mask & (hex_x >= x_left) & (hex_x < x_right)
        bar_counts.append(counts[mask].sum() if mask.any() else 0)

    # Normalize as fraction of total band counts → [0, 1]
    bar_counts = np.array(bar_counts, dtype=float)
    if total_in_band > 0:
        bar_counts /= total_in_band

    # Second pass: apply colors
    for patch, intensity in zip(ax_hist.patches, bar_counts):
        patch.set_facecolor(cmap(intensity))
        patch.set_edgecolor('gray')


def plot_age_dist(scatter_values, top_hist_values, bottom_hist_values,
                  max_age_young, min_age_old, bins=np.linspace(0, 1, 32),
                  kind='hex',
                  xlabel_young="Beta (≤21)", xlabel_old="Beta (≥75)",
                  title=""):
    from matplotlib import gridspec

    fig = plt.figure(figsize=(5, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])
    axes = np.array([fig.add_subplot(gs[row]) for row in range(3)])

    # --- Row 0: Young histogram (top) ---
    sns.histplot(x=top_hist_values, ax=axes[0], bins=bins, stat='density', palette='c0')
    axes[0].set_xlim(0, 1)
    axes[0].xaxis.set_ticks_position('top')
    axes[0].xaxis.set_label_position('top')
    axes[0].set_xlabel(xlabel_young, labelpad=8)
    fig.text(0.5, 0.935, title, ha='center', fontsize=13, fontweight='bold')

    # --- Row 1: Hexbin (middle) ---
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(13.5, 100)
    axes[1].invert_yaxis()
    axes[1].tick_params(axis='x', labelbottom=False, labeltop=False)
    axes[1].set_xlabel("")
    axes[1].set_ylabel('Age (years)')

    for rect_kwargs in [
        dict(xy=(0, 0), width=1, height=max_age_young),
        dict(xy=(0, min_age_old), width=1, height=100),
    ]:
        axes[1].add_patch(Rectangle(**rect_kwargs, linewidth=1, edgecolor=None,
                                     facecolor='lightgray', alpha=0.33, zorder=1))

    axes[1].axline((0, min_age_old), (1, min_age_old), color='#8B0000', linestyle='dashed', linewidth=0.75)
    axes[1].axline((0, max_age_young), (1, max_age_young), color='#8B0000', linestyle='dashed', linewidth=0.75)

    color = "C0"
    color_rgb = mpl.colors.colorConverter.to_rgb(color)
    colors = [set_hls_values(color_rgb, l=val) for val in np.linspace(1, 0, 12)]
    cmap = blend_palette(colors, as_cmap=True)

    if kind == 'hex':
        hb = axes[1].hexbin(x=scatter_values[:, 0], y=scatter_values[:, 1],
                            cmap=cmap, gridsize=len(bins), linewidths=0,
                            extent=[0, 1, 13.5, 100], zorder=3)
    elif kind == 'scatter':
        hb = axes[1].hexbin(x=scatter_values[:, 0], y=scatter_values[:, 1],
                    cmap=cmap, gridsize=len(bins), linewidths=0,
                    extent=[0, 1, 13.5, 100], zorder=3)
        hb.set_alpha(0)

        axes[1].scatter(x=scatter_values[:, 0], y=scatter_values[:, 1],
                        marker='+', alpha=0.5,
                        zorder=3)

    # --- Row 2: Old histogram (bottom) ---
    sns.histplot(x=bottom_hist_values, ax=axes[2], bins=bins, stat='density')
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 15)
    axes[2].invert_yaxis()
    axes[2].set_xlabel(xlabel_old, labelpad=8)
    fig.text(0.5, 0.03, "", ha='center', fontsize=13, fontweight='bold')

    if kind == 'hex':
        color_nonempty_edges(hb, 'gray')

    color_histbars_from_hexbin(axes[0], hb, max_age=max_age_young)
    color_histbars_from_hexbin(axes[2], hb, min_age=min_age_old)

    max_density = np.max(
        [p.get_height() for p in axes[0].patches] +
        [p.get_height() for p in axes[2].patches]
    )
    axes[0].set_ylim(0, max_density * 1.05)
    axes[2].set_ylim(max_density * 1.05, 0)

    plt.subplots_adjust(hspace=0)
    plt.show()


def _build_design_row(age: float, female: float,
                      a_maturity: float, l_max: float,
                      n_power: float,
                      orth_coef: float | None = None,
                      t_linear_ref: np.ndarray | None = None) -> np.ndarray:
    """
    Build a single-row design vector [1, t_lin, t_pow_orth, female].
    orth_coef is the scalar projection coefficient used during training.
    If not available it is approximated from t_linear_ref.
    """
    t_lin = float(np.clip((age - a_maturity) / (l_max - a_maturity), 0, 1))
    t_pow = t_lin ** n_power
    # orthogonalise using the population-level projection coefficient
    if orth_coef is not None:
        t_pow_orth = t_pow - orth_coef * t_lin
    else:
        t_pow_orth = t_pow  # fallback – small error at extreme ages
    return np.array([1.0, t_lin, t_pow_orth, float(female)])


def _predict_ab(model, cpg_col_idx: int, X_row: np.ndarray):
    """Return (alpha, beta_param) for a single design row and CpG column."""
    g_alpha = model.gamma_alpha[:, cpg_col_idx]   # shape (4,)
    g_beta  = model.gamma_beta[:, cpg_col_idx]
    log_a   = float(X_row @ g_alpha)
    log_b   = float(X_row @ g_beta)
    return np.exp(np.clip(log_a, -10, 10)), np.exp(np.clip(log_b, -10, 10))


def _predicted_mean_var(a: float, b: float):
    mu  = a / (a + b)
    var = a * b / ((a + b) ** 2 * (a + b + 1))
    return mu, var


def _beta_pdf_curve(a: float, b: float, x: np.ndarray) -> np.ndarray:
    """Evaluate Beta(a, b) PDF over x."""
    return beta_dist.pdf(x, a, b)


def plot_age_dist_model(
    adata,
    model,
    cpg_name: str,
    max_age_young: float = 35,
    min_age_old:   float = 70,
    a_maturity:    float = 20.0,
    l_max:         float = 90.0,
    n_power:       float = 2.0,
    bins:          np.ndarray = np.linspace(0, 1, 32),
    kind:          str  = 'hex',
    xlabel_young:  str  = "Beta (young)",
    xlabel_old:    str  = "Beta (old)",
    title:         str  = "",
    female_val:    float = 0.5,   # sex value used for predicted curve (0/1/0.5)
    n_age_curve:   int  = 200,    # resolution of predicted mean/variance ribbon
):
    """
    Plot observed methylation with BetaRegressionAlphaBeta model predictions.

    Parameters
    ----------
    adata        : AnnData  (obs: 'age', 'female';  X: clipped methylation)
    model        : BetaRegressionAlphaBeta  (gamma_alpha / gamma_beta shape (4, n_cpgs))
    cpg_name     : CpG to inspect (must be in adata.var_names)
    max_age_young: upper age for "young" histogram
    min_age_old  : lower age for "old" histogram
    a_maturity   : age normalisation lower bound used during training
    l_max        : age normalisation upper bound used during training
    n_power      : exponent used for the power term during training
    female_val   : sex covariate used when drawing predicted curves
                   (0 = male, 1 = female, 0.5 = population average)
    """

    # ── locate CpG ────────────────────────────────────────────────────────────
    var_names = list(adata.var_names)
    if cpg_name not in var_names:
        raise ValueError(f"'{cpg_name}' not found in adata.var_names")
    cpg_col = var_names.index(cpg_name)

    # ── extract observations ──────────────────────────────────────────────────
    ages   = adata.obs['age'].values.astype(float)
    female = adata.obs['female'].values.astype(float)

    import scipy.sparse
    X_raw = adata.X
    if scipy.sparse.issparse(X_raw):
        meth = np.asarray(X_raw[:, cpg_col].todense()).ravel()
    else:
        meth = np.asarray(X_raw)[:, cpg_col]
    meth = np.clip(meth, 1e-6, 1 - 1e-6)

    young_mask = ages <= max_age_young
    old_mask   = ages >= min_age_old

    scatter_values    = np.column_stack([meth, ages])
    top_hist_values   = meth[young_mask]
    bottom_hist_values = meth[old_mask]

    t_lin_all = np.clip((ages - a_maturity) / (l_max - a_maturity), 0, 1)
    t_pow_all = t_lin_all ** n_power
    t_power_orth, orth_coef = orthogonalize_power_term(t_lin_all, t_pow_all)

    age_grid  = np.linspace(a_maturity, l_max, n_age_curve)
    pred_mu   = np.zeros(n_age_curve)
    pred_sd   = np.zeros(n_age_curve)
    pred_a_arr = np.zeros(n_age_curve)
    pred_b_arr = np.zeros(n_age_curve)

    for i, ag in enumerate(age_grid):
        xrow = _build_design_row(ag, female_val, a_maturity, l_max,
                                 n_power, orth_coef)
        a, b        = _predict_ab(model, cpg_col, xrow)
        mu, var     = _predicted_mean_var(a, b)
        pred_mu[i]  = mu
        pred_sd[i]  = np.sqrt(var)
        pred_a_arr[i] = a
        pred_b_arr[i] = b

    # ── predicted PDF at representative young / old age ───────────────────────
    rep_young = np.median(ages[young_mask]) if young_mask.any() else max_age_young
    rep_old   = np.median(ages[old_mask])   if old_mask.any()   else min_age_old

    xrow_y = _build_design_row(rep_young, female_val, a_maturity, l_max,
                                n_power, orth_coef)
    xrow_o = _build_design_row(rep_old,   female_val, a_maturity, l_max,
                                n_power, orth_coef)

    a_y, b_y = _predict_ab(model, cpg_col, xrow_y)
    a_o, b_o = _predict_ab(model, cpg_col, xrow_o)

    x_pdf   = np.linspace(0.001, 0.999, 400)
    pdf_y   = _beta_pdf_curve(a_y, b_y, x_pdf)
    pdf_o   = _beta_pdf_curve(a_o, b_o, x_pdf)

    mu_y, var_y = _predicted_mean_var(a_y, b_y)
    mu_o, var_o = _predicted_mean_var(a_o, b_o)

    # ── build figure ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(5, 12))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])
    axes = np.array([fig.add_subplot(gs[row]) for row in range(3)])

    color = "C0"
    color_rgb = mpl.colors.colorConverter.to_rgb(color)
    colors = [set_hls_values(color_rgb, l=val) for val in np.linspace(1, 0, 12)]
    cmap = blend_palette(colors, as_cmap=True)

    # ── Row 0: Young histogram + predicted PDF ────────────────────────────────
    # scale PDF so its peak matches histogram scale (density already normalised)
    ax0_twin = axes[0].twinx()
    ax0_twin.set_zorder(axes[0].get_zorder() - 1)  # twin behind
    axes[0].patch.set_visible(
        False)  # histogram axes background transparent
    ax0_twin.plot(x_pdf, pdf_y, color='#171717', lw=1.8, ls='--',
                  label=f'Predicted\n(age {rep_young:.0f}y)')
    ax0_twin.fill_between(x_pdf, pdf_y, alpha=0.25, color='#8B0000')
    ax0_twin.set_ylabel('Pred. density', fontsize=7)
    ax0_twin.tick_params(axis='y', labelcolor='gray', labelsize=6)
    ax0_twin.set_ylim(bottom=0)

    # annotate predicted moments
    axes[0].axvline(mu_y, color='black', lw=1.0, ls=':')
    axes[0].text(mu_y, axes[0].get_ylim()[0] * 0.9,
                 f'μ={mu_y:.2f}\nσ²={var_y:.4f}',
                 fontsize=7, color='black', va='top', ha='center')

    axes[0].set_xlim(0, 1)
    axes[0].xaxis.set_ticks_position('top')
    axes[0].xaxis.set_label_position('top')
    axes[0].set_xlabel(xlabel_young, labelpad=8)
    axes[0].legend(fontsize=7, loc='upper left')
    fig.suptitle(title or cpg_name, x=0.85, y=0.96, ha='center',
             fontsize=9, fontweight='bold')

    sns.histplot(x=top_hist_values, ax=axes[0], bins=bins,
                 stat='density', color='C0', alpha=1, label='Observed')



    # ── Row 1: Hexbin / scatter + predicted mean ribbon ───────────────────────
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(13.5, 100)
    axes[1].invert_yaxis()
    axes[1].tick_params(axis='x', labelbottom=False, labeltop=False)
    axes[1].set_xlabel("")
    axes[1].set_ylabel('Age (years)')

    for rect_kwargs in [
        dict(xy=(0, 0),          width=1, height=max_age_young),
        dict(xy=(0, min_age_old), width=1, height=100),
    ]:
        axes[1].add_patch(Rectangle(**rect_kwargs, linewidth=1, edgecolor=None,
                                     facecolor='lightgray', alpha=0.33, zorder=1))

    axes[1].axline((0, min_age_old), (1, min_age_old),
                   color='#8B0000', linestyle='dashed', linewidth=0.75)
    axes[1].axline((0, max_age_young), (1, max_age_young),
                   color='#8B0000', linestyle='dashed', linewidth=0.75)

    hb = axes[1].hexbin(x=scatter_values[:, 0], y=scatter_values[:, 1],
                        cmap=cmap, gridsize=len(bins), linewidths=0,
                        extent=[0, 1, 13.5, 100], zorder=3)

    if kind == 'scatter':
        hb.set_alpha(0)
        axes[1].scatter(x=scatter_values[:, 0], y=scatter_values[:, 1],
                        marker='+', alpha=0.5, zorder=3)

    # predicted mean ribbon (note: y-axis is age, x-axis is methylation)
    axes[1].plot(pred_mu, age_grid,
                 color='#c0392b', lw=2.0, zorder=5, label='Pred. mean')
    axes[1].fill_betweenx(age_grid,
                          pred_mu - pred_sd,
                          pred_mu + pred_sd,
                          color='#c0392b', alpha=0.18, zorder=4,
                          label='±1 SD')
    axes[1].legend(fontsize=7, loc='lower right')

    # ── Row 2: Old histogram + predicted PDF ──────────────────────────────────

    ax2_twin = axes[2].twinx()
    ax2_twin.set_zorder(axes[2].get_zorder() - 1)  # twin behind
    axes[2].patch.set_visible(
        False)  # histogram axes background transparent
    ax2_twin.plot(x_pdf, pdf_o, color='#171717', lw=1.8, ls='--',
                  label=f'Predicted\n(age {rep_old:.0f}y)')
    ax2_twin.fill_between(x_pdf, pdf_o, alpha=0.25, color='#8B0000')

    axes[2].axvline(mu_o, color='black', lw=1.0, ls=':')
    axes[2].text(mu_o, axes[2].get_ylim()[0] * 0.9,
                 f'μ={mu_o:.2f}\nσ²={var_o:.4f}',
                 fontsize=7, color='black', va='bottom', ha='center')

    axes[2].set_xlim(0, 1)

    axes[2].invert_yaxis()
    axes[2].set_xlabel(xlabel_old, labelpad=8)
    axes[2].legend(fontsize=7, loc='upper left')
    ax2_twin.set_ylabel('Pred. density', fontsize=7)
    ax2_twin.tick_params(axis='y', labelsize=6)
    ax2_twin.invert_yaxis()
    ax2_twin.set_ylim(top=0)

    sns.histplot(x=bottom_hist_values, ax=axes[2], bins=bins,
                 stat='density', color='C0', alpha=1, label='Observed')
    # ── post-render coloring (matches original helpers) ───────────────────────
    if kind == 'hex':
        color_nonempty_edges(hb, 'gray')

    color_histbars_from_hexbin(axes[0], hb, max_age=max_age_young)
    color_histbars_from_hexbin(axes[2], hb, min_age=min_age_old)

    max_density = np.max(
        [p.get_height() for p in axes[0].patches] +
        [p.get_height() for p in axes[2].patches]
    )

    axes[0].set_ylim(0, max_density * 1.05)
    ax0_twin.set_ylim(axes[0].get_ylim())
    axes[2].set_ylim(max_density * 1.05, 0)
    ax2_twin.set_ylim(axes[2].get_ylim())

    plt.subplots_adjust(hspace=0)
    plt.tight_layout()
    plt.show()
    return fig


def inspect_top_cpgs(results_df, adata, model, score_col='clonality_index',
                     n=6, ascending=False, **kwargs):
    """
    Loop over the top-n CpGs ranked by score_col and call plot_age_dist_model.
    kwargs are forwarded to plot_age_dist_model.
    """
    ranked = results_df.dropna(subset=[score_col]).sort_values(
        score_col, ascending=ascending
    )
    for cpg in ranked.index[:n]:
        print(f"\n── {cpg}  {score_col}={ranked.loc[cpg, score_col]:.4f} ──")
        plot_age_dist_model(adata, model, cpg_name=cpg,
                            title=cpg, **kwargs)