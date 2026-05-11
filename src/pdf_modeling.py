from numbers import Number

import numpy as np
from itertools import product
from logging import getLogger
from scipy import stats
from .util import hardy_weinberg_probabilities



logger = getLogger(__name__)


def compute_theoretical_pdf(xs, original, clonal_fracs, distribution_func=hardy_weinberg_probabilities,
                                       distribution_func_params=None):
    if distribution_func_params is None:
        distribution_func_params = {}

    if callable(original):
        original_pdf = original
    else:
        kde = stats.gaussian_kde(original)
        original_pdf = kde  # gaussian_kde instances are directly callable on arrays

    if isinstance(clonal_fracs, Number):
        clonal_fracs = [clonal_fracs]

    clonal_fracs = np.asarray(clonal_fracs)
    n_vafs = len(clonal_fracs)
    hw_outcomes = np.array([0.0, 0.5, 1.0])
    all_paths = np.array(list(product(hw_outcomes, repeat=n_vafs)))  # (3^n, n)

    xs = np.asarray(xs, dtype=float)
    pdf_result = np.zeros(len(xs))

    for path in all_paths:
        current_y = xs.copy()
        jacobian = np.ones(len(xs))
        path_prob = np.ones(len(xs))
        valid = np.ones(len(xs), dtype=bool)

        for j in range(n_vafs - 1, -1, -1):
            vaf = clonal_fracs[j]
            hw_outcome = path[j]
            one_minus_vaf = 1.0 - vaf

            if one_minus_vaf == 0.0:
                valid &= (current_y == hw_outcome)
                prev_x = current_y.copy()
            else:
                prev_x = (current_y - vaf * hw_outcome) / one_minus_vaf
                jacobian = np.where(valid, jacobian / one_minus_vaf, jacobian)

            valid &= (prev_x >= 0.0) & (prev_x <= 1.0)

            # Evaluate distribution only on valid points; use 0.5 as safe dummy elsewhere
            safe_prev_x = np.where(valid, prev_x, 0.5)
            prob_UU, prob_MU, prob_MM = distribution_func(safe_prev_x, **distribution_func_params)

            # Select the right probability for this path's hw_outcome
            if hw_outcome == 0.0:
                path_prob = np.where(valid, path_prob * prob_UU, path_prob)
            elif hw_outcome == 0.5:
                path_prob = np.where(valid, path_prob * prob_MU, path_prob)
            else:
                path_prob = np.where(valid, path_prob * prob_MM, path_prob)

            current_y = prev_x

        final_valid = valid & (current_y >= 0.0) & (current_y <= 1.0)

        # Evaluate original PDF only where valid
        safe_current_y = np.where(final_valid, current_y, 0.0)
        orig_pdf_vals = original_pdf(safe_current_y)

        pdf_result += np.where(final_valid, orig_pdf_vals * path_prob * jacobian, 0.0)

    return pdf_result


def compute_theoretical_pdf_mc(xs, original, clonal_fracs, n_samples=100000,
                                distribution_func=hardy_weinberg_probabilities,
                                distribution_func_params=None):
    """
    Monte Carlo approach for multiple VAFs.
    Simulates the exact sequential sampling process.

    Parameters
    ----------
    xs : array-like
        Evaluation points for the output PDF.
    original : array-like or callable
        Either an array of samples (KDE will be constructed) or a callable
        representing the original PDF/KDE (e.g. stats.beta(a, b).pdf).
        If callable, inverse-CDF sampling is used via a fine grid.
    vafs : array-like
        VAFs to apply sequentially. All must be <= 0.5.
    n_samples : int
        Number of Monte Carlo samples.
    distribution_func : callable
        Hardy-Weinberg probability function.
    distribution_func_params : dict, optional
        Extra kwargs forwarded to distribution_func.
    """
    if isinstance(clonal_fracs, Number):
        clonal_fracs = [clonal_fracs]

    if distribution_func_params is None:
        distribution_func_params = {}

    # ------------------------------------------------------------------
    # Sample from original — accepts samples array or PDF callable
    # ------------------------------------------------------------------
    if callable(original):
        # Inverse-CDF sampling from an arbitrary PDF on [0, 1]
        grid = np.linspace(1e-6, 1 - 1e-6, 10_000)
        pdf_vals = np.asarray(original(grid), dtype=float)
        pdf_vals = np.maximum(pdf_vals, 0)
        cdf_vals = np.cumsum(pdf_vals)
        cdf_vals /= cdf_vals[-1]
        u = np.random.uniform(0, 1, n_samples)
        mc_samples = np.interp(u, cdf_vals, grid)
    else:
        original_kde = stats.gaussian_kde(original)
        mc_samples = original_kde.resample(n_samples)[0]

    mc_samples = np.clip(mc_samples, 0, 1)

    # ------------------------------------------------------------------
    # Vectorised sequential HW transformation (no Python loop over samples)
    # ------------------------------------------------------------------
    current_samples = mc_samples.copy()

    for clonal_fraction in clonal_fracs:
        probs = np.array([
            distribution_func(b, **distribution_func_params)
            for b in current_samples
        ])                                      # (n_samples, 3)
        prob_UU = probs[:, 0]
        prob_MU = probs[:, 1]

        rand_vals = np.random.random(n_samples)
        hw_outcomes = np.where(
            rand_vals < prob_UU, 0.0,
            np.where(rand_vals < prob_UU + prob_MU, 0.5, 1.0)
        )

        current_samples = (1 - clonal_fraction) * current_samples + clonal_fraction * hw_outcomes

    result_kde = stats.gaussian_kde(current_samples)
    return result_kde.pdf(xs)


def generate_diploid_methylation_states(methylation_probs_df,
                                        distribution_func=hardy_weinberg_probabilities,
                                        n_cells=100, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    n_individuals, n_cpgs = methylation_probs_df.shape
    beta = methylation_probs_df.values  # (n_individuals, n_cpgs)

    prob_UU, prob_MU, prob_MM = distribution_func(beta)  # each (n_individuals, n_cpgs)

    # Random samples: (n_cells, n_individuals, n_cpgs)
    random_vals = np.random.random((n_cells, n_individuals, n_cpgs))

    # Broadcast prob arrays to (1, n_individuals, n_cpgs)
    prob_UU = prob_UU[np.newaxis, :, :]
    prob_MU = prob_MU[np.newaxis, :, :]

    # Assign states
    methylation_tensor = np.zeros((n_cells, n_individuals, n_cpgs))
    methylation_tensor[random_vals >= prob_UU] = 0.5
    methylation_tensor[random_vals >= (prob_UU + prob_MU)] = 1.0

    state_counts = {
        'UU': int(np.sum(methylation_tensor == 0.0)),
        'MU': int(np.sum(methylation_tensor == 0.5)),
        'MM': int(np.sum(methylation_tensor == 1.0)),
    }

    return methylation_tensor, state_counts