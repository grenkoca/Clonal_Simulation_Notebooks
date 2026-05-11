"""
CHIPSimulation_V2 — mechanistic clonal haematopoiesis simulation.

Division-symmetry model
-----------------------
Each lineage is characterised by three division-outcome probabilities:
  p_ss  symmetric self-renewal  (both daughters remain HSC)
  p_sd  asymmetric division     (one HSC, one differentiating cell)
  p_dd  symmetric differentiation (both daughters exit the HSC pool)

Lotka-Volterra niche competition
---------------------------------
Population pressure is computed as:
  pressure = clip(1 - dot(a, counts) / K, 0, 1)

Life-event hook
---------------
Pass ``event_fn(sim, t)`` to the constructor to schedule clinical events
(therapy, new clone injection, parameter changes) without subclassing.

Example hook patterns
~~~~~~~~~~~~~~~~~~~~~
1. Add a new lineage::

    def hook(sim, t):
        if t == 30:
            sim.add_lineage(p_ss=0.40, p_sd=0.33, p_dd=0.27)

2. Niche collapse (therapy)::

    def hook(sim, t):
        if t == 30:
            sim.K *= 0.2
        elif t == 35:
            sim.K = sim._K_init

3. Division-rate scaling::

    def hook(sim, t):
        if t == 30:
            sim.tau *= 0.5

4. Affinity boost for a specific lineage::

    def hook(sim, t):
        if t == 30:
            sim.a[idx] = 2.0
"""
import warnings
from typing import Union, Callable

import numpy as np
import pandas as pd
from numbers import Number
try:
    from anndata import AnnData
    _ANNDATA_AVAILABLE = True
except ImportError:
    AnnData = None  # type: ignore
    _ANNDATA_AVAILABLE = False

CLONAL_MARKER_MUTATION_RATE = 0


def orthogonalize_power_term(t_linear: np.ndarray, t_power: np.ndarray):
    proj_coef = np.dot(t_power, t_linear) / np.dot(t_linear, t_linear)
    t_power_orth = t_power - proj_coef * t_linear
    return t_power_orth, proj_coef


class ClonalSim:
    """
    Mechanistic simulation of clonal haematopoiesis (CHIP) in the HSC niche.

    The HSC pool is partitioned into lineages that compete for a finite niche.
    Each lineage is governed by three division-outcome probabilities (p_ss, p_sd,
    p_dd) that determine whether it expands, holds steady, or contracts over time.
    New mutant lineages arise stochastically via somatic mutation and may carry a
    net self-renewal advantage (p_ss > p_dd).

    Parameters
    ----------
    N_init : int, default 1000
        Number of background (neutral) HSC lineages that partition the niche at
        initialisation. Larger N → finer-grained neutral diversity.

    p_sym_renewal : float, default 0.34
        Per-division probability that both daughter cells remain HSCs
        (symmetric self-renewal, p_ss). Drives clonal expansion when above the
        neutral point relative to p_diff.

    p_asym : float, default 0.33
        Per-division probability of asymmetric division (p_sd): one HSC daughter,
        one differentiating daughter. Maintains pool size with no net growth bias.

    p_diff : float, default 0.33
        Per-division probability that both daughters exit the HSC pool
        (symmetric differentiation, p_dd). Drives clonal contraction.

        .. note::
            The three probabilities are automatically renormalised to sum to 1.
            Net growth bias for a lineage is ``r = p_ss - p_dd``. This is also
            impacted by `phi`, which is calculated from the Lotka–Volterra
            equations

    tau : float, default 1.0
        Mean HSC division time in years. Scales how quickly the clonal dynamics
        unfold; smaller tau → faster turnover and faster clonal sweeps.

    mu : float, default 1e-8
        Somatic mutation rate: probability per cell division of a new CHIP-relevant
        variant arising in a lineage.

    max_variants : int or None, default None
        Hard cap on the number of new mutant lineages allowed to arise during the
        simulation. ``None`` means unlimited.

    niche_capacity : float or None, default None
        Carrying capacity K of the HSC niche used in the Lotka-Volterra competition
        term. Defaults to ``total_population`` when ``None``. Reducing K mid-run
        (via the event hook) simulates niche contraction (e.g. therapy).

    niche_affinity : float, default 1.0
        Competitive weight of each lineage in the niche pressure calculation
        ``pressure = clip(1 - dot(a, counts) / K, 0, 1)``. Values > 1 make a
        lineage consume more niche space per cell; < 1 makes it less competitive.
        Per-lineage values can be set directly on ``sim.a[idx]`` after construction.

    division_dist : callable or None, default None
        A zero-argument callable ``() -> (p_ss, p_sd, p_dd)`` that draws division
        probabilities for each newly arising mutant lineage. Defaults to a
        Beta(1, 30) draw for p_ss (skewed toward near-neutral variants) with p_sd
        fixed at ~0.33 and p_dd as the remainder.

    symmetry_drift_alpha : float, default 1000.0
        Dirichlet concentration parameter for stochastic drift of (p_ss, p_sd, p_dd)
        each timestep. Higher values → tighter, slower drift around the current
        symmetry; lower values → more volatile division-symmetry fluctuations.

    drift_mutants : bool, default False
        Whether to apply symmetry drift to mutant lineages as well as background
        lineages. When ``False``, mutant lineages retain fixed division probabilities.

    adata : AnnData or None, default None
        Initial methylation state stored as AnnData object. When provided:
        - `.X` shape must be `(N, n_cpg)` with methylation beta values in [0, 1]
        - `.var_names` should contain CpG identifiers (e.g., "cgXXXXXX" or "CpG_0")
        - `.obs_names` should contain lineage IDs as strings
        - Optional layers: `'methylated_reads'`, `'total_reads'` for read-level data

        Deprecated: Use ``adata`` instead of ``methylation_matrix``.

    methylation_matrix : ndarray of shape (N, n_cpg) or None, default None
        Initial methylation state of each background lineage across CpG sites.
        Required for methylation-clock analyses; ignored if ``None``.

        .. deprecated::
            Use ``adata`` parameter instead. This will be removed in a future version.

    p_u_to_m : float or None, default None
        Per-division probability that an unmethylated CpG site becomes methylated.
        Both ``p_u_to_m`` and ``p_m_to_u`` must be set to enable methylation drift.

    transition_generator : float or None, default None
        Per-division probability that a methylated CpG site becomes unmethylated.

    cpg_types : ndarray of str or None, default None
        Array of shape ``(n_cpg,)`` specifying the type of each CpG site:
        - ``'clonal_marker'``: Near-zero transition probabilities (1e-5) for faithful transmission
        - ``'drift'``: Uses scalar ``p_u_to_m`` / ``p_m_to_u`` values from constructor

        When provided, creates per-CpG transition probability vectors for fine-grained control.

    cpg_means : ndarray of float or None, default None
        Array of shape ``(n_cpg,)`` with population mean methylation level for each CpG.
        Used for HWE-aware initialization and downstream AUC computation.

    event_fn : callable or None, default None
        Hook called at every simulated timestep as ``event_fn(sim, t)``. Use this
        to inject clinical events (therapy, new clone introduction, parameter
        changes) without subclassing. See module-level docstring for patterns.
    """

    def __init__(self, N_init=1000, counts_init=1, K_init=1000,
                 p_sym_renewal=0.34, p_asym=0.33, p_diff=0.33,
                 tau=1.0, mu=1e-8, dt=None, max_variants=None,
                 adata=None, methylation_matrix=None, transition_generator=None,
                 cpg_types=None, cpg_means=None, niche_affinity=1.0,
                 division_dist=None,
                 symmetry_drift_alpha=1000.0, drift_mutants=False,
                 event_fn=None, empirical_methylation_model=None,
                 a_maturity=13.5, l_max=122.5, age_exponent=2.0,
                 proj_coef=None,
                 X_grid=None):
        self.N_init = N_init
        self._n_background_lineages = N_init

        raw = np.array([p_sym_renewal, p_asym, p_diff], dtype=np.float64)
        raw = raw / raw.sum()  # normalise to sum-to-1
        self.p_ss = np.full(N_init, raw[0], dtype=np.float64)
        self.p_sd = np.full(N_init, raw[1], dtype=np.float64)
        self.p_dd = np.full(N_init, raw[2], dtype=np.float64)
        if X_grid is not None:
            self.X_grid = X_grid
        else:
            self.X_grid = None

        # --- Division time (per lineage) ---
        self._tau_init = float(tau)
        self.tau = np.full(N_init, float(tau), dtype=np.float64)

        # --- Niche parameters ---
        self.K = K_init if K_init is not None else N_init
        self._K_init = self.K
        self.a = np.full(N_init, float(niche_affinity), dtype=np.float64)
        self.dt = dt

        # --- Division-distribution for new mutant lineages ---
        if division_dist is None:
            def _default_division_dist():
                p_ss_new = float(np.random.beta(1, 30))
                p_dd_new = float(np.clip(1.0 - p_ss_new - 0.33, 0.0, 1.0))
                p_sd_new = float(np.clip(1.0 - p_ss_new - p_dd_new, 0.0, 1.0))
                # renormalise
                s = p_ss_new + p_sd_new + p_dd_new
                return p_ss_new / s, p_sd_new / s, p_dd_new / s
            self.division_dist = _default_division_dist
        else:
            self.division_dist = division_dist

        # --- Symmetry drift parameters ---
        self.symmetry_drift_alpha = float(symmetry_drift_alpha)
        self.drift_mutants = bool(drift_mutants)

        # --- Event hook ---
        self.event_fn = event_fn

        # --- SCARLET translation parameters ---
        self.a_maturity = a_maturity
        self.l_max = l_max
        self.age_exponent = age_exponent
        self.proj_coef = proj_coef
        self.scarlet_params = None

        # --- Core state ---
        self._mu_init = mu
        self.mu = np.full(N_init, float(mu), dtype=np.float32)
        self.counts = np.full(shape=N_init, fill_value=counts_init, dtype=np.float32)
        self.lineage_ages = np.zeros(N_init, dtype=np.float32)
        self.current_time = 0.0

        # Handle AnnData backend or deprecated methylation_matrix
        self._adata: AnnData | None = None
        self._adata_history: list[AnnData | None] = [None]
        self.initial_methylation_matrix = None  # Kept for backward compatibility

        self.cpg_types = cpg_types

        # Initialise transition-rate attributes so all code paths can safely
        # reference them without risking AttributeError.
        self.p_u_to_m = None
        self.p_m_to_u = None
        self.p_u_to_m_vec = None
        self.p_m_to_u_vec = None

        if adata is not None:
            if not _ANNDATA_AVAILABLE:
                raise ImportError("anndata must be installed to use the 'adata' parameter")
            if not isinstance(adata, AnnData):
                raise TypeError("'adata' must be an AnnData object")
            if adata.X.shape[0] != N_init:
                raise ValueError(f"adata.X must have {N_init} rows (n_lineages), got {adata.X.shape[0]}")
            # Store AnnData reference and extract matrix for backward compatibility
            self._adata = adata
            self.initial_methylation_matrix = np.asarray(adata.X).copy()
            self.methylation_updates_history = [None]  # For tracking new lineage additions
            # Methylation fidelity parameters

            if isinstance(transition_generator, Number):
                self.p_u_to_m = transition_generator
                self.p_m_to_u = transition_generator
            elif isinstance(transition_generator, Callable):
                try:
                    self.p_u_to_m_vec, self.p_m_to_u_vec = transition_generator(self._adata.X.mean(axis=0))
                except AttributeError as e:
                    raise AttributeError("Unable to load transition generator")

        else:
            # No methylation data provided
            self.methylation_updates_history = [None]


        if cpg_types is not None:
            cpg_types_arr = np.asarray(cpg_types, dtype=object)
            n_cpg = len(cpg_types_arr)

            # Validate cpg_types values
            valid_types = {'clonal_marker', 'drift'}
            unique_types = set(cpg_types_arr)
            if not unique_types.issubset(valid_types):
                invalid = unique_types - valid_types
                raise ValueError(f"Invalid cpg_type(s): {invalid}. Must be one of {valid_types}")

            # Resolve cpg_means: use provided array or fall back to initial matrix mean
            if cpg_means is not None:
                cpg_means_arr = np.asarray(cpg_means, dtype=np.float64)
            elif self.initial_methylation_matrix is not None:
                cpg_means_arr = np.mean(self.initial_methylation_matrix, axis=0).astype(np.float32)
            else:
                cpg_means_arr = np.full(n_cpg, 0.5, dtype=np.float32)

            self.CLONAL_MARKER_MUTATION_RATE = CLONAL_MARKER_MUTATION_RATE

            clonal_mask = cpg_types_arr == 'clonal_marker'

            self.empirical_methylation_model = empirical_methylation_model

            # If the callable path was not taken, expand scalar rates to per-CpG
            # vectors so the rest of this block always operates on arrays.
            if self.p_u_to_m_vec is None:
                rate_u = float(self.p_u_to_m) if self.p_u_to_m is not None else 0.0
                rate_m = float(self.p_m_to_u) if self.p_m_to_u is not None else rate_u
                self.p_u_to_m_vec = np.full(n_cpg, rate_u, dtype=np.float32)
                self.p_m_to_u_vec = np.full(n_cpg, rate_m, dtype=np.float32)

            total = self.p_u_to_m_vec + self.p_m_to_u_vec
            self.methylation_eq_prob = np.where(total > 0, self.p_u_to_m_vec / total, 0.5)
            self.methylation_eq_variance = self.methylation_eq_prob * (1 - self.methylation_eq_prob)

            self.p_u_to_m_vec[clonal_mask] = CLONAL_MARKER_MUTATION_RATE
            self.p_m_to_u_vec[clonal_mask] = CLONAL_MARKER_MUTATION_RATE

            self.cpg_means = np.mean(self.initial_methylation_matrix, axis=0).astype(np.float32)

            # --- Analytical variance trajectory attributes (SCARLET-based) ---
            omega = self.p_u_to_m_vec + self.p_m_to_u_vec
            eta = np.where(omega > 0, self.p_u_to_m_vec / omega, cpg_means_arr)
            self.cpg_omega = omega
            self.cpg_eta = eta

            if self.scarlet_params is not None:
                # Use fully-vectorized SCARLET variance (processes all CpGs in one call)
                from scarlet_translation import _var_Z_order_1_vec as _scarlet_var_vec

                _sp = self.scarlet_params

                def cpg_var_at_time(t: float) -> np.ndarray:
                    """Expected methylation variance at time t under SCARLET model."""
                    # Fully vectorized: pass t as shape-(1,) array, get (1, n_cpg) back
                    result = _scarlet_var_vec(
                        np.array([max(t, 0.0)]), _sp.s, _sp.N,
                        _sp.eta, _sp.omega, _sp.p, _sp.c
                    )[0]
                    return np.where(np.isfinite(result), result, 0.0)

                self.cpg_var_at_time = cpg_var_at_time
            else:
                # Fallback: simple two-state Markov variance
                self.cpg_var_stationary = eta * (1.0 - eta)

                def cpg_var_at_time(t: float) -> np.ndarray:
                    """Expected methylation variance at time t under two-state Markov."""
                    return self.cpg_var_stationary * (1.0 - np.exp(-2.0 * self.cpg_omega * t))

                self.cpg_var_at_time = cpg_var_at_time


            # Store cpg_means if provided, otherwise infer from initial matrix
            if cpg_means is not None:
                self.cpg_means = np.asarray(cpg_means, dtype=np.float32)
                if len(self.cpg_means) != n_cpg:
                    raise ValueError(f"cpg_means length ({len(self.cpg_means)}) must match cpg_types length ({n_cpg})")
            elif self.initial_methylation_matrix is not None:
                # Infer means from initial matrix (average across lineages)
                self.cpg_means = np.mean(self.initial_methylation_matrix, axis=0).astype(np.float32)

        # --- History ---
        self.time_history = [0]
        self._time_arr = np.array([0.0], dtype=np.float64)  # fast numpy mirror of time_history
        self.counts_history = [self.counts.copy()]
        self.K_history = [self.K]
        # symmetry_history stores np.stack([p_ss, p_sd, p_dd], axis=-1) per timestep
        self.symmetry_history = [np.stack([self.p_ss.copy(), self.p_sd.copy(), self.p_dd.copy()], axis=-1)]
        self.lineage_ids = list(range(N_init))
        self.next_lineage_id = N_init

        assert max_variants is None or isinstance(max_variants, int), \
            "Parameter `max_variants` must be an integer."
        self.max_new_variants = max_variants
        self.num_new_variants = 0

        self.color_exceptions = {}

        # --- Lineage class system ---
        # Each lineage belongs to a named class that controls coloring in plots.
        #   class_palettes      : class_name -> [hex, ...]  cycled by position within class
        #   class_color_fns     : class_name -> callable(lineage_id, sim) -> color string
        #   lineage_classes     : lineage_id -> class_name
        #   _class_member_order : class_name -> [lineage_id, ...]  (insertion order)
        self.class_palettes: dict = {
            "initial": ["#AAAAAA"],
            "derived": [
                '#FFB0B8'
                # "#E45756", "#F28522", "#4C78A8", "#72B7B2", "#54A24B",
                # "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC",
            ],
        }
        self.class_color_fns: dict = {}
        self.lineage_classes: dict = {}
        self._class_member_order: dict = {"initial": [], "derived": []}
        # O(1) position lookup: class_name -> {lineage_id -> index_in_member_order}
        self._class_member_position: dict = {"initial": {}, "derived": {}}
        for lid in range(N_init):
            self.lineage_classes[lid] = "initial"
            self._class_member_position["initial"][lid] = len(self._class_member_order["initial"])
            self._class_member_order["initial"].append(lid)

        # parent-child lineage graph: child_id -> parent_id (None for background)
        self.lineage_parents = {}
        self.run_properties = {}

        self.verbose=False

    def load_empirical_data(self, model) -> None:
        """Load an empirical beta regression model to calibrate per-CpG transition rates.

        Must be called before the cpg_types block executes (i.e. before __init__
        completes), or transition rates must be recomputed manually afterward.
        Stores the model as self.empirical_methylation_model for use during
        rate calibration.

        Parameters
        ----------
        model : BetaRegressionAlphaBeta
            Fitted beta regression model with gamma_alpha and gamma_beta attributes,
            each of shape (4, n_cpg): [intercept, age_linear, age_power, sex].
        """
        self.empirical_methylation_model = model


    def _assign_lineage_class(self, lineage_id: int, class_name: str) -> None:
        """Register *lineage_id* as a member of *class_name*."""
        self.lineage_classes[lineage_id] = class_name
        if class_name not in self._class_member_order:
            self._class_member_order[class_name] = []
            self._class_member_position[class_name] = {}
        if lineage_id not in self._class_member_position.get(class_name, {}):
            pos = len(self._class_member_order[class_name])
            self._class_member_order[class_name].append(lineage_id)
            self._class_member_position[class_name][lineage_id] = pos

    def _resolve_lineage_color(self, lineage_id: int, palette_overrides: dict | None = None) -> str:
        """Return a color string for *lineage_id* using the class system.

        Resolution order (highest priority first):

        1. ``color_exceptions`` — explicit per-lineage override set via
           ``add_lineage(color=...)`` or ``sim.color_exceptions[id] = ...``.
        2. ``class_color_fns`` — per-class callable ``(lineage_id, sim) -> color``
           registered via :meth:`register_class`.
        3. ``class_palettes`` — list of hex colors cycled by insertion order within
           the class.  *palette_overrides* (a ``{class_name: [colors]}`` dict) takes
           precedence over ``self.class_palettes`` when provided.
        """
        if lineage_id in self.color_exceptions:
            return self.color_exceptions[lineage_id]

        cls = self.lineage_classes.get(lineage_id, "derived")

        if cls in self.class_color_fns:
            return self.class_color_fns[cls](lineage_id, self)

        if palette_overrides and cls in palette_overrides:
            palette = palette_overrides[cls]
        else:
            palette = self.class_palettes.get(cls, self.class_palettes.get("derived", ["#E45756"]))

        pos = self._class_member_position.get(cls, {}).get(lineage_id, 0)
        return palette[pos % len(palette)]

    def register_class(self, class_name: str, palette: list | None = None,
                       color_fn=None) -> None:
        """Register a lineage class with a color palette or callable.

        Parameters
        ----------
        class_name : str
            Identifier for the class (e.g. ``"mutant_fit"``).
        palette : list of str or None
            Hex color strings cycled by member order within the class.
            When ``color_fn`` is also provided, ``palette`` is stored but
            ``color_fn`` takes precedence.
        color_fn : callable or None
            ``(lineage_id, sim) -> color`` — called instead of palette when set.
            Receives the lineage id and the simulation object so it can inspect
            any attribute (fitness, VAF, parent, etc.) to compute a color.

        Examples
        --------
        Fitness-based coloring::

            import matplotlib.pyplot as plt
            def fitness_color(lid, sim):
                idx = sim.lineage_ids.index(lid)
                r = sim.p_ss[idx] - sim.p_dd[idx]
                return plt.cm.RdYlGn(plt.Normalize(-0.1, 0.1)(r))

            sim.register_class("mutant_fit", color_fn=fitness_color)
        """
        if palette is not None:
            self.class_palettes[class_name] = list(palette)
        if color_fn is not None:
            self.class_color_fns[class_name] = color_fn
        if class_name not in self._class_member_order:
            self._class_member_order[class_name] = []
            self._class_member_position[class_name] = {}

    def apply_methylation_drift(self, methylation_matrix, counts, dt, step_type='tau_leap'):
        """
        Apply stochastic methylation state transitions.

        Each CpG site in each cell of each lineage undergoes transitions:
        - Unmethylated (0) -> Methylated (1) with probability p_u_to_m per division
        - Methylated (1) -> Unmethylated (0) with probability p_m_to_u per division

        When step_type='tau_leap', transition events are drawn from exact Poisson
        distributions, mirroring the tau-leap used for lineage birth/death.

        When step_type='gaussian', a single Gaussian draw replaces the two Poisson
        draws via the CLT approximation:
            Poisson(λ_g) - Poisson(λ_l) ≈ N(λ_g - λ_l, λ_g + λ_l)
        This is exact for large clones (λ >> 1) and negligibly different for small
        clones where both λ ≈ 0 regardless of the distribution used. It halves the
        number of random draws and keeps all computation in float32.

        Parameters:
        - methylation_matrix: Current methylation matrix (lineages × CpG sites),
                              values are mean methylation fraction in [0, 1]
        - counts: Current cell counts per lineage, shape (N_lineages,)
        - dt: Timestep size (same units as tau)
        - step_type: 'tau_leap' for exact Poisson draws, 'gaussian' for CLT approximation

        Returns:
        - Updated methylation matrix after one timestep of stochastic drift
        """
        if self.p_u_to_m_vec is not None and self.p_m_to_u_vec is not None:
            p_u_to_m_eff = self.p_u_to_m_vec  # shape (n_cpg,
            p_m_to_u_eff = self.p_m_to_u_vec
        elif self.p_u_to_m is not None and self.p_m_to_u is not None:
            p_u_to_m_eff = self.p_u_to_m  # scalar
            p_m_to_u_eff = self.p_m_to_u
        else:
            return methylation_matrix

        # Skip extinct lineages entirely — their rows carry weight 0 in any bulk
        # computation, but processing them wastes O(n_extinct × n_cpg) work.
        active = counts > 0
        if not np.any(active):
            return methylation_matrix

        mat = methylation_matrix[active]
        # Cast to float32 explicitly: self.tau is float64, and Python literal 2.0 is
        # a float64 scalar — either would silently upcast the entire (n_active, n_cpg)
        # computation to float64, doubling memory and bandwidth.
        cnt  = counts[active].astype(np.float32)
        tau  = self.tau[active].astype(np.float32)

        n_alleles      = np.float32(2.0) * cnt[:, None]
        n_methylated   = mat * n_alleles
        n_unmethylated = n_alleles - n_methylated  # avoids (1 - mat) * n_alleles

        # Compute scale (dt/tau) once; shape (n_active, 1) broadcasts over CpGs.
        scale    = (np.float32(dt) / tau)[:, None]
        lam_gain = n_unmethylated * p_u_to_m_eff * scale
        lam_loss = n_methylated   * p_m_to_u_eff * scale

        if step_type == 'gaussian':
            # CLT: mean = λ_g - λ_l, variance = λ_g + λ_l (independent Poissons).
            # One normal draw instead of two Poisson draws.
            lam_net = lam_gain - lam_loss
            lam_std = np.sqrt(np.maximum(lam_gain + lam_loss, np.float32(0.0)))
            delta = lam_net + lam_std * np.random.standard_normal(mat.shape).astype(np.float32)
        else:
            # tau_leap: exact Poisson draws (preserves integer allele-flip semantics)
            gains  = np.random.poisson(lam_gain).astype(np.float32)
            losses = np.random.poisson(lam_loss).astype(np.float32)
            delta  = gains - losses

        new_n_methylated = np.clip(n_methylated + delta, np.float32(0.0), n_alleles)
        safe_alleles = np.maximum(n_alleles, np.float32(1.0))
        updated = (new_n_methylated / safe_alleles).astype(methylation_matrix.dtype)

        if active.all():
            return updated

        new_matrix = methylation_matrix.copy()
        new_matrix[active] = updated
        return new_matrix

    def simulate_timestep(self, step_type='gaussian'):
        if self.event_fn is not None:
            self.event_fn(self, self.current_time)

        dt = float(np.min(self.tau)) if self.dt is None else self.dt
        assert dt <= np.min(self.tau), f"self.dt must be <= min(tau). dt={dt}, min(tau)={np.min(self.tau)}"

        self.current_time += dt

        current_occupancy = np.dot(self.a, self.counts)
        phi = (self.K - current_occupancy) / self.K


        eff_p_ss = np.maximum(self.p_ss * (1.00 + phi
                                           ), 0)
        eff_p_dd = np.maximum(self.p_dd * (1.00 - phi), 0)
        if self.verbose:
            print(f"t={self.current_time:.3g} (dt={dt})")
            print(f"   Current count: {np.sum(self.counts):.2f} (K={self.K})\n"
                  f"   Pressure     : {phi}\n"
                  f"   p_ss -> eff_p_ss : {np.mean(self.p_ss):.3f}, {np.mean(eff_p_ss):.3f}\n"
                  f"   p_dd -> eff_p_dd : {np.mean(self.p_dd):.3f}, {np.mean(eff_p_dd):.3f}")

        if step_type == 'gaussian':
            # 3. Deterministic Flux (The Signal)
            activity = self.counts * dt / self.tau
            delta = (eff_p_ss - eff_p_dd) * activity

            variance = (eff_p_ss + eff_p_dd) * activity
            sigma = np.sqrt(np.maximum(variance, 0))
            # This is the Gaussian approximation to the Poisson, or the
            # diffusion coefficient of a Moran process
            diffusion_coef = np.random.normal(0, sigma)
            adjustment = delta + diffusion_coef


            if self.verbose:
                      print(f"   Activity     : {np.sum(activity)}\n"
                      f"   Total adjust.: {np.sum(adjustment):.2f} (delta: {np.sum(delta):.2f}, noise: {np.std(diffusion_coef):.2f})\n"
                      f"                  ({min(adjustment):.2f}, {max(adjustment):.2f})")

            # 5. Update
            self.counts = np.clip(self.counts + adjustment, 0.0, None)
            self.counts[self.counts < 0.5] = 0.0  # hard absorption; no half cells
        elif step_type == 'tau_leap':
            birth_rate = self.counts * eff_p_ss / self.tau
            death_rate = self.counts * eff_p_dd / self.tau

            # Poisson draws — exact count of birth and death events
            births = np.random.poisson(birth_rate * dt)
            deaths = np.random.poisson(death_rate * dt)
            adjustment = births-deaths
            if self.verbose:

                print(f"   Avg. Births : {np.mean(births)} ({np.min(births)}, {np.max(births)})\n"
                      f"   Avg. Deaths : {np.mean(deaths)} ({np.min(deaths)}, {np.max(deaths)})\n"
                      f"   Total adjust.: {np.sum(adjustment)} ({min(adjustment)}, {max(adjustment)})")

            self.counts = np.maximum(self.counts + adjustment, 0)

        # Apply methylation drift to existing lineages.
        # Use the cached matrix from the previous step rather than rebuilding via
        # get_methylation_matrix_at_time (which vstack-reconstructs from history each call).
        if self.initial_methylation_matrix is not None:
            current_matrix = (
                self._current_drifted_matrix
                if hasattr(self, '_current_drifted_matrix') and self._current_drifted_matrix is not None
                else self.get_methylation_matrix_at_time(self.current_time)
            )
            self._current_drifted_matrix = self.apply_methylation_drift(
                current_matrix, self.counts, dt, step_type=step_type
            )
        else:
            self._current_drifted_matrix = None

        # --- Mutation phase: spawn new lineages ---
        new_lineage_counts = []
        new_p_ss_list, new_p_sd_list, new_p_dd_list = [], [], []
        new_tau_list, new_a_list, new_mu_list = [], [], []
        new_ages, new_ids, new_parent_ids = [], [], []
        new_methylation_rows = []

        should_create_variants = (self.max_new_variants is None or
                                  self.num_new_variants < self.max_new_variants)

        if should_create_variants:
            # Vectorised mutation test: draw all probabilities in one numpy call
            mutation_probs = self.mu * self.counts * dt / self.tau
            mutated_mask = np.random.random(len(self.counts)) < mutation_probs
            mutated_indices = np.where(mutated_mask)[0]

            # Respect variant cap
            if self.max_new_variants is not None:
                remaining = self.max_new_variants - self.num_new_variants
                mutated_indices = mutated_indices[:remaining]

            if len(mutated_indices) > 0:
                # Pre-compute medians once for all new lineages this step
                median_tau = float(np.median(self.tau))
                median_mu  = float(np.median(self.mu))

                # Resolve current methylation matrix once (used by all mutations)
                if self.initial_methylation_matrix is not None:
                    meth_matrix_now = (
                        self._current_drifted_matrix
                        if self._current_drifted_matrix is not None
                        else self.get_methylation_matrix_at_time(self.current_time)
                    )
                else:
                    meth_matrix_now = None

                for i in mutated_indices:
                    p_ss_new, p_sd_new, p_dd_new = self.division_dist()
                    new_lineage_counts.append(1)
                    new_p_ss_list.append(p_ss_new)
                    new_p_sd_list.append(p_sd_new)
                    new_p_dd_list.append(p_dd_new)
                    new_tau_list.append(median_tau)
                    new_mu_list.append(median_mu)
                    new_a_list.append(1.0)
                    new_ages.append(self.current_time)
                    new_ids.append(self.next_lineage_id)
                    new_parent_ids.append(self.lineage_ids[i])
                    self.next_lineage_id += 1
                    self.num_new_variants += 1

                    if meth_matrix_now is not None and i < len(meth_matrix_now):
                        parent_methylation_row = meth_matrix_now[i, :]
                        # Single founding cell: draw diploid genotype from HWE giving
                        # {0.0, 0.5, 1.0} for {UU, MU, MM}.
                        p_parent = np.clip(parent_methylation_row, 0.0, 1.0)
                        probs_uu = (1 - p_parent) ** 2
                        probs_mu_hwe = 2 * p_parent * (1 - p_parent)
                        r = np.random.uniform(size=p_parent.shape)
                        genotypes = np.where(r < probs_uu, 0,
                                    np.where(r < probs_uu + probs_mu_hwe, 1, 2))
                        child_methylation = (genotypes * 0.5).astype(np.float32)
                        new_methylation_rows.append(child_methylation)

        # Record methylation updates
        if new_methylation_rows:
            self.methylation_updates_history.append(np.array(new_methylation_rows))
        else:
            self.methylation_updates_history.append(None)

        # Record parent-child relationships for newly spawned lineages
        for new_id, parent_id in zip(new_ids, new_parent_ids):
            self.lineage_parents[new_id] = parent_id

        # Append new lineages to simulation state
        if new_lineage_counts:
            if self.verbose:
                print(f"\nNew lineage (index: {new_ids})")
                print(f"\tTime: {self.current_time:.2g}")
                print(f"\tcounts: {new_lineage_counts}")
                print(f"\tp_ss/p_sd/p_dd: {list(zip(new_p_ss_list, new_p_sd_list, new_p_dd_list))}")
                print(f"\tTotal variants created so far: "
                      f"{self.num_new_variants}/{self.max_new_variants if self.max_new_variants is not None else 'unlimited'}")

            self.counts = np.concatenate([self.counts, new_lineage_counts])
            self.p_ss = np.concatenate([self.p_ss, new_p_ss_list])
            self.p_sd = np.concatenate([self.p_sd, new_p_sd_list])
            self.p_dd = np.concatenate([self.p_dd, new_p_dd_list])
            self.tau = np.concatenate([self.tau, new_tau_list])
            self.mu = np.concatenate([self.mu, new_mu_list])
            self.a = np.concatenate([self.a, new_a_list])
            self.lineage_ages = np.concatenate([self.lineage_ages, new_ages])
            self.lineage_ids.extend(new_ids)
            for new_id in new_ids:
                self._assign_lineage_class(new_id, "derived")

            if self._current_drifted_matrix is not None and new_methylation_rows:
                self._current_drifted_matrix = np.vstack(
                    [self._current_drifted_matrix, np.array(new_methylation_rows)]
                )

        self._apply_symmetry_drift()

        # Record history
        self.time_history.append(self.current_time)
        self._time_arr = np.append(self._time_arr, self.current_time)
        self.counts_history.append(self.counts.copy())
        self.K_history.append(self.K)
        self.symmetry_history.append(np.stack([self.p_ss.copy(), self.p_sd.copy(), self.p_dd.copy()], axis=-1))


    def _apply_symmetry_drift(self):
        """Apply Dirichlet random walk to per-lineage division-symmetry probabilities.

        Background lineages (indices 0..self._n_background_lineages-1) are always drifted.
        If self.drift_mutants is True, drift is applied to all lineages including newly spawned ones.
        """
        pass
        # n_total = len(self.p_ss)
        # n_drift = n_total if self.drift_mutants else min(self._n_background_lineages, n_total)
        #
        # for i in range(n_drift):
        #     probs = np.array([self.p_ss[i], self.p_sd[i], self.p_dd[i]], dtype=np.float16)
        #     # Guard against zero/negative concentrations which would make Dirichlet undefined
        #     probs = np.clip(probs, 1e-12, None)
        #     new_probs = np.random.dirichlet(self.symmetry_drift_alpha * probs)
        #     self.p_ss[i], self.p_sd[i], self.p_dd[i] = new_probs

    def add_lineage(self, p_ss, p_sd, p_dd, tau=None, a=1.0, mu=None, initial_count=None, methylation_row=None,
                    color=None, parent_id=None, lineage_class="derived"):
        """Add a new lineage to the simulation atomically.

        Parameters
        ----------
        p_ss, p_sd, p_dd : float
            Division-symmetry probabilities. Normalised to sum to 1.
        tau : float or None
            Division time. Defaults to ``self._tau_init``.
        a : float
            Niche affinity. Default 1.0.
        initial_count : float or None
            Initial cell count. Defaults to ``self.total_pop / self._n_background_lineages``.
        methylation_row : array-like or None
            Methylation state of shape ``(n_cpg,)``.  If None, ``None`` is appended to
            ``methylation_updates_history``; otherwise the row is stored as ``(1, n_cpg)``.
        lineage_class : str, default ``"derived"``
            Class name used for color resolution in :meth:`plot_lineage_evolution`.
            Register custom classes with :meth:`register_class`.

        Returns
        -------
        int
            The new lineage ID.
        """
        raw = np.array([p_ss, p_sd, p_dd], dtype=np.float64)
        raw = raw / raw.sum()

        if tau is None:
            tau = self._tau_init

        if initial_count is None:
            initial_count = self._n_background_lineages

        if mu is None:
            mu = self._mu_init

        new_id = self.next_lineage_id
        self.next_lineage_id += 1

        if color is not None:
            self.color_exceptions[new_id] = color

        if parent_id is not None:
            self.lineage_parents[new_id] = parent_id

        self.p_ss = np.append(self.p_ss, raw[0])
        self.p_sd = np.append(self.p_sd, raw[1])
        self.p_dd = np.append(self.p_dd, raw[2])
        self.tau = np.append(self.tau, float(tau))
        self.mu = np.append(self.mu, float(mu))
        self.a = np.append(self.a, float(a))
        self.counts = np.append(self.counts, float(initial_count))
        # self.counts_history[-1] = self.counts
        self.lineage_ages = np.append(self.lineage_ages, self.current_time)
        self.lineage_ids.append(new_id)
        self._assign_lineage_class(new_id, lineage_class)

        if methylation_row is None:
            self.methylation_updates_history.append(None)
        else:
            mrow = np.asarray(methylation_row, dtype=float).reshape(1, -1)
            self.methylation_updates_history.append(mrow)
            # Keep the cached drifted matrix in sync so the new lineage is
            # immediately visible to get_bulk_methylation and the mutation-
            # spawning loop without needing a full history rebuild.
            if hasattr(self, '_current_drifted_matrix') and self._current_drifted_matrix is not None:
                self._current_drifted_matrix = np.vstack(
                    [self._current_drifted_matrix, mrow.astype(self._current_drifted_matrix.dtype)]
                )

        return new_id

    def get_methylation_matrix_at_time(self, time_point):
        if self.initial_methylation_matrix is None:
            return None

        # searchsorted is O(log N) vs O(N) argmin; _time_arr is kept sorted
        t_idx = int(np.clip(
            np.searchsorted(self._time_arr, time_point, side='right') - 1,
            0, len(self._time_arr) - 1
        ))

        # Only apply updates up to the timestep corresponding to t_idx
        updates_to_apply = [
            upd for upd in self.methylation_updates_history[1:t_idx + 1]
            if upd is not None
        ]

        if not updates_to_apply:
            # No new lineages added — return original directly (read-only, no copy needed)
            return self.initial_methylation_matrix

        return np.vstack([self.initial_methylation_matrix] + updates_to_apply)


    def get_vafs(self):
        vafs = []
        lineage_info = []

        for i in range(len(self.counts)):
            r = self.p_ss[i] - self.p_dd[i]
            if r > 0:  # Only lineages with net growth bias
                clonality = self.counts[i]   # Fraction of total cells
                vaf = clonality / 2.0  # VAF is half of clonality for heterozygous variants
                vafs.append(vaf)
                lineage_info.append({
                    'vaf': vaf,
                    'clonality': clonality,
                    'p_ss': self.p_ss[i],
                    'p_sd': self.p_sd[i],
                    'p_dd': self.p_dd[i],
                    'tau': self.tau[i],
                    'a': self.a[i],
                    'is_mutated': True,
                    'age': self.current_time - self.lineage_ages[i],
                    'lineage_id': i
                })

        return np.array(vafs), lineage_info

    def run_simulation(self, time_points):
        """Run simulation for specified time points"""
        results = {}
        max_t = max(time_points)
        while self.current_time < max_t:
            self.simulate_timestep()

        for t in time_points:
            t_idx = int(np.clip(
                np.searchsorted(self._time_arr, t, side='right') - 1,
                0, len(self._time_arr) - 1
            ))
            vafs, lineage_info = self.get_vafs()
            sym = self.symmetry_history[t_idx]
            results[t] = {
                'vafs': vafs,
                'lineage_info': lineage_info,
                'total_lineages': len(self.counts_history[t_idx]),
                'mutated_lineages': int(np.sum(sym[:, 0] - sym[:, 2] > 0))
            }
        return results

    def get_all_vafs_at_timepoint(self, timepoint=None):
        """
        Get VAFs for ALL lineages at a specific timepoint.

        Parameters:
        - timepoint: float, the time at which to retrieve VAFs. If None, uses current time.

        Returns:
        - vafs: numpy array of VAFs for all lineages
        - lineage_info: list of dictionaries containing detailed info for each lineage
        """
        if timepoint is None:
            # Use current state
            counts = self.counts
            p_ss = self.p_ss
            p_sd = self.p_sd
            p_dd = self.p_dd
            tau = self.tau
            a = self.a
            lineage_ages = self.lineage_ages
            lineage_ids = self.lineage_ids
            actual_time = self.current_time
        else:
            # Find the closest timepoint in history
            if not self.time_history:
                raise ValueError("No simulation history available. Run simulation first.")

            t_idx = int(np.clip(
                np.searchsorted(self._time_arr, timepoint, side='right') - 1,
                0, len(self._time_arr) - 1
            ))
            actual_time = float(self._time_arr[t_idx])

            # Get state at that timepoint from history
            counts = self.counts_history[t_idx]
            n = len(counts)
            symmetry = self.symmetry_history[t_idx]  # shape (n_lineages, 3)
            p_ss = symmetry[:, 0]
            p_sd = symmetry[:, 1]
            p_dd = symmetry[:, 2]
            tau = self.tau[:n]
            a = self.a[:n]
            lineage_ages = self.lineage_ages[:n]
            lineage_ids = self.lineage_ids[:n]

        # Calculate clonality and VAF for all lineages
        clonality = counts   # Fraction of total cells
        vafs = clonality / 2.0  # VAF is half of clonality for heterozygous variants

        # Create detailed lineage information
        lineage_info = []
        for i in range(len(counts)):
            age_at_timepoint = actual_time - lineage_ages[i] if i < len(lineage_ages) else actual_time
            p_ss_i = float(p_ss[i]) if i < len(p_ss) else 0.34
            p_sd_i = float(p_sd[i]) if i < len(p_sd) else 0.33
            p_dd_i = float(p_dd[i]) if i < len(p_dd) else 0.33
            tau_i = float(tau[i]) if i < len(tau) else self._tau_init
            a_i = float(a[i]) if i < len(a) else 1.0

            info = {
                'lineage_index': i,
                'lineage_id': lineage_ids[i] if i < len(lineage_ids) else i,
                'vaf': vafs[i],
                'clonality': clonality[i],
                'count': counts[i],
                'p_ss': p_ss_i,
                'p_sd': p_sd_i,
                'p_dd': p_dd_i,
                'tau': tau_i,
                'a': a_i,
                'age': age_at_timepoint,
                'is_mutated': p_ss_i - p_dd_i > 0,
                'timepoint': actual_time
            }
            lineage_info.append(info)

        return vafs, lineage_info

    def sample_cells_for_methylation_analysis(self, n_cells=1000, timepoint=None, lineage_filter=None):
        """
        Sample individual cells from the population for methylation analysis.

        Converts the per-lineage mean methylation representation into diploid
        per-cell allele counts {0, 1, 2} (UU / MU / MM), incorporating
        intra-clone epimutation variance via beta-binomial sampling where a
        variance model is available.

        For each lineage, the within-clone methylation distribution is modelled
        as Beta(α, β) parameterised by the lineage mean and the expected
        variance at that lineage's age under the two-state Markov model
        (cpg_var_at_time).  A single draw from this beta gives the allele
        fraction for that cell; a Binomial(2, p) draw then gives the diploid
        state.  When no variance model is available the beta collapses to a
        point mass and the draw reduces to Binomial(2, mean).

        Parameters:
        - n_cells: Number of cells to sample (default 1000)
        - timepoint: Time at which to sample. If None, uses current time.
        - lineage_filter: Optional function to filter lineages (e.g., lambda info: info['is_mutated'])

        Returns:
        - methylation_samples: Array of shape (n_cells, n_CpG_sites) with
                               diploid allele counts {0, 1, 2}
        - cell_info: List of dictionaries containing lineage information for each sampled cell
        """
        if self.initial_methylation_matrix is None:
            raise ValueError("No methylation matrix available. Initialize simulation with methylation_matrix parameter.")

        # Get current population state
        vafs, lineage_info = self.get_all_vafs_at_timepoint(timepoint)

        # Apply lineage filter if provided
        if lineage_filter is not None:
            valid_indices = [i for i, info in enumerate(lineage_info) if lineage_filter(info)]
            lineage_info = [lineage_info[i] for i in valid_indices]
            vafs = vafs[valid_indices]
        else:
            valid_indices = list(range(len(lineage_info)))

        if len(lineage_info) == 0:
            raise ValueError("No lineages match the filter criteria.")

        # Get methylation matrix at this timepoint
        methylation_matrix = self.get_methylation_matrix_at_time(
            timepoint if timepoint is not None else self.current_time
        )
        methylation_matrix = methylation_matrix[valid_indices, :]

        # Calculate sampling probabilities based on lineage sizes
        counts = np.array([info['count'] for info in lineage_info])
        sampling_probs = counts / np.sum(counts)

        # Sample cells from lineages
        sampled_lineage_indices = np.random.choice(
            len(lineage_info),
            size=n_cells,
            p=sampling_probs,
            replace=True
        )

        n_cpg_sites = methylation_matrix.shape[1]

        # ------------------------------------------------------------------
        # Precompute beta-binomial parameters per lineage.
        #
        # For each lineage i and CpG j:
        #   φ   = m(1-m) / v  - 1        (precision; → ∞ as v → 0)
        #   α   = m · φ,  β = (1-m) · φ
        #
        # When v ≈ 0 (young lineage or clonal marker), φ → large and the
        # Beta concentrates on m, recovering Binomial(2, m) in the limit.
        # When v → m(1-m) (old lineage at equilibrium), φ → 0 and the Beta
        # becomes maximally dispersed (U-shaped).
        # ------------------------------------------------------------------
        has_var_model = (
            hasattr(self, 'cpg_var_at_time') and
            self.cpg_types is not None
        )

        if has_var_model:
            # ------------------------------------------------------------------
            # Vectorised beta-binomial parameter computation.
            # Evaluate variance for ALL lineages × ALL CpGs in one shot.
            # ------------------------------------------------------------------
            ages_arr = np.array([max(float(info['age']), 0.0) for info in lineage_info])

            if hasattr(self, 'scarlet_params') and self.scarlet_params is not None:
                from scarlet_translation import _var_Z_order_1_vec as _sv
                _sp = self.scarlet_params
                # (n_lineages, n_cpg) in one vectorised call
                all_cpg_vars = _sv(ages_arr, _sp.s, _sp.N, _sp.eta, _sp.omega, _sp.p, _sp.c)
                all_cpg_vars = np.where(np.isfinite(all_cpg_vars), all_cpg_vars, 0.0)
            else:
                # Two-state Markov: outer product of stationary variance × decay envelope
                # cpg_var_stationary: (n_cpg,); ages_arr: (n_lineages,)
                all_cpg_vars = (
                    self.cpg_var_stationary[np.newaxis, :]
                    * (1.0 - np.exp(-2.0 * self.cpg_omega[np.newaxis, :] * ages_arr[:, np.newaxis]))
                )  # (n_lineages, n_cpg)

            m = np.clip(methylation_matrix.astype(np.float64), 1e-6, 1.0 - 1e-6)
            max_var = m * (1.0 - m)
            v   = np.minimum(all_cpg_vars.astype(np.float64), max_var * (1.0 - 1e-7))
            phi = np.where(v < 1e-10, 1e6, max_var / np.maximum(v, 1e-12) - 1.0)
            bb_alpha = np.maximum(m * phi, 1e-6)   # (n_lineages, n_cpg)
            bb_beta  = np.maximum((1.0 - m) * phi, 1e-6)

        # ------------------------------------------------------------------
        # Draw cell states — fully vectorised: no per-cell Python loop
        # ------------------------------------------------------------------
        if has_var_model:
            # Gather per-cell beta parameters by fancy indexing, then draw in one call
            alpha_cells = bb_alpha[sampled_lineage_indices]   # (n_cells, n_cpg)
            beta_cells  = bb_beta[sampled_lineage_indices]    # (n_cells, n_cpg)
            p_draws = np.random.beta(alpha_cells, beta_cells)
            methylation_samples = np.random.binomial(2, p_draws).astype(np.int8)
        else:
            probs = methylation_matrix[sampled_lineage_indices].astype(np.float64)
            methylation_samples = np.random.binomial(2, probs).astype(np.int8)

        # Build cell_info list (dict building is unavoidable but cheap vs. the draws)
        cell_info = [
            {
                'cell_id': cell_idx,
                'lineage_index': lineage_info[lin_idx]['lineage_index'],
                'lineage_id':    lineage_info[lin_idx]['lineage_id'],
                'p_ss':          lineage_info[lin_idx]['p_ss'],
                'p_sd':          lineage_info[lin_idx]['p_sd'],
                'p_dd':          lineage_info[lin_idx]['p_dd'],
                'is_mutated':    lineage_info[lin_idx]['is_mutated'],
                'vaf':           lineage_info[lin_idx]['vaf'],
                'lineage_age':   lineage_info[lin_idx]['age'],
            }
            for cell_idx, lin_idx in enumerate(sampled_lineage_indices)
        ]

        return methylation_samples, cell_info

    # ------------------------------------------------------------------
    # AnnData backend methods (US-001)
    # ------------------------------------------------------------------

    def get_adata_at_time(self, time_point: float | None = None) -> AnnData | None:
        """Get the AnnData object at a specific time point.

        Parameters
        ----------
        time_point : float or None
            Time point to retrieve. If None, returns current state.

        Returns
        -------
        AnnData or None
            AnnData object with shape (n_lineages, n_cpg), or None if no methylation data.

        Notes
        -----
        The returned AnnData has:
        - .X: methylation beta values in [0, 1]
        - .var_names: CpG identifiers
        - .obs_names: lineage IDs as strings
        """
        if self._adata is None and self.initial_methylation_matrix is None:
            return None

        # Get the methylation matrix at the specified time
        if time_point is None:
            matrix = self.get_current_methylation_matrix()
        else:
            matrix = self.get_methylation_matrix_at_time(time_point)

        if matrix is None:
            return None

        n_lineages, n_cpg = matrix.shape

        # Get var_names and obs_names from stored _adata if available
        if self._adata is not None:
            var_names = list(self._adata.var_names)
            # Extend var_names if needed (for new lineages added during simulation)
            while len(var_names) < n_cpg:
                var_names.append(f"CpG_{len(var_names)}")

            obs_names = [str(lid) for lid in self.lineage_ids[:n_lineages]]
        else:
            # Fallback to default names
            var_names = [f"CpG_{i}" for i in range(n_cpg)]
            obs_names = [str(i) for i in range(n_lineages)]

        return AnnData(
            X=matrix,
            obs=pd.DataFrame(index=obs_names[:n_lineages]),
            var=pd.DataFrame(index=var_names[:n_cpg])
        )

    def get_current_methylation_matrix(self):
        if hasattr(self, '_current_drifted_matrix') and self._current_drifted_matrix is not None:
            return self._current_drifted_matrix
        return self.get_methylation_matrix_at_time(self.current_time)


    def update_adata_history(self):
        """Update the AnnData history with current state.

        This should be called after each simulation timestep to record
        the state of methylation data at that time point.
        """
        if self._adata is not None or self.initial_methylation_matrix is not None:
            current_adata = self.get_adata_at_time()
            self._adata_history.append(current_adata)

    def get_adata_history(self, include_current: bool = True) -> list[AnnData | None]:
        """Get the history of AnnData snapshots.

        Parameters
        ----------
        include_current : bool, default True
            Whether to include the current state (not yet in history).

        Returns
        -------
        list of AnnData or None
            List of AnnData objects at each recorded time point.
        """
        if include_current and len(self._adata_history) > 0:
            return self._adata_history.copy()
        return self._adata_history[:-1].copy() if self._adata_history else []

    # ------------------------------------------------------------------
    # HWE-aware methylation initializer (US-003)
    # ------------------------------------------------------------------

    @staticmethod
    def init_methylation_matrix(
        n_lineages: int,
        n_cpg: int,
        cpg_types: np.ndarray,
        cpg_means: np.ndarray,
        fidelity_noise: float = 0.05
    ) -> np.ndarray:
        """Initialize methylation matrix using HWE for clonal markers.

        For clonal marker CpGs, draws genotypes from Hardy-Weinberg equilibrium
        states {UU, MU, MM} with probabilities [(1-p)², 2p(1-p), p²], then maps
        to methylation beta values {0.0, 0.5, 1.0}.

        For drift CpGs, initializes all lineages to the population mean plus small noise.

        Parameters
        ----------
        n_lineages : int
            Number of lineages (rows in output matrix).
        n_cpg : int
            Number of CpG sites (columns in output matrix).
        cpg_types : np.ndarray of shape (n_cpg,)
            Array of 'clonal_marker' or 'drift' strings.
        cpg_means : np.ndarray of shape (n_cpg,)
            Population mean methylation level for each CpG.
        fidelity_noise : float, default 0.05
            Standard deviation of Gaussian noise added to clonal marker values.

        Returns
        -------
        np.ndarray of shape (n_lineages, n_cpg), dtype float16
            Initialized methylation matrix with values in [0, 1].
        """
        matrix = np.zeros((n_lineages, n_cpg), dtype=np.float16)

        cpg_types_arr = np.asarray(cpg_types)
        # Use float64 for intermediate HWE calculations to avoid precision loss
        cpg_means_arr = np.asarray(cpg_means, dtype=np.float64)

        # Separate clonal marker and drift CpGs
        clonal_mask = cpg_types_arr == 'clonal_marker'
        drift_mask = ~clonal_mask

        if np.any(clonal_mask):
            # Vectorised HWE draw for all clonal-marker CpGs at once.
            # p: (n_clonal,); r: (n_lineages, n_clonal)
            p = cpg_means_arr[clonal_mask]                  # (n_clonal,)
            cum_uu = (1.0 - p) ** 2                         # P(UU)
            cum_mu = cum_uu + 2.0 * p * (1.0 - p)           # P(UU) + P(MU)
            r = np.random.random((n_lineages, p.shape[0]))
            genotypes = np.where(r < cum_uu, 0, np.where(r < cum_mu, 1, 2))
            matrix[:, clonal_mask] = (genotypes * 0.5).astype(np.float16)
            # Add fidelity noise and clip
            noise = np.random.normal(0.0, fidelity_noise, size=(n_lineages, p.shape[0]))
            matrix[:, clonal_mask] = np.clip(
                matrix[:, clonal_mask].astype(np.float32) + noise, 0.0, 1.0
            ).astype(np.float16)

        if np.any(drift_mask):
            # Vectorised HWE draw for all drift CpGs at once.
            p = cpg_means_arr[drift_mask]                   # (n_drift,)
            cum_uu = (1.0 - p) ** 2
            cum_mu = cum_uu + 2.0 * p * (1.0 - p)
            r = np.random.random((n_lineages, p.shape[0]))
            genotypes = np.where(r < cum_uu, 0, np.where(r < cum_mu, 1, 2))
            matrix[:, drift_mask] = (genotypes * 0.5).astype(np.float16)

        return matrix


    def get_bulk_methylation(self) -> np.ndarray | None:
        current_matrix = self.get_current_methylation_matrix()
        if current_matrix is None:
            return None

        total_count = np.sum(self.counts)
        if total_count == 0:
            return np.zeros(current_matrix.shape[1], dtype=np.float16)

        n = min(len(self.counts), current_matrix.shape[0])
        weights = self.counts[:n]
        bulk = np.sum(weights[:, np.newaxis] * current_matrix[:n, :], axis=0) / np.sum(weights)
        return bulk.astype(np.float16)


    def get_lineage_vafs(
        self,
        background_class: str = 'initial'
    ) -> dict[str | int, float]:
        """Get ground truth VAF for all non-background lineages.

        Parameters
        ----------
        background_class : str, default 'initial'
            Class name to filter out as background (not included in output).

        Returns
        -------
        dict[str|int, float]
            Dictionary mapping lineage_id → vaf for each non-background lineage.
            VAF computed as counts[i] / sum(counts) for heterozygous variants.
            Empty dict if no non-background lineages exist.

        Notes
        -----
        VAF is calculated as half the clonality (fraction of total cells),
        assuming heterozygous mutations. For homozygous mutations, multiply by 2.
        """
        result: dict[str | int, float] = {}

        total_count = np.sum(self.counts)
        if total_count == 0:
            return result

        for idx, lid in enumerate(self.lineage_ids):
            lineage_class = self.lineage_classes.get(lid, 'derived')
            if lineage_class != background_class:
                # VAF = clonality / 2 (for heterozygous variant)
                vaf = float(self.counts[idx]) / total_count / 2.0
                result[lid] = vaf

        return result

    # ------------------------------------------------------------------
    # AnnData export (US-007)
    # ------------------------------------------------------------------

    def to_ann_data(
        self,
        obs_metadata: dict[str, np.ndarray] | None = None
    ) -> AnnData:
        """Export current simulation state as AnnData.

        Creates an AnnData object containing bulk methylation values with
        CpG metadata in .var and optional observation-level metadata in .obs.

        Parameters
        ----------
        obs_metadata : dict or None, default None
            Optional dictionary of observation-level metadata to add to .obs.
            Keys become column names, values should be arrays of appropriate length.
            For single individual export, use scalar-wrapping arrays like {'age': [45.0]}.

        Returns
        -------
        AnnData
            AnnData object with:
            - .X: bulk methylation (shape (n_cpg,) for single sample)
            - .var_names: CpG identifiers
            - .var['cpg_type']: 'clonal_marker' or 'drift' for each CpG
            - .var['cpg_mean']: population mean methylation per CpG
            - .obs: observation metadata if provided

        Notes
        -----
        Layers 'methylated_reads' and 'total_reads' are populated if available
        from the underlying AnnData backend.
        """
        if not _ANNDATA_AVAILABLE:
            raise ImportError("anndata must be installed to use to_ann_data()")

        # Get bulk methylation as .X (transposed to shape (n_cpg,) for single sample)
        bulk_meth = self.get_bulk_methylation()
        if bulk_meth is None:
            raise ValueError("No methylation data available. Initialize with adata parameter.")

        n_cpg = len(bulk_meth)

        # Determine CpG identifiers
        if hasattr(self, 'cpg_ids') and self.cpg_ids is not None:
            var_names = list(self.cpg_ids)
        elif self._adata is not None and len(self._adata.var_names) > 0:
            var_names = list(self._adata.var_names)
        else:
            var_names = [f"CpG_{i}" for i in range(n_cpg)]

        # Build var DataFrame with CpG metadata
        var_data: dict[str, np.ndarray] = {}

        if self.cpg_types is not None:
            var_data['cpg_type'] = np.asarray(self.cpg_types)[:n_cpg]

        var_data['cpg_mean'] = np.asarray(self.cpg_means, dtype=np.float16)[:n_cpg]

        # Build obs DataFrame with optional metadata
        obs_data: dict[str, np.ndarray] = {}
        if obs_metadata is not None:
            for key, value in obs_metadata.items():
                obs_data[key] = np.asarray(value)

        # Create AnnData object with shape (n_obs=1, n_vars=n_cpg) for single sample
        X_data = bulk_meth.reshape(1, -1)  # Shape (1, n_cpg)

        adata = AnnData(
            X=X_data,
            var=pd.DataFrame(index=var_names[:n_cpg], data={k: v for k, v in var_data.items() if len(v) == n_cpg}),
            obs=pd.DataFrame(data=obs_data) if obs_data else pd.DataFrame(index=[0])
        )

        # Add layers if available from current state
        current_matrix = self.get_current_methylation_matrix()
        if current_matrix is not None and self._adata is not None:
            try:
                # Extract read-level data if stored in layers
                if 'methylated_reads' in self._adata.layers:
                    methylated_reads_bulk = np.sum(
                        self.counts[:, np.newaxis] * self._adata.layers['methylated_reads'], axis=0
                    ) / np.sum(self.counts)
                    adata.layers['methylated_reads'] = methylated_reads_bulk.reshape(1, -1)

                if 'total_reads' in self._adata.layers:
                    total_reads_bulk = np.sum(
                        self.counts[:, np.newaxis] * self._adata.layers['total_reads'], axis=0
                    ) / np.sum(self.counts)
                    adata.layers['total_reads'] = total_reads_bulk.reshape(1, -1)
            except (KeyError, IndexError):
                pass  # Layers not available or incompatible shapes

        return adata


# ============================================================================
# Module-level functions
# ============================================================================

def generate_synthetic_cohort(
    n_individuals: int,
    age_min: float,
    age_max: float,
    sim_factory: callable,
    n_jobs: int = -1,
    retry_attempts: int = 3,
    sim_factory_kwargs=None,
    return_sims=True
) -> tuple[np.ndarray, np.ndarray, list[dict], Union[list[ClonalSim], list[None]]]:
    """Generate a cross-sectional synthetic cohort with variable ages.

    For each individual, draws an age uniformly from [age_min, age_max],
    runs a ClonalSim simulation until that age, and extracts bulk methylation
    and ground truth VAFs.

    Parameters
    ----------
    n_individuals : int
        Number of individuals to simulate.
    age_min : float
        Minimum age in years (inclusive).
    age_max : float
        Maximum age in years (inclusive).
    sim_factory : callable
        Zero-argument callable that returns a fully initialized ClonalSim instance.
        The factory is called once per individual (potentially in parallel).
    n_jobs : int, default -1
        Number of parallel jobs. -1 means use all available CPUs.
        Ignored if joblib is not available; falls back to serial execution.
    retry_attempts : int, default 3
        Maximum retries per individual on exception. If all attempts fail,
        the entire cohort generation raises immediately.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[dict]]
        - ages: shape (n_individuals,), dtype float16, age of each individual
        - bulk_methylation: shape (n_individuals, n_cpg), dtype float16
        - ground_truth_vafs: list of dicts, VAF dictionaries for each individual

    Raises
    ------
    Exception
        If sim_factory fails after retry_attempts for any individual.

    Examples
    --------
    >>> def make_sim():
    ...     return ClonalSim(N_init=100, adata=my_adata, p_u_to_m=1e-8, transition_generator=1e-8)
    ...
    >>> ages, bulk_meth, vafs, sims = generate_synthetic_cohort(
    ...     n_individuals=50, age_min=30, age_max=90, sim_factory=make_sim
    ... )
    """
    if sim_factory_kwargs is None:
        sim_factory_kwargs = {}

    def _simulate_individual(args: tuple) -> tuple[float, np.ndarray, dict, ClonalSim]:
        """Simulate a single individual with retry logic.

        Parameters
        ----------
        args : tuple
            (individual_idx, age_min, age_max, sim_factory, retry_attempts)

        Returns
        -------
        tuple[float, np.ndarray, dict]
            (age, bulk_methylation_array, vaf_dict)
        """
        individual_idx, amin, amax, factory, retries, sim_factory_kwargs, return_sims = args

        # Draw random age
        target_age = np.random.uniform(amin, amax)

        last_exc = None
        for attempt in range(retries):
            try:
                # Create fresh simulation instance
                sim = factory(**sim_factory_kwargs)

                # Run simulation until reaching target age
                while sim.current_time < target_age:
                    sim.simulate_timestep()

                # Extract bulk methylation and VAFs
                bulk_meth = sim.get_bulk_methylation()
                if bulk_meth is None:
                    raise RuntimeError("get_bulk_methylation() returned None")

                vafs = sim.get_lineage_vafs(background_class='initial')

                if not return_sims:
                    del sim
                    sim = None

                vafs_arr = np.array(sorted(list(vafs.values())))

                print(f"Age {target_age:.1f}, max VAF: {np.max(vafs_arr):.3g}, Num. VAFS: {(vafs_arr > 0).sum()} ")

                return float(target_age), np.asarray(bulk_meth, dtype=np.float16), vafs, sim

            except Exception as e:
                last_exc = e
                warnings.warn(f"Error in individual {individual_idx}:\n{e}")
                if attempt < retries - 1:
                    continue  # Retry
                raise RuntimeError(
                    f"Individual {individual_idx} failed after {retries} attempts. "
                    f"Last error: {e}"
                ) from last_exc

    # Prepare arguments for parallel execution
    args_list = [
        (i, age_min, age_max, sim_factory, retry_attempts, sim_factory_kwargs, return_sims)
        for i in range(n_individuals)
    ]

    # Try joblib for parallelization, fall back to serial
    try:
        if n_jobs != 1:
            from joblib import Parallel, delayed
            from tqdm import tqdm
            try:
                from tqdm_joblib import tqdm_joblib
                with tqdm_joblib(tqdm(total=len(args_list), desc="Simulating")):
                    results = Parallel(n_jobs=n_jobs)(
                        delayed(_simulate_individual)(args) for args in args_list
                    )
            except ImportError:
                # tqdm_joblib not available, run parallel without progress bar
                results = Parallel(n_jobs=n_jobs)(
                    delayed(_simulate_individual)(args) for args in args_list
                )
        else:
            from tqdm import tqdm
            results = [_simulate_individual(args) for args in tqdm(args_list, desc="Simulating")]

    except ImportError:
        # joblib not available, serial fallback
        results = [_simulate_individual(args) for args in args_list]
    # Unpack results
    ages = np.array([r[0] for r in results], dtype=np.float16)
    bulk_methylation = np.vstack([r[1] for r in results]).astype(np.float16)
    ground_truth_vafs = [r[2] for r in results]
    sims = [r[3] for r in results]


    return ages, bulk_methylation, ground_truth_vafs, sims