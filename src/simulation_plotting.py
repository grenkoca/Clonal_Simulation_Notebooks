import numpy as np
import matplotlib.pyplot as plt
from .clonal_evolution import ClonalSim
from .util import gini

def plot_lineage_dendrogram(sim: ClonalSim, min_count=0.0, show_background=False,
                            color_by='type', linewidth_by='count',
                            figsize=(14, 8), ax=None):
    """
    Plot a dendrogram of the simulated lineage histories.

    Each lineage is drawn as a horizontal branch spanning its lifetime from
    birth to the end of the simulation.  Vertical connectors link a parent
    lineage to each of its offspring at the moment the offspring arose.
    The tree layout places leaves at integer y-positions and internal nodes
    at the midpoint of their children's y-range, mirroring the standard
    phylogenetic dendrogram convention.

    Parameters
    ----------
    sim : ClonalSim
        Simulation object
    min_count : float, default 0.0
        Minimum final cell count a lineage must have to appear in the plot.
    show_background : bool, default False
        If True, include all ``N_init`` background (neutral) lineages even
        when they never spawned a mutant.  When False (default) only
        background lineages that are direct ancestors of at least one
        visible mutant are shown.
    color_by : {'type', 'r', 'count'}, default 'type'
        Colour scheme applied to branches.

        * ``'type'``  – blue for background lineages, red for mutants.
        * ``'r'``     – colour encodes the net growth bias *r = p_ss − p_dd*
          on a blue→red diverging scale (neutral = white/grey).
        * ``'count'`` – colour encodes final cell count on a viridis scale.
    linewidth_by : {'count', 'uniform'}, default 'count'
        ``'count'`` scales branch width proportionally to the square-root of
        the final cell count so dominant clones are visually prominent.
        ``'uniform'`` draws all branches at the same width.
    figsize : tuple of float, default (14, 8)
        Figure size in inches (used only when *ax* is ``None``).
    ax : matplotlib.axes.Axes or None, default None
        Target axes object.  A new figure is created when ``None``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes

    Notes
    -----
    Parent–child relationships are recorded automatically during
    :meth:`simulate_timestep`.  Running the dendrogram on a simulation
    object that was run *before* this tracking was added will produce a
    flat layout (all mutants shown as independent roots).
    """
    from matplotlib.lines import Line2D

    end_time = sim.current_time

    # ── Lookup helpers ────────────────────────────────────────────────────
    id_to_idx = {lid: i for i, lid in enumerate(sim.lineage_ids)}
    final_counts = {lid: float(sim.counts[i]) for i, lid in enumerate(sim.lineage_ids)}
    birth_times = {lid: float(sim.lineage_ages[i]) for i, lid in enumerate(sim.lineage_ids)}
    is_bg = {lid: (lid < sim.N_init) for lid in sim.lineage_ids}

    # ── Determine the visible set of lineage IDs ──────────────────────────
    visible = {lid for lid in sim.lineage_ids if final_counts.get(lid, 0) >= min_count}

    if not show_background:
        # Keep only mutant lineages + their background ancestors
        mutant_visible = {lid for lid in visible if not is_bg[lid]}
        ancestor_bg = set()
        for lid in mutant_visible:
            pid = sim.lineage_parents.get(lid)
            while pid is not None:
                if is_bg.get(pid, False):
                    ancestor_bg.add(pid)
                pid = sim.lineage_parents.get(pid)
        visible = mutant_visible | ancestor_bg

    if not visible:
        print("No lineages to display. Lower min_count or set show_background=True.")
        return None, None

    # ── Build children map for visible nodes ──────────────────────────────
    children_map = {lid: [] for lid in visible}
    for lid in visible:
        pid = sim.lineage_parents.get(lid)
        if pid is not None and pid in visible:
            children_map[pid].append(lid)

    roots = [lid for lid in visible if sim.lineage_parents.get(lid) not in visible]

    # ── Assign y-positions: leaves get integer slots, parents get midpoint ─
    y_positions = {}
    y_counter = [0]

    def _assign_y(node):
        kids = sorted(children_map.get(node, []), key=lambda c: birth_times.get(c, 0))
        if not kids:
            y_positions[node] = float(y_counter[0])
            y_counter[0] += 1
        else:
            for kid in kids:
                _assign_y(kid)
            child_ys = [y_positions[c] for c in kids]
            y_positions[node] = (min(child_ys) + max(child_ys)) / 2.0

    for root in sorted(roots, key=lambda lid: birth_times.get(lid, 0)):
        _assign_y(root)

    # ── Colour helpers ────────────────────────────────────────────────────
    final_sym = sim.symmetry_history[-1]
    n_sym = final_sym.shape[0]
    max_count = max(final_counts.values()) if final_counts else 1.0

    def _color(lid):
        if color_by == 'r':
            idx = id_to_idx.get(lid, 0)
            r = float(final_sym[idx, 0] - final_sym[idx, 2]) if idx < n_sym else 0.0
            norm = plt.Normalize(-0.15, 0.15)
            return plt.cm.RdBu_r(norm(r))
        elif color_by == 'count':
            norm = plt.Normalize(0, np.log1p(max_count))
            return plt.cm.viridis(norm(np.log1p(final_counts.get(lid, 0))))
        else:  # 'type'
            return '#4C78A8' if is_bg.get(lid, False) else '#E45756'

    def _lw(lid):
        if linewidth_by == 'count':
            count = max(final_counts.get(lid, 0), 0.1)
            return 0.6 + 4.5 * (count / max_count) ** 0.5
        return 1.5

    # ── Draw ──────────────────────────────────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    for lid in visible:
        x0 = birth_times.get(lid, 0.0)
        y = y_positions[lid]
        color = _color(lid)
        lw = _lw(lid)

        # Horizontal branch: birth → simulation end
        ax.plot([x0, end_time], [y, y], color=color, linewidth=lw,
                solid_capstyle='round', zorder=2)

        # Vertical connector from parent y to this lineage's y at birth time
        pid = sim.lineage_parents.get(lid)
        if pid is not None and pid in visible:
            y_parent = y_positions[pid]
            ax.plot([x0, x0], [y_parent, y], color=color,
                    linewidth=max(lw * 0.5, 0.8), alpha=0.75,
                    solid_capstyle='round', zorder=2)

        # Dot at birth for mutant lineages
        if not is_bg.get(lid, False):
            ax.plot(x0, y, 'o', color=color, markersize=5,
                    markeredgecolor='white', markeredgewidth=0.8, zorder=4)

    # ── Formatting ────────────────────────────────────────────────────────
    ax.set_xlabel('Time (years)', fontsize=12)
    ax.set_yticks([])
    ax.set_title('Lineage Dendrogram', fontsize=14, fontweight='bold')
    ax.set_xlim(-end_time * 0.02, end_time * 1.08)
    ax.spines[['left', 'right', 'top']].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Legend
    if color_by == 'type':
        legend_handles = [
            Line2D([0], [0], color='#4C78A8', linewidth=2, label='Background (neutral)'),
            Line2D([0], [0], color='#E45756', linewidth=2,
                   marker='o', markersize=5, markeredgecolor='white',
                   label='Mutant'),
        ]
        ax.legend(handles=legend_handles, loc='upper left', fontsize=10)
    elif color_by == 'r':
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=plt.Normalize(-0.15, 0.15))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Net growth bias r = p_ss − p_dd', shrink=0.6)
    elif color_by == 'count':
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                   norm=plt.Normalize(0, np.log1p(max_count)))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='log(1 + final cell count)', shrink=0.6)

    # Parameter annotation
    n_mutants = sum(1 for lid in visible if not is_bg.get(lid, False))
    n_bg_shown = sum(1 for lid in visible if is_bg.get(lid, False))
    param_text = (f'Mutant lineages: {n_mutants}\n'
                  f'Background shown: {n_bg_shown}\n'
                  f'$t_{{end}}$={end_time:.1f} yr')
    ax.text(0.98, 0.98, param_text, transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
            verticalalignment='top', horizontalalignment='right', fontsize=9)

    plt.tight_layout()
    return fig, ax


def plot_lineage_evolution(sim: ClonalSim, min_proportion=0.001, max_lineages=20,
                           timepoints_to_evaluate=None, shuffle=False, log_y=False, background_palette=None,
                           figsize=(12, 8)):
    if len(sim.time_history) < 2:
        print("Need to run simulation first to generate time history")
        return

    # background_palette overrides the "initial" class palette for this plot only
    _palette_overrides: dict = {}
    if background_palette is not None:
        _palette_overrides["initial"] = list(background_palette)

    # Convert to numpy arrays for easier manipulation
    time_array = np.array(sim.time_history)

    # Create matrix of proportions over time
    max_lineages_count = max(len(counts) for counts in sim.counts_history)
    proportion_matrix = np.zeros((len(sim.time_history), max_lineages_count))

    # Fill the matrix — normalise by K at each timestep so niche contractions
    # (e.g. therapy events) are reflected in the proportions
    K_array = np.array(sim.K_history)
    for t_idx, counts in enumerate(sim.counts_history):
        # norm = K_array[t_idx] if t_idx < len(K_array) else self.total_pop
        proportions = counts
        proportion_matrix[t_idx, :len(proportions)] = counts

    # Identify significant lineages
    final_proportions = proportion_matrix[-1, :]
    # Net growth bias (r = p_ss - p_dd) from final symmetry snapshot
    final_sym = sim.symmetry_history[-1]  # shape (n_lineages, 3)
    final_r = final_sym[:, 0] - final_sym[:, 2]
    if len(final_r) < len(final_proportions):
        final_r = np.pad(final_r, (0, len(final_proportions) - len(final_r)), 'constant')

    max_proportions = np.max(proportion_matrix, axis=0)
    significant_mask = max_proportions >= min_proportion

    significant_indices = np.where(significant_mask)[0]
    if len(significant_indices) > max_lineages:
        final_props_significant = final_proportions[significant_indices]
        top_indices = significant_indices[np.argsort(final_props_significant)[::-1][:max_lineages]]
    else:
        top_indices = significant_indices


    fig, ax_stackplot = plt.subplots(figsize=figsize)

    plot_data = []
    labels = []


    for i, idx in enumerate(top_indices):
        lineage_data = proportion_matrix[:, idx]
        plot_data.append(lineage_data)

        lid = sim.lineage_ids[idx] if idx < len(sim.lineage_ids) else idx
        cls = sim.lineage_classes.get(lid, "derived")
        if idx < sim.N_init:
            labels.append(f'[{cls}] {idx}')
        else:
            r_val = final_r[idx] if idx < len(final_r) else 0.0
            labels.append(f'[{cls}] r={r_val:.2g}')

    other_mask = np.ones(proportion_matrix.shape[1], dtype=bool)
    other_mask[top_indices] = False
    other_data = np.sum(proportion_matrix[:, other_mask], axis=1)

    colors = []
    for idx in top_indices:
        lid = sim.lineage_ids[idx] if idx < len(sim.lineage_ids) else idx
        colors.append(sim._resolve_lineage_color(lid, _palette_overrides))

    if np.any(other_data > 0):
        colors.append('#989898')

    if np.any(other_data > 0):
        plot_data.append(other_data)
        labels.append('Other lineages')

    plot_data = np.array(plot_data).T  # shape: (time_steps, n_plotted_cols)

    n_plot_cols = plot_data.shape[1]

    if shuffle:
        shuffle_indices = np.argsort(np.random.random(n_plot_cols))
        plot_data = plot_data[:, shuffle_indices]
        # Shuffle colors and labels to match — but don't shuffle the "Other" entry if present
        has_other = np.any(other_data > 0)
        n_main = n_plot_cols - (1 if has_other else 0)
        main_colors = [colors[i] for i in shuffle_indices[:n_main]]
        main_labels = [labels[i] for i in shuffle_indices[:n_main]]
        if has_other:
            colors = main_colors + [colors[-1]]
            labels = main_labels + [labels[-1]]
        else:
            colors = main_colors
            labels = main_labels
    else:
        shuffle_indices = np.arange(n_plot_cols)

    ax_stackplot.stackplot(time_array, *plot_data.T,
                           labels=labels, colors=colors[:len(labels)],
                           alpha=0.8, linewidth=0.25,
                           edgecolor='black', zorder=1)
    if log_y:
        ax_stackplot.semilogy()

    # Identify blocks of times that K diverged from K_init
    block_markers = [[time_array[0], K_array[0]]]
    _running_k = sim._K_init
    for idx, (k, t) in enumerate(zip(K_array, time_array)):
        if k != _running_k:
            block_markers.append([t, k])  # Draw next index for text
            _running_k = k

    if len(block_markers) % 2 != 0 and block_markers[-1][1] != sim._K_init:  # If unmatched, add last time point
        block_markers.append([time_array[-1], K_array[-1]])

    for idx in range(len(block_markers) - 1):
        if idx == 0:
            continue  # Skip starting point
        x0 = block_markers[idx][0]
        height = np.max(np.sum(proportion_matrix, axis=1)) * 1.0
        width = block_markers[idx + 1][0] - block_markers[idx][0]

        rect = plt.Rectangle((x0, 0),
                             width,
                             height,
                             facecolor='lightgray',
                             edgecolor='black',
                             linestyle='dashed',
                             linewidth=2,
                             zorder=0)

        # plt.annotate(xy=(x0 + width / 2, height * 0.98),
        #              text=f"$K$({block_markers[idx - 1][1]}"
        #                   f"\n  $\\rightarrow$ {block_markers[idx][1]})",
        #              va='top', ha='center',
        #              fontsize=8,
        #              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='black'))
        # plt.gca().add_patch(rect)
        idx += 1  # skip next index

    if timepoints_to_evaluate is None:
        timepoints_to_evaluate = []

    for t in timepoints_to_evaluate:
        t_idx_hist = np.argmin(np.abs(time_array - t))
        t_actual = time_array[t_idx_hist]

        all_vafs = sim.get_all_vafs_at_timepoint(t_actual)[0]

        # Build a VAF array aligned with plot_data columns (before shuffle):
        # columns 0..len(top_indices)-1 map to top_indices; last column is "Other" (VAF=0)
        pre_shuffle_vafs = np.zeros(n_plot_cols)
        for col_i, lin_idx in enumerate(top_indices):
            if lin_idx < len(all_vafs):
                pre_shuffle_vafs[col_i] = all_vafs[lin_idx]

        # Apply the same shuffle so VAFs align with the (shuffled) plot_data columns
        plot_vafs = pre_shuffle_vafs[shuffle_indices]

        row_proportions = plot_data[t_idx_hist, :]  # already shuffled

        max_vaf_idx = plot_vafs.argmax()
        max_vaf = plot_vafs[max_vaf_idx]

        # Vertical position: middle of the max-VAF lineage's band in the stacked plot
        pos = np.sum(row_proportions[:max_vaf_idx]) + row_proportions[max_vaf_idx] / 2

        # Vertical line at the timepoint
        ax_stackplot.axvline(x=t_actual, color='red', linestyle='-', alpha=0.6, linewidth=2)

        # Point at the timepoint and max VAF level
        ax_stackplot.plot(t_actual, pos, 'ro', markersize=8, markerfacecolor='red', markeredgecolor='darkred',
                          markeredgewidth=2, linewidth=0.05)

        # Label with max VAF value (rounded to 3 decimal places)
        ax_stackplot.annotate(f'{max_vaf:.1f}\n({max_vaf / np.sum(plot_vafs) * 100:.3g}%)',
                              xy=(t_actual, pos),
                              xytext=(5, 10),
                              textcoords='offset points',
                              fontsize=9,
                              color='darkred',
                              fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='red'))

    # Formatting
    ax_stackplot.set_xlabel('Time (years)', fontsize=12)
    ax_stackplot.set_ylabel('$N$ (Total)', fontsize=12)
    # ax.set_ylim(0, 1)
    ax_stackplot.grid(True, alpha=0.3)
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    param_text = (f'$N_{{init}}$={sim.N_init}\n'
                  f'$K_{{init}}$={sim._K_init}\n'
                  f'$τ$={sim._tau_init} yr')
    ax_stackplot.text(0.02, 0.98, param_text, transform=ax_stackplot.transAxes,
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                      verticalalignment='top', fontsize=10)

    plt.tight_layout()
    plt.show()

N_over_s_spec = lambda counts, sim, t: np.sum(counts) * np.mean(sim.tau[:len(counts)]), 'Avg. $N/s$', {}
gini_spec = lambda counts, sim, t: gini(counts), 'Gini Coef.', {'ylim': (0, 1)}
nonzero_spec = lambda counts, sim, t: np.count_nonzero(counts), 'Nonzero Lineages', {'log_y': False}



def plot_population_metrics(
        sim,
        xs,
        metric_specs,  # list of (callable, ylabel, ylim) — ylim=None → auto
        figsize_per_panel=(6, 1.5),
        sharex=True,
        **kwargs
):
    """
    Parameters
    ----------
    sim          : simulation object with .counts_history and .tau
    xs           : x-axis values (e.g. timepoints)
    metric_specs : list of tuples, each:
                     (metric_fn, ylabel, ylim)
                   where metric_fn(counts, sim, t_idx, **kwargs) -> scalar
                   and ylim is (ymin, ymax) or None for auto
    figsize_per_panel : (w, h) per subplot row
    sharex       : share x axis across panels
    **kwargs     : forwarded to stackplot styling + metric_fn calls

    Returns
    -------
    fig, axes

    Example
    -------
    specs = [
        (lambda counts, sim, t: np.sum(counts) * np.mean(sim.tau[:len(counts)]),
         'Avg. $N/s$', None),
        (lambda counts, sim, t: gini(counts),
         'Gini Coef.', (0, 1)),
        (lambda counts, sim, t: np.count_nonzero(counts),
         'Nonzero Lineages', None),
    ]
    fig, axes = plot_population_metrics(sim, xs, specs)
    """
    n = len(metric_specs)
    fig, axes = plt.subplots(
        n, 1,
        figsize=(figsize_per_panel[0], figsize_per_panel[1] * n),
        sharex=sharex,
        squeeze=False  # always 2-D so indexing is consistent
    )
    axes = axes[:, 0]  # flatten to 1-D

    for ax, (metric_fn, ylabel, plot_kwargs) in zip(axes, metric_specs):
        values = [
            metric_fn(sim.get_all_vafs_at_timepoint(x)[0], sim, t_idx, **kwargs)
            for t_idx, x in enumerate(xs)
        ]

        ax.stackplot(
            xs,
            values,
            color=plot_kwargs.get('color', 'lightgray'),
            edgecolor=plot_kwargs.get('edgecolor', 'black'),
            alpha=plot_kwargs.get('alpha', 0.67),
        )

        if 'log_y' in plot_kwargs.keys():
            log_y = plot_kwargs.pop('log_y')
            if log_y:
                ax.semilogy()

        if 'ylim' in plot_kwargs.keys():
            ax.set_ylim(plot_kwargs.pop('ylim'))

        ax.set_ylabel(ylabel, **plot_kwargs)

    if sharex:
        axes[-1].set_xlabel(kwargs.get('xlabel', 'Time (years)'))

    fig.tight_layout()
    return fig, axes


"""
Muller-style lineage evolution plot for ClonalSim.

Derived lineages spawn from the vertical midpoint of their parent's band,
following the classic Muller plot convention used in evolutionary biology.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict


# ---------------------------------------------------------------------------
# Tree / ordering utilities
# ---------------------------------------------------------------------------

def _build_children_map(sim):
    """Return dict: parent_id -> [child_id, ...] in order of emergence."""
    children = defaultdict(list)
    for child_id, parent_id in sim.lineage_parents.items():
        children[parent_id].append(child_id)
    # Sort children by their lineage_age (time of first appearance)
    for parent_id in children:
        children[parent_id].sort(
            key=lambda cid: sim.lineage_ages[sim.lineage_ids.index(cid)]
            if cid in sim.lineage_ids else 0
        )
    return children


def _muller_order(roots, children_map):
    """
    Return a flat list of lineage IDs in 'Muller order':
    children interleaved around their parent so the parent appears in the
    centre of its own band.

    For a parent with children [c1, c2, c3] (by emergence time):
      Layout (bottom to top):  c1, c2, PARENT, c3
      i.e. first half below, last half above.
    This way the parent's midpoint is bracketed symmetrically.
    """
    result = []

    def _place(lid):
        kids = children_map.get(lid, [])
        n = len(kids)
        below = kids[:n // 2]       # earlier half → below parent
        above = kids[n // 2:]       # later half  → above parent

        for k in below:
            _place(k)
        result.append(lid)
        for k in above:
            _place(k)

    for root in roots:
        _place(root)

    return result


# ---------------------------------------------------------------------------
# Core Muller offset computation
# ---------------------------------------------------------------------------

def _compute_muller_offsets(sim, ordered_ids, proportion_matrix, time_array):
    """
    For each (lineage, timestep) compute the *bottom* y-coordinate of the
    lineage's band using the Muller convention.

    The rule is:
        bottom[child, t] = mid[parent, t] - half_height_of_child_subtree[t] / 2

    where mid[parent, t] = bottom[parent, t] + count[parent, t] / 2.

    We implement this by computing, for each node in Muller order, the
    cumulative running baseline so that children are centred on their parent.
    """
    n_time = len(time_array)
    n_lin = len(ordered_ids)

    id_to_idx = {lid: sim.lineage_ids.index(lid) for lid in ordered_ids
                 if lid in sim.lineage_ids}
    order_pos = {lid: i for i, lid in enumerate(ordered_ids)}

    # counts[i, t] for lineage at position i in Muller order
    counts = np.zeros((n_lin, n_time))
    for i, lid in enumerate(ordered_ids):
        col = id_to_idx.get(lid)
        if col is not None and col < proportion_matrix.shape[1]:
            counts[i, :] = proportion_matrix[:, col]

    # Build children map in *order* space
    children_map = _build_children_map(sim)

    def _subtree_ids(lid):
        """All lineage IDs in the subtree rooted at lid (including lid)."""
        result = [lid]
        for c in children_map.get(lid, []):
            result.extend(_subtree_ids(c))
        return result

    # Total 'height' consumed by a subtree at each timestep
    def _subtree_height(lid, t_slice=None):
        ids = _subtree_ids(lid)
        h = np.zeros(n_time)
        for sid in ids:
            col = id_to_idx.get(sid)
            if col is not None and col < proportion_matrix.shape[1]:
                if t_slice is None:
                    h += proportion_matrix[:, col]
                else:
                    h += proportion_matrix[t_slice, col]
        return h

    # Identify roots (background / initial lineages with no parent)
    all_lid_set = set(ordered_ids)
    roots = [lid for lid in ordered_ids
             if lid not in sim.lineage_parents and
             sim.lineage_classes.get(lid, "derived") == "initial"]

    # -----------------------------------------------------------------------
    # Compute bottom offsets recursively
    # -----------------------------------------------------------------------
    bottoms = np.full((n_lin, n_time), np.nan)

    def _assign_bottoms(lid, baseline):
        """
        baseline: array (n_time,) — y-coordinate of the bottom of the
                  *entire subtree* rooted at lid.
        """
        pos = order_pos[lid]
        kids = children_map.get(lid, [])
        n = len(kids)
        below_kids = kids[:n // 2]
        above_kids = kids[n // 2:]

        running = baseline.copy()

        # Place below-children first (bottom → up)
        for k in below_kids:
            _assign_bottoms(k, running)
            running = running + _subtree_height(k)

        # Place the parent itself
        bottoms[pos, :] = running
        running = running + counts[pos, :]

        # Place above-children
        for k in above_kids:
            _assign_bottoms(k, running)
            running = running + _subtree_height(k)

    # Roots sit sequentially from y=0
    baseline = np.zeros(n_time)
    for root in roots:
        _assign_bottoms(root, baseline)
        baseline = baseline + _subtree_height(root)

    # Any lineages not yet placed (derived without tracked parent in ordered_ids)
    for i, lid in enumerate(ordered_ids):
        if np.any(np.isnan(bottoms[i, :])):
            bottoms[i, :] = baseline
            baseline = baseline + counts[i, :]

    return counts, bottoms, id_to_idx


def plot_muller_evolution(sim,
                          min_proportion=0.001,
                          max_lineages=50,
                          timepoints_to_evaluate=None,
                          log_y=False,
                          background_palette=None,
                          figsize=(14, 7),
                          show_legend=True,
                          alpha=0.85,
                          edge_linewidth=0.3):
    """
    Muller-style lineage evolution plot for a ``ClonalSim`` instance.

    Derived lineages spawn from the vertical midpoint of their parent's band,
    following the Muller plot convention: child clones "bud" out of the centre
    of their parent, preserving the visual parent-child relationship.

    Parameters
    ----------
    sim : ClonalSim
    min_proportion : float
        Minimum peak count for a lineage to be shown individually.
    max_lineages : int
        Hard cap on individually shown lineages (beyond this → aggregated into
        "Other derived" and background).
    timepoints_to_evaluate : list of float or None
        Time points at which to annotate the dominant VAF / clone.
    log_y : bool
        Use a logarithmic y-axis.
    background_palette : list of str or None
        Override the default palette for 'initial' class lineages.
    figsize : tuple
    show_legend : bool
    alpha : float
        Fill transparency.
    edge_linewidth : float
        Width of the black edge drawn on each band.
    """
    if len(sim.time_history) < 2:
        print("Need to run simulation first to generate time history")
        return

    _palette_overrides = {}
    if background_palette is not None:
        _palette_overrides["initial"] = list(background_palette)

    time_array = np.array(sim.time_history)
    n_t = len(time_array)
    max_lin = max(len(c) for c in sim.counts_history)

    proportion_matrix = np.zeros((n_t, max_lin))
    for t_idx, counts in enumerate(sim.counts_history):
        proportion_matrix[t_idx, :len(counts)] = counts

    max_proportions = np.max(proportion_matrix, axis=0)
    significant_mask = max_proportions >= min_proportion
    significant_indices = np.where(significant_mask)[0]

    final_proportions = proportion_matrix[-1, :]
    if len(significant_indices) > max_lineages:
        fps = final_proportions[significant_indices]
        top_indices = set(significant_indices[np.argsort(fps)[::-1][:max_lineages]])
    else:
        top_indices = set(significant_indices)

    # Ensure every child whose parent is shown is also shown (for proper Muller tree)
    # (We only enforce one level deep; deeper levels collapse into "Other".)
    children_map = _build_children_map(sim)

    shown_ids = [lid for lid in sim.lineage_ids
                 if sim.lineage_ids.index(lid) in top_indices]

    # Background lineages = roots
    roots = [lid for lid in shown_ids
             if sim.lineage_classes.get(lid, "derived") == "initial"]

    ordered_ids = _muller_order(roots, {
        lid: [c for c in children_map.get(lid, []) if c in shown_ids]
        for lid in shown_ids
    })

    counts_ordered, bottoms, id_to_idx = _compute_muller_offsets(
        sim, ordered_ids, proportion_matrix, time_array
    )

    shown_set = set(id_to_idx.keys())
    other_counts = np.zeros(n_t)
    for col_idx, lid in enumerate(sim.lineage_ids):
        if lid not in shown_set and col_idx < proportion_matrix.shape[1]:
            other_counts += proportion_matrix[:, col_idx]

    fig, ax = plt.subplots(figsize=figsize)
    handles = []

    for i, lid in enumerate(ordered_ids):
        y0 = bottoms[i, :]
        y1 = y0 + counts_ordered[i, :]
        color = sim._resolve_lineage_color(lid, _palette_overrides)
        cls = sim.lineage_classes.get(lid, "derived")

        col = id_to_idx.get(lid)
        final_r = 0.0
        if col is not None and col < len(sim.lineage_ids):
            # Net growth bias at last timepoint
            last_sym = sim.symmetry_history[-1]
            if col < last_sym.shape[0]:
                final_r = last_sym[col, 0] - last_sym[col, 2]

        if cls == "initial":
            label = f"[bg] {lid}"
        else:
            label = f"[derived] r={final_r:.2g}"

        ax.fill_between(time_array, y0, y1,
                        color=color, alpha=alpha,
                        linewidth=edge_linewidth,
                        edgecolor='black',
                        zorder=2)

        patch = mpatches.Patch(color=color, alpha=alpha, label=label)
        handles.append(patch)

    # Other residual — stacked on top
    if np.any(other_counts > min_proportion):
        top_of_shown = np.zeros(n_t)
        for i in range(len(ordered_ids)):
            top_of_shown = np.maximum(top_of_shown, bottoms[i, :] + counts_ordered[i, :])

        ax.fill_between(time_array,
                        top_of_shown,
                        top_of_shown + other_counts,
                        color='#555555', alpha=0.5,
                        linewidth=edge_linewidth,
                        edgecolor='blaclineagesk',
                        label='Other ',
                        zorder=1)
        handles.append(mpatches.Patch(color='#555555', alpha=0.5, label='Other lineages'))

    if timepoints_to_evaluate:
        for t in timepoints_to_evaluate:
            t_idx_hist = np.argmin(np.abs(time_array - t))
            t_actual = time_array[t_idx_hist]

            all_vafs = sim.get_all_vafs_at_timepoint(t_actual)[0]

            # Find the shown lineage with the highest VAF
            best_vaf, best_i = 0.0, None
            for i, lid in enumerate(ordered_ids):
                col = id_to_idx.get(lid)
                if col is not None and col < len(all_vafs):
                    v = all_vafs[col]
                    if v > best_vaf:
                        best_vaf = v
                        best_i = i

            ax.axvline(x=t_actual, color='#ff4444', linestyle='--',
                       alpha=0.8, linewidth=1.5, zorder=5)

            if best_i is not None:
                y_mid = bottoms[best_i, t_idx_hist] + counts_ordered[best_i, t_idx_hist] / 2
                ax.plot(t_actual, y_mid, 'o',
                        color='#ff4444', markersize=8,
                        markeredgecolor='white', markeredgewidth=1.5,
                        zorder=6)
                ax.annotate(f'VAF={best_vaf:.3f}',
                            xy=(t_actual, y_mid),
                            xytext=(8, 0), textcoords='offset points',
                            fontsize=8, color='#ff8888', fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.25',
                                      facecolor='#1a1a2e', alpha=0.9,
                                      edgecolor='#ff4444'),
                            zorder=7)

    if log_y:
        ax.set_yscale('log')

    ax.set_xlabel('Time (years)', fontsize=12, color='#cccccc')
    ax.set_ylabel('N (cells)', fontsize=12, color='#cccccc')
    ax.tick_params(colors='#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')

    param_text = (f'$N_{{init}}$={sim.N_init}\n'
                  f'$τ$={sim._tau_init} yr')
    ax.text(0.02, 0.98, param_text,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e2030",
                      alpha=0.9, edgecolor="#444455"),
            verticalalignment='top', fontsize=9, color='#cccccc')

    if show_legend and len(handles) <= 20:
        leg = ax.legend(handles=handles,
                        bbox_to_anchor=(1.01, 1), loc='upper left',
                        fontsize=8, framealpha=0.15,
                        facecolor='#1a1a2e', edgecolor='#333355',
                        labelcolor='#cccccc')

    ax.set_title('Muller Plot — Clonal Evolution', fontsize=14,
                 color='#eeeeee', pad=12)
    ax.grid(True, alpha=0.12, color='#555555', linestyle=':')

    plt.tight_layout()
    plt.show()
    return fig, ax

def plot_neutral_drift_prediction(
    sim: ClonalSim,
    t_horizon: float,
    t_start: float | None = None,
    n_points: int = 300,
    top_n_selected: int = 5,
    show_actual: bool = True,
    figsize: tuple = (12, 11),
    ax_array=None,
):
    """Plot the analytical neutral drift prediction from a given simulation state.

    Calls :meth:`ClonalSim.predict_neutral_drift_trajectory` and renders three
    stacked panels:

    **Panel 1 — Heterozygosity decay**
        Normalised diversity H(t)/H0 on a log scale, with vertical markers for the
        half-life and the estimated fixation time. If ``show_actual=True``, the
        empirical Simpson heterozygosity computed from ``sim.counts_history`` is
        overlaid as a scatter.

    **Panel 2 — Expected maximum clone frequency**
        The analytical :math:`E[\\max_i f_i(t)]` curve, optionally overlaid with
        the actual observed maximum frequency from history. Also shows the logistic
        deterministic trajectories of any selected (s > 0) lineages.

    **Panel 3 — Fitness landscape**
        Scatter of all lineages at t_start in :math:`(r_i, \\tau_i)` space, where
        :math:`r_i = p^{ss}_i - p^{dd}_i`. Point size encodes initial frequency,
        colour encodes selective advantage :math:`s_i = r_i / \\tau_i`. Iso-s
        contours and vertical/horizontal reference lines are drawn. Lineages with
        :math:`w_i > 1` (expansion faster than drift) are annotated.

    Parameters
    ----------
    sim : ClonalSim
        A simulation object that has been run for at least one timestep.
    t_horizon : float
        End year for the prediction.
    t_start : float or None, default None
        Start year for the prediction. Defaults to ``sim.current_time``.
    n_points : int, default 300
        Resolution of analytical curves.
    top_n_selected : int, default 5
        Maximum number of selected-lineage logistic trajectories to draw in
        panel 2.
    show_actual : bool, default True
        Overlay empirical history on panels 1 and 2 when available.
    figsize : tuple, default (12, 11)
        Figure size.
    ax_array : array-like of 3 Axes or None
        Pre-existing axes to draw into. A new figure is created when ``None``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of 3 matplotlib.axes.Axes
        [ax_diversity, ax_maxfreq, ax_fitness]

    Notes
    -----
    The heterozygosity curve is normalised to 1 at ``t_start`` so that the decay
    rate is interpretable independently of initial clone count. Multiply by
    ``result['heterozygosity_H0']`` to get absolute values.

    The fitness landscape uses a diverging colormap centred at s=0. Lineages to
    the right of the vertical dashed line at r=0 have net self-renewal bias; those
    above the horizontal dashed line at tau=tau_bar divide faster than average.
    """
    result = sim.predict_neutral_drift_trajectory(
        t_horizon=t_horizon, t_start=t_start, n_points=n_points
    )

    t         = result["t"]
    t0        = result["t_start"]
    H_norm    = result["heterozygosity"]
    max_freq  = result["expected_max_freq"]
    T_half    = result["T_halflife"]
    T_fix     = result["T_fix"]
    T_drift   = result["T_drift"]
    N_eff     = result["N_eff"]
    tau_bar   = result["tau_bar"]
    beta_bar  = result["beta_bar"]
    mean_d    = result["mean_d"]
    K         = result["K"]
    L         = result["n_lineages"]

    pl = result["per_lineage"]
    freq_0    = pl["freq"]
    r_0       = pl["r"]
    tau_0     = pl["tau"]
    s_0       = pl["s"]
    d_0       = pl["d"]
    w_0       = pl["w"]
    lids      = pl["lineage_id"]
    sel_ids   = pl["selected_ids"]
    freq_traj = pl["freq_trajectory"]

    # ── Figure layout ─────────────────────────────────────────────────────
    if ax_array is None:
        fig, axes = plt.subplots(3, 1, figsize=figsize,
                                 gridspec_kw={"height_ratios": [1.1, 1.1, 1.3]})
    else:
        axes = list(ax_array)
        fig = axes[0].get_figure()

    ax_div, ax_max, ax_fit = axes

    _PRED_COLOR   = "#4C78A8"   # analytical prediction
    _ACTUAL_COLOR = "#E45756"   # empirical overlay
    _SEL_CMAP     = plt.cm.YlOrRd

    # ═══════════════════════════════════════════════════════════════════════
    # Panel 1 — Heterozygosity decay
    # ═══════════════════════════════════════════════════════════════════════
    ax_div.semilogy(t, H_norm, color=_PRED_COLOR, lw=2.2, label="Analytical $H(t)/H_0$", zorder=3)
    ax_div.axhline(1.0, color="grey", lw=0.8, linestyle="--", alpha=0.5)

    # Half-life marker
    if t0 < t0 + T_half < t_horizon:
        ax_div.axvline(t0 + T_half, color="#F28522", lw=1.4, linestyle=":", alpha=0.8)
        ax_div.text(
            t0 + T_half, ax_div.get_ylim()[0] * 3 if ax_div.get_ylim()[0] > 0 else 0.3,
            f" $T_{{1/2}}$={T_half:.0f} yr",
            color="#F28522", fontsize=8, va="center",
        )

    # Fixation time marker (may be far off plot)
    if t0 < t0 + T_fix < t_horizon * 2:
        ax_div.axvline(min(t0 + T_fix, t_horizon), color="#B279A2",
                       lw=1.4, linestyle=":", alpha=0.8)
        ax_div.text(
            min(t0 + T_fix, t_horizon) * 0.99,
            0.5,
            f"$T_{{fix}}$≈{T_fix:.0f} yr ",
            color="#B279A2", fontsize=8, va="center", ha="right",
        )

    # Empirical heterozygosity overlay
    if show_actual and len(sim.counts_history) > 1:
        time_hist = np.array(sim.time_history)
        mask_hist = (time_hist >= t0) & (time_hist <= t_horizon)
        H_empirical = []
        for counts in [sim.counts_history[i]
                       for i, flag in enumerate(mask_hist) if flag]:
            total = np.sum(counts)
            if total > 0:
                f = counts / total
                H_empirical.append(1.0 - np.sum(f ** 2))
            else:
                H_empirical.append(np.nan)

        H0_abs = result["heterozygosity_H0"]
        if H0_abs > 0:
            H_empirical_norm = np.array(H_empirical) / H0_abs
            ax_div.scatter(
                time_hist[mask_hist], H_empirical_norm,
                color=_ACTUAL_COLOR, s=6, alpha=0.5, zorder=2,
                label="Empirical $H(t)/H_0$", linewidths=0,
            )

    ax_div.set_ylabel("$H(t) / H_0$", fontsize=11)
    ax_div.set_title("Neutral Drift Prediction", fontsize=13, fontweight="bold")
    ax_div.legend(fontsize=9, loc="upper right")
    ax_div.set_xlim(t[0], t[-1])
    ax_div.grid(True, alpha=0.25, axis="both")
    ax_div.spines[["top", "right"]].set_visible(False)

    # Annotation box
    _info = (
        f"$N_{{\\mathrm{{eff}}}}={N_eff:.0f}$\n"
        f"$T_{{\\mathrm{{drift}}}}={T_drift:.0f}$ yr\n"
        f"$K={K:.0f}$,  $L={L}$\n"
        f"$\\bar{{\\tau}}={tau_bar:.2f}$ yr,  $\\bar{{\\beta}}={beta_bar:.2f}$"
    )
    ax_div.text(
        0.02, 0.05, _info,
        transform=ax_div.transAxes,
        fontsize=8.5,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )

    # ═══════════════════════════════════════════════════════════════════════
    # Panel 2 — Expected maximum clone frequency
    # ═══════════════════════════════════════════════════════════════════════
    ax_max.plot(t, max_freq, color=_PRED_COLOR, lw=2.2,
                label="$E[\\max_i f_i(t)]$ (neutral drift)", zorder=3)

    # Logistic trajectories of selected lineages (top_n_selected by s)
    if len(sel_ids) > 0:
        sel_s = s_0[r_0 > 0]
        top_sel_indices = np.argsort(sel_s)[::-1][:top_n_selected]
        cmap = plt.cm.YlOrRd
        for rank, si_idx in enumerate(top_sel_indices):
            fi_t = freq_traj[si_idx]
            s_val = sel_s[si_idx]
            w_val = float(w_0[np.where(r_0 > 0)[0][si_idx]])
            lid = sel_ids[si_idx]
            color = cmap(0.3 + 0.6 * rank / max(len(top_sel_indices) - 1, 1))
            ax_max.plot(
                t, fi_t,
                color=color, lw=1.5, linestyle="--", alpha=0.85, zorder=2,
                label=f"Clone {lid}: $s$={s_val:.3f}, $w$={w_val:.2f}",
            )

    # Empirical max frequency overlay
    if show_actual and len(sim.counts_history) > 1:
        time_hist = np.array(sim.time_history)
        mask_hist = (time_hist >= t0) & (time_hist <= t_horizon)
        max_freqs_empirical = []
        for counts in [sim.counts_history[i]
                       for i, flag in enumerate(mask_hist) if flag]:
            total = np.sum(counts)
            max_freqs_empirical.append(float(np.max(counts) / total) if total > 0 else 0.0)

        ax_max.scatter(
            time_hist[mask_hist], max_freqs_empirical,
            color=_ACTUAL_COLOR, s=6, alpha=0.5, zorder=2,
            label="Empirical max freq", linewidths=0,
        )

    ax_max.axhline(1.0 / L, color="grey", lw=0.9, linestyle="--", alpha=0.5)
    ax_max.text(
        t[1], 1.0 / L * 1.05, f"  $1/L = {1/L:.3f}$",
        fontsize=7.5, color="grey", va="bottom",
    )
    ax_max.set_ylabel("Max clone frequency $f_{{\\max}}$", fontsize=11)
    ax_max.legend(fontsize=8, loc="upper left")
    ax_max.set_xlim(t[0], t[-1])
    ax_max.set_ylim(bottom=0.0)
    ax_max.grid(True, alpha=0.25)
    ax_max.spines[["top", "right"]].set_visible(False)

    # ═══════════════════════════════════════════════════════════════════════
    # Panel 3 — Fitness landscape (r vs tau, coloured by s)
    # ═══════════════════════════════════════════════════════════════════════
    # Size encodes initial frequency (log-scaled so small clones are still visible)
    size_raw = np.sqrt(np.maximum(freq_0, 0)) * 800
    size_raw = np.clip(size_raw, 8, 250)

    s_abs_max = max(float(np.max(np.abs(s_0))), 1e-6)
    norm_s = plt.Normalize(-s_abs_max, s_abs_max)
    cmap_s = plt.cm.RdBu_r

    sc = ax_fit.scatter(
        r_0, tau_0,
        c=s_0, cmap=cmap_s, norm=norm_s,
        s=size_raw, alpha=0.75, edgecolors="white", linewidths=0.4,
        zorder=3,
    )
    plt.colorbar(sc, ax=ax_fit, label="Selective advantage $s_i = r_i / \\tau_i$  [yr$^{-1}$]",
                 fraction=0.03, pad=0.02)

    # Reference lines
    ax_fit.axvline(0.0, color="grey", lw=1.0, linestyle="--", alpha=0.6)
    ax_fit.axhline(tau_bar, color="#72B7B2", lw=1.0, linestyle=":", alpha=0.8,
                   label=f"$\\bar{{\\tau}}={tau_bar:.2f}$ yr")

    # Iso-s contours: s = r/tau → tau = r/s for fixed s values
    r_contour = np.linspace(
        max(float(np.min(r_0)) - 0.02, -0.01),
        float(np.max(r_0)) + 0.02,
        200,
    )
    s_contour_vals = np.array([0.01, 0.05, 0.1, 0.2]) * s_abs_max / 0.1
    # Only draw contours where they cross the visible tau range
    tau_range = ax_fit.get_ylim() if ax_fit.get_ylim() != (0.0, 1.0) else (
        float(np.min(tau_0)) * 0.8, float(np.max(tau_0)) * 1.2
    )
    for s_c in np.linspace(s_abs_max * 0.1, s_abs_max * 0.8, 4):
        tau_c = r_contour / (s_c + 1e-30)
        valid = (tau_c >= 0) & (tau_c < float(np.max(tau_0)) * 2.5)
        if np.any(valid):
            ax_fit.plot(
                r_contour[valid], tau_c[valid],
                color=cmap_s(norm_s(s_c)), lw=0.9, linestyle="--", alpha=0.45,
            )
            # Label the rightmost point
            idx_lab = np.where(valid)[0][-1]
            ax_fit.text(
                r_contour[idx_lab], tau_c[idx_lab],
                f" $s$={s_c:.3f}",
                fontsize=7, color=cmap_s(norm_s(s_c)), alpha=0.9,
            )

    # Annotate w > 1 lineages (expansion faster than drift)
    high_w_mask = w_0 > 1.0
    for i in np.where(high_w_mask)[0]:
        ax_fit.annotate(
            f"$w$={w_0[i]:.1f}",
            xy=(r_0[i], tau_0[i]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=7, color="#E45756",
            arrowprops=dict(arrowstyle="->", color="#E45756", lw=0.7),
        )

    ax_fit.set_xlabel("Net growth bias  $r_i = p^{ss}_i - p^{dd}_i$", fontsize=11)
    ax_fit.set_ylabel("Division time  $\\tau_i$  [yr]", fontsize=11)
    ax_fit.set_title(
        "Fitness landscape at $t_0$  —  point size $\\propto \\sqrt{f_0}$, "
        "colour = $s_i$",
        fontsize=10,
    )
    ax_fit.legend(fontsize=8, loc="upper right")
    ax_fit.grid(True, alpha=0.2)
    ax_fit.spines[["top", "right"]].set_visible(False)

    # Shared x-axis label for panels 1 & 2
    ax_div.set_xticklabels([])
    ax_max.set_xlabel("Time (years)", fontsize=10)

    fig.tight_layout(h_pad=1.8)
    return fig, axes


def plot_interactive(sims, ages, clonal_burden):
    N_INDIVIDUALS = len(sims)
    import io, base64, matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import unittest.mock as mock
    import pandas as pd

    from bokeh.plotting import figure, show, output_notebook
    from bokeh.models import HoverTool, ColumnDataSource
    from bokeh.transform import linear_cmap
    from bokeh.palettes import Viridis256

    output_notebook()

    def plot_lineage_evolution_b64(sim, **kwargs):
        plt.close('all')
        with mock.patch('matplotlib.pyplot.show', lambda: None):
            plot_lineage_evolution(sim, **kwargs)
        fig = plt.gcf()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{b64}"

    pl_kwargs = dict(max_lineages=10000, figsize=(6, 3), shuffle=True)
    sim_images = [plot_lineage_evolution_b64(sim, **pl_kwargs) for sim in sims]

    df = pd.DataFrame({
        'ages':          ages,
        'clonal_burden': clonal_burden,
        'individual_id': [f"ID_{i}" for i in range(N_INDIVIDUALS)],
        'sim_image':     sim_images,
    })

    source = ColumnDataSource(df)

    hover = HoverTool(tooltips="""
        <div style="padding:6px; background:#1a1a2e; border-radius:6px; border:1px solid #444;">
            <div style="font-family:monospace; color:#e0e0e0; font-size:12px; margin-bottom:4px;">
                <b>@individual_id</b> &nbsp;|&nbsp; Age: <b>@ages{0.1f}</b> &nbsp;|&nbsp; Burden: <b>@clonal_burden{0.3f}</b>
            </div>
            <img src="@sim_image" width="420" style="border-radius:4px; display:block;"/>
        </div>
    """, attachment="vertical")

    p = figure(
        title="Clonal Burden vs Age",
        x_axis_label="Age (years)",
        y_axis_label="Clonal Burden",
        width=750, height=480,
        tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset', 'save'],
        toolbar_location="above",
        background_fill_color="#0f0f1a",
        border_fill_color="#0f0f1a",
    )

    # Style axes and grid
    p.xaxis.axis_label_text_color = "#aaaacc"
    p.yaxis.axis_label_text_color = "#aaaacc"
    p.xaxis.major_label_text_color = "#888899"
    p.yaxis.major_label_text_color = "#888899"
    p.xaxis.axis_line_color = "#333355"
    p.yaxis.axis_line_color = "#333355"
    p.xgrid.grid_line_color = "#1e1e3a"
    p.ygrid.grid_line_color = "#1e1e3a"
    p.title.text_color = "#ddddff"
    p.title.text_font = "Georgia"
    p.title.text_font_size = "15px"

    # Color-map points by clonal burden
    mapper = linear_cmap(
        field_name='clonal_burden',
        palette=Viridis256,
        low=df['clonal_burden'].min(),
        high=df['clonal_burden'].max(),
    )

    p.circle(
        x='ages', y='clonal_burden',
        source=source,
        size=9,
        color=mapper,
        line_color="white",
        line_width=0.5,
        fill_alpha=0.85,
        hover_fill_alpha=1.0,
        hover_line_color="#00ffcc",
        hover_line_width=2,
        hover_fill_color=mapper,
    )
    matplotlib.use('inline')

    show(p)
    matplotlib.use('inline')
