"""
260930 analysis.py  --  the full analysis pipeline for the npj Complexity paper.

Model: learning by testing + recency communication (model_learningbytesting.py),
N = 100, K = 30, alpha = 2, tau = 1, T = 2000, long run = last 500 ticks,
communication normalised so the average agent receives one message per tick.

Network selection, the same in every analysis.  Main set: five ideal types
that change one thing at a time, all with N = 100.

    even      in-degree 4 for every agent (random 4-regular digraph): no
              positional differences are possible
    uneven    heavy-tailed in-degree with mean 4; out-degree unrelated to
              in-degree (cov(k_in, k_out) ~ 0)
    aligned   same in-degrees; out-degree rank-matched to in-degree, so the
              agents who hear most also speak most (cov > 0)
    opposed   same in-degrees; out-degree rank-reversed (cov < 0)
    dense     the same in-degree sequence times three (mean 12); out-degree
              unrelated: the density check with the shape held fixed

Appendix set: the four generators used in earlier drafts (random, small
world, scale free, layered) at N = 100 and <k> ~ 4, to show that the ideal
types cover them.

Experiments
    longrun   T = 2000; stale share and performance over time; per-agent
              long-run violations against in-degree; steady-state cohort
              curves (share of agents believing / disbelieving a retired
              clause against its age)
    shock     same seeds; at t = 1500 the whole constraint set is replaced
              at once; recovery of performance, flushing of the old clauses,
              spread of the new ones

Usage (PYTHONHASHSEED is pinned to 0 automatically):
    py "260930 analysis.py" networks     describe the networks (table, figure)
    py "260930 analysis.py" run          run every simulation, in parallel
    py "260930 analysis.py" figures      tables and figures from saved results
    py "260930 analysis.py" all
    add --quick for a one-seed, short-horizon smoke test (separate outputs)

All outputs go to output/ with the prefix "260930 ".
"""
import contextlib
import io
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

# Python's per-process hash randomisation feeds the model's random stream, so
# every run (parent and workers) is pinned to the same hash seed.
if os.environ.get("PYTHONHASHSEED") != "0":
    _env = dict(os.environ, PYTHONHASHSEED="0")
    sys.exit(subprocess.call([sys.executable] + sys.argv, env=_env))

import numpy as np
import networkx as nx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model_learningbytesting import LearningByTestingModel  # noqa: E402

OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
QUICK = "--quick" in sys.argv
PFX = "260930 quick" if QUICK else "260930"

# ----------------------------------------------------------------- settings
N, K, ALPHA, TAU = 100, 30, 2, 1
BASE = dict(K=K, alpha=ALPHA, obs_prob=0.01, clause_interval=TAU)
if QUICK:
    T, WINDOW, SEEDS, SHOCK_T = 200, 50, [1], 120
    COHORT_START, COHORT_END, COHORT_AGE, SPREAD_AGE = 120, 140, 40, 60
else:
    T, WINDOW, SEEDS, SHOCK_T = 2000, 500, list(range(1, 11)), 1500
    COHORT_START, COHORT_END, COHORT_AGE, SPREAD_AGE = 1500, 1700, 300, 150
SHOCK_FRAC = 1.0          # share of the constraint set replaced at the shock
WORKERS = max(1, min(12, (os.cpu_count() or 2) - 2))


# ---------------------------------------------------------------- networks
def power_law_sequence(n, mean, beta=0.65, kmax=None):
    """Deterministic heavy-tailed degree sequence: k_r ~ c r^(-beta) for
    rank r = 1..n, floored, min 1, capped at kmax, adjusted to sum exactly
    n * mean.  Returned in descending order."""
    target = int(round(n * mean))
    ranks = np.arange(1, n + 1, dtype=float)
    kmax = kmax or n - 1
    best = None
    for c in np.linspace(1.0, 40.0 * mean, 40000):
        seq = np.minimum(kmax, np.maximum(1, np.floor(c * ranks ** (-beta)))).astype(int)
        s = int(seq.sum())
        if best is None or abs(s - target) < abs(best[1] - target):
            best = (seq.copy(), s)
        if s >= target:
            break
    seq = best[0]
    i = 1
    while seq.sum() < target:            # top up just below the top of the ranking
        j = i % n
        if seq[j] < kmax:
            seq[j] += 1
        i += 1
    i = n - 1
    while seq.sum() > target:            # trim from the tail, keeping min 1
        j = i % n
        if seq[j] > 1:
            seq[j] -= 1
        i -= 1
    return seq


def scaled_sequence(seq, factor, kmax):
    """The same degree sequence multiplied by `factor`, capped at kmax, with
    the capped surplus handed to the next ranks so that the sum is exact."""
    s = np.minimum(kmax, seq * factor).astype(int)
    i = 1
    while s.sum() < seq.sum() * factor:
        j = i % len(s)
        if s[j] < kmax:
            s[j] += 1
        i += 1
    return s


KIN4 = power_law_sequence(N, 4, beta=0.65, kmax=40)
KIN12 = scaled_sequence(KIN4, 3, kmax=N - 1)

MAIN = {
    "even":    dict(label="Even (k = 4 for all)",                kind="ideal", kin=np.full(N, 4), corr="aligned"),
    "uneven":  dict(label="Uneven (heavy tail, mean 4)",          kind="ideal", kin=KIN4,  corr="random"),
    "aligned": dict(label="Uneven, aligned (hubs also talk)",     kind="ideal", kin=KIN4,  corr="aligned"),
    "opposed": dict(label="Uneven, opposed (hubs listen only)",   kind="ideal", kin=KIN4,  corr="opposed"),
    "dense":   dict(label="Uneven, dense (same shape, mean 12)", kind="ideal", kin=KIN12, corr="random"),
}
APPENDIX = {
    "random":     dict(label="Random (ER)",       kind="gen", gen=dict(type_network="Random", connect_prob=0.04)),
    "smallworld": dict(label="Small world (WS)",  kind="gen", gen=dict(type_network="Small World", n_size=4, rewire_prob=0.1)),
    "scalefree":  dict(label="Scale free (BA)",   kind="gen", gen=dict(type_network="Scale Free", min_deg=4)),
    "layered":    dict(label="Layered (3 layers)", kind="gen", gen=dict(type_network="Hierarchical", nlayers=3,
                                                                         intra_layer_connectance=0.08,
                                                                         inter_layer_connectance=0.02)),
}
SPECS = {**MAIN, **APPENDIX}
COLORS = {"even": "#7f7f7f", "uneven": "#1f77b4", "aligned": "#d62728", "opposed": "#2ca02c", "dense": "#9467bd",
          "random": "#c9a227", "smallworld": "#17becf", "scalefree": "#e377c2", "layered": "#8c564b"}
STYLES = {k: "-" for k in MAIN} | {k: "--" for k in APPENDIX}


def build_ideal(spec, seed):
    """Directed graph with the given in-degree sequence and an out-degree
    sequence set by the correlation rule, randomised by degree-preserving
    edge swaps.  Node labels are shuffled so that the label carries no
    information about degree."""
    rng = np.random.default_rng(seed * 7919 + 17)
    kin = np.asarray(spec["kin"], int)
    if spec["corr"] == "random":
        kout = rng.permutation(kin)
    elif spec["corr"] == "aligned":
        kout = kin.copy()
    else:
        kout = kin[::-1].copy()
    G = nx.directed_havel_hakimi_graph([int(x) for x in kin], [int(x) for x in kout])
    E = G.number_of_edges()
    nx.directed_edge_swap(G, nswap=10 * E, max_tries=1000 * E, seed=int(rng.integers(2 ** 31 - 1)))
    perm = rng.permutation(G.number_of_nodes())
    return nx.relabel_nodes(G, {i: int(perm[i]) for i in range(G.number_of_nodes())})


class AnalysisModel(LearningByTestingModel):
    """Default model plus a log of clause replacements (tick, old, new)."""

    def __init__(self, *args, **kwargs):
        self.retire_log = []
        super().__init__(*args, **kwargs)

    def replace_universal_clause(self):
        u, old, new = super().replace_universal_clause()
        self.retire_log.append((self.steps, old, new, old not in self.C))
        return (u, old, new)


def make_model(name, seed, horizon):
    spec = SPECS[name]
    with contextlib.redirect_stdout(io.StringIO()):
        if spec["kind"] == "ideal":
            G = build_ideal(spec, seed)
            return AnalysisModel(N=N, R=horizon, seed=seed, setup_source="graph", input_graph=G, **BASE)
        return AnalysisModel(N=N, R=horizon, seed=seed, setup_source="generate", **spec["gen"], **BASE)


def network_stats(G):
    kin = np.array([d for _, d in G.in_degree()], float)
    kout = np.array([d for _, d in G.out_degree()], float)
    kbar = kin.mean()
    lam = float(np.max(np.real(np.linalg.eigvals(nx.to_numpy_array(G)))))
    corr = np.corrcoef(kin, kout)[0, 1] if kin.std() > 0 and kout.std() > 0 else float("nan")
    return dict(N=G.number_of_nodes(), E=G.number_of_edges(), k_mean=kbar, kin_sd=kin.std(),
                kin_max=kin.max(), kin_zero=(kin == 0).mean(), cv2=kin.var() / kbar ** 2,
                inout_cov_norm=float(np.mean((kin - kbar) * (kout - kout.mean())) / kbar ** 2),
                inout_corr=corr, lam_over_k=lam / kbar, reciprocity=nx.overall_reciprocity(G),
                clustering=nx.average_clustering(G.to_undirected()))


# -------------------------------------------------------------- simulation
def stale_share(m, Cs):
    tot = stale = 0
    for a in m.agent_list:
        tot += len(a.kb)
        stale += sum(1 for c in a.kb if c not in Cs)
    return stale / tot if tot else 0.0


def holders(m, c, sign):
    return sum(1 for a in m.agent_list if a.sign.get(c) == sign) / m.N


def run_one(args):
    name, seed, mode = args
    import random
    m = make_model(name, seed, T)
    Nn = m.N
    stats = network_stats(m.network)
    kin = np.array([m.network.in_degree(i) for i in range(Nn)], float)
    kout = np.array([m.network.out_degree(i) for i in range(Nn)], float)
    avgV = np.zeros(T + 1); minV = np.zeros(T + 1); hom = np.zeros(T + 1); stale = np.zeros(T + 1)
    avgV[0], minV[0], hom[0] = m.avg_true_V, m.min_true_V, m.homogeneity
    stale[0] = stale_share(m, set(m.C))
    vsum = np.zeros(Nn)
    h_acc = np.zeros(COHORT_AGE + 1); n_acc = np.zeros(COHORT_AGE + 1); c_acc = np.zeros(COHORT_AGE + 1)
    tracked = []
    spread = np.full(SPREAD_AGE + 1, np.nan); live_new = np.zeros(SPREAD_AGE + 1)
    old_pos = np.full(SPREAD_AGE + 1, np.nan); old_neg = np.full(SPREAD_AGE + 1, np.nan)
    new_clauses = old_clauses = None
    for t in range(1, T + 1):
        m.step()
        if mode == "shock" and t == SHOCK_T:
            before = set(m.C)
            n_rep = int(round(SHOCK_FRAC * m.M))
            for u in random.sample(range(m.M), n_rep):
                m.C[u] = m.random_clause()
            after = set(m.C)
            new_clauses = sorted(after - before)
            old_clauses = sorted(before - after)
            m._C_set = after
            m.calc_performances()
        avgV[t], minV[t], hom[t] = m.avg_true_V, m.min_true_V, m.homogeneity
        Cs = set(m.C)
        stale[t] = stale_share(m, Cs)
        if t > T - WINDOW:
            vsum += np.array([a.true_violations for a in m.agent_list])
        if mode == "longrun":
            if m.retire_log:
                tick, old, new, gone = m.retire_log[-1]
                if tick == t and gone and COHORT_START <= t < COHORT_END:
                    tracked.append((t, old))
            for t0, c in tracked:
                a = t - t0
                if a <= COHORT_AGE and c not in Cs:
                    h_acc[a] += holders(m, c, 1); n_acc[a] += holders(m, c, -1); c_acc[a] += 1
        elif t >= SHOCK_T:
            a = t - SHOCK_T
            if a <= SPREAD_AGE:
                live = [c for c in new_clauses if c in Cs]
                live_new[a] = len(live)
                if live:
                    spread[a] = np.mean([holders(m, c, 1) for c in live])
                dead = [c for c in old_clauses if c not in Cs]
                if dead:
                    old_pos[a] = np.mean([holders(m, c, 1) for c in dead])
                    old_neg[a] = np.mean([holders(m, c, -1) for c in dead])
    out = dict(name=name, seed=seed, mode=mode, stats=stats, kin=kin, kout=kout,
               avgV=avgV, minV=minV, hom=hom, stale=stale, vmean=vsum / WINDOW)
    if mode == "longrun":
        with np.errstate(invalid="ignore", divide="ignore"):
            out["cohort_h"] = h_acc / c_acc; out["cohort_n"] = n_acc / c_acc
        out["cohort_count"] = c_acc
    else:
        out.update(spread=spread, live_new=live_new, old_pos=old_pos, old_neg=old_neg)
    return out


def do_run():
    jobs = list(product(SPECS, SEEDS, ["longrun", "shock"]))
    print(f"{len(jobs)} runs (T={T}, {len(SEEDS)} seeds, {len(SPECS)} networks) on {WORKERS} workers")
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for k, f in enumerate(as_completed(futs), 1):
            r = f.result()
            results.append(r)
            if k % 10 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} done, {time.time() - t0:.0f}s", flush=True)
    ts = {}
    agents = ["network,seed,agent,k_in,k_out,V_longrun"]
    net_rows = []
    for r in results:
        key = f"{r['name']}|{r['seed']}|{r['mode']}"
        for q in ("avgV", "minV", "hom", "stale"):
            ts[f"{key}|{q}"] = r[q]
        if r["mode"] == "longrun":
            for q in ("cohort_h", "cohort_n", "cohort_count"):
                ts[f"{key}|{q}"] = r[q]
            for i in range(len(r["kin"])):
                agents.append(f"{r['name']},{r['seed']},{i},{int(r['kin'][i])},{int(r['kout'][i])},{r['vmean'][i]:.4f}")
            net_rows.append((r["name"], r["seed"], r["stats"]))
        else:
            for q in ("spread", "live_new", "old_pos", "old_neg"):
                ts[f"{key}|{q}"] = r[q]
    np.savez_compressed(OUT / f"{PFX} timeseries.npz", **ts)
    (OUT / f"{PFX} agents.csv").write_text("\n".join(agents) + "\n", encoding="utf-8")
    cols = list(net_rows[0][2].keys())
    lines = ["network,seed," + ",".join(cols)]
    for name, seed, st in sorted(net_rows):
        lines.append(f"{name},{seed}," + ",".join(f"{st[c]:.4f}" for c in cols))
    (OUT / f"{PFX} networks.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {PFX} timeseries.npz, agents.csv, networks.csv in {OUT}  ({time.time() - t0:.0f}s)")


# ------------------------------------------------------------ statistics
def rankdata(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def load_agents():
    rows = (OUT / f"{PFX} agents.csv").read_text(encoding="utf-8").strip().split("\n")[1:]
    d = {}
    for r in rows:
        name, seed, i, kin, kout, v = r.split(",")
        d.setdefault(name, {}).setdefault(int(seed), []).append((int(kin), int(kout), float(v)))
    return {n: {s: np.array(v) for s, v in per.items()} for n, per in d.items()}


def load_networks():
    lines = (OUT / f"{PFX} networks.csv").read_text(encoding="utf-8").strip().split("\n")
    cols = lines[0].split(",")[2:]
    d = {}
    for r in lines[1:]:
        parts = r.split(",")
        d.setdefault(parts[0], []).append([float(x) for x in parts[2:]])
    return cols, {n: np.array(v) for n, v in d.items()}


def series(ts, name, mode, q):
    return np.array([ts[f"{name}|{s}|{mode}|{q}"] for s in SEEDS])


def msd(x):
    x = np.asarray(x, float)
    return f"{np.nanmean(x):.2f} +- {np.nanstd(x):.2f}"


def md_table(header, rows):
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
    line = lambda r: "| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |"
    return "\n".join([line(header), "|" + "|".join("-" * (x + 2) for x in w) + "|"] + [line(r) for r in rows])


def first_within(x, level, margin, start=0):
    idx = np.where(x[start:] <= level + margin)[0]
    return int(idx[0]) if len(idx) else -1


# --------------------------------------------------------------- figures
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8.5,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "figure.dpi": 100, "savefig.bbox": "tight"})
    return plt


def save(fig, stem):
    fig.savefig(OUT / f"{PFX} {stem}.pdf")
    fig.savefig(OUT / f"{PFX} {stem}.png", dpi=160)


def legend_line(ax, names, **kw):
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, **kw)


def fig_networks(plt):
    """In-degree distributions and the in/out-degree relation of the main set (seed 1)."""
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.4))
    for j, name in enumerate(MAIN):
        with contextlib.redirect_stdout(io.StringIO()):
            G = build_ideal(SPECS[name], SEEDS[0])
        kin = np.array([d for _, d in G.in_degree()]); kout = np.array([d for _, d in G.out_degree()])
        ax = axes[0, j]
        ax.hist(kin, bins=np.arange(-0.5, kin.max() + 1.5), color=COLORS[name])
        ax.set_title(SPECS[name]["label"], fontsize=9.5)
        ax.set_xlabel("in-degree"); ax.set_ylabel("agents" if j == 0 else "")
        ax = axes[1, j]
        ax.scatter(kin, kout, s=12, color=COLORS[name], alpha=0.5)   # no jitter: degrees are integers
        if kin.std() == 0 and kout.std() == 0:                        # the even network is one point
            ax.set_xlim(0, 2 * kin[0]); ax.set_ylim(0, 2 * kout[0])
        ax.set_xlabel("in-degree"); ax.set_ylabel("out-degree" if j == 0 else "")
        st = network_stats(G)
        ax.text(0.03, 0.95, f"corr = {st['inout_corr']:.2f}\n$\\Lambda/\\langle k\\rangle$ = {st['lam_over_k']:.2f}",
                transform=ax.transAxes, va="top", fontsize=8.5)
    fig.suptitle("The five ideal-type networks (one realisation each)", y=1.01)
    fig.tight_layout()
    save(fig, "fig_networks")
    plt.close(fig)


def fig_misinformation(plt, ts):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    for name in MAIN:
        s = series(ts, name, "longrun", "stale")
        ax.plot(np.arange(T + 1), s.mean(0), color=COLORS[name], label=SPECS[name]["label"])
    ax.set_xlabel("tick"); ax.set_ylabel("share of positive beliefs that are stale")
    ax.set_title("(a) Stale beliefs over a run"); ax.set_ylim(0, None)
    legend_line(ax, MAIN, loc="lower right")
    ax = axes[1]
    for name in MAIN:
        h = np.nanmean(series(ts, name, "longrun", "cohort_h"), 0)
        n = np.nanmean(series(ts, name, "longrun", "cohort_n"), 0)
        ax.plot(h, color=COLORS[name], label=SPECS[name]["label"])
        ax.plot(n, color=COLORS[name], ls=":")
    ax.set_xlabel("ticks since the clause was retired"); ax.set_ylabel("share of agents")
    ax.set_title("(b) After a retirement: still believing (solid), disbelieving (dotted)")
    ax.set_ylim(0, None)
    fig.tight_layout()
    save(fig, "fig_misinformation")
    plt.close(fig)


def fig_longrun(plt, ts):
    names = list(SPECS)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, q, title in zip(axes, ("avgV", "minV", "hom"),
                            ("Average violations", "Best-agent violations", "Homogeneity")):
        for x, name in enumerate(names):
            vals = series(ts, name, "longrun", q)[:, -WINDOW:].mean(1)
            ax.errorbar(x, vals.mean(), yerr=vals.std(), fmt="o", color=COLORS[name], capsize=3,
                        mfc="white" if name in APPENDIX else COLORS[name])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n if n in MAIN else n + "\n(app.)" for n in names], rotation=45, ha="right", fontsize=8.5)
        ax.set_title(title)
        ax.axvline(len(MAIN) - 0.5, color="k", lw=0.6, alpha=0.4)
    axes[0].set_ylabel("long-run mean (last 500 ticks), +- 1 s.d. over seeds")
    fig.suptitle("Long-run population outcomes by network (filled: main set; open: appendix set)", y=1.02)
    fig.tight_layout()
    save(fig, "fig_longrun")
    plt.close(fig)


def binned(x, y, edges, min_n=5):
    xs, ys, es = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x >= lo) & (x < hi)
        if sel.sum() >= min_n:
            xs.append(x[sel].mean()); ys.append(y[sel].mean()); es.append(y[sel].std() / np.sqrt(sel.sum()))
    return np.array(xs), np.array(ys), np.array(es)


def fig_position(plt, agents, nets):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
    edges_k = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 45, 70, 101])
    edges_l = np.logspace(np.log10(0.03), np.log10(12), 18)
    for name in SPECS:
        per = agents[name]
        x = np.concatenate([per[s][:, 0] for s in per]); y = np.concatenate([per[s][:, 2] for s in per])
        kb = np.concatenate([np.full(len(per[s]), per[s][:, 0].mean()) for s in per])
        lam = x / kb
        kw = dict(color=COLORS[name], ls=STYLES[name], marker="o" if name in MAIN else "s", ms=4,
                  lw=1.8 if name in MAIN else 1.1, alpha=1 if name in MAIN else 0.8, label=SPECS[name]["label"])
        if name == "even":
            axes[0].errorbar([4], [y.mean()], yerr=[y.std() / np.sqrt(len(y))], **kw)
            axes[1].errorbar([1.0], [y.mean()], yerr=[y.std() / np.sqrt(len(y))], **kw)
            continue
        bx, by, be = binned(x, y, edges_k); axes[0].errorbar(bx, by, yerr=be, capsize=2, **kw)
        bx, by, be = binned(lam, y, edges_l); axes[1].errorbar(bx, by, yerr=be, capsize=2, **kw)
    ax = axes[0]
    ax.set_xlabel("in-degree $k$"); ax.set_ylabel("long-run violations")
    ax.set_title("(a) Performance against in-degree"); ax.set_xscale("symlog", linthresh=10)
    kt = [0, 1, 2, 3, 5, 10, 20, 40, 100]
    ax.set_xticks(kt); ax.set_xticklabels([str(k) for k in kt]); ax.minorticks_off()
    ax = axes[1]
    ax.set_xscale("log"); ax.set_xlabel("intake $k/\\langle k\\rangle$ (messages per tick)")
    ax.set_title("(b) The same, against intake"); ax.axvline(1, color="k", lw=0.6, alpha=0.4)
    lt = [0.1, 0.2, 0.5, 1, 2, 5, 10]
    ax.set_xticks(lt); ax.set_xticklabels([str(x) for x in lt]); ax.minorticks_off()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    save(fig, "fig_position")
    plt.close(fig)


def fig_shortrun(plt, ts):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    lim = min(600, T)
    ax = axes[0, 0]
    for name in MAIN:
        ax.plot(np.arange(lim + 1), series(ts, name, "longrun", "avgV").mean(0)[:lim + 1],
                color=COLORS[name], label=SPECS[name]["label"])
    ax.set_title("(a) Start-up: average violations from empty belief stores"); ax.set_xlabel("tick")
    ax.set_ylabel("average violations"); legend_line(ax, MAIN)
    ax = axes[0, 1]
    lo, hi = max(0, SHOCK_T - 100), min(T, SHOCK_T + SPREAD_AGE + 100)
    for name in MAIN:
        ax.plot(np.arange(lo, hi + 1), series(ts, name, "shock", "avgV").mean(0)[lo:hi + 1], color=COLORS[name])
    ax.axvline(SHOCK_T, color="k", lw=0.8, ls="--"); ax.set_title("(b) Shock: the whole constraint set replaced at once")
    ax.set_xlabel("tick"); ax.set_ylabel("average violations")
    ax = axes[1, 0]
    for name in MAIN:
        ax.plot(np.nanmean(series(ts, name, "shock", "spread"), 0), color=COLORS[name])
    ax.set_title("(c) Spread of a new clause (share of agents holding it, while it lives)")
    ax.set_xlabel("ticks since the clause was introduced"); ax.set_ylabel("share of agents")
    ax = axes[1, 1]
    for name in MAIN:
        ax.plot(np.nanmean(series(ts, name, "shock", "old_pos"), 0), color=COLORS[name])
        ax.plot(np.nanmean(series(ts, name, "shock", "old_neg"), 0), color=COLORS[name], ls=":")
    ax.set_title("(d) Fate of the retired clauses: still believed (solid), disbelieved (dotted)")
    ax.set_xlabel("ticks since the shock"); ax.set_ylabel("share of agents")
    for a in axes.flat:
        a.set_ylim(0, None) if a is not axes[0, 0] and a is not axes[0, 1] else None
    fig.tight_layout()
    save(fig, "fig_shortrun")
    plt.close(fig)


def do_figures():
    plt = setup_mpl()
    ts = dict(np.load(OUT / f"{PFX} timeseries.npz"))
    agents = load_agents()
    cols, nets = load_networks()
    fig_networks(plt)
    fig_misinformation(plt, ts)
    fig_longrun(plt, ts)
    fig_position(plt, agents, nets)
    fig_shortrun(plt, ts)

    # ---- tables ----
    md = [f"# {PFX} analysis summary", "",
          f"Default model (learning by testing + recency), N={N}, K={K}, alpha={ALPHA}, tau={TAU}, "
          f"T={T}, long run = last {WINDOW} ticks, seeds = {SEEDS[0]}..{SEEDS[-1]}. "
          f"Shock at t={SHOCK_T}: {int(SHOCK_FRAC*100)}% of the constraint set replaced.", ""]
    ci = {c: i for i, c in enumerate(cols)}
    rows = []
    for name in SPECS:
        st = nets[name].mean(0)
        rows.append([name, SPECS[name]["label"], f"{st[ci['k_mean']]:.1f}", f"{st[ci['kin_sd']]:.1f}",
                     f"{st[ci['kin_max']]:.0f}", f"{st[ci['cv2']]:.2f}", f"{st[ci['inout_corr']]:.2f}",
                     f"{st[ci['lam_over_k']]:.2f}", f"{st[ci['reciprocity']]:.2f}", f"{st[ci['clustering']]:.2f}",
                     f"{100*st[ci['kin_zero']]:.0f}%"])
    md += ["## Networks (means over seeds)", "",
           md_table(["key", "network", "<k>", "sd k_in", "max k_in", "Var(k)/<k>^2", "corr(k_in,k_out)",
                     "Lambda/<k>", "reciprocity", "clustering", "k_in = 0"], rows), ""]

    rows = []
    for name in SPECS:
        a = series(ts, name, "longrun", "avgV")[:, -WINDOW:].mean(1)
        b = series(ts, name, "longrun", "minV")[:, -WINDOW:].mean(1)
        h = series(ts, name, "longrun", "hom")[:, -WINDOW:].mean(1)
        s = series(ts, name, "longrun", "stale")[:, -WINDOW:].mean(1)
        rows.append([name, msd(a), msd(b), msd(h), msd(100 * s)])
    md += ["## Long run (mean +- s.d. over seeds of the last-500-tick average)", "",
           md_table(["network", "average violations", "best-agent violations", "homogeneity", "stale share (%)"], rows), ""]

    # Second-order check: with mean intake fixed at 1, structure can move the
    # average only through the spread of intake, by V''(1) Var(lambda) / 2.
    xs, ys = [], []
    for name in MAIN:
        for s, arr in agents[name].items():
            xs.append(arr[:, 0] / arr[:, 0].mean()); ys.append(arr[:, 2])
    x = np.concatenate(xs); y = np.concatenate(ys); sel = x > 0
    L = np.log(x[sel]); A = np.vstack([np.ones_like(L), L, L ** 2]).T
    a0, b, c = np.linalg.lstsq(A, y[sel], rcond=None)[0]
    Vpp = 2 * c - b
    rows = []
    for name in SPECS:
        varlam = np.mean([(arr[:, 0] / arr[:, 0].mean()).var() for arr in agents[name].values()])
        meas = series(ts, name, "longrun", "avgV")[:, -WINDOW:].mean(1).mean()
        rows.append([name, f"{varlam:.2f}", f"{a0 + 0.5 * Vpp * varlam:.2f}", f"{meas:.2f}"])
    md += ["## Second-order check: the average against the spread of intake", "",
           f"Intake curve fitted on the main set's agents as V = a + b ln(lambda) + c ln(lambda)^2: "
           f"V(1) = {a0:.2f}, V'(1) = {b:.2f}, V''(1) = {Vpp:.2f}. "
           f"Predicted average = V(1) + V''(1) Var(lambda) / 2.", "",
           md_table(["network", "Var(intake)", "predicted average", "measured average"], rows), ""]

    rows = []
    for name in SPECS:
        per = agents[name]
        rho, gap, pct = [], [], []
        for s, arr in per.items():
            kin, v = arr[:, 0], arr[:, 2]
            rho.append(spearman(kin, v))
            top = np.argsort(-kin)[:8]; bot = np.argsort(kin)[:8]
            gap.append(v[top].mean() - v[bot].mean())
            best = int(np.argmin(v))
            pct.append(100 * ((kin < kin[best]).mean() + 0.5 * (kin == kin[best]).mean()))
        rows.append([name, msd(rho) if not np.all(np.isnan(rho)) else "n/a (no spread)", msd(gap), f"{np.mean(pct):.0f}%"])
    md += ["## Position (per run, then mean +- s.d. over seeds)", "",
           md_table(["network", "Spearman rho(k_in, V)", "gap: top 8 minus bottom 8", "best agent's k_in percentile"], rows),
           "", "Negative rho and gap mean that better-connected agents do better.", ""]

    rows = []
    for name in SPECS:
        a = series(ts, name, "longrun", "avgV").mean(0)
        level = a[-WINDOW:].mean()
        t_start = first_within(a, level, 1.0)
        sh = series(ts, name, "shock", "avgV").mean(0)
        pre = sh[SHOCK_T - 100:SHOCK_T].mean()
        peak = sh[SHOCK_T:SHOCK_T + 50].max()
        half = np.where(sh[SHOCK_T + 1:] - pre <= 0.5 * (peak - pre))[0]
        t_rec = int(half[0]) + 1 if len(half) else -1
        sp = np.nanmean(series(ts, name, "shock", "spread"), 0)
        i25 = np.where(sp >= 0.25)[0]; t25 = int(i25[0]) if len(i25) else -1
        s30 = sp[30] if len(sp) > 30 else float("nan"); s60 = sp[60] if len(sp) > 60 else float("nan")
        op = np.nanmean(series(ts, name, "shock", "old_pos"), 0)
        ih = np.where(op <= 0.5 * np.nanmax(op))[0]; thalf = int(ih[0]) if len(ih) else -1
        rows.append([name, t_start, f"{pre:.1f}", f"{peak:.1f}", t_rec, t25, f"{100*s30:.0f}%", f"{100*s60:.0f}%", thalf])
    md += ["## Short run (averages over seeds)", "",
           md_table(["network", "start-up: ticks to within 1 of long-run level", "pre-shock level",
                     "peak after shock", "shock half-life (ticks)", "new clause: ticks to reach 25% of agents",
                     "held by, at 30 ticks", "at 60 ticks", "old clause: ticks to halve believers"], rows),
           "", "-1 means the threshold was not reached within the recorded window.", ""]
    md += ["## Files", "", f"- `{PFX} timeseries.npz`: per (network, seed, mode) time series and cohort curves",
           f"- `{PFX} agents.csv`: per-agent in-degree, out-degree and long-run violations",
           f"- `{PFX} networks.csv`: structural statistics per (network, seed)",
           f"- `{PFX} fig_networks / fig_misinformation / fig_longrun / fig_position / fig_shortrun` (.pdf and .png)", ""]
    (OUT / f"{PFX} summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


def do_networks():
    plt = setup_mpl()
    fig_networks(plt)
    rows = []
    for name in SPECS:
        with contextlib.redirect_stdout(io.StringIO()):
            m = make_model(name, SEEDS[0], 1)
        st = network_stats(m.network)
        rows.append([name, f"{st['k_mean']:.2f}", f"{st['kin_sd']:.2f}", f"{st['kin_max']:.0f}", f"{st['cv2']:.2f}",
                     f"{st['inout_corr']:.2f}", f"{st['lam_over_k']:.2f}", f"{st['reciprocity']:.2f}",
                     f"{st['clustering']:.2f}", f"{100*st['kin_zero']:.0f}%"])
    print(md_table(["key", "<k>", "sd k_in", "max k_in", "Var/<k>^2", "corr(in,out)", "Lambda/<k>",
                    "reciprocity", "clustering", "k_in=0"], rows))
    print(f"\nKIN4 (top 10): {KIN4[:10].tolist()} ... min {KIN4.min()}, sum {KIN4.sum()}")
    print(f"KIN12 (top 10): {KIN12[:10].tolist()} ... min {KIN12.min()}, sum {KIN12.sum()}")


if __name__ == "__main__":
    cmds = [a for a in sys.argv[1:] if not a.startswith("--")] or ["all"]
    for cmd in cmds:
        if cmd in ("networks", "all"):
            do_networks()
        if cmd in ("run", "all"):
            do_run()
        if cmd in ("figures", "all"):
            do_figures()
