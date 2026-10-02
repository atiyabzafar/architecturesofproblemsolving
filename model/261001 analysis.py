"""
261001 analysis.py  --  the full analysis pipeline for the npj Complexity paper.

Model: learning by testing + recency communication (model_learningbytesting.py),
N = 100, K = 30, alpha = 2, tau = 1, communication normalised so the average
agent receives one message per tick.

Network selection, the same in every analysis.

  Main set: three networks that differ in one property at a time.
    even     in-degree 4 and out-degree 4 for every agent: no positional
             differences are possible
    uneven   heavy-tailed in-degree with mean 4; out-degrees are a random
             permutation of the in-degree sequence
    dense    the same in-degree sequence times three (mean 12): density with
             the shape of the distribution held fixed
  Every main-set graph is required to be strongly connected.

  Appendix set: four standard generators at <k> ~ 4 (random, small world,
  scale free, layered), analysed in exactly the same way and reported
  separately.

Experiments, 40 seeds per network
    longrun   T = 2000; long-run statistics over the last 500 ticks; stale
              share over time; per-agent violations against in-degree;
              cohort curves (share of agents believing / disbelieving a
              retired clause against its age)
    shock     same seeds; at t = 1500 the whole constraint set is replaced at
              once; the run continues to t = 2500; recovery of performance,
              spread of the new clauses, fate of the old ones

Uncertainty is always computed with the RUN as the unit of observation
(agents of one run share one environment), never by pooling agents.

Usage (PYTHONHASHSEED is pinned to 0 automatically):
    py "261001 analysis.py" networks     describe the networks
    py "261001 analysis.py" run          run every simulation, in parallel;
                                         resumes if interrupted (--fresh to restart)
    py "261001 analysis.py" figures      tables and figures from saved results
    py "261001 analysis.py" all
    add --quick for a two-seed, short-horizon smoke test (separate outputs)

All outputs go to output/ with the prefix "261001 ".
"""
import contextlib
import io
import math
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
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
PFX = "261001 quick" if QUICK else "261001"

# ----------------------------------------------------------------- settings
N, K, ALPHA, TAU = 100, 30, 2, 1
BASE = dict(K=K, alpha=ALPHA, obs_prob=0.01, clause_interval=TAU)
if QUICK:
    T_LONG, T_SHOCK, WINDOW, SEEDS, SHOCK_T, BATCH = 200, 260, 50, [1, 2, 3], 120, 1
    COHORT_START, COHORT_END, COHORT_AGE, SPREAD_AGE = 120, 140, 40, 60
else:
    T_LONG, T_SHOCK, WINDOW, SEEDS, SHOCK_T, BATCH = 2000, 2500, 500, list(range(1, 41)), 1500, 10
    COHORT_START, COHORT_END, COHORT_AGE, SPREAD_AGE = 1500, 1700, 300, 150
SHOCK_FRAC = 1.0          # share of the constraint set replaced at the shock
WORKERS = max(1, min(16, (os.cpu_count() or 2) - 4))
Z95 = 1.96


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
    "even":   dict(label="Even (in-degree 4 for all)",          kind="ideal", kin=np.full(N, 4), out="same"),
    "uneven": dict(label="Uneven (heavy tail, mean 4)",          kind="ideal", kin=KIN4,  out="shuffled"),
    "dense":  dict(label="Uneven, dense (same shape, mean 12)",  kind="ideal", kin=KIN12, out="shuffled"),
}
APPENDIX = {
    "random":     dict(label="Random (Erdos-Renyi)",   kind="gen", gen=dict(type_network="Random", connect_prob=0.04)),
    "smallworld": dict(label="Small world (Watts-Strogatz)", kind="gen", gen=dict(type_network="Small World", n_size=4, rewire_prob=0.1)),
    "scalefree":  dict(label="Scale free (Barabasi-Albert)", kind="gen", gen=dict(type_network="Scale Free", min_deg=4)),
    "layered":    dict(label="Layered (3 layers)",     kind="gen", gen=dict(type_network="Hierarchical", nlayers=3,
                                                                           intra_layer_connectance=0.08,
                                                                           inter_layer_connectance=0.02)),
}
SPECS = {**MAIN, **APPENDIX}
COLORS = {"even": "#7f7f7f", "uneven": "#1f77b4", "dense": "#9467bd",
          "random": "#c9a227", "smallworld": "#17becf", "scalefree": "#e377c2", "layered": "#8c564b"}
SHORT = {"even": "Even", "uneven": "Uneven", "dense": "Uneven, dense",
         "random": "Random", "smallworld": "Small world", "scalefree": "Scale free", "layered": "Layered"}


def build_ideal(spec, seed):
    """Directed graph with the given in-degree sequence; out-degrees equal to
    the in-degrees ("same") or a random permutation of them ("shuffled").
    Built as a Havel-Hakimi realisation, randomised by degree-preserving edge
    swaps, redrawn until strongly connected, and relabelled so that the node
    label carries no information about degree."""
    kin = np.asarray(spec["kin"], int)
    for attempt in range(200):
        rng = np.random.default_rng(seed * 7919 + 17 + 100003 * attempt)
        kout = kin.copy() if spec["out"] == "same" else rng.permutation(kin)
        G = nx.directed_havel_hakimi_graph([int(x) for x in kin], [int(x) for x in kout])
        E = G.number_of_edges()
        nx.directed_edge_swap(G, nswap=10 * E, max_tries=1000 * E, seed=int(rng.integers(2 ** 31 - 1)))
        if nx.is_strongly_connected(G):
            perm = rng.permutation(G.number_of_nodes())
            return nx.relabel_nodes(G, {i: int(perm[i]) for i in range(G.number_of_nodes())})
    raise RuntimeError("no strongly connected realisation found")


class AnalysisModel(LearningByTestingModel):
    """Default model plus a log of clause replacements (tick, old, new, gone)."""

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
    corr = np.corrcoef(kin, kout)[0, 1] if kin.std() > 0 and kout.std() > 0 else float("nan")
    scc = max(len(c) for c in nx.strongly_connected_components(G))
    return dict(N=G.number_of_nodes(), E=G.number_of_edges(), k_mean=kbar, kin_sd=kin.std(),
                kin_max=kin.max(), kin_zero=(kin == 0).mean(), cv2=kin.var() / kbar ** 2,
                inout_corr=corr, largest_scc=scc / G.number_of_nodes(),
                reciprocity=nx.overall_reciprocity(G), clustering=nx.average_clustering(G.to_undirected()))


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
    T = T_LONG if mode == "longrun" else T_SHOCK
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
    spread = np.full(SPREAD_AGE + 1, np.nan)
    old_pos = np.full(SPREAD_AGE + 1, np.nan); old_neg = np.full(SPREAD_AGE + 1, np.nan)
    new_clauses = old_clauses = None
    for t in range(1, T + 1):
        m.step()
        if mode == "shock" and t == SHOCK_T:
            before = set(m.C)
            for u in random.sample(range(m.M), int(round(SHOCK_FRAC * m.M))):
                m.C[u] = m.random_clause()
            after = set(m.C)
            new_clauses = sorted(after - before)
            old_clauses = sorted(before - after)
            m._C_set = after
            m.calc_performances()
        avgV[t], minV[t], hom[t] = m.avg_true_V, m.min_true_V, m.homogeneity
        Cs = set(m.C)
        stale[t] = stale_share(m, Cs)
        if mode == "longrun":
            if t > T - WINDOW:
                vsum += np.array([a.true_violations for a in m.agent_list])
            if m.retire_log:
                tick, old, new, gone = m.retire_log[-1]
                if tick == t and gone and COHORT_START <= t < COHORT_END:
                    tracked.append((t, old))
            for t0, c in tracked:
                a = t - t0
                if a <= COHORT_AGE and c not in Cs:
                    h_acc[a] += holders(m, c, 1); n_acc[a] += holders(m, c, -1); c_acc[a] += 1
        elif SHOCK_T <= t <= SHOCK_T + SPREAD_AGE:
            a = t - SHOCK_T
            live = [c for c in new_clauses if c in Cs]
            if live:
                spread[a] = np.mean([holders(m, c, 1) for c in live])
            dead = [c for c in old_clauses if c not in Cs]
            if dead:
                old_pos[a] = np.mean([holders(m, c, 1) for c in dead])
                old_neg[a] = np.mean([holders(m, c, -1) for c in dead])
    out = dict(name=name, seed=seed, mode=mode, avgV=avgV, minV=minV, hom=hom, stale=stale)
    if mode == "longrun":
        with np.errstate(invalid="ignore", divide="ignore"):
            out.update(cohort_h=h_acc / c_acc, cohort_n=n_acc / c_acc)
        out.update(stats=stats, kin=kin, kout=kout, vmean=vsum / WINDOW)
    else:
        out.update(spread=spread, old_pos=old_pos, old_neg=old_neg)
    return out


def do_run():
    cache = Path(tempfile.gettempdir()) / (PFX.replace(" ", "_") + "_analysis_cache")
    if "--fresh" in sys.argv and cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(exist_ok=True)
    batches = [SEEDS[i:i + BATCH] for i in range(0, len(SEEDS), BATCH)]
    per = len(SPECS) * 2
    print(f"{len(SEEDS) * per} runs: {len(SPECS)} networks x {len(SEEDS)} seeds x 2 experiments, on {WORKERS} workers", flush=True)
    t0 = time.time()
    for b, seeds in enumerate(batches):
        f = cache / f"batch_{b:02d}.pkl"
        if f.exists():
            print(f"  batch {b + 1}/{len(batches)} already done", flush=True)
            continue
        jobs = list(product(SPECS, seeds, ["shock", "longrun"]))
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            res = list(ex.map(run_one, jobs))
        tmp = f.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(res, fh)
        tmp.replace(f)
        print(f"  batch {b + 1}/{len(batches)} done ({len(jobs)} runs), {time.time() - t0:.0f}s", flush=True)
    results = []
    for b in range(len(batches)):
        with open(cache / f"batch_{b:02d}.pkl", "rb") as fh:
            results += pickle.load(fh)
    ts = {}
    agents = ["network,seed,agent,k_in,k_out,V_longrun"]
    net_rows = []
    for r in results:
        key = f"{r['name']}|{r['seed']}|{r['mode']}"
        for q in ("avgV", "minV", "hom", "stale"):
            ts[f"{key}|{q}"] = r[q]
        if r["mode"] == "longrun":
            for q in ("cohort_h", "cohort_n"):
                ts[f"{key}|{q}"] = r[q]
            for i in range(len(r["kin"])):
                agents.append(f"{r['name']},{r['seed']},{i},{int(r['kin'][i])},{int(r['kout'][i])},{r['vmean'][i]:.4f}")
            net_rows.append((r["name"], r["seed"], r["stats"]))
        else:
            for q in ("spread", "old_pos", "old_neg"):
                ts[f"{key}|{q}"] = r[q]
    np.savez_compressed(OUT / f"{PFX} timeseries.npz", **ts)
    (OUT / f"{PFX} agents.csv").write_text("\n".join(agents) + "\n", encoding="utf-8")
    cols = list(net_rows[0][2].keys())
    lines = ["network,seed," + ",".join(cols)]
    for name, seed, st in sorted(net_rows, key=lambda x: (x[0], x[1])):
        lines.append(f"{name},{seed}," + ",".join(f"{st[c]:.4f}" for c in cols))
    (OUT / f"{PFX} networks.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.rmtree(cache)
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


def se(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float("nan")


def mse(x, d=2):
    return f"{np.nanmean(x):.{d}f} +- {se(x):.{d}f}"


def welch(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a.mean() - b.mean()
    s = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return d, s, d / s if s > 0 else float("nan")


def md_table(header, rows):
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
    line = lambda r: "| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |"
    return "\n".join([line(header), "|" + "|".join("-" * (x + 2) for x in w) + "|"] + [line(r) for r in rows])


def intake_fit(agents, names):
    """V = a + b ln(lambda) + c ln(lambda)^2 on all agents of `names` with an in-link."""
    xs, ys = [], []
    for name in names:
        for arr in agents[name].values():
            lam = arr[:, 0] / arr[:, 0].mean()
            xs.append(lam[lam > 0]); ys.append(arr[lam > 0, 2])
    L = np.log(np.concatenate(xs)); y = np.concatenate(ys)
    return np.linalg.lstsq(np.vstack([np.ones_like(L), L, L ** 2]).T, y, rcond=None)[0]


def binned_runs(per, xfun, edges, min_n=5, min_runs=3):
    """Bin agents by x; the point is the pooled mean, the interval comes from
    the spread of the per-run bin means (the run is the unit of observation)."""
    xs, ys, es = [], [], []
    X = {s: xfun(a) for s, a in per.items()}
    for lo, hi in zip(edges[:-1], edges[1:]):
        px, py, runm = [], [], []
        for s, a in per.items():
            sel = (X[s] >= lo) & (X[s] < hi)
            if sel.any():
                px.append(X[s][sel]); py.append(a[sel, 2]); runm.append(a[sel, 2].mean())
        if len(runm) < min_runs or sum(len(v) for v in py) < min_n:
            continue
        xs.append(np.concatenate(px).mean()); ys.append(np.concatenate(py).mean())
        es.append(Z95 * np.std(runm, ddof=1) / math.sqrt(len(runm)))
    return np.array(xs), np.array(ys), np.array(es)


# --------------------------------------------------------------- figures
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "figure.dpi": 100, "savefig.bbox": "tight"})
    return plt


def save(fig, stem):
    fig.savefig(OUT / f"{PFX} {stem}.pdf")
    fig.savefig(OUT / f"{PFX} {stem}.png", dpi=160)


def fig_networks(plt, names, stem):
    """In-degree distributions and the in/out-degree relation (first seed)."""
    n = len(names)
    fig, axes = plt.subplots(2, n, figsize=(3.3 * n + 0.4, 5.6), squeeze=False)
    for j, name in enumerate(names):
        G = make_model(name, SEEDS[0], 1).network
        kin = np.array([d for _, d in G.in_degree()]); kout = np.array([d for _, d in G.out_degree()])
        ax = axes[0, j]
        ax.hist(kin, bins=np.arange(-0.5, kin.max() + 1.5), color=COLORS[name])
        ax.set_title(SPECS[name]["label"], fontsize=10)
        ax.set_xlabel("in-degree"); ax.set_ylabel("agents" if j == 0 else "")
        ax = axes[1, j]
        ax.scatter(kin, kout, s=12, color=COLORS[name], alpha=0.5)   # no jitter: degrees are integers
        if kin.std() == 0 and kout.std() == 0:                        # the even network is one point
            ax.set_xlim(0, 2 * kin[0]); ax.set_ylim(0, 2 * kout[0])
        ax.set_xlabel("in-degree"); ax.set_ylabel("out-degree" if j == 0 else "")
    fig.tight_layout()
    save(fig, stem)
    plt.close(fig)


def fig_misinformation(plt, ts, names, stem):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    ax = axes[0]
    for name in names:
        ax.plot(np.arange(T_LONG + 1), series(ts, name, "longrun", "stale").mean(0), color=COLORS[name],
                label=SPECS[name]["label"])
    ax.set_xlabel("tick"); ax.set_ylabel("share of positive beliefs that are stale")
    ax.set_title("(a) Stale beliefs over a run"); ax.set_ylim(0, None)
    ax.legend(frameon=False, loc="lower right")
    ax = axes[1]
    for name in names:
        ax.plot(np.nanmean(series(ts, name, "longrun", "cohort_h"), 0), color=COLORS[name])
        ax.plot(np.nanmean(series(ts, name, "longrun", "cohort_n"), 0), color=COLORS[name], ls=":")
    ax.set_xlabel("ticks since the clause was retired"); ax.set_ylabel("share of agents")
    ax.set_title("(b) After a retirement: still believing (solid), disbelieving (dotted)")
    ax.set_ylim(0, None)
    fig.tight_layout()
    save(fig, stem)
    plt.close(fig)


def fig_longrun(plt, ts, names, stem):
    fig, axes = plt.subplots(1, 3, figsize=(2.2 * len(names) + 4.5, 3.6))
    for ax, q, title in zip(axes, ("avgV", "minV", "hom"),
                            ("Average violations", "Best-agent violations", "Homogeneity")):
        for x, name in enumerate(names):
            vals = series(ts, name, "longrun", q)[:, -WINDOW:].mean(1)
            ax.errorbar(x, vals.mean(), yerr=Z95 * se(vals), fmt="o", color=COLORS[name], capsize=4, ms=7)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([SHORT[n] for n in names], rotation=20, ha="right")
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_title(title)
    axes[0].set_ylabel("long-run mean, 95% interval over seeds")
    fig.tight_layout()
    save(fig, stem)
    plt.close(fig)


def fig_position(plt, agents, names, stem, ref=None):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    edges_k = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 45, 70, 101])
    edges_l = np.logspace(np.log10(0.03), np.log10(12), 18)
    for name in names:
        per = agents[name]
        kw = dict(color=COLORS[name], marker="o", ms=4, lw=1.8, capsize=2, label=SPECS[name]["label"])
        if all(a[:, 0].std() == 0 for a in per.values()):          # no spread of in-degree: one point
            runm = [a[:, 2].mean() for a in per.values()]
            k0 = next(iter(per.values()))[0, 0]
            for ax, x0 in ((axes[0], k0), (axes[1], 1.0)):
                ax.errorbar([x0], [np.mean(runm)], yerr=[Z95 * se(runm)], **kw)
            continue
        bx, by, be = binned_runs(per, lambda a: a[:, 0], edges_k)
        axes[0].errorbar(bx, by, yerr=be, **kw)
        bx, by, be = binned_runs(per, lambda a: a[:, 0] / a[:, 0].mean(), edges_l)
        axes[1].errorbar(bx, by, yerr=be, **kw)
    if ref is not None:                                            # the main set's fitted curve, for reference
        lam = np.logspace(np.log10(0.2), np.log10(10), 100)
        v = ref[0] + ref[1] * np.log(lam) + ref[2] * np.log(lam) ** 2
        axes[0].plot(4 * lam, v, color="k", ls="--", lw=1.2, label="curve fitted on the main set")
        axes[1].plot(lam, v, color="k", ls="--", lw=1.2, label="curve fitted on the main set")
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
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    save(fig, stem)
    plt.close(fig)


def fig_shortrun(plt, ts, names, stem):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4))
    lim = min(600, T_LONG)
    ax = axes[0, 0]
    for name in names:
        ax.plot(np.arange(lim + 1), series(ts, name, "longrun", "avgV").mean(0)[:lim + 1],
                color=COLORS[name], label=SPECS[name]["label"])
    ax.set_title("(a) Start-up: average violations from empty knowledge bases"); ax.set_xlabel("tick")
    ax.set_ylabel("average violations"); ax.legend(frameon=False)
    ax = axes[0, 1]
    lo = max(0, SHOCK_T - 100)
    for name in names:
        ax.plot(np.arange(lo, T_SHOCK + 1), series(ts, name, "shock", "avgV").mean(0)[lo:], color=COLORS[name])
    ax.axvline(SHOCK_T, color="k", lw=0.8, ls="--")
    ax.set_title("(b) Shock: the whole constraint set replaced at once")
    ax.set_xlabel("tick"); ax.set_ylabel("average violations")
    ax = axes[1, 0]
    for name in names:
        ax.plot(np.nanmean(series(ts, name, "shock", "spread"), 0), color=COLORS[name])
    ax.set_title("(c) Spread of a new clause (share of agents holding it, while it lives)")
    ax.set_xlabel("ticks since the clause was introduced"); ax.set_ylabel("share of agents"); ax.set_ylim(0, None)
    ax = axes[1, 1]
    for name in names:
        ax.plot(np.nanmean(series(ts, name, "shock", "old_pos"), 0), color=COLORS[name])
        ax.plot(np.nanmean(series(ts, name, "shock", "old_neg"), 0), color=COLORS[name], ls=":")
    ax.set_title("(d) Retired clauses: still believed (solid), disbelieved (dotted)")
    ax.set_xlabel("ticks since the shock"); ax.set_ylabel("share of agents"); ax.set_ylim(0, None)
    fig.tight_layout()
    save(fig, stem)
    plt.close(fig)


# ---------------------------------------------------------------- tables
SHOCK_WINDOWS = [(1, 20), (21, 60), (61, 120), (121, 200), (201, 400), (401, 1000)]


def tables(ts, agents, cols, nets, names, fit, title):
    md = [f"# {title}", ""]
    ci = {c: i for i, c in enumerate(cols)}
    rows = []
    for name in names:
        with warnings.catch_warnings():               # the even network has no in/out correlation (all nan)
            warnings.simplefilter("ignore", RuntimeWarning)
            st = np.nanmean(nets[name], 0)
        corr = "none" if np.isnan(st[ci["inout_corr"]]) else f"{st[ci['inout_corr']]:.2f}"
        rows.append([name, f"{st[ci['k_mean']]:.1f}", f"{st[ci['kin_sd']]:.1f}", f"{st[ci['kin_max']]:.0f}",
                     f"{st[ci['cv2']]:.2f}", corr, f"{st[ci['clustering']]:.2f}",
                     f"{100 * st[ci['kin_zero']]:.0f}%", f"{100 * st[ci['largest_scc']]:.0f}%"])
    md += ["## Networks (means over seeds)", "",
           md_table(["network", "<k>", "sd k_in", "max k_in", "Var(intake)", "corr(k_in,k_out)", "clustering",
                     "no in-link", "largest strongly connected part"], rows), ""]

    rows = []
    lr = {}
    for name in names:
        lr[name] = {q: series(ts, name, "longrun", q)[:, -WINDOW:].mean(1) for q in ("avgV", "minV", "hom", "stale")}
        rows.append([name, mse(lr[name]["avgV"]), mse(lr[name]["minV"]), mse(lr[name]["hom"], 3), mse(100 * lr[name]["stale"], 1)])
    md += [f"## Long run (last {WINDOW} ticks; mean +- 1 s.e. over {len(SEEDS)} seeds)", "",
           md_table(["network", "average violations", "best-agent violations", "homogeneity", "stale beliefs (%)"], rows), ""]

    rows = []
    for name in names:
        varlam = np.mean([(a[:, 0] / a[:, 0].mean()).var() for a in agents[name].values()])
        rows.append([name, f"{varlam:.2f}", f"{fit[0] + 0.5 * (2 * fit[2] - fit[1]) * varlam:.2f}", mse(lr[name]["avgV"])])
    md += ["## Second-order check", "",
           f"Curve fitted on the main set: V(1) = {fit[0]:.2f}, V'(1) = {fit[1]:.2f}, V''(1) = {2 * fit[2] - fit[1]:.2f}; "
           "predicted average = V(1) + V''(1) Var(intake) / 2.", "",
           md_table(["network", "Var(intake)", "predicted average", "measured average"], rows), ""]

    rows = []
    for name in names:
        rho, gap, pct = [], [], []
        for arr in agents[name].values():
            kin, v = arr[:, 0], arr[:, 2]
            rho.append(spearman(kin, v))
            top = np.argsort(-kin)[:8]; bot = np.argsort(kin)[:8]
            gap.append(v[top].mean() - v[bot].mean())
            best = int(np.argmin(v))
            pct.append(100 * ((kin < kin[best]).mean() + 0.5 * (kin == kin[best]).mean()))
        if np.all(np.isnan(rho)):
            rows.append([name, "none (no spread)", "none", "none", "-"])
        else:
            rows.append([name, mse(rho), f"{100 * np.mean(np.array(rho) < 0):.0f}%", mse(gap), f"{np.mean(pct):.0f}%"])
    md += ["## Position (per run; mean +- 1 s.e. over seeds)", "",
           md_table(["network", "Spearman rho(k_in, V)", "runs with rho < 0", "gap: top 8 minus bottom 8",
                     "best agent's in-degree percentile"], rows), ""]

    rows, sr = [], {}
    for name in names:
        a = series(ts, name, "longrun", "avgV").mean(0)
        idx = np.where(a <= a[-WINDOW:].mean() + 1.0)[0]
        sh = series(ts, name, "shock", "avgV")
        pre = sh[:, SHOCK_T - 100:SHOCK_T].mean(1)
        lv = [sh[:, SHOCK_T + lo:SHOCK_T + hi + 1].mean(1) for lo, hi in SHOCK_WINDOWS if SHOCK_T + hi <= T_SHOCK]
        sr[name] = dict(pre=pre, lv=lv)
        rows.append([name, int(idx[0]) if len(idx) else -1, mse(pre)] + [mse(x) for x in lv])
    md += ["## Short run: average violations around the shock (mean +- 1 s.e. over seeds)", "",
           md_table(["network", "start-up: ticks to within 1 of long-run level", "100 ticks before"]
                    + [f"ticks {lo}-{hi} after" for lo, hi in SHOCK_WINDOWS if SHOCK_T + hi <= T_SHOCK], rows), ""]

    rows = []
    for name in names:
        sp = series(ts, name, "shock", "spread"); op = series(ts, name, "shock", "old_pos")
        on = series(ts, name, "shock", "old_neg")
        m = np.nanmean(sp, 0); i25 = np.where(m >= 0.25)[0]
        mo = np.nanmean(op, 0); ih = np.where(mo <= 0.5 * np.nanmax(mo))[0]
        pts = [a for a in (30, 60, 100) if a <= SPREAD_AGE]
        sr[name].update({f"sp{a}": sp[:, a] for a in pts})
        rows.append([name] + [mse(100 * sp[:, a], 1) for a in pts]
                    + [int(i25[0]) if len(i25) else -1, f"{100 * np.nanmax(m):.0f}%", int(ih[0]) if len(ih) else -1,
                       f"{100 * np.nanmax(np.nanmean(on, 0)):.0f}%"])
    md += ["## Short run: new and retired clauses after the shock", "",
           md_table(["network"] + [f"new clause held by (%), {a} ticks on" for a in (30, 60, 100) if a <= SPREAD_AGE]
                    + ["ticks to 25% of agents", "peak share", "retired clause: ticks to halve believers",
                       "peak share disbelieving"], rows), ""]
    return md, lr, sr


def do_figures():
    plt = setup_mpl()
    ts = dict(np.load(OUT / f"{PFX} timeseries.npz"))
    agents = load_agents()
    cols, nets = load_networks()
    fit = intake_fit(agents, MAIN)
    main, app = list(MAIN), list(APPENDIX)
    for names, pre in ((main, "fig"), (app, "figA")):
        fig_networks(plt, names, f"{pre}_networks")
        fig_misinformation(plt, ts, names, f"{pre}_misinformation")
        fig_longrun(plt, ts, names, f"{pre}_longrun")
        fig_position(plt, agents, names, f"{pre}_position", ref=None if pre == "fig" else fit)
        fig_shortrun(plt, ts, names, f"{pre}_shortrun")

    head = [f"Default model (learning by testing + recency), N={N}, K={K}, alpha={ALPHA}, tau={TAU}; long-run "
            f"experiment T={T_LONG}, last {WINDOW} ticks; shock experiment: {int(SHOCK_FRAC * 100)}% of the constraint "
            f"set replaced at t={SHOCK_T}, run to t={T_SHOCK}; seeds {SEEDS[0]}..{SEEDS[-1]}. "
            "All uncertainties use the run as the unit of observation.", ""]
    md1, lr, sr = tables(ts, agents, cols, nets, main, fit, f"{PFX} analysis summary: main set")
    md2, lrA, srA = tables(ts, agents, cols, nets, app, fit, "Appendix set: standard generators")

    rows = []
    for a, b in (("even", "uneven"), ("dense", "uneven"), ("even", "dense")):
        if a not in lr or b not in lr:
            continue
        tests = [("average violations", lr[a]["avgV"], lr[b]["avgV"]), ("best-agent violations", lr[a]["minV"], lr[b]["minV"]),
                 ("stale beliefs (%)", 100 * lr[a]["stale"], 100 * lr[b]["stale"]),
                 ("pre-shock level", sr[a]["pre"], sr[b]["pre"])]
        tests += [(f"level, ticks {lo}-{hi} after shock", x, y) for (lo, hi), x, y in zip(SHOCK_WINDOWS, sr[a]["lv"], sr[b]["lv"])]
        tests += [(f"excess over own pre-shock level, ticks {lo}-{hi}", x - sr[a]["pre"], y - sr[b]["pre"])
                  for (lo, hi), x, y in zip(SHOCK_WINDOWS, sr[a]["lv"], sr[b]["lv"])]
        tests += [(f"new clause held by (%), {k[2:]} ticks on", 100 * sr[a][k], 100 * sr[b][k]) for k in sr[a] if k.startswith("sp")]
        for lab, x, y in tests:
            d, s, t = welch(x, y)
            rows.append([f"{a} - {b}", lab, f"{d:+.2f}", f"{s:.2f}", f"{t:+.1f}"])
    md3 = ["## Differences between main-set networks (Welch; the run is the unit)", "",
           md_table(["pair", "quantity", "difference", "s.e.", "t"], rows), ""]
    files = ["## Files", "", f"- `{PFX} timeseries.npz`, `{PFX} agents.csv`, `{PFX} networks.csv`: raw results",
             f"- `{PFX} fig_*`: main-set figures; `{PFX} figA_*`: the same figures for the appendix set (.pdf and .png)", ""]
    out = md1[:2] + head + md1[2:] + md3 + md2 + files
    (OUT / f"{PFX} summary.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


def do_networks():
    rows = []
    for name in SPECS:
        st = network_stats(make_model(name, SEEDS[0], 1).network)
        corr = "none" if np.isnan(st["inout_corr"]) else f"{st['inout_corr']:.2f}"
        rows.append([name, f"{st['k_mean']:.2f}", f"{st['kin_sd']:.2f}", f"{st['kin_max']:.0f}", f"{st['cv2']:.2f}",
                     corr, f"{st['clustering']:.2f}", f"{100 * st['kin_zero']:.0f}%", f"{100 * st['largest_scc']:.0f}%"])
    print(md_table(["key", "<k>", "sd k_in", "max k_in", "Var(intake)", "corr(in,out)", "clustering",
                    "no in-link", "largest SCC"], rows))
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
