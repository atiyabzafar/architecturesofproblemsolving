"""
261111 fig_s6_meanfield.py  --  Figure S6: the reduced mean-field model against the simulations,
one panel per main-text result.

    python "261111 fig_s6_meanfield.py" RUNDIR

RUNDIR holds main.npz and ablation.npz written by "261109 results runs.py". Put this file next to
"261001 analysis.py" (for the in-degree sequences) and reduced_ebm.py. Writes to RUNDIR:
    261111 figS6_meanfield.pdf / .png
    261111 meanfield_numbers.txt      every number quoted in the supplement
    261111 meanfield_ebm.json         the model's long-run values per class (for reuse)

Panels
  a  violations against intake (main text Fig. 4a): simulation bins, model lines
  b  current and outdated positive beliefs against intake, Scale Free (Fig. 4b)
  c  current and outdated positive beliefs per agent, by network (Fig. 4c)
  d  recency rule against random sharing, Regular and Scale Free (Fig. 5b and Section 3.1.3)

The model tracks beliefs, not violations. Violations are computed from the model's beliefs with
per-constraint violation rates measured on the simulation agents (pooled over the three networks):
    V = M [ (1 - T) u(S) + T v_T ],   v_T = share of believed current constraints that are violated,
    u(S) = c0 + c1 S                  share of unbelieved current constraints that are violated,
                                      as a straight line in the agent's outdated beliefs S.
So any gap between model and simulation in panel a comes from the beliefs, not from the conversion.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ana", HERE / "261001 analysis.py")
ana = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ana)                    # in-degree sequences; pins PYTHONHASHSEED as usual
sys.path.insert(0, str(HERE))
from reduced_ebm import ReducedEBM               # noqa: E402

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
M_ = 60
NSEEDS=100
S40 = range(1, NSEEDS+1)
MAIN3 = ["even", "uneven", "dense"]
SHORT = {"even": "Regular", "uneven": "Scale Free", "dense": "Scale Free, dense"}
NETC = {"even": "#000000", "uneven": "#0072B2", "dense": "#E69F00"}      # as in the main-text figures
NETM = {"even": "o", "uneven": "s", "dense": "^"}
EDGES = [0.12, 0.2, 0.3, 0.45, 0.65, 0.9, 1.25, 1.8, 2.6, 3.8, 5.5, 12]   # intake bins, as in Fig. 4
LOG = []


def say(s=""):
    print(s); LOG.append(s)


def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return (x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else (np.nan, np.nan)


# ------------------------------------------------------------------ simulations
Mz = np.load(RUN / "main.npz")
Az = np.load(RUN / "ablation.npz")
G = lambda n, q: np.array([Mz[f"{n}|{s}|{q}"] for s in S40])
H = lambda n, v, q: np.array([Az[f"{n}|{v}|{s}|{q}"] for s in S40])

abm = {}
for n in MAIN3:
    kin = G(n, "kin"); lam = kin / kin.mean(1, keepdims=True)
    abm[n] = dict(lam=lam, V=G(n, "win")[:, 10:].mean(1), true=G(n, "true"), stale=G(n, "stale"),
                  cov=G(n, "cov"), vb=G(n, "vb"), vu=G(n, "vu"))


def binned(n, q):
    """Pooled bin means; 95% interval from the spread of per-run bin means (the run is the unit)."""
    a = abm[n]; lam, y = a["lam"], a[q]
    out = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        s = (lam > lo) & (lam <= hi)
        if s.sum() < 20:
            continue
        runm = [y[i][s[i]].mean() for i in range(len(y)) if s[i].any()]
        out.append((np.exp(np.log(lam[s]).mean()), y[s].mean(), ci(runm)[1]))
    return np.array(out)


# conversion from beliefs to violations, measured on the simulation agents
T_all = np.concatenate([abm[n]["true"].ravel() for n in MAIN3]) / M_
S_all = np.concatenate([abm[n]["stale"].ravel() for n in MAIN3]) / M_
cov = np.concatenate([abm[n]["cov"].ravel() for n in MAIN3])
vb = np.concatenate([abm[n]["vb"].ravel() for n in MAIN3])
vu = np.concatenate([abm[n]["vu"].ravel() for n in MAIN3])
V_all = np.concatenate([abm[n]["V"].ravel() for n in MAIN3])
vT = vb.sum() / cov.sum()
u = vu / np.maximum(M_ - cov, 1)
c1, c0 = np.polyfit(S_all, u, 1)
conv = lambda T, S: M_ * ((1 - T) * (c0 + c1 * S) + T * vT)
r2 = np.corrcoef(conv(T_all, S_all), V_all)[0, 1] ** 2
say("CONVERSION FROM BELIEFS TO VIOLATIONS (simulation agents, three main networks pooled)")
say(f"  believed current constraints violated: v_T = {vT:.3f}")
say(f"  unbelieved current constraints violated: u = {c0:.3f} + {c1:.3f} x (outdated positives / M)")
say(f"  check on the simulation agents: R^2 = {r2:.2f} between converted and measured violations")

# ------------------------------------------------------------------ model
ebm = {}
for n in MAIN3:
    kin = np.asarray(ana.MAIN[n]["kin"])
    for rule in (("recency", "stock") if n != "dense" else ("recency",)):
        m = ReducedEBM(kin, rule=rule)
        t, Y = m.run(1500, t_eval=np.arange(1000, 1501, 5.0))
        o = m.unpack(Y)
        T, S, p = (o[q].mean(0) for q in "TSp")
        ebm[(n, rule)] = dict(k=m.k.tolist(), lam=m.lam.tolist(), P=m.P.tolist(), T=T.tolist(), S=S.tolist(),
                              p=p.tolist(), V=conv(T, S).tolist())
        say(f"  model ({rule:7s}) {SHORT[n]:18s} solved")
E = lambda n, q, rule="recency": np.array(ebm[(n, rule)][q])
Epop = lambda n, q, rule="recency": float(E(n, q, rule) @ E(n, "P", rule))

# ------------------------------------------------------------------ numbers for the text
say("\nMODEL AGAINST SIMULATION")
for n in ("uneven", "dense"):
    lam, V = E(n, "lam"), E(n, "V")
    b_m = np.polyfit(np.log(lam), V, 1)[0] * np.log(2)
    a = abm[n]; x = np.log(a["lam"]).ravel(); ok = np.isfinite(x)
    b_s = np.polyfit(x[ok], a["V"].ravel()[ok], 1)[0] * np.log(2)
    say(f"  {SHORT[n]:18s} violations per doubling of intake: model {b_m:+.2f}, simulation {b_s:+.2f}")
for n in MAIN3:
    say(f"  {SHORT[n]:18s} current / outdated per agent: model {M_ * Epop(n, 'T'):5.1f} / {M_ * Epop(n, 'S'):5.1f}"
        f" | simulation {abm[n]['true'].mean():5.1f} / {abm[n]['stale'].mean():5.1f}"
        f" | violations model {Epop(n, 'V'):5.2f}, simulation {abm[n]['V'].mean():5.2f}"
        f" | share of 1s model {Epop(n, 'p'):.3f}")
dm = Epop("uneven", "V") - Epop("even", "V"); ds = abm["uneven"]["V"].mean() - abm["even"]["V"].mean()
say(f"  Scale Free minus Regular, violations: model {dm:+.2f}, simulation {ds:+.2f}")
lam, T, S = E("uneven", "lam"), E("uneven", "T"), E("uneven", "S")
say(f"  Scale Free, lowest to highest intake: current {M_ * T[0]:.1f} -> {M_ * T[-1]:.1f}, outdated "
    f"{M_ * S[0]:.1f} -> {M_ * S[-1]:.1f}, total {M_ * (T[0] + S[0]):.1f} -> {M_ * (T[-1] + S[-1]):.1f}, "
    f"current share {T[0] / (T[0] + S[0]):.0%} -> {T[-1] / (T[-1] + S[-1]):.0%}")
abl = {}
for n in ("even", "uneven"):
    for v in ("full", "norec"):
        abl[(n, v)] = dict(true=H(n, v, "ts_true")[:, 151:].mean(1), stale=H(n, v, "ts_stale")[:, 151:].mean(1))
    for q, lab in (("T", "current"), ("S", "outdated")):
        key = "true" if q == "T" else "stale"
        m0, m1 = Epop(n, q, "recency"), Epop(n, q, "stock")
        s0, s1 = abl[(n, "full")][key].mean(), abl[(n, "norec")][key].mean()
        say(f"  {SHORT[n]:18s} random sharing instead of recency, {lab:8s}: model {M_ * m0:5.1f} -> {M_ * m1:5.1f} "
            f"({m1 / m0 - 1:+.0%}) | simulation {s0:5.1f} -> {s1:5.1f} ({s1 / s0 - 1:+.0%})")
    T1 = E(n, "T", "stock")
    if len(T1) > 1:
        say(f"  {SHORT[n]:18s} random sharing: model current beliefs from lowest to highest intake "
            f"{M_ * T1[0]:.1f} -> {M_ * T1[-1]:.1f} (intake buys nothing)")

# ------------------------------------------------------------------ figure
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 7.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
                     "axes.linewidth": 0.7, "lines.linewidth": 1.4, "legend.frameon": False,
                     "savefig.bbox": "tight"})


def letter(ax, t):
    ax.set_title(t, loc="left", fontweight="bold", fontsize=9, pad=4)


rng = np.random.default_rng(0)
fig, axs = plt.subplots(2, 2, figsize=(7.08, 5.2), gridspec_kw=dict(wspace=0.32, hspace=0.5))
a1, a2, a3, a4 = axs.ravel()

# a: violations against intake
for n, ls in (("uneven", "-"), ("dense", "--")):
    b = binned(n, "V"); sh = 1.0 if n == "uneven" else 1.04
    a1.errorbar(b[:, 0] * sh, b[:, 1], yerr=b[:, 2], fmt=NETM[n], ms=3.5, color=NETC[n], ecolor=NETC[n],
                elinewidth=0.7, capsize=0)
    a1.plot(E(n, "lam"), E(n, "V"), ls=ls, color=NETC[n], lw=1.3)
mu, e = ci(abm["even"]["V"].mean(1))
a1.errorbar([1], [mu], yerr=[e], fmt=NETM["even"], ms=4.5, color=NETC["even"])
a1.plot([1], [Epop("even", "V")], "D", ms=5, mfc="white", mec=NETC["even"], mew=1.2)
a1.set_xscale("log"); a1.set_xticks([0.25, 0.5, 1, 2, 4, 8]); a1.set_xticklabels(["0.25", "0.5", "1", "2", "4", "8"])
a1.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
a1.set_xlabel(r"Communication intake, $\lambda_i$"); a1.set_ylabel("Mean number of violations")
a1.add_artist(a1.legend([Line2D([], [], color=NETC[n], marker=NETM[n], ms=3.5, lw=1.3) for n in MAIN3],
                        [SHORT[n] for n in MAIN3], loc="lower left", handlelength=1.8, labelspacing=0.25))
a1.legend([Line2D([], [], color="#555", marker="o", ls="none", ms=3.5), Line2D([], [], color="#555", lw=1.3),
           Line2D([], [], color="#555", marker="D", mfc="white", ls="none", ms=4.5)],
          ["simulation (95% interval)", "model", "model, Regular"], loc="upper right", handlelength=1.5,
          labelspacing=0.25)
letter(a1, "a")

# b: current and outdated against intake, Scale Free
bt, bs = binned("uneven", "true"), binned("uneven", "stale")
a2.errorbar(bt[:, 0], bt[:, 1], yerr=bt[:, 2], fmt="o", ms=3.5, color="#000000", elinewidth=0.7, capsize=0)
a2.errorbar(bs[:, 0], bs[:, 1], yerr=bs[:, 2], fmt="o", ms=3.5, mfc="white", color="#000000", elinewidth=0.7, capsize=0)
a2.plot(E("uneven", "lam"), M_ * E("uneven", "T"), color="#000000", lw=1.3)
a2.plot(E("uneven", "lam"), M_ * E("uneven", "S"), color="#000000", lw=1.3, ls="--")
a2.set_xscale("log"); a2.set_xticks([0.25, 0.5, 1, 2, 4, 8]); a2.set_xticklabels(["0.25", "0.5", "1", "2", "4", "8"])
a2.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); a2.set_ylim(0, None)
a2.set_xlabel(r"Communication intake, $\lambda_i$ (Scale Free)"); a2.set_ylabel("Positive beliefs per agent")
a2.legend([Line2D([], [], color="k", marker="o", ls="none", ms=3.5), Line2D([], [], color="k", marker="o", mfc="white", ls="none", ms=3.5),
           Line2D([], [], color="k"), Line2D([], [], color="k", ls="--")],
          ["current, simulation", "outdated, simulation", "current, model", "outdated, model"],
          loc="lower left", ncol=2, handlelength=1.8, columnspacing=1.0, labelspacing=0.2)
letter(a2, "b")


def dots(ax, x0, vals, col, filled=True):
    ax.scatter(x0 + rng.uniform(-0.07, 0.07, len(vals)), vals, s=5, color=col, alpha=0.35, lw=0)
    ax.plot([x0 - 0.1, x0 + 0.1], [np.mean(vals)] * 2, color=col, lw=1.5)


# c: current and outdated per agent, by network
for gi, (key, q, lab) in enumerate((("true", "T", "Current"), ("stale", "S", "Outdated"))):
    for j, n in enumerate(MAIN3):
        x0 = gi + (j - 1) * 0.27
        dots(a3, x0, abm[n][key].mean(1), NETC[n])
        a3.plot([x0], [M_ * Epop(n, q)], "D", ms=5, mfc="white", mec=NETC[n], mew=1.2)
a3.set_xticks([0, 1], ["Current", "Outdated"]); a3.set_xlim(-0.55, 1.55); a3.set_ylim(0, 1.55 * a3.get_ylim()[1])
a3.set_ylabel("Positive beliefs per agent")
a3.legend([Line2D([], [], color=NETC[n], lw=3) for n in MAIN3]
          + [Line2D([], [], color="#555", marker="o", ls="none", ms=3, alpha=0.5), Line2D([], [], color="#555", lw=1.5),
             Line2D([], [], color="#555", marker="D", mfc="white", ls="none", ms=4.5)],
          [SHORT[n] for n in MAIN3] + ["simulation, one dot per run", "simulation mean", "model"],
          loc="upper center", ncol=2, handlelength=1.4, columnspacing=0.8, labelspacing=0.25, fontsize=6.5)
letter(a3, "c")

# d: recency rule against random choice of the belief to send. x: network, grouped by kind of belief;
#    colour shade: sharing rule (full colour = recency rule, light = random choice).
LIGHT = {"even": "#a6a6a6", "uneven": "#8fc3e6"}
ticks, labels = [], []
for gi, (key, q) in enumerate((("true", "T"), ("stale", "S"))):
    for j, n in enumerate(("even", "uneven")):
        xc = gi * 2.3 + j * 1.0
        ticks.append(xc); labels.append(SHORT[n])
        for r, (v, rule, col) in enumerate((("full", "recency", NETC[n]), ("norec", "stock", LIGHT[n]))):
            x0 = xc + (r - 0.5) * 0.42
            dots(a4, x0, abl[(n, v)][key], col)
            a4.plot([x0], [M_ * Epop(n, q, rule)], "D", ms=5, mfc="white", mec=col, mew=1.2)
a4.set_xticks(ticks, labels)
for gi, lab in enumerate(("Current", "Outdated")):
    a4.text(gi * 2.3 + 0.5, 0.66, lab, transform=a4.get_xaxis_transform(), ha="center", va="bottom", fontsize=7.5)
a4.set_xlim(-0.6, 3.9); a4.set_ylim(0, 2 * a4.get_ylim()[1]); a4.set_ylabel("Positive beliefs per agent")
a4.legend([Line2D([], [], color=c, lw=3) for c in (NETC["even"], LIGHT["even"], NETC["uneven"], LIGHT["uneven"])],
          ["Regular, recency", "Regular, random", "Scale Free, recency", "Scale Free, random "],
          loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.4, columnspacing=0.8,
          labelspacing=0.25, fontsize=6.5)
letter(a4, "d")

for ext in ("pdf", "png"):
    fig.savefig(RUN / f"261111 figS6_meanfield.{ext}", dpi=300)
(RUN / "261111 meanfield_numbers.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
(RUN / "261111 meanfield_ebm.json").write_text(json.dumps({f"{n}|{r}": v for (n, r), v in ebm.items()}), encoding="utf-8")
print("wrote", RUN / "261111 figS6_meanfield.pdf", "and the numbers file")