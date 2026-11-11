import sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = {**dict(np.load(sys.argv[1])), **dict(np.load(sys.argv[2]))}; OUT = sys.argv[3]; S = range(1, 41)
plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 8, "legend.fontsize": 6.8, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "axes.linewidth": 0.7, "lines.linewidth": 1.4})
tot = lambda n, q: np.sum([D[f"{n}|{s}|{q}"] for s in S], 0)
def rho(n, k=1, c=None):
    h, ca = tot(n, "hold"), tot(n, "cacc")
    if c is None: return 100 * h[:, k].sum(1) / ca / 100
    ncl = np.array([D[f"{n}|{s}|ncl_in"] for s in S]).mean(0); return 100 * h[:, k, c] / ca / ncl[c]
mn = lambda n, a, b: (np.array(rho(n))[a:b + 1]).min()
print("SF rho- min/max over 100-200: %.1f / %.1f" % (mn("uneven", 100, 200), rho("uneven")[100:201].max()))
tau = np.arange(401)
fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.08, 2.6), gridspec_kw=dict(wspace=0.42))
for n, col, ls, lab in (("even", "#000000", "-", "Regular"), ("uneven", "#0072B2", "-", "Scale Free"),
                        ("uneven_eq", "#0072B2", ":", "Scale Free, equal intake"), ("dense", "#E69F00", "--", "Scale Free, dense")):
    a.plot(tau, rho(n)[:401], color=col, ls=ls, label=lab)
a.set_xlabel("Periods since retirement, $\\tau$"); a.set_ylabel("Agents with a negative belief, $\\rho_-$ (%)"); a.set_ylim(0, 55); a.set_yticks([0, 10, 20, 30, 40])
a.legend(loc="upper right", borderaxespad=0.1); a.set_title("a", loc="left", fontweight="bold")
for cc, col, lab in ((2, "#000000", "High intake, memory 37"), (1, "#0072B2", "Middle intake, memory 107"),
                     (0, "#E69F00", "Low intake, memory 176")):
    b.plot(tau, rho("uneven", 1, cc)[:401], color=col, label=lab)
b.plot(tau, rho("uneven")[:401], color="#888888", ls="--", lw=1.0, label="All agents")
b.set_xlabel("Periods since retirement, $\\tau$"); b.set_ylabel("Agents with a negative belief (%)"); b.set_ylim(0, 48); b.set_yticks([0, 10, 20, 30])
b.legend(loc="upper right", borderaxespad=0.1, title="Scale Free", title_fontsize=7); b.set_title("b", loc="left", fontweight="bold")
for n, col, lab in (("even", "#000000", "Regular"), ("uneven", "#0072B2", "Scale Free")):
    ev, ca = tot(n, "ev"), tot(n, "cacc"); r = ev[:, 6].sum(1) / ca
    k = np.ones(11) / 11; c.plot(tau[5:396], np.convolve(r, k, "valid")[:391], color=col, label=lab)
c.set_xlabel("Periods since retirement, $\\tau$"); c.set_ylabel("Negative beliefs evicted per period"); c.legend(loc="upper right")
c.set_title("c", loc="left", fontweight="bold")
fig.savefig(OUT, dpi=250, bbox_inches="tight"); print("wrote", OUT)
