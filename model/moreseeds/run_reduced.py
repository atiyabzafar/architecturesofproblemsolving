"""run_reduced.py -- runs the reduced EBM on the three networks of the paper and draws the check figure.
Only the in-degree sequences are needed. Writes res.json and reduced_ebm_check.png/.pdf (about 1 minute)."""
import numpy as np, json
from reduced_ebm import ReducedEBM, v0

def heavy_tail_sequence(n=100, mean=4, expo=0.65):
    """In-degrees k_r = floor(c r^-expo), floored at 1, summing to n*mean (as in the paper)."""
    target = n * mean; lo, hi = 0.1, 1000.0
    for _ in range(100):
        c = 0.5 * (lo + hi)
        k = np.maximum(1, np.floor(c * np.arange(1, n + 1) ** -expo)).astype(int)
        lo, hi = (c, hi) if k.sum() < target else (lo, c)
    k[0] += target - k.sum()
    return k

vT, vS = 0.13, 0.12
seqs = {"even": np.full(100,4), "uneven": heavy_tail_sequence(100,4,0.65)}
seqs["dense"] = np.minimum(3*seqs["uneven"], 99)
res = {}
for name, kin in seqs.items():
    m = ReducedEBM(kin, rule="recency"); M=m.M
    te = np.unique(np.r_[np.arange(0,50,0.25), np.arange(50,1501,1.0)])
    t, Y = m.run(1500, t_eval=te)
    ys = m.shock(Y[:,-1])
    t2, Y2 = m.run(2500, y0=ys, t0=1500, t_eval=np.arange(1500,2501,1.0))
    o, o2 = m.unpack(Y), m.unpack(Y2)
    V  = M*((1-o["T"])*v0(o["p"]) + o["T"]*vT + o["S"]*vS)
    V2 = M*((1-o2["T"])*v0(o2["p"]) + o2["T"]*vT + o2["S"]*vS)
    late = t > 1000
    r = dict(k=m.k.tolist(), lam=m.lam.tolist(), P=m.P.tolist())
    for q in "TSNWp": r[q+"_k"] = o[q][late].mean(0).tolist(); r[q+"_pop"] = float(o[q][late].mean(0)@m.P)
    r["V_k"] = V[late].mean(0).tolist(); r["V_pop"] = float(V[late].mean(0)@m.P)
    r["stale_pop"] = float(((o["S"]/(o["T"]+o["S"]))[late].mean(0))@m.P)
    Vp = V@m.P; Vs = V2@m.P
    r["t"]=t.tolist(); r["Vpop_t"]=Vp.tolist(); r["ppop_t"]=(o["p"]@m.P).tolist(); r["Tpop_t"]=(o["T"]@m.P).tolist()
    r["p_t"]=o["p"].tolist(); r["t2"]=t2.tolist(); r["Vpop_t2"]=Vs.tolist(); r["Tpop_t2"]=(o2["T"]@m.P).tolist()
    pre = Vp[-1]; jump = Vs[1:21].mean(); base = r["V_pop"]
    rec = lambda a,b: float(1-(Vs[(t2-1500>=a)&(t2-1500<=b)].mean()-base)/(jump-base))
    r["shock"] = dict(before=float(pre), w1_20=float(jump),
                      windows={f"{a}-{b}": float(Vs[(t2-1500>=a)&(t2-1500<=b)].mean()) for a,b in [(1,20),(21,60),(61,120),(121,200),(201,400),(401,1000)]})
    # time to recover 3/4 of the jump
    peak = Vs[1:5].max(); frac = (peak - Vs)/(peak - base)
    r["shock"]["t_75"] = float(t2[np.argmax(frac>=0.75)]-1500)
    # early split
    pk = o["p"]; sel = t <= 10
    sl = []
    for i in np.where(sel)[0][1::4]:
        if t[i] in (1,2,3,5,8,10,20,50):
            d = pk[i]-pk[i,0]; dl = m.lam-m.lam[0]
            sl.append((float(t[i]), float(np.corrcoef(dl,d)[0,1]**2) if d.std()>0 else 1.0, float(np.polyfit(dl,d,1)[0])))
    r["early"] = sl
    res[name]=r
    print(name, {q: round(r[q+"_pop"],3) for q in "TSNWp"}, "stale", round(r["stale_pop"],3), "V", round(r["V_pop"],2))
    print("   shock", {k: round(v,2) for k,v in r["shock"]["windows"].items()}, "t75", r["shock"]["t_75"])
json.dump(res, open("res.json","w"))
# per-class V vs intake
for name in ("uneven","dense"):
    r=res[name]; lam=np.array(r["lam"]); V=np.array(r["V_k"]); T=np.array(r["T_k"]); S=np.array(r["S_k"]); p=np.array(r["p_k"])
    b = np.polyfit(np.log(lam), V, 1)[0]
    print(name, "slope dV/dln(lam) =", round(b,2), " V at lam .25/1/2.5/max:",
          [round(float(np.interp(np.log(x), np.log(lam), V)),2) for x in (0.25,1,2.5,lam.max())],
          " p range", round(p.min(),3), round(p.max(),3), " T range", round(T.min(),3), round(T.max(),3), " S range", round(S.min(),3), round(S.max(),3))

# ------------------------------------------------------------------ figure
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
r = res
C = {"even": "#6b6b6b", "uneven": "#2a78d6", "dense": "#eb6834"}
L = {"even": "regular", "uneven": "scale free", "dense": "scale free, dense"}
LS = {"even": "-", "uneven": "-", "dense": "--"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.2, "grid.linewidth": 0.6, "legend.frameon": False,
                     "lines.linewidth": 2})
fig, ax = plt.subplots(2, 2, figsize=(10, 7.8))

# (a) approach to p*: classes of the scale-free network + regular
a = ax[0, 0]; x = r["uneven"]; t = np.array(x["t"]); p = np.array(x["p_t"]); lam = np.array(x["lam"])
pick = [0, int(np.argmin(abs(lam - 1))), len(lam) - 1]
blues = ["#9ec5f4", "#3987e5", "#104281"]
for c, col in zip(pick, blues):
    a.plot(np.maximum(t, .25), p[:, c], color=col, label=f"in-degree {x['k'][c]} (intake {lam[c]:.2f})")
te = np.array(r["even"]["t"]); a.plot(np.maximum(te, .25), np.array(r["even"]["p_t"])[:, 0], color=C["even"], ls=":", label="regular (intake 1)")
a.axhspan(0.686, 0.691, color="#eda100", alpha=0.25, lw=0); a.text(0.3, 0.694, "ABM, all networks", fontsize=8, color="#7a5300")
a.set_xscale("log"); a.set_xlabel("periods (log)"); a.set_ylabel("share of choices set to 1, $p_k$")
#a.set_title("(a) same attractor, approached at rate set by intake", loc="left", fontsize=10); 
a.set_title("(a)", loc="left", fontsize=10); 
a.legend(fontsize=8, loc="lower right")

# (b) early split linear in intake
a = ax[0, 1]
shades = [
#    "#e6f1fd",  # 1
#    "#cde2fb",  # 2
    "#a9cdf6",  # 3
    "#86b6ef",  # 5
    "#5b9eea",  # 8
    "#3987e5",  # 10
    "#1c5cab",  # 20
    "#0d366b",  # 50
]

for tt, col in zip((1, 2, 3,5,8, 250), shades):
    print(tt,"\n")
    i = int(np.argmin(abs(t - tt))); a.plot(lam, p[i] - p[i, 0], "o-", ms=4, lw=1.5, color=col, label=f"t = {tt}")
a.set_xlabel("intake $\\lambda_k=k/\\langle k\\rangle$"); a.set_ylabel("$p_k - p_{k_{\\min}}$")
a.set_title("(b)", loc="left", fontsize=10); 
#a.set_title("(b) early split grows linearly in intake\n(scale free)", loc="left", fontsize=10); 
a.legend(fontsize=8, loc="upper left")

# (e) shock
a = ax[1, 0]
for name in ("even", "uneven", "dense"):
    x = r[name]; a.plot(np.array(x["t2"]) - 1500, x["Vpop_t2"], color=C[name], ls=LS[name], label=L[name])
a.set_xlabel("periods after the shock"); a.set_ylabel("average violations")
a.set_xlim(-5, 600); a.legend(fontsize=8)
#a.set_title("(c) after the shock: transients different,\nregular advantage", loc="left", fontsize=10)
a.set_title("(c)", loc="left", fontsize=10)

# (f) long-run population values vs ABM
a = ax[1, 1]
abm = {"even": (0.237, 0.180, 0.691, 0.432), "uneven": (0.183, 0.179, 0.686, 0.496), "dense": (0.179, 0.174, 0.686, 0.493)}
xs = np.arange(4); w = 0.13
plt.rcParams["hatch.linewidth"] = 0.8
for j, name in enumerate(("even", "uneven", "dense")):
    e = [r[name]["T_pop"], r[name]["S_pop"], r[name]["p_pop"], r[name]["stale_pop"]]
    off = (j - 1) * 2.2 * w
    a.bar(xs + off - w / 2, abm[name], w * 0.9, facecolor="white", edgecolor=C[name], hatch="xxxx", lw=1.0)
    a.bar(xs + off + w / 2, e, w * 0.9, color=C[name], edgecolor=C[name], lw=1.0)
from matplotlib.patches import Patch
hand = [Patch(facecolor=C[n], edgecolor=C[n], label=L[n]) for n in ("even", "uneven", "dense")]
hand += [Patch(facecolor="white", edgecolor="#444", hatch="xxxx", label="ABM"),
         Patch(facecolor="#444", edgecolor="#444", label="reduced EBM")]
a.set_xticks(xs, ["live T", "stale S", "p", "stale share"]); a.set_ylabel("long-run value"); a.set_ylim(0, 0.82)
a.legend(handles=hand, fontsize=8, loc="upper left", ncol=2)
#a.set_title("(d) population, long run: ABM (hatched) vs\nreduced EBM (solid)", loc="left", fontsize=10)
a.set_title("(d)", loc="left", fontsize=10)
fig.tight_layout()
fig.savefig("reduced_ebm_check.png", dpi=200); fig.savefig("reduced_ebm_check.pdf")
