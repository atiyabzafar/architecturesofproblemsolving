"""
reduced_ebm.py -- the reduced compartment model (same equations as the full EBM, simpler terms).

Per in-degree class k (fractions of M):  T, S, N, W, p;  plus R, the number of retired clauses.
    dT = D + th W + A^T + Phi^NT - Phi^TN - T/M + S/|U| - E^T
    dS = T/M - th S + A^S + Phi^NS - Phi^SN - S/|U| - E^S
    dN = th S - th W + A^N + A^W + Phi^SN + Phi^TN - Phi^NS - Phi^NT - E^N
    dW = -th W - W/M + (N-W)/|U| + A^W + Phi^TN - Phi^NT - E^W
    dp = (2/K) Pi [(1-p) a_up - p a_down]     (local-optimum bracket)
    dR = 1 - R/|U|
Communication, simplified:
  * a class-k agent receives lambda_k = k/<k> messages per period from a random member of the
    population, who has something to say with probability a_k = 1 - exp(-Delta_k),
    Delta_k = M sum_X Xbar d_X,k (population-average sender);
  * the message is of kind X in {T, W, S, N-W} with probability
        m_X,k = gamma_X d_X,k / sum_Y gamma_Y d_Y,k,
    d_X,k = probability that the receiver disagrees;  gamma_X = what is in circulation:
        rule="stock"   : gamma = population stock (Tbar, Wbar, Sbar, (N-W)bar)    [uniform rule]
        rule="recency" : gamma = rate at which beliefs of kind X are touched (tested, discovered or
                         received) in the population, solved self-consistently.  Only tests create
                         touches of T and N-W; stale positives and wrong negatives are touched only by
                         being received, so at the self-consistent point gamma_S = gamma_W = 0.
  * an arriving message is an addition A if the receiver held no belief about the clause and a
    flip Phi if it held the opposite sign.
Eviction: new beliefs (discoveries and additions, not flips) evict in proportion to stock once memory
is full, switched on by a logistic in the occupancy B/2 (kappa = 500, midpoint B0 = 0.99).
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import skellam # for Z(a)


def classes_from_degrees(kin):
    ks, cnt = np.unique(np.asarray(kin), return_counts=True)
    P = cnt / cnt.sum()
    return ks, P, ks / (ks * P).sum()


def flip_probs(mu, p):
    mu = np.maximum(mu, 1e-9); p = np.clip(p, 1e-9, 1 - 1e-9)
    z0 = skellam.pmf(0, mu, mu * p)
    up = 0.5 * z0 / np.maximum(skellam.cdf(0, mu, mu * p), 1e-12)
    dn = 0.5 * p * z0 / np.maximum(skellam.sf(-1, mu, mu * p), 1e-12)
    return up, dn


def v0(p):
    return 1 - p + p ** 2 / 2


def split(x, y):
    tot = x + y
    return np.divide(x, tot, out=np.zeros_like(tot), where=tot > 1e-15), \
           np.divide(y, tot, out=np.zeros_like(tot), where=tot > 1e-15)


class ReducedEBM:
    def __init__(self, kin, K=30, alpha=2, rule="recency", kappa=500.0, B0=0.99, n_fix=300):
        self.k, self.P, self.lam = classes_from_degrees(kin)
        self.nc = len(self.k)
        self.K, self.alpha = K, alpha
        self.M, self.U = int(round(alpha * K)), K * (K - 1)
        self.rule, self.kappa, self.B0, self.n_fix = rule, kappa, B0, n_fix

    def terms(self, y):
        nc, M, U, P, lam = self.nc, self.M, self.U, self.P, self.lam
        cl = lambda v: np.clip(v, 0, None)
        T, S, N = cl(y[:nc]), cl(y[nc:2*nc]), cl(y[2*nc:3*nc])
        W = np.minimum(cl(y[3*nc:4*nc]), N)
        p = np.clip(y[4*nc:5*nc], 0, 1)
        R = max(float(y[5*nc]), 0.0)
        Nd = N - W
        B = T + S + N
        th = 1 / (2 * M * np.maximum(B, 1e-12))
        uL = np.clip(1 - T - W, 0, 1)
        if R > 1e-9:
            s, f = np.clip(M * S / R, 0, 1), np.clip(M * Nd / R, 0, 1)
        else:
            s, f = 0 * T, 0 * T
        uD = np.clip(1 - s - f, 0, 1)
        D = uL / (2 * (U - M * np.minimum(B, 2)))
        d = np.clip(np.array([1 - T, 1 - W, 1 - s, 1 - f]), 0, 1)            # receiver disagrees
        stock = np.array([T, W, S, Nd])
        a = -np.expm1(-M * ((stock @ P) @ d))                     # sender has something to say
        flow = a * lam / M                                         # messages delivered, fraction of M

        if self.rule == "stock":
            g = stock @ P
        else:                                                      # recency: touch rates
            priv = np.array([P @ (D + th * (T + W)), 0.0, 0.0, P @ (th * (S + Nd))])
            g = priv + 1e-12
            for _ in range(self.n_fix):
                m = g[:, None] * d
                m = np.divide(m, m.sum(0), out=np.zeros_like(m), where=m.sum(0) > 1e-15)
                g_new = priv + 1e-12 + (m * flow) @ P
                if np.abs(g_new - g).max() < 1e-12 * max(g_new.max(), 1e-12):
                    g = g_new; break
                g = g_new
        g = g + 1e-12                                              # numerical floor
        m = g[:, None] * d
        m = np.divide(m, m.sum(0), out=np.zeros_like(m), where=m.sum(0) > 1e-15)
        cT, cW, cS, cN = m * flow                                  # messages of each kind received

        add, flp = split(uL, W);  AT, PhiNT = cT * add, cT * flp   # "this live clause holds"
        add, flp = split(uL, T);  AW, PhiTN = cW * add, cW * flp   # "this live clause is gone"
        add, flp = split(uD, f);  AS, PhiNS = cS * add, cS * flp   # "this retired clause holds"
        add, flp = split(uD, s);  AN, PhiSN = cN * add, cN * flp   # "this retired clause is gone"

        # logistic switch on eviction, 1/(1+exp(-kappa(B/2-B0))), written overflow-safe
        sig = 0.5 * (1 + np.tanh(0.5 * self.kappa * (B / 2 - self.B0)))
        Ek = sig * (D + AT + AW + AS + AN) / np.maximum(B, 1e-12)
        return dict(T=T, S=S, N=N, W=W, Nd=Nd, p=p, R=R, th=th, D=D, g=g,
                    AT=AT, AW=AW, AS=AS, AN=AN, PhiNT=PhiNT, PhiTN=PhiTN, PhiNS=PhiNS, PhiSN=PhiSN,
                    ET=Ek*T, ES=Ek*S, EN=Ek*N, EW=Ek*W)

    def rhs(self, t, y):
        x = self.terms(y); M, U = self.M, self.U
        T, S, N, W, Nd, th = x["T"], x["S"], x["N"], x["W"], x["Nd"], x["th"]
        dT = x["D"] + th*W + x["AT"] + x["PhiNT"] - x["PhiTN"] - T/M + S/U - x["ET"]
        dS = T/M - th*S + x["AS"] + x["PhiNS"] - x["PhiSN"] - S/U - x["ES"]
        dN = th*S - th*W + x["AN"] + x["AW"] + x["PhiSN"] + x["PhiTN"] - x["PhiNS"] - x["PhiNT"] - x["EN"]
        dW = -th*W - W/M + Nd/U + x["AW"] + x["PhiTN"] - x["PhiNT"] - x["EW"]
        Pi = M * (x["D"] + th*W + x["AT"] + x["PhiNT"] + x["AS"] + x["PhiNS"])
        up, dn = flip_probs(self.alpha * (T + S), x["p"])
        dp = (2 / self.K) * Pi * ((1 - x["p"]) * up - x["p"] * dn)
        return np.concatenate([dT, dS, dN, dW, dp, [1 - x["R"] / U]])

    def y0(self):
        y = np.zeros(5 * self.nc + 1); y[4*self.nc:5*self.nc] = 0.5; return y

    def run(self, t_end, y0=None, t0=0.0, t_eval=None):
        y0 = self.y0() if y0 is None else y0
        if t_eval is None:
            t_eval = np.arange(t0, t_end + 1e-9, 1.0)
        sol = solve_ivp(self.rhs, (t0, t_end), y0, t_eval=t_eval, method="LSODA", rtol=1e-6, atol=1e-9)
        return sol.t, sol.y

    def shock(self, y):
        """Every clause replaced at once: live positives turn stale, wrong negatives turn correct."""
        nc = self.nc; y = y.copy()
        y[nc:2*nc] += y[:nc]; y[:nc] = 0.0; y[3*nc:4*nc] = 0.0
        y[5*nc] = min(y[5*nc] + self.M, self.U)
        return y

    def unpack(self, Y):
        nc = self.nc
        return {q: Y[i*nc:(i+1)*nc].T for i, q in enumerate("TSNWp")}
