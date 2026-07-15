# Corrected single-variable mean-field steady state.
# Greedy flips ONE variable at a time. For a variable in alpha AND + alpha XOR clauses,
# with partners independently 1 w.p. p:
#   A1 ~ Bin(alpha, p)   (AND clauses, partner=1)
#   X1 ~ Bin(alpha, p)   (XOR clauses, partner=1)
#   X0 ~ Bin(alpha, 1-p) (XOR clauses, partner=0)
# Atiyab's exact rules:  accept 0->1 iff A1+X0 > X1 ;  accept 1->0 iff X1 > A1+X0.
# Let S = A1 + X0 - X1.  F_up = Pr(S>0),  F_dn = Pr(S<0).
# Rate eq (single variable):  dp/dt ~ (1-p) F_up - p F_dn  =>  p_bar = F_up / (F_up + F_dn).
from math import comb

def binpmf(n, k, q):
    return comb(n, k) * q**k * (1-q)**(n-k)

def F_up_dn(alpha, p):
    fup = fdn = 0.0
    for a1 in range(alpha+1):
        pa1 = binpmf(alpha, a1, p)
        for x1 in range(alpha+1):
            px1 = binpmf(alpha, x1, p)
            for x0 in range(alpha+1):
                px0 = binpmf(alpha, x0, 1-p)
                w = pa1*px1*px0
                S = a1 + x0 - x1
                if S > 0: fup += w
                elif S < 0: fdn += w
    return fup, fdn

def solve(alpha, p0=0.5, iters=200):
    p = p0
    for _ in range(iters):
        fup, fdn = F_up_dn(alpha, p)
        p = fup/(fup+fdn) if (fup+fdn) > 0 else p
    return p

def Vfrac(p):  # naive independent-bit fraction (same one everyone uses)
    return 0.5 + 0.5*(1-p)**2

print(f"{'alpha':>5} {'p_corrected':>12} {'V/M(naive p)':>13}   (Juan p~0.66@4, sim p=0.74 V/M=0.49)")
for a in (2,4,8,16,50):
    pc = solve(a)
    print(f"{a:>5} {pc:>12.3f} {Vfrac(pc):>13.3f}")
