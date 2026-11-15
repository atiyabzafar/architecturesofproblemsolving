import numpy as np
from scipy.optimize import root
from scipy.stats import skellam

# Pair-level steady-state closure for the ER / well-mixed case
# Variables:
#   uA = q11^A
#   uX = q00^X
#   vX = q11^X
# with symmetry q01^C = q10^C = (1 - diagonal terms)/2.
#
# This code closes the pair equations by using conditional partner
# probabilities and Skellam-distributed background fields.
#
# It solves for stationary pair observables and predicts
#   V* = 1/2 (1 - q11^A) + 1/2 (q00^X + q11^X)
# and V_abs = alpha * K * V*.


def clip01(x, eps=1e-10):
    return min(1.0 - eps, max(eps, x))


def pair_probabilities(uA, uX, vX):
    q11A = clip01(uA)
    q00X = clip01(uX)
    q11X = clip01(vX)
    q01A = max(1e-12, (1.0 - q11A) / 2.0)
    q01X = max(1e-12, (1.0 - q00X - q11X) / 2.0)
    pA = q11A + q01A
    pX = q11X + q01X
    p = 0.5 * (pA + pX)
    p = clip01(p)
    return {
        'q11A': q11A,
        'q01A': q01A,
        'q00X': q00X,
        'q01X': q01X,
        'q11X': q11X,
        'pA': pA,
        'pX': pX,
        'p': p,
    }


def conditional_thetas(probs):
    q11A, q01A = probs['q11A'], probs['q01A']
    q00X, q01X, q11X = probs['q00X'], probs['q01X'], probs['q11X']
    p = clip01(probs['p'])
    p0 = clip01(1.0 - p)
    return {
        'A10': q01A / p0,
        'A11': q11A / p,
        'X00': q00X / p0,
        'X10': q01X / p0,
        'X01': q01X / p,
        'X11': q11X / p,
    }


def skellam_tail_ge(k, mu_plus, mu_minus):
    return skellam.sf(k - 1, mu_plus, mu_minus)


def skellam_tail_le(k, mu_plus, mu_minus):
    return skellam.cdf(k, mu_plus, mu_minus)


def gamma_rates(alpha, probs):
    th = conditional_thetas(probs)
    mu_plus_0 = alpha * (th['A10'] + th['X00'])
    mu_minus_0 = alpha * th['X10']
    mu_plus_1 = alpha * (th['A11'] + th['X01'])
    mu_minus_1 = alpha * th['X11']
    return {
        'G_01_A1': skellam_tail_ge(0,  mu_plus_0, mu_minus_0),
        'G_01_A0': skellam_tail_ge(1,  mu_plus_0, mu_minus_0),
        'G_01_X0': skellam_tail_ge(0,  mu_plus_0, mu_minus_0),
        'G_01_X1': skellam_tail_ge(2,  mu_plus_0, mu_minus_0),
        'G_10_A1': skellam_tail_le(-2, mu_plus_1, mu_minus_1),
        'G_10_X0': skellam_tail_le(-2, mu_plus_1, mu_minus_1),
        'G_10_X1': skellam_tail_le(0,  mu_plus_1, mu_minus_1),
        'mu_plus_0': mu_plus_0,
        'mu_minus_0': mu_minus_0,
        'mu_plus_1': mu_plus_1,
        'mu_minus_1': mu_minus_1,
    }


def stationary_equations(vec, alpha):
    uA, uX, vX = vec
    probs = pair_probabilities(uA, uX, vX)
    g = gamma_rates(alpha, probs)
    q11A, q01A = probs['q11A'], probs['q01A']
    q00X, q01X, q11X = probs['q00X'], probs['q01X'], probs['q11X']
    f1 = q01A * g['G_01_A1'] - q11A * g['G_10_A1']
    f2 = q01X * g['G_10_X0'] - q00X * g['G_01_X0']
    f3 = q01X * g['G_01_X1'] - q11X * g['G_10_X1']
    return np.array([f1, f2, f3])


def solve_pair_steady_state(alpha, guess=(0.7, 0.25, 0.25)):
    sol = root(lambda z: stationary_equations(z, alpha), np.array(guess), method='hybr')
    probs = pair_probabilities(*sol.x)
    Vfrac = 0.5 * (1.0 - probs['q11A']) + 0.5 * (probs['q00X'] + probs['q11X'])
    return {
        'success': bool(sol.success),
        'message': sol.message,
        'alpha': alpha,
        'q11A': probs['q11A'],
        'q01A': probs['q01A'],
        'q00X': probs['q00X'],
        'q01X': probs['q01X'],
        'q11X': probs['q11X'],
        'p': probs['p'],
        'Vfrac': Vfrac,
        'abs_residual': float(np.linalg.norm(stationary_equations(sol.x, alpha))),
        'raw_solution': sol,
    }


def solve_alpha_grid(alphas, K=20, guess=(0.7, 0.25, 0.25), verbose=True):
    out = []
    current_guess = guess
    for alpha in alphas:
        ans = solve_pair_steady_state(alpha, guess=current_guess)
        ans['Vabs'] = alpha * K * ans['Vfrac']
        out.append(ans)
        current_guess = (ans['q11A'], ans['q00X'], ans['q11X'])
        if verbose:
            print(
                f"alpha={alpha:4.1f} | success={ans['success']} | "
                f"p*={ans['p']:.4f} | V*={ans['Vfrac']:.4f} | abs={ans['Vabs']:.2f} | "
                f"res={ans['abs_residual']:.2e}"
            )
    return out


if __name__ == '__main__':
    alphas = [1, 2, 3, 4, 6, 8, 10]
    results = solve_alpha_grid(alphas, K=20)
    print('\nDetailed first solution:')
    r = results[0]
    for k in ['q11A', 'q01A', 'q00X', 'q01X', 'q11X', 'p', 'Vfrac', 'Vabs', 'abs_residual']:
        print(f'{k}: {r[k]}')
