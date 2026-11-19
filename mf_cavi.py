"""Uncertainty-aware matrix factorisation: mean-field CAVI vs. MAP on cold-start users.

Model
    u_i, v_j ~ N(0, I_K)
    r_ij | u_i, v_j ~ N(u_i^T v_j, sigma^2),   (i, j) observed

Variational family (mean field)
    q(u_i) = N(mu_i, S_i),   q(v_j) = N(nu_j, T_j)

Evaluation
    Hold out 20% of observed ratings, fit on the rest, and report the average held-out
    predictive log likelihood (1/|out|) sum log p(r_ij | R_in):
      VI:  N(mu_i^T nu_j, sigma^2 + Var_q[u_i^T v_j])
      MAP: N(u_hat_i^T v_hat_j, sigma^2)   (plug-in point estimate)
    Results are broken down by how many in-sample ratings the user has.

Usage
    python mf_cavi.py               # main run (seed 1) + figure
    python mf_cavi.py --seeds 10    # also repeat over 10 seeds
"""
import argparse
import os

import numpy as np
from scipy.stats import norm

N_USERS, N_ITEMS, K, SIGMA = 300, 200, 3, 0.5
N_ITERS = 50
BINS = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 10_000)]
I_K = np.eye(K)


# ---------------------------------------------------------------- data
def simulate(rng):
    """Ratings from the model; users differ a lot in how many items they rate."""
    U = rng.standard_normal((N_USERS, K))
    V = rng.standard_normal((N_ITEMS, K))
    R = U @ V.T + SIGMA * rng.standard_normal((N_USERS, N_ITEMS))
    rated = rng.random((N_USERS, N_ITEMS)) < rng.beta(0.7, 4.0, size=N_USERS)[:, None]
    held = rated & (rng.random((N_USERS, N_ITEMS)) < 0.2)
    return R, rated & ~held, held


# ---------------------------------------------------------------- inference
def outer(m):
    return np.einsum("nk,nl->nkl", m, m)


def update_side(means_other, second_other, neighbours, ratings):
    """Gaussian block update for one side, given E[w] and E[w w^T] of the other side.

    With second_other = E[w w^T] this is the CAVI update; with second_other = w w^T
    (point estimates) the mean is the exact ridge-regression (MAP) maximiser.
    """
    n = len(neighbours)
    mean, cov = np.empty((n, K)), np.empty((n, K, K))
    for x, nb in enumerate(neighbours):
        cov[x] = np.linalg.inv(I_K + second_other[nb].sum(0) / SIGMA**2)
        mean[x] = cov[x] @ (means_other[nb].T @ ratings[x]) / SIGMA**2
    return mean, cov


def pred_var(m_i, S_i, m_j, T_j):
    """Var_q[u_i^T v_j] for independent Gaussians."""
    return (np.einsum("nkl,nlk->n", S_i, T_j)
            + np.einsum("nk,nkl,nl->n", m_i, T_j, m_i)
            + np.einsum("nk,nkl,nl->n", m_j, S_i, m_j))


def kl_std(m, C):
    """sum_n KL(N(m_n, C_n) || N(0, I))."""
    _, logdet = np.linalg.slogdet(C)
    return 0.5 * np.sum(np.trace(C, axis1=1, axis2=2) + (m**2).sum(1) - K - logdet)


def run(seed):
    rng = np.random.default_rng(seed)
    R, mask, held = simulate(rng)
    obs_i, obs_j = np.nonzero(mask)
    te_i, te_j = np.nonzero(held)
    r_obs, r_te = R[obs_i, obs_j], R[te_i, te_j]

    by_user = [np.nonzero(mask[i])[0] for i in range(N_USERS)]
    by_item = [np.nonzero(mask[:, j])[0] for j in range(N_ITEMS)]
    user_r = [R[i, by_user[i]] for i in range(N_USERS)]
    item_r = [R[by_item[j], j] for j in range(N_ITEMS)]

    def elbo(mu, S, nu, T):
        sq = ((r_obs - (mu[obs_i] * nu[obs_j]).sum(1)) ** 2
              + pred_var(mu[obs_i], S[obs_i], nu[obs_j], T[obs_j]))
        ll = -0.5 * len(r_obs) * np.log(2 * np.pi * SIGMA**2) - 0.5 * sq.sum() / SIGMA**2
        return ll - kl_std(mu, S) - kl_std(nu, T)

    init_u = 0.1 * rng.standard_normal((N_USERS, K))
    init_v = 0.1 * rng.standard_normal((N_ITEMS, K))

    # CAVI
    mu, S = init_u.copy(), np.tile(I_K, (N_USERS, 1, 1))
    nu, T = init_v.copy(), np.tile(I_K, (N_ITEMS, 1, 1))
    elbos = []
    for _ in range(N_ITERS):
        mu, S = update_side(nu, T + outer(nu), by_user, user_r)
        nu, T = update_side(mu, S + outer(mu), by_item, item_r)
        elbos.append(elbo(mu, S, nu, T))
    elbos = np.array(elbos)
    assert np.all(np.diff(elbos) > -1e-6), "ELBO decreased: bug in the updates"

    # MAP by coordinate ascent (same initialisation)
    u_hat, v_hat = init_u.copy(), init_v.copy()
    for _ in range(N_ITERS):
        u_hat = update_side(v_hat, outer(v_hat), by_user, user_r)[0]
        v_hat = update_side(u_hat, outer(u_hat), by_item, item_r)[0]

    # held-out predictive log likelihood per rating
    sd_vi = np.sqrt(SIGMA**2 + pred_var(mu[te_i], S[te_i], nu[te_j], T[te_j]))
    ll_vi = norm.logpdf(r_te, (mu[te_i] * nu[te_j]).sum(1), sd_vi)
    ll_map = norm.logpdf(r_te, (u_hat[te_i] * v_hat[te_j]).sum(1), SIGMA)

    n_in = mask.sum(1)[te_i]  # in-sample ratings of the user behind each held-out rating
    by_bin = []
    for lo, hi in BINS:
        sel = (n_in >= lo) & (n_in < hi)
        by_bin.append((int(sel.sum()), ll_vi[sel].mean(), ll_map[sel].mean()))
    return dict(elbos=elbos, vi=ll_vi.mean(), map=ll_map.mean(), by_bin=by_bin,
                n_train=int(mask.sum()), n_test=int(held.sum()))


def bin_label(lo, hi):
    return f"{lo}-{hi - 1}" if hi < 10_000 else f"{lo}+"


# ---------------------------------------------------------------- reporting
def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    axes[0].plot(np.arange(1, N_ITERS + 1), res["elbos"], color="#2a6fdb")
    axes[0].set_xlabel("CAVI iteration")
    axes[0].set_ylabel("ELBO")
    axes[0].set_title("ELBO increases every iteration", fontsize=10)

    labels = [bin_label(lo, hi) for lo, hi in BINS]
    vi_b = [b[1] for b in res["by_bin"]]
    map_b = [b[2] for b in res["by_bin"]]
    x, w = np.arange(len(BINS)), 0.38
    axes[1].bar(x - w / 2, vi_b, w, label="VI (CAVI)", color="#2a6fdb")
    axes[1].bar(x + w / 2, map_b, w, label="MAP", color="#d9822b")
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel("user's number of in-sample ratings")
    axes[1].set_ylabel("held-out log lik per rating (higher is better)")
    axes[1].set_title("The gap is concentrated in cold-start users", fontsize=10)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=0, help="also repeat over this many seeds")
    args = ap.parse_args()

    res = run(args.seed)
    e = res["elbos"]
    print(f"seed {args.seed}: {res['n_train']} in-sample ratings, {res['n_test']} held out")
    print(f"ELBO {e[0]:.1f} -> {e[-1]:.1f} (monotone)")
    print(f"held-out log lik per rating: VI {res['vi']:.3f}   MAP {res['map']:.3f}")
    for (lo, hi), (n, v, m) in zip(BINS, res["by_bin"]):
        print(f"  users with {bin_label(lo, hi):>6} ratings (n={n:4d}): VI {v:7.3f}   MAP {m:7.3f}")
    os.makedirs("figs", exist_ok=True)
    plot(res, os.path.join("figs", "results.png"))

    if args.seeds:
        runs = [run(s) for s in range(1, args.seeds + 1)]
        d = np.array([r["vi"] - r["map"] for r in runs])
        d0 = np.array([r["by_bin"][0][1] - r["by_bin"][0][2] for r in runs])
        print(f"\nover {args.seeds} seeds (VI minus MAP, mean +/- sd):")
        print(f"  all held-out ratings : {d.mean():+.3f} +/- {d.std(ddof=1):.3f}  (VI better in {(d > 0).sum()}/{len(d)})")
        print(f"  users with <5 ratings: {d0.mean():+.3f} +/- {d0.std(ddof=1):.3f}  (VI better in {(d0 > 0).sum()}/{len(d0)})")


if __name__ == "__main__":
    main()
