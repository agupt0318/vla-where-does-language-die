"""Cross-check probe numerics with scalar-loop einsum and an independent objective.

Apple Accelerate may set floating-point warning flags on finite matmul results.
This audit checks values, finite coefficients, solver termination, decision scores,
and predictions using a separately implemented multinomial logistic objective.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scripts.analyze_execution_probe import split_episodes


def audit(run: Path) -> dict:
    d = np.load(run / "activations.npz", allow_pickle=False)
    y = d["y"]
    tr, te, nv = split_episodes(d["groups"], d["novel_mask"].astype(bool), 0)
    test = np.concatenate([te, nv[d["executed_y"][nv] >= 0]])
    predictions = np.load(run / "predictions_seed0.npz", allow_pickle=False)
    out = {
        "description": (
            "Independent einsum-based multinomial logistic objective and score verification."
        ),
        "sites": {},
    }
    for site in sorted(k for k in d.files if k.startswith(("vlm.", "expert."))):
        scaler = StandardScaler().fit(d[site][tr])
        x = scaler.transform(d[site][tr])
        z = scaler.transform(d[site][test])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clf = LogisticRegression(C=0.1, max_iter=2000, random_state=0, tol=1e-8).fit(x, y[tr])
            original_logits = clf.decision_function(z)
            original_pred = clf.predict(z)
        einsum_logits = np.einsum("ij,kj->ik", z, clf.coef_, optimize=False) + clf.intercept_
        np.testing.assert_allclose(original_logits, einsum_logits, rtol=1e-10, atol=1e-10)
        assert np.isfinite(clf.coef_).all() and np.isfinite(original_logits).all()
        assert clf.n_iter_.max() < 2000
        np.testing.assert_array_equal(original_pred, np.argmax(einsum_logits, axis=1))
        np.testing.assert_array_equal(original_pred[: len(te)], predictions[site + "__trained"])
        np.testing.assert_array_equal(original_pred[len(te) :], predictions[site])
        n, dim = x.shape
        target = np.eye(4)[y[tr]]
        reg = 1 / (0.1 * n)

        def objective(parameters, dim=dim, x=x, target=target, n=n, reg=reg):
            params = parameters.reshape(4, dim + 1)
            weights, intercept = params[:, :dim], params[:, dim]
            logits = np.einsum("ij,kj->ik", x, weights, optimize=False) + intercept
            logp = logits - logsumexp(logits, axis=1, keepdims=True)
            loss = -(target * logp).sum() / n + reg * (weights * weights).sum() / 2
            error = (np.exp(logp) - target) / n
            grad_w = np.einsum("ik,ij->kj", error, x, optimize=False) + reg * weights
            gradient = np.column_stack([grad_w, error.sum(axis=0)])
            return loss, gradient.ravel()

        result = minimize(
            objective,
            np.zeros(4 * (dim + 1)),
            jac=True,
            method="L-BFGS-B",
            options={"gtol": 1e-7, "ftol": 1e-12, "maxiter": 2000},
        )
        assert result.success, result.message
        params = result.x.reshape(4, dim + 1)
        independent_logits = (
            np.einsum("ij,kj->ik", z, params[:, :dim], optimize=False) + params[:, dim]
        )
        independent_pred = independent_logits.argmax(axis=1)
        assert np.isfinite(independent_logits).all()
        np.testing.assert_array_equal(original_pred, independent_pred)
        out["sites"][site] = {
            "sklearn_warning_count": len(caught),
            "warning_messages": sorted({str(w.message) for w in caught}),
            "coefficients_and_scores_finite": True,
            "sklearn_iterations": int(clf.n_iter_.max()),
            "max_blas_einsum_logit_error": float(abs(original_logits - einsum_logits).max()),
            "independent_optimizer_success": bool(result.success),
            "independent_optimizer_iterations": int(result.nit),
            "all_heldout_predictions_match_independent_optimizer": True,
            "all_heldout_predictions_match_saved_analysis": True,
            "n_compared": len(test),
        }
    (run / "numerical_audit.json").write_text(json.dumps(out, indent=2, allow_nan=False))
    print(run / "numerical_audit.json")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    audit(p.parse_args().run)
