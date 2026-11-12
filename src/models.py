import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baseline import load
from config import OUTPUTS, HIGH_RISK_THRESHOLD

SEED = 42
N_BOOT = 2000
N_PAIR_PERM = 2000
MIN_DIFF = 0.10


def feature_sets(jr):
    return {
        "M0": [jr],
        "M1": [jr, "jeonse_ppsqm_yoy", f"{jr}_chg"],
        "M2": [jr, "new_share"],
    }


MODELS = {
    "linear": (make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
               make_pipeline(StandardScaler(), LogisticRegression(C=1.0))),
    "rf": (RandomForestRegressor(n_estimators=200, max_depth=3, min_samples_leaf=3, random_state=SEED),
           RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=3,
                                  class_weight="balanced", random_state=SEED)),
    "gbm": (GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=2,
                                      subsample=0.8, random_state=SEED),
            GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=2,
                                       subsample=0.8, random_state=SEED)),
}


def loocv(model, X, y_rate, y_cls):
    reg, clf = clone(MODELS[model][0]), clone(MODELS[model][1])
    p_reg, p_cls = np.zeros(len(X)), np.zeros(len(X))
    for tr, te in LeaveOneOut().split(X):
        reg.fit(X[tr], np.log(y_rate[tr]))
        p_reg[te] = reg.predict(X[te])
        if y_cls[tr].min() == y_cls[tr].max():
            p_cls[te] = y_cls[tr].mean()
        else:
            clf.fit(X[tr], y_cls[tr])
            p_cls[te] = clf.predict_proba(X[te])[:, 1]
    return p_reg, p_cls


def score(p_reg, p_cls, y_rate, y_cls):
    rho = spearmanr(p_reg, y_rate).statistic
    auc = roc_auc_score(y_cls, p_cls) if 0 < y_cls.sum() < len(y_cls) else np.nan
    return rho, auc


def _perm_once(model, X, y_rate, y_cls, seed):
    idx = np.random.default_rng(seed).permutation(len(y_rate))
    return score(*loocv(model, X, y_rate[idx], y_cls[idx]), y_rate[idx], y_cls[idx])


def paired_boot(a, b, y_rate, y_cls, rng):
    n, out = len(y_rate), []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if not 0 < y_cls[i].sum() < n:
            continue
        ra, aa = score(a[0][i], a[1][i], y_rate[i], y_cls[i])
        rb, ab = score(b[0][i], b[1][i], y_rate[i], y_cls[i])
        out.append((ra - rb, aa - ab))
    return np.array(out)


def _to_rank(p):
    return tuple(pd.Series(x).rank().to_numpy() / len(x) for x in p)


def paired_perm(a, b, y_rate, y_cls, rng):
    a, b = _to_rank(a), _to_rank(b)
    obs = np.subtract(score(*a, y_rate, y_cls), score(*b, y_rate, y_cls))
    hits = np.zeros(2)
    for _ in range(N_PAIR_PERM):
        s = rng.random(len(y_rate)) < 0.5
        pa = tuple(np.where(s, x, y) for x, y in zip(a, b))
        pb = tuple(np.where(s, y, x) for x, y in zip(a, b))
        d = np.subtract(score(*pa, y_rate, y_cls), score(*pb, y_rate, y_cls))
        hits += np.abs(d) >= np.abs(obs) - 1e-12
    return (hits + 1) / (N_PAIR_PERM + 1)


def paired_diffs(preds, sets, y_rate, y_cls, jr):
    rng, rng_perm = np.random.default_rng(SEED), np.random.default_rng(SEED + 1)
    comps = [((s, m), (s, "linear"), "모델: 같은 피처셋의 선형 대비")
             for s in sets for m in MODELS if m != "linear"]
    comps += [((s, m), ("M0", "linear"), "사다리: M0 선형 대비")
              for s in sets for m in MODELS if (s, m) != ("M0", "linear")]
    diffs = []
    for a, b, kind in comps:
        ra, aa = score(*preds[a], y_rate, y_cls)
        rb, ab = score(*preds[b], y_rate, y_cls)
        bt = paired_boot(preds[a], preds[b], y_rate, y_cls, rng)
        lo, hi = np.nanpercentile(bt, [2.5, 97.5], axis=0)
        p_rho, p_auc = paired_perm(preds[a], preds[b], y_rate, y_cls, rng_perm)
        diffs.append({"jr": jr, "comparison": kind, "a": "/".join(a), "b": "/".join(b),
                      "diff_spearman": ra - rb, "ci_spearman_lo": lo[0], "ci_spearman_hi": hi[0],
                      "p_paired_perm_spearman": p_rho,
                      "diff_auc": aa - ab, "ci_auc_lo": lo[1], "ci_auc_hi": hi[1],
                      "p_paired_perm_auc": p_auc,
                      "meaningful_spearman": abs(ra - rb) >= MIN_DIFF and (lo[0] > 0 or hi[0] < 0),
                      "meaningful_auc": abs(aa - ab) >= MIN_DIFF and (lo[1] > 0 or hi[1] < 0)})
    return pd.DataFrame(diffs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jr", default="jr_cell", help="사다리의 전세가율 피처")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--from-oof", action="store_true",
                    help="모형을 다시 학습하지 않고 저장된 model_oof*.csv로 쌍체 비교만 다시 계산")
    args = ap.parse_args()
    tag = "" if args.jr == "jr_cell" else f"_{args.jr}"
    sets = feature_sets(args.jr)

    if args.from_oof:
        o = pd.read_csv(OUTPUTS / f"model_oof{tag}.csv")
        preds = {k: (g.pred_log_rate.to_numpy(), g.pred_prob_high.to_numpy())
                 for k, g in o.groupby(["feature_set", "model"], sort=False)}
        first = o.groupby(["feature_set", "model"], sort=False).get_group(next(iter(preds)))
        dif = paired_diffs(preds, sets, first.accident_rate.to_numpy(float), first.high_risk.to_numpy(int), args.jr)
        dif.to_csv(OUTPUTS / f"model_paired_diff{tag}.csv", index=False, encoding="utf-8-sig")
        print(dif.drop(columns=["jr"]).round(3).to_string(index=False))
        return

    d = load(sorted({f for fs in sets.values() for f in fs}))
    y_rate = d["accident_rate"].to_numpy(float)
    y_cls = (y_rate >= HIGH_RISK_THRESHOLD).astype(int)
    print(f"n={len(d)}, 고위험 {y_cls.sum()}개 구, 전세가율 = {args.jr}")

    rows, oofs, preds = [], [], {}
    for sname, feats in sets.items():
        X = d[feats].to_numpy(float)
        for model in MODELS:
            p = loocv(model, X, y_rate, y_cls)
            preds[(sname, model)] = p
            rho, auc = score(*p, y_rate, y_cls)
            null = np.array(Parallel(n_jobs=-1)(
                delayed(_perm_once)(model, X, y_rate, y_cls, SEED + k) for k in range(args.n_perm)))
            rows.append({"jr": args.jr, "feature_set": sname, "model": model, "features": "+".join(feats),
                         "n": len(d), "n_high": int(y_cls.sum()),
                         "loocv_spearman": rho,
                         "p_perm_spearman": (np.sum(null[:, 0] >= rho) + 1) / (args.n_perm + 1),
                         "loocv_auc": auc,
                         "p_perm_auc": (np.sum(null[:, 1] >= auc) + 1) / (args.n_perm + 1)})
            oofs.append(pd.DataFrame({"gu": d.gu, "jr": args.jr, "feature_set": sname, "model": model,
                                      "accident_rate": y_rate, "high_risk": y_cls,
                                      "pred_log_rate": p[0], "pred_prob_high": p[1]}))
            print(f"  {sname:3s} {model:7s} ρ={rho:+.3f}  AUC={auc:.3f}")

    res, dif = pd.DataFrame(rows), paired_diffs(preds, sets, y_rate, y_cls, args.jr)
    res.to_csv(OUTPUTS / f"model_comparison{tag}.csv", index=False, encoding="utf-8-sig")
    dif.to_csv(OUTPUTS / f"model_paired_diff{tag}.csv", index=False, encoding="utf-8-sig")
    pd.concat(oofs).to_csv(OUTPUTS / f"model_oof{tag}.csv", index=False, encoding="utf-8-sig")
    print("\n" + res.drop(columns=["jr", "features", "n", "n_high"]).round(3).to_string(index=False))
    print("\n" + dif.drop(columns=["jr"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
