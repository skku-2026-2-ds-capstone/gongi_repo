import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import PROCESSED, EXTERNAL, OUTPUTS, TARGET_START, HIGH_RISK_THRESHOLD

RNG = np.random.default_rng(42)
N_PERM = 1000


def load(features):
    X = pd.read_csv(PROCESSED / "gu_features.csv")
    assert pd.to_datetime(X["window_end"]).max() < TARGET_START, "타깃 이후 자료로 만든 피처"
    y = pd.read_csv(EXTERNAL / "hug_target.csv")
    d = X.merge(y, on="gu", how="inner", validate="one_to_one")
    if len(d) != 25:
        print(f"[경고] 결합된 구 {len(d)}개 (25개 아님): 누락 {set(X.gu) ^ set(d.gu)}")
    miss = d[features].isna().any(axis=1)
    if miss.any():
        print(f"[경고] 피처 결측으로 제외: {d.loc[miss, 'gu'].tolist()}")
        d = d[~miss]
    return d.reset_index(drop=True)


def loocv(X, y_rate, y_cls):
    reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0))
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


def metrics(p_reg, p_cls, y_rate, y_cls):
    rho = spearmanr(p_reg, y_rate).statistic
    auc = roc_auc_score(y_cls, p_cls) if 0 < y_cls.sum() < len(y_cls) else np.nan
    pred = (p_cls >= y_cls.mean()).astype(int)
    bacc = 0.5 * ((pred[y_cls == 1] == 1).mean() + (pred[y_cls == 0] == 0).mean())
    return rho, auc, bacc


def evaluate(d, features, label, setting):
    d = d.dropna(subset=features).reset_index(drop=True)
    X = d[features].to_numpy(float)
    y_rate = d["accident_rate"].to_numpy(float)
    y_cls = (y_rate >= HIGH_RISK_THRESHOLD).astype(int)
    p_reg, p_cls = loocv(X, y_rate, y_cls)
    rho, auc, bacc = metrics(p_reg, p_cls, y_rate, y_cls)
    null_rho, null_auc = [], []
    for _ in range(N_PERM):
        idx = RNG.permutation(len(y_rate))
        pr, pc = loocv(X, y_rate[idx], y_cls[idx])
        r_, a_, _ = metrics(pr, pc, y_rate[idx], y_cls[idx])
        null_rho.append(r_)
        null_auc.append(a_)
    raw_rho = spearmanr(d[features[0]], y_rate).statistic if len(features) == 1 else np.nan
    row = {"model": label, "setting": setting, "features": "+".join(features), "n": len(d),
           "n_high": int(y_cls.sum()),
           "raw_spearman_feature": raw_rho,
           "loocv_spearman": rho, "p_perm_spearman": (np.sum(np.array(null_rho) >= rho) + 1) / (N_PERM + 1),
           "loocv_auc": auc, "p_perm_auc": (np.sum(np.array(null_auc) >= auc) + 1) / (N_PERM + 1),
           "loocv_balanced_acc": bacc}
    oof = pd.DataFrame({"gu": d.gu, "setting": setting, "model": label, "accident_rate": y_rate,
                        "high_risk": y_cls, "pred_log_rate": p_reg, "pred_prob_high": p_cls})
    return row, oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m0", nargs="+", default=["jr_cell"], help="M0 피처. 여러 개면 각각 단독 M0로 평가")
    ap.add_argument("--n-perm", type=int, default=1000, help="순열 횟수 (빠른 점검은 200)")
    args = ap.parse_args()
    global N_PERM
    N_PERM = args.n_perm
    d_all = load([])

    rows, oofs = [], []
    rows.append({"model": "naive", "setting": "기본", "features": "-", "n": len(d_all),
                 "n_high": int((d_all.accident_rate >= HIGH_RISK_THRESHOLD).sum()),
                 "loocv_spearman": np.nan, "loocv_auc": 0.5, "loocv_balanced_acc": 0.5})
    for m0 in args.m0:
        f = m0.split("+")
        d = load(f)
        settings = [("기본", d, f)]
        if "accident_count" in d:
            settings.append(("사고 3건 이하 구 제외", d[d.accident_count > 3].reset_index(drop=True), f))
        if f"{m0}_aux" in d:
            settings.append(("보조 시차 t-48~t-24", d, [f"{m0}_aux"]))
        if f"{m0}_long" in d:
            settings.append(("입력 창 4년 2020~2023", d, [f"{m0}_long"]))
        for name, dd, ff in settings:
            r, o = evaluate(dd, ff, "M0" if len(f) == 1 else "M-aux", name)
            rows.append(r)
            oofs.append(o)

    res = pd.DataFrame(rows)
    res.to_csv(OUTPUTS / "baseline_results.csv", index=False, encoding="utf-8-sig")
    pd.concat(oofs).to_csv(OUTPUTS / "baseline_oof.csv", index=False, encoding="utf-8-sig")
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
