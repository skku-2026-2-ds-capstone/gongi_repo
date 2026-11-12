import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scipy.stats import spearmanr

from baseline import load
from config import OUTPUTS, PROCESSED, HIGH_RISK_THRESHOLD
from models import loocv

SEED = 42
CUTOFFS = [1.0, 1.2, 1.3, 1.4, 1.5, 1.8, 2.1, 2.7]
SPECS = [
    ("M0 jr_cell 1년(주)", ["jr_cell"]),
    ("M0 jr_cell 4년", ["jr_cell_long"]),
    ("M0 jr_cell_all 1년", ["jr_cell_all"]),
    ("M1 1년", ["jr_cell", "jeonse_ppsqm_yoy", "jr_cell_chg"]),
]


def aucs(X, y_rate, cut):
    y_cls = (y_rate >= cut).astype(int)
    p_reg, p_cls = loocv("linear", X, y_rate, y_cls)
    return roc_auc_score(y_cls, p_cls), roc_auc_score(y_cls, p_reg), p_reg


def drop_one(d, y):
    rows = []
    for name, feats in SPECS:
        X = d[feats].to_numpy(float)
        for i, gu in [(None, "(전체)")] + list(enumerate(d.gu)):
            keep = np.ones(len(y), bool) if i is None else np.arange(len(y)) != i
            yc = (y[keep] >= HIGH_RISK_THRESHOLD).astype(int)
            p_reg, p_cls = loocv("linear", X[keep], y[keep], yc)
            rows.append({"model": name, "dropped": gu, "n": int(keep.sum()), "n_high": int(yc.sum()),
                         "loocv_spearman": spearmanr(p_reg, y[keep]).statistic,
                         "loocv_auc": roc_auc_score(yc, p_cls)})
    return pd.DataFrame(rows)


def rq2_judgment(y_all, d):
    ladders = {
        "기본 (1.8%, 2023)": (HIGH_RISK_THRESHOLD, "jr_cell", "new_share", True),
        "민감도 1: 기준 중앙값 1.0%": (1.0, "jr_cell", "new_share", True),
        "민감도 2: 입력 창 2021H2~2023H2": (HIGH_RISK_THRESHOLD, "jr_cell_prop", "new_share_prop", True),
        "참고: 입력 창 4년 2020~2023": (HIGH_RISK_THRESHOLD, "jr_cell_long", "new_share_long", False),
    }
    rows = []
    for cond, (cut, jr, new, counts) in ladders.items():
        sets = {"M0": [jr], "M1": [jr, "jeonse_ppsqm_yoy", "jr_cell_chg"], "M2": [jr, new]}
        if "youth_share" in d:
            sets["M-aux"] = [jr, "youth_share"]
        y_cls = (y_all >= cut).astype(int)
        score = {}
        for m, feats in sets.items():
            p_reg, p_cls = loocv("linear", d[feats].to_numpy(float), y_all, y_cls)
            score[m] = (spearmanr(p_reg, y_all).statistic, roc_auc_score(y_cls, p_cls))
        for m in [k for k in sets if k != "M0"]:
            rows.append({"condition": cond, "in_verdict": counts, "cutoff": cut, "n_high": int(y_cls.sum()), "model": m,
                         "features": "+".join(sets[m]),
                         "rho_m0": score["M0"][0], "rho": score[m][0], "rho_higher": score[m][0] > score["M0"][0],
                         "auc_m0": score["M0"][1], "auc": score[m][1], "auc_higher": score[m][1] > score["M0"][1]})
    res = pd.DataFrame(rows)
    res["both_higher"] = res.rho_higher & res.auc_higher
    verdict = res[res.in_verdict].groupby("model").both_higher.all().map({True: "RQ2 긍정", False: "RQ2 긍정 아님"})
    return res, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--rq2-only", action="store_true", help="프로포절 RQ2 판정 규칙만 계산 (순열 없음, 수 초)")
    args = ap.parse_args()

    if args.rq2_only:
        feats = ["jr_cell", "jr_cell_prop", "jr_cell_long", "new_share", "new_share_prop", "new_share_long",
                 "jeonse_ppsqm_yoy", "jr_cell_chg"]
        if "youth_share" in pd.read_csv(PROCESSED / "gu_features.csv", nrows=0).columns:
            feats.append("youth_share")
        d = load(feats)
        res, verdict = rq2_judgment(d["accident_rate"].to_numpy(float), d)
        res.to_csv(OUTPUTS / "rq2_judgment.csv", index=False, encoding="utf-8-sig")
        print(res.drop(columns=["features", "cutoff"]).round(3).to_string(index=False))
        print("\n판정\n" + verdict.to_string())
        return

    d = load(sorted({f for _, fs in SPECS for f in fs}))
    y = d["accident_rate"].to_numpy(float)
    rng = np.random.default_rng(SEED)
    rows, rank = [], d[["gu", "accident_rate", "accident_count"]].copy()
    rank["rank_actual"] = rank["accident_rate"].rank(ascending=False, method="min").astype(int)

    for name, feats in SPECS:
        X = d[feats].to_numpy(float)
        for cut in CUTOFFS:
            auc_clf, auc_rank, p_reg = aucs(X, y, cut)
            null = []
            for _ in range(args.n_perm):
                idx = rng.permutation(len(y))
                a, b, _ = aucs(X, y[idx], cut)
                null.append((a, b))
            null = np.array(null)
            rows.append({"model": name, "features": "+".join(feats), "cutoff": cut,
                         "n_high": int((y >= cut).sum()), "preregistered": cut == HIGH_RISK_THRESHOLD,
                         "auc_clf": auc_clf, "p_perm_auc_clf": (np.sum(null[:, 0] >= auc_clf) + 1) / (args.n_perm + 1),
                         "auc_rank": auc_rank, "p_perm_auc_rank": (np.sum(null[:, 1] >= auc_rank) + 1) / (args.n_perm + 1)})
            if cut == HIGH_RISK_THRESHOLD:
                rank[f"pred_rank | {name}"] = pd.Series(-p_reg).rank(method="min").astype(int).to_numpy()
        print(f"{name} 완료")

    drop = drop_one(d, y)
    drop.to_csv(OUTPUTS / "sensitivity_drop_one.csv", index=False, encoding="utf-8-sig")
    res = pd.DataFrame(rows)
    res.to_csv(OUTPUTS / "sensitivity_cutoff.csv", index=False, encoding="utf-8-sig")
    rank.sort_values("rank_actual").to_csv(OUTPUTS / "district_ranking.csv", index=False, encoding="utf-8-sig")
    wide = res.pivot(index=["cutoff", "n_high"], columns="model", values="auc_clf").round(3)
    print("\nAUC (로지스틱 재적합) — 행: 기준선(%) / 고위험 구 수\n" + wide.to_string())
    wide = res.pivot(index=["cutoff", "n_high"], columns="model", values="auc_rank").round(3)
    print("\nAUC (회귀 예측 순위)\n" + wide.to_string())
    print("\n" + rank.sort_values("rank_actual").to_string(index=False))
    full = drop[drop.dropped == "(전체)"].set_index("model")
    part = drop[drop.dropped != "(전체)"]
    worst = lambda col: part.loc[part.groupby("model")[col].idxmin()].set_index("model")
    kangseo = part[part.dropped == "강서구"].set_index("model")
    summ = pd.DataFrame({"rho_full": full.loocv_spearman, "rho_min": worst("loocv_spearman").loocv_spearman,
                         "rho_min_drop": worst("loocv_spearman").dropped, "rho_drop_강서": kangseo.loocv_spearman,
                         "auc_full": full.loocv_auc, "auc_min": worst("loocv_auc").loocv_auc,
                         "auc_min_drop": worst("loocv_auc").dropped, "auc_drop_강서": kangseo.loocv_auc})
    print("\n구 하나씩 빼기\n" + summ.round(3).to_string())


if __name__ == "__main__":
    main()
