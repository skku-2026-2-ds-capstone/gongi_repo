import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import EXTERNAL, OUTPUTS, PROCESSED, HIGH_RISK_THRESHOLD, SEOUL_GU
from features import load, window_features

TRAIN = {"main": ("2023-01-01", "2023-12-31"), "prev": ("2022-01-01", "2022-12-31")}
TEST = {"main": ("2024-01-01", "2024-12-31"), "prev": ("2023-01-01", "2023-12-31")}
TEST_TARGET_START = pd.Timestamp("2026-06-01")
SETS = {
    "M0": ["jr_cell"],
    "M1": ["jr_cell", "jeonse_ppsqm_yoy", "jr_cell_chg"],
    "M2": ["jr_cell", "new_share"],
    "M-aux": ["jr_cell", "youth_share"],
}
N_PERM = 10_000
RNG = np.random.default_rng(42)


def ladder_features(t, r, win):
    (ms, me), (ps, pe) = [tuple(map(pd.Timestamp, win[k])) for k in ("main", "prev")]
    main, prev = window_features(t, r, ms, me), window_features(t, r, ps, pe)
    f = pd.DataFrame({
        "jr_cell": main["jr_cell"],
        "jeonse_ppsqm_yoy": main["jeonse_ppsqm"] / prev["jeonse_ppsqm"] - 1,
        "jr_cell_chg": main["jr_cell"] - prev["jr_cell"],
        "new_share": main["new_share"],
        "n_jeonse": main["n_jeonse"],
    })
    f.index.name = "gu"
    return f


def perm_p(stat_fn, y, observed):
    null = np.array([stat_fn(RNG.permutation(y)) for _ in range(N_PERM)])
    return (np.sum(null >= observed) + 1) / (N_PERM + 1)


def evaluate(name, pred_rank_score, pred_cls_score, y26, hi26, n_top=6):
    rho = spearmanr(pred_rank_score, y26).statistic
    auc = roc_auc_score(hi26, pred_cls_score)
    cut = np.quantile(y26, 0.75)
    cut6 = np.sort(y26)[::-1][n_top - 1]
    auc6 = roc_auc_score(y26 >= cut6, pred_cls_score)
    return {"model": name, "spearman_2026": rho,
            "p_perm_spearman": perm_p(lambda yy: spearmanr(pred_rank_score, yy).statistic, y26, rho),
            "auc_2026_top_quartile": auc,
            "p_perm_auc": perm_p(lambda yy: roc_auc_score(yy >= cut, pred_cls_score), y26, auc),
            "auc_2026_top6": auc6,
            "p_perm_auc_top6": perm_p(lambda yy: roc_auc_score(yy >= cut6, pred_cls_score), y26, auc6)}


def main():
    t, r = load()
    assert t["contract_date"].max() >= pd.Timestamp("2024-12-01"), "2024 연립·다세대 자료가 없음 (collect_molit.py로 수집)"
    X23, X24 = ladder_features(t, r, TRAIN), ladder_features(t, r, TEST)
    ref = pd.read_csv(PROCESSED / "gu_features.csv").set_index("gu")
    for c in ["jr_cell", "jeonse_ppsqm_yoy", "jr_cell_chg", "new_share"]:
        assert np.allclose(X23[c], ref.loc[X23.index, c]), f"{c}: features.py와 다름"
    assert pd.Timestamp(TEST["main"][1]) < TEST_TARGET_START
    pop = pd.read_csv(EXTERNAL / "population.csv").set_index(["ym", "gu"])["youth_share"]
    X23["youth_share"], X24["youth_share"] = pop.loc[202312], pop.loc[202412]

    y25 = pd.read_csv(EXTERNAL / "hug_target.csv").set_index("gu")
    y26 = pd.read_csv(EXTERNAL / "hug_target_2026.csv").set_index("gu")
    gu = sorted(SEOUL_GU.values())
    y25r, y26r = y25.loc[gu, "accident_rate"].to_numpy(float), y26.loc[gu, "accident_rate"].to_numpy(float)
    hi25 = (y25r >= HIGH_RISK_THRESHOLD).astype(int)
    hi26 = (y26r >= np.quantile(y26r, 0.75)).astype(int)
    print(f"2026 상위 사분위 기준 {np.quantile(y26r, 0.75):.2f}% → 고위험 {hi26.sum()}개 구, "
          f"상위 6개 기준 {np.sort(y26r)[::-1][5]:.2f}% → {(y26r >= np.sort(y26r)[::-1][5]).sum()}개 구")

    rows, preds = [], pd.DataFrame({"gu": gu, "rate_2025": y25r, "rate_2026": y26r, "high_2026": hi26})
    rows.append(evaluate("기준값: 2025 사고율 그대로", y25r, y25r, y26r, hi26))
    rows.append(evaluate("기준값: 2024 전세가율 원값", X24.loc[gu, "jr_cell"].to_numpy(), X24.loc[gu, "jr_cell"].to_numpy(), y26r, hi26))
    for m, feats in SETS.items():
        Xtr, Xte = X23.loc[gu, feats].to_numpy(float), X24.loc[gu, feats].to_numpy(float)
        reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xtr, np.log(y25r))
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0)).fit(Xtr, hi25)
        p_reg, p_cls = reg.predict(Xte), clf.predict_proba(Xte)[:, 1]
        rows.append(evaluate(f"{m} ({'+'.join(feats)})", p_reg, p_cls, y26r, hi26))
        preds[f"pred_rank_{m}"] = pd.Series(-p_reg).rank(method="min").astype(int).to_numpy()

    res = pd.DataFrame(rows)
    res.to_csv(OUTPUTS / "temporal_validation.csv", index=False, encoding="utf-8-sig")
    preds["rank_2026"] = pd.Series(-y26r).rank(method="min").astype(int).to_numpy()
    preds.sort_values("rank_2026").to_csv(OUTPUTS / "temporal_predictions.csv", index=False, encoding="utf-8-sig")

    chg = pd.DataFrame({
        "gu": gu,
        "rate_2025": y25r, "rate_2026": y26r, "rate_change": y26r - y25r,
        "count_2025": y25.loc[gu, "accident_count"].to_numpy(), "count_2026": y26.loc[gu, "accident_count"].to_numpy(),
        "jr_cell_2023": X23.loc[gu, "jr_cell"].to_numpy(), "jr_cell_2024": X24.loc[gu, "jr_cell"].to_numpy(),
        "n_jeonse_2023": X23.loc[gu, "n_jeonse"].to_numpy(), "n_jeonse_2024": X24.loc[gu, "n_jeonse"].to_numpy(),
    })
    chg["jr_change"] = chg.jr_cell_2024 - chg.jr_cell_2023
    chg["jeonse_count_change"] = chg.n_jeonse_2024 / chg.n_jeonse_2023 - 1
    chg.sort_values("rate_change").to_csv(OUTPUTS / "accident_rate_change.csv", index=False, encoding="utf-8-sig")

    print(res.round(3).to_string(index=False))
    print(f"\n사고율 변화와 전세가율 변화의 순위 상관 ρ = {spearmanr(chg.rate_change, chg.jr_change).statistic:+.3f}")
    print(chg.sort_values("rate_change").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
