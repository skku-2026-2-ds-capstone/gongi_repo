import numpy as np
import pandas as pd

from config import (INTERIM, PROCESSED, OUTPUTS, EXTERNAL, SEOUL_GU, WINDOWS, INPUT_CUTOFF,
                    TARGET_START, TARGET_END, RENT_REPORTING_BREAK, AREA_BINS, AREA_LABELS,
                    BUILD_BINS, BUILD_LABELS, MIN_CELL_N, MIN_GU_N, MATCH_AREA_TOL,
                    MATCH_MAX_GAP_DAYS, HOUSING_TYPES, JR_TYPES, dataset_name)

_W_MAIN = "2023.01~12 (타깃 17~29개월 전)"
_W_YOY = "2023 vs 2022 (타깃 17~41개월 전)"
_W_AUX = "2021.07~2023.06 (타깃 23~47개월 전)"
_W_LONG = "2020.01~2023.12 (타깃 17~65개월 전)"
_K_RENT = "예. 전월세 신고기한 30일이라 2024.01 말까지 모두 공개"
_K_TRADE = ("대부분 예. 매매 신고기한 30일이라 2024.01 말까지 공개. 단 해제 표시(R3)는 나중에 등록될 수 있어 "
            "지금 받은 자료가 당시보다 해제를 조금 더 걸러냄")
_K_NONE = "해당 없음 (품질 표시용, 모형 입력 아님)"

FEATURE_DICT = [
    ("jr_cell", "전세가율", "동×면적대×건축연대 셀별 (전세 ㎡당 보증금 중앙값 ÷ 매매 ㎡당 가격 중앙값), 셀 전세건수 가중평균. "
     "셀은 매매·전세 각 5건 이상, 구 전세의 50%를 못 덮으면 동×면적(L2) → 면적×연대(L3)로 거칠게", "비율(0~1)", "매매+전세", _W_MAIN, "+",
     "보증금이 집값에 가까울수록 작은 하락에도 보증금을 못 돌려줌(깡통전세). 근거 강: 김진유 2022, 안선영·이상엽 2025", "M0·M1·M2", _K_TRADE),
    ("jr_match_med", "전세가율", "동일건물(법정동+지번+건물명)·면적±5%·계약일 1년 이내 매매와 짝지은 개별 전세가율의 중앙값",
     "비율(0~1)", "매매+전세", _W_MAIN, "+", "jr_cell과 같은 개념의 다른 측정. 매칭률 9~29%라 대표성 낮음", "민감도", _K_TRADE),
    ("jr_match_gt80", "전세가율", "짝지은 전세 중 개별 전세가율 > 0.8 비중", "비율(0~1)", "매매+전세", _W_MAIN, "+",
     "80%는 학계의 고위험 전세 기준(김기중 외 2023)", "참고", _K_TRADE),
    ("jr_match_gt90", "전세가율", "짝지은 전세 중 개별 전세가율 > 0.9 비중", "비율(0~1)", "매매+전세", _W_MAIN, "+",
     "90%는 2023.5 이후 HUG 보증 가입 상한", "참고", _K_TRADE),
    ("match_rate", "품질", "전세 중 동일건물 매매와 짝지어진 비율 (jr_match_* 의 신뢰도)", "비율(0~1)", "전세", _W_MAIN, "없음",
     "-", "품질", _K_NONE),
    ("jr_simple", "전세가율", "구 전체 전세 ㎡당 보증금 중앙값 ÷ 매매 ㎡당 가격 중앙값 (매매·전세 주택 구성 차이를 통제 못 함)",
     "비율(0~1)", "매매+전세", _W_MAIN, "+", "셀 방식과 비교하기 위한 단순 버전", "참고", _K_TRADE),
    ("sale_ppsqm", "가격수준", "매매 ㎡당 가격 중앙값", "만원/㎡", "매매", _W_MAIN, "-",
     "저가 빌라는 시세가 불투명해 감정가 부풀리기가 쉬움. 근거 중(KDI 문윤상 2023). 프로포절은 '수준보다 변화·비율'을 내세웠으므로 주 모형이 아닌 민감도",
     "민감도", _K_TRADE),
    ("jeonse_ppsqm", "가격수준", "전세 ㎡당 보증금 중앙값", "만원/㎡", "전세", _W_MAIN, "-",
     "sale_ppsqm과 같은 논리(r=0.92로 거의 같은 정보)", "참고", _K_RENT),
    ("sale_ppsqm_yoy", "변화", "매매 ㎡당 가격 중앙값 전년 대비 변화율", "비율(-1~)", "매매", _W_YOY, "-",
     "매매가 하락은 담보 부족으로 이어짐. 근거 중. 단 jr_cell과 r=-0.78이라 같은 모형에 못 넣음", "제외(공선성)", _K_TRADE),
    ("jeonse_ppsqm_yoy", "변화", "전세 ㎡당 보증금 중앙값 전년 대비 변화율", "비율(-1~)", "전세", _W_YOY, "-",
     "전세가 하락 시 새 보증금으로 이전 보증금을 못 돌려줌(역전세). 근거 중~약, 연구 간 혼재(김성찬·선주한 2025)", "M1", _K_RENT),
    ("n_trade_yoy", "변화", "매매 거래량 전년 대비 변화율", "비율(-1~)", "매매", _W_YOY, "?",
     "거래량과 사고의 관계를 직접 검증한 국내 연구 없음", "제외(가설 없음)", _K_TRADE),
    ("n_jeonse_yoy", "변화", "전세 거래량 전년 대비 변화율 (2021.6 공개범위 단절을 넘으면 NaN)", "비율(-1~)", "전세", _W_YOY, "?",
     "거래량과 사고의 관계를 직접 검증한 국내 연구 없음", "제외(가설 없음)", _K_RENT),
    ("jr_cell_chg", "변화", "jr_cell(2023) − jr_cell(2022)", "비율 차(%p/100)", "매매+전세", _W_YOY, "+",
     "전세가율이 빠르게 오르는 지역은 담보 여력이 빠르게 줄어듦. 직접 검증 연구 없는 추론 가설(근거 약)", "M1", _K_TRADE),
    ("wolse_share", "구조", "월세 ÷ (전세+월세) 계약 수", "비율(0~1)", "전월세", _W_MAIN, "?",
     "계약 단위로는 보증부월세가 사고 위험을 낮추지만 구 단위 근거 없음", "제외(가설 없음)", _K_RENT),
    ("renewal_share", "구조", "전세 중 계약구분='갱신' 비중 (계약구분 기재분 기준)", "비율(0~1)", "전세", _W_MAIN, "?",
     "보증사고와 직접 연결한 연구 없음", "제외(가설 없음)", _K_RENT),
    ("build_year_med", "노후·신축", "전세 계약 건물의 건축년도 중앙값", "연도", "전세", _W_MAIN, "+",
     "new_share와 r=0.95로 같은 정보", "참고", _K_RENT),
    ("old_share", "노후·신축", "전세 계약 중 2000년 이전 준공 비중", "비율(0~1)", "전세", _W_MAIN, "?",
     "시세가 쌓여 덜 위험(-)하다는 논리와 저가라 위험(+)하다는 논리가 맞섬. 직접 근거 없음", "민감도", _K_RENT),
    ("new_share", "노후·신축", "전세 계약 중 계약연도−준공연도 ≤ 5년 비중 (신축빌라)", "비율(0~1)", "전세", _W_MAIN, "+",
     "신축 빌라 동시진행·감정가 부풀리기가 전세사기 핵심 수법. 근거 중(권혁·성현곤 2025)", "M2", _K_RENT),
    ("jr_cell_aux", "전세가율", "jr_cell과 같은 정의, 보조 시차 창", "비율(0~1)", "매매+전세", _W_AUX, "+",
     "계약 후 약 2년 뒤 만기에 사고가 난다는 시차 구조 확인용", "민감도", _K_TRADE),
    ("n_trade", "표본", "창 안 매매 건수", "건", "매매", _W_MAIN, "없음", "-", "품질", _K_NONE),
    ("n_jeonse", "표본", "창 안 전세 건수", "건", "전세", _W_MAIN, "없음", "-", "품질", _K_NONE),
    ("low_n_flag", "표본", f"매매 또는 전세 건수 < {MIN_GU_N} 이면 1 (해당 구 피처 불안정)", "0/1", "-", _W_MAIN, "없음", "-", "품질", _K_NONE),
    ("jr_cell_apt", "전세가율", "jr_cell과 같은 정의를 아파트 매매·전세에 적용 (동은 같은 단지가 섞이도록 그대로 사용)",
     "비율(0~1)", "아파트 매매+전세", _W_MAIN, "+", "아파트 전세가율. 아파트는 시세가 투명해 사고 경로가 빌라와 다를 수 있음", "M0 대안", _K_TRADE),
    ("jr_cell_offi", "전세가율", "jr_cell과 같은 정의를 오피스텔 매매·전세에 적용", "비율(0~1)", "오피스텔 매매+전세", _W_MAIN, "+",
     "오피스텔은 빌라와 함께 전세사기 집중 유형", "M0 대안", _K_TRADE),
    ("jr_cell_all", "전세가율", "유형별 jr_cell(연립다세대·아파트·오피스텔)을 각 유형 전세 건수로 가중평균. 단독다가구 제외",
     "비율(0~1)", "3개 유형 매매+전세", _W_MAIN, "+", "HUG 타깃이 유형 구분 없는 구 전체 사고율이므로 범위를 맞춘 전세가율", "M0 대안", _K_TRADE),
    ("jr_cell_all_chg", "변화", "jr_cell_all(2023) − jr_cell_all(2022)", "비율 차(%p/100)", "3개 유형 매매+전세", _W_YOY, "+",
     "jr_cell_chg의 전 유형 버전", "참고", _K_TRADE),
    ("jr_cell_all_aux", "전세가율", "jr_cell_all과 같은 정의, 보조 시차 창", "비율(0~1)", "3개 유형 매매+전세", _W_AUX, "+",
     "jr_cell_aux의 전 유형 버전", "민감도", _K_TRADE),
    ("jr_cell_long", "전세가율", "jr_cell과 같은 정의, 4년 창. 2021.6 이전 전월세는 확정일자 기반이라 표본이 얇음", "비율(0~1)",
     "매매+전세", _W_LONG, "+", "1년 창의 표본 부족·연도 특수성을 줄였을 때도 같은 결론인지 확인 (_apt, _offi, _all 도 같은 창)",
     "민감도", _K_TRADE),
    ("jr_cell_prop", "전세가율", "jr_cell과 같은 정의, 2021.07~2023.12 창 (new_share_prop 도 같은 창)", "비율(0~1)",
     "매매+전세", "2021.07~2023.12 (타깃 17~47개월 전)", "+",
     "프로포절 RQ2 판정 규칙의 민감도 2 '입력 창 2021H2~2023H2로 확장'", "판정 민감도", _K_TRADE),
    ("youth_share", "인구", "주민등록 인구 중 20~39세 비율 (주 입력 창 마지막 달 말, 2023.12)", "비율(0~1)",
     "행정안전부 법정동별 연령별 주민등록 인구", "2023.12 (타깃 18개월 전)", "+",
     "전세사기 피해자 76%가 20~30대(w7). 청년이 많은 구일수록 피해 노출이 크다는 가설. SC3 교수 제안", "M-aux",
     "예. 주민등록 인구는 매월 공개"),
    ("n_jeonse_apt", "표본", "창 안 아파트 전세 건수 (_offi, _sh 도 같은 정의)", "건", "전세", _W_MAIN, "없음", "-", "품질", _K_NONE),
    ("jeonse_share_rh", "구조", "전세 계약 중 연립다세대 비중 (분모: 4개 유형 전세 합. _apt, _offi, _sh 도 같은 정의)",
     "비율(0~1)", "4개 유형 전세", _W_MAIN, "+", "빌라 전세 비중이 높은 구일수록 사고 노출이 큼(사고의 대부분이 빌라·오피스텔)", "참고", _K_RENT),
]
DICT_COLUMNS = ["feature", "group", "definition", "unit", "source", "input_window", "hypothesis_sign",
                "rationale", "model_role", "known_at_prediction"]


def _read(name):
    d = pd.read_csv(INTERIM / f"{name}_clean.csv", parse_dates=["contract_date"],
                    dtype={"lawd_cd": str}, low_memory=False)
    d["area_bin"] = pd.cut(d["area"], AREA_BINS, labels=AREA_LABELS, right=False)
    d["build_bin"] = pd.cut(d["build_year"], BUILD_BINS, labels=BUILD_LABELS, right=False)
    return d


def load(htype="rh"):
    return _read(dataset_name(htype, "trade")), _read(dataset_name(htype, "rent"))


def load_other_types():
    out = {}
    for h in HOUSING_TYPES:
        if h == "rh" or not (INTERIM / f"{dataset_name(h, 'rent')}_clean.csv").exists():
            continue
        has_trade = (INTERIM / f"{dataset_name(h, 'trade')}_clean.csv").exists()
        out[h] = (_read(dataset_name(h, "trade")) if has_trade else None, _read(dataset_name(h, "rent")))
    return out


def window_slice(df, start, end):
    s = df[(df["contract_date"] >= start) & (df["contract_date"] <= end)]
    assert s.empty or s["contract_date"].max() <= end, "창 밖 계약 혼입"
    return s


CELL_LEVELS = [
    ("L1_동×면적×연대", ["dong", "area_bin", "build_bin"]),
    ("L2_동×면적", ["dong", "area_bin"]),
    ("L3_면적×연대", ["area_bin", "build_bin"]),
]
MIN_COVERAGE = 0.5


def _cell_ratio(t, j, keys):
    k = ["gu"] + keys
    a = t.groupby(k, observed=True)["ppsqm"].agg(sale_med="median", n_sale="size")
    b = j.groupby(k, observed=True)["ppsqm"].agg(jeon_med="median", n_jeon="size")
    c = a.join(b, how="inner")
    c = c[(c.n_sale >= MIN_CELL_N) & (c.n_jeon >= MIN_CELL_N)]
    if c.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    c = c.assign(w=c.jeon_med / c.sale_med * c.n_jeon)
    g = c.groupby("gu")
    return g.w.sum() / g.n_jeon.sum(), g.n_jeon.sum() / j.groupby("gu").size()


def jr_cell(t, r):
    j = r[r["is_jeonse"]]
    out = pd.DataFrame(index=sorted(j["gu"].unique()), columns=["jr_cell", "cell_coverage", "cell_level"])
    for name, keys in CELL_LEVELS:
        ratio, cover = _cell_ratio(t, j, keys)
        todo = out.index[out["jr_cell"].isna()]
        ok = [g for g in todo if g in cover.index and cover[g] >= MIN_COVERAGE]
        out.loc[ok, "jr_cell"] = ratio[ok]
        out.loc[ok, "cell_coverage"] = cover[ok]
        out.loc[ok, "cell_level"] = name
    return out.astype({"jr_cell": float, "cell_coverage": float})


def jr_match(t, r):
    j = r[r["is_jeonse"]][["gu", "bldg_key", "area", "contract_date", "ppsqm"]].reset_index(names="jid")
    s = t[["bldg_key", "area", "contract_date", "ppsqm"]]
    m = j.merge(s, on="bldg_key", suffixes=("", "_s"))
    m = m[((m.area_s - m.area).abs() / m.area <= MATCH_AREA_TOL) &
          ((m.contract_date_s - m.contract_date).dt.days.abs() <= MATCH_MAX_GAP_DAYS)]
    m["gap"] = (m.contract_date_s - m.contract_date).dt.days.abs()
    m = m.sort_values("gap").drop_duplicates("jid")
    m["jr"] = m.ppsqm / m.ppsqm_s
    g = m.groupby("gu")["jr"]
    res = pd.DataFrame({"jr_match_med": g.median(),
                        "jr_match_gt80": g.apply(lambda x: (x > 0.8).mean()),
                        "jr_match_gt90": g.apply(lambda x: (x > 0.9).mean()),
                        "n_matched": g.size()})
    res["match_rate"] = res.n_matched / j.groupby("gu").size()
    return res


def basic(t, r):
    j, w = r[r["is_jeonse"]], r[~r["is_jeonse"]]
    out = pd.DataFrame({
        "n_trade": t.groupby("gu").size(),
        "n_jeonse": j.groupby("gu").size(),
        "n_wolse": w.groupby("gu").size(),
        "sale_ppsqm": t.groupby("gu")["ppsqm"].median(),
        "jeonse_ppsqm": j.groupby("gu")["ppsqm"].median(),
    })
    out["jr_simple"] = out.jeonse_ppsqm / out.sale_ppsqm
    out["wolse_share"] = out.n_wolse / (out.n_jeonse + out.n_wolse)
    known = j[j["contract_kind"].isin(["신규", "갱신"])]
    out["renewal_share"] = known.groupby("gu")["contract_kind"].apply(lambda s: (s == "갱신").mean())
    age = j["contract_date"].dt.year - j["build_year"]
    out["build_year_med"] = j.groupby("gu")["build_year"].median()
    out["old_share"] = (j["build_year"] < 2000).where(j["build_year"].notna()).groupby(j["gu"]).mean()
    out["new_share"] = (age <= 5).where(age.notna()).groupby(j["gu"]).mean()
    return out


def window_features(t, r, start, end):
    ts, rs = window_slice(t, start, end), window_slice(r, start, end)
    f = basic(ts, rs)
    f = f.join(jr_cell(ts, rs)).join(jr_match(ts, rs))
    f["low_n_flag"] = ((f.n_trade < MIN_GU_N) | (f.n_jeonse < MIN_GU_N)).astype(int)
    return f.reindex(sorted(SEOUL_GU.values()))


def cross_section(t, r):
    (ms, me), (ps, pe), (as_, ae) = WINDOWS["main"], WINDOWS["prev"], WINDOWS["aux"]
    main = window_features(t, r, ms, me)
    prev = window_features(t, r, ps, pe)
    aux = window_features(t, r, as_, ae)
    long = window_features(t, r, *WINDOWS["long"])
    prop = window_features(t, r, *WINDOWS["prop"])
    f = main.copy()
    for c, base in [("sale_ppsqm_yoy", "sale_ppsqm"), ("jeonse_ppsqm_yoy", "jeonse_ppsqm"),
                    ("n_trade_yoy", "n_trade"), ("n_jeonse_yoy", "n_jeonse")]:
        f[c] = main[base] / prev[base] - 1
    if ps < RENT_REPORTING_BREAK <= me:
        f["n_jeonse_yoy"] = np.nan
    f["jr_cell_chg"] = main["jr_cell"] - prev["jr_cell"]
    for c in ["jr_cell", "jr_match_med", "jr_match_gt90"]:
        f[f"{c}_aux"] = aux[c]
        f[f"{c}_long"] = long[c]
    for c in ["jr_cell", "new_share"]:
        f[f"{c}_prop"] = prop[c]
    f["new_share_long"] = long["new_share"]
    f.index.name = "gu"
    f["window_start"], f["window_end"] = ms.date(), me.date()
    f["input_cutoff"] = INPUT_CUTOFF.date()
    return f.reset_index()


def _type_window(t, r, start, end, h):
    rs = window_slice(r, start, end)
    f = pd.DataFrame({f"n_jeonse_{h}": rs[rs["is_jeonse"]].groupby("gu").size()})
    if h in JR_TYPES and t is not None:
        jr = jr_cell(window_slice(t, start, end), rs)
        f = f.join(jr[["jr_cell", "cell_coverage"]].add_suffix(f"_{h}"), how="outer")
    return f.reindex(sorted(SEOUL_GU.values()))


def type_cross_section(r_rh, rh_jr, others):
    f = pd.DataFrame(index=sorted(SEOUL_GU.values()))
    for wname, suffix in [("main", ""), ("prev", "_prev"), ("aux", "_aux"), ("long", "_long")]:
        start, end = WINDOWS[wname]
        rr = window_slice(r_rh, start, end)
        w = pd.DataFrame({"n_jeonse_rh": rr[rr["is_jeonse"]].groupby("gu").size(),
                          "jr_cell_rh": rh_jr[wname]}).reindex(f.index)
        for h, (t, r) in others.items():
            w = w.join(_type_window(t, r, start, end, h))
        jr_types = [h for h in JR_TYPES if f"jr_cell_{h}" in w]
        jr = w[[f"jr_cell_{h}" for h in jr_types]].to_numpy()
        n = w[[f"n_jeonse_{h}" for h in jr_types]].fillna(0).to_numpy()
        n = np.where(np.isnan(jr), 0, n)
        w["jr_cell_all"] = np.nansum(jr * n, axis=1) / np.where(n.sum(axis=1) > 0, n.sum(axis=1), np.nan)
        if wname == "main":
            types = [h for h in HOUSING_TYPES if f"n_jeonse_{h}" in w]
            tot = w[[f"n_jeonse_{h}" for h in types]].fillna(0).sum(axis=1)
            for h in types:
                w[f"jeonse_share_{h}"] = w[f"n_jeonse_{h}"].fillna(0) / tot
            f = f.join(w.drop(columns=["n_jeonse_rh", "jr_cell_rh"]))
        else:
            for h in ["all"] + [h for h in jr_types if h != "rh"]:
                f[f"jr_cell_{h}{suffix}"] = w[f"jr_cell_{h}"]
    f["jr_cell_all_chg"] = f["jr_cell_all"] - f["jr_cell_all_prev"]
    f.index.name = "gu"
    return f.reset_index()


def halfyear_panel(t, r):
    t, r = t.copy(), r.copy()
    for d in (t, r):
        d["half"] = d["contract_date"].dt.year.astype(str) + "H" + np.where(d["contract_date"].dt.month <= 6, "1", "2")
    rows = []
    for h in sorted(set(t["half"]) | set(r["half"])):
        y, k = int(h[:4]), h[-1]
        start = pd.Timestamp(f"{y}-01-01") if k == "1" else pd.Timestamp(f"{y}-07-01")
        end = pd.Timestamp(f"{y}-06-30") if k == "1" else pd.Timestamp(f"{y}-12-31")
        f = window_features(t, r, start, end)
        f["half"] = h
        if end <= INPUT_CUTOFF:
            role = "input_eligible"
        elif end < TARGET_START:
            role = "gap"
        elif start <= TARGET_END:
            role = "target_period"
        else:
            role = "post_target"
        f["period_role"] = role
        rows.append(f.reset_index(names="gu"))
    return pd.concat(rows, ignore_index=True)


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    t, r = load()
    cs = cross_section(t, r)
    others = load_other_types()
    if others:
        rh = cs.set_index("gu")
        rh_jr = {"main": rh.jr_cell, "prev": rh.jr_cell - rh.jr_cell_chg, "aux": rh.jr_cell_aux,
                 "long": rh.jr_cell_long}
        cs = cs.merge(type_cross_section(r, rh_jr, others), on="gu", how="left", validate="one_to_one")
        print(f"유형 확장: {', '.join(HOUSING_TYPES[h] for h in others)}")
    pop_path = EXTERNAL / "population.csv"
    if pop_path.exists():
        ym = int(WINDOWS["main"][1].strftime("%Y%m"))
        pop = pd.read_csv(pop_path)
        cs = cs.merge(pop.loc[pop.ym == ym, ["gu", "youth_share"]], on="gu", how="left", validate="one_to_one")
    assert pd.to_datetime(cs["window_end"]).max() < TARGET_START
    cs.to_csv(PROCESSED / "gu_features.csv", index=False, encoding="utf-8-sig")
    panel = halfyear_panel(t, r)
    panel.to_csv(PROCESSED / "gu_halfyear_panel.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(FEATURE_DICT, columns=DICT_COLUMNS) \
        .to_csv(OUTPUTS / "feature_dictionary.csv", index=False, encoding="utf-8-sig")
    show = ["gu", "n_trade", "n_jeonse", "jr_cell", "cell_coverage", "cell_level", "jr_match_med", "match_rate", "low_n_flag"]
    print(cs[show].round(3).to_string(index=False))
    print(f"\n패널: {panel['half'].nunique()}개 반기, period_role 분포\n{panel.groupby('period_role').half.nunique()}")


if __name__ == "__main__":
    main()
