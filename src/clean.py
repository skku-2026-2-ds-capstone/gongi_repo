import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import (RAW, INTERIM, OUTPUTS, SEOUL_GU, GU_TO_CODE, AREA_MIN, AREA_MAX, MAD_K,
                    HOUSING_TYPES, dataset_name)

_API_COMMON = {"sggCd": "lawd_cd", "umdNm": "dong", "jibun": "jibun", "mhouseNm": "bldg",
               "aptNm": "bldg", "offiNm": "bldg", "excluUseAr": "area", "totalFloorAr": "area",
               "floor": "floor", "buildYear": "build_year", "houseType": "house_type"}
API_TRADE = {**_API_COMMON, "landAr": "land_area", "plottageAr": "land_area",
             "dealAmount": "price", "cdealDay": "cancel_date", "dealingGbn": "deal_type"}
API_RENT = {**_API_COMMON, "deposit": "deposit", "monthlyRent": "monthly_rent", "contractTerm": "term_raw",
            "contractType": "contract_kind", "useRRRight": "rr_used",
            "preDeposit": "pre_deposit", "preMonthlyRent": "pre_rent"}
CSV_COMMON = {"번지": "jibun", "건물명": "bldg", "층": "floor", "건축년도": "build_year",
              "전용면적(㎡)": "area", "대지권면적(㎡)": "land_area"}
CSV_TRADE = {**CSV_COMMON, "거래금액(만원)": "price", "해제사유발생일": "cancel_date",
             "거래유형": "deal_type"}
CSV_RENT = {**CSV_COMMON, "보증금(만원)": "deposit", "월세금(만원)": "monthly_rent",
            "계약기간": "term_raw", "계약구분": "contract_kind", "갱신요구권 사용": "rr_used",
            "종전계약 보증금(만원)": "pre_deposit", "종전계약 월세(만원)": "pre_rent"}


def to_num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip()
                         .replace({"": np.nan, "-": np.nan, "nan": np.nan}), errors="coerce")


def parse_term_months(s: pd.Series) -> pd.Series:
    def one(x):
        if not isinstance(x, str) or "~" not in x:
            return np.nan
        a, b = [re.sub(r"\D", "", t) for t in x.split("~")]
        try:
            if len(a) == 4:
                a, b = "20" + a, "20" + b
            ya, ma, yb, mb = int(a[:4]), int(a[4:6]), int(b[:4]), int(b[4:6])
            return (yb - ya) * 12 + (mb - ma)
        except ValueError:
            return np.nan
    return s.map(one)


def read_api_dir(name: str, kind: str) -> pd.DataFrame:
    files = sorted((RAW / name).glob("*.csv"))
    frames = [pd.read_csv(f, dtype=str) for f in files if f.stat().st_size > 10]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    mapping = API_TRADE if kind == "trade" else API_RENT
    df["contract_date"] = pd.to_datetime(
        df["dealYear"].str.zfill(4) + "-" + df["dealMonth"].str.zfill(2) + "-" + df["dealDay"].str.zfill(2),
        errors="coerce")
    df = df.rename(columns=mapping)
    df["source"] = "api"
    return df


def read_csv_export(path: Path, kind: str) -> pd.DataFrame:
    for enc in ("cp949", "utf-8-sig"):
        try:
            lines = path.read_text(encoding=enc).splitlines()
            break
        except UnicodeDecodeError:
            continue
    header = next(i for i, l in enumerate(lines) if l.replace('"', "").startswith(("NO,", "시군구")))
    df = pd.read_csv(path, skiprows=header, encoding=enc, dtype=str)
    parts = df["시군구"].str.split()
    df["gu"], df["dong"] = parts.str[1], parts.str[2:].str.join(" ")
    df["lawd_cd"] = df["gu"].map(GU_TO_CODE)
    df["contract_date"] = pd.to_datetime(df["계약년월"].str.strip() + df["계약일"].str.strip().str.zfill(2),
                                         format="%Y%m%d", errors="coerce")
    if kind == "rent" and "전월세구분" in df:
        df["monthly_rent"] = df["월세금(만원)"]
    df = df.rename(columns=CSV_TRADE if kind == "trade" else CSV_RENT)
    df["source"] = f"csv:{path.name}"
    return df


def standardize(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    df = df.copy()
    df["lawd_cd"] = df["lawd_cd"].astype(str).str[:5]
    df["gu"] = df["lawd_cd"].map(SEOUL_GU)
    for c in ["area", "land_area", "floor", "build_year", "price", "deposit",
              "monthly_rent", "pre_deposit", "pre_rent"]:
        if c in df:
            df[c] = to_num(df[c])
    if kind == "trade":
        df["cancel_date"] = df.get("cancel_date", pd.Series(index=df.index, dtype=str))
        df["cancelled"] = df["cancel_date"].fillna("").astype(str).str.strip().replace("nan", "") != ""
    else:
        df["term_months"] = parse_term_months(df.get("term_raw", pd.Series(index=df.index, dtype=str)))
        df["monthly_rent"] = df["monthly_rent"].fillna(0)
        df["is_jeonse"] = df["monthly_rent"] == 0
        df["contract_kind"] = df.get("contract_kind", pd.Series(index=df.index)).fillna("미기재")
    for c in ["dong", "jibun", "bldg"]:
        if c not in df:
            df[c] = ""
    df["bldg_key"] = (df["lawd_cd"] + "|" + df["dong"].fillna("") + "|" +
                      df["jibun"].fillna("").astype(str) + "|" + df["bldg"].fillna("").astype(str))
    return df


class Logger:
    def __init__(self):
        self.rows = []

    def step(self, dataset, rule, before, after, note=""):
        b = before.groupby("gu").size()
        a = after.groupby("gu").size().reindex(b.index, fill_value=0)
        for gu in b.index:
            self.rows.append({"dataset": dataset, "rule": rule, "gu": gu,
                              "n_before": int(b[gu]), "n_removed": int(b[gu] - a[gu]),
                              "n_after": int(a[gu]), "note": note})


def mad_filter(df, value_col, group_cols):
    logv = np.log(df[value_col].where(df[value_col] > 0))
    med = logv.groupby([df[c] for c in group_cols]).transform("median")
    mad = (logv - med).abs().groupby([df[c] for c in group_cols]).transform("median")
    mad = mad.replace(0, np.nan).fillna((logv - med).abs().median())
    keep = (logv - med).abs() <= MAD_K * 1.4826 * mad
    return keep.fillna(False)


def clean(df: pd.DataFrame, kind: str, log: Logger, name: str = None) -> pd.DataFrame:
    name = name or kind
    d0 = df
    d = d0[d0["gu"].notna()]
    log.step(name, "R1_서울25구외", d0, d)

    d = d.copy()
    d["ym"] = d["contract_date"].dt.to_period("M")
    days = d.groupby(["gu", "ym"])["contract_date"].transform("nunique")
    first = d.groupby("gu")["ym"].transform("min")
    last = d.groupby("gu")["ym"].transform("max")
    partial = ((d["ym"] == first) | (d["ym"] == last)) & (days < 10) & (d["source"] != "api")
    before = d
    d = d[~partial]
    log.step(name, "R2_부분월", before, d, "공개시스템 CSV의 구별 첫·마지막 달 중 계약일 10일 미만 (API 제외)")

    if kind == "trade":
        before = d
        d = d[~d["cancelled"]]
        log.step(name, "R3_해제", before, d)

    before = d
    dup_cols = [c for c in d.columns if c not in ("NO", "source", "ym")]
    d = d.drop_duplicates(subset=dup_cols)
    log.step(name, "R4_완전중복", before, d, "동일건물 다른 호수 가능성은 한계로 명시")

    core = ["contract_date", "area"] + (["price"] if kind == "trade" else ["deposit"])
    before = d
    d = d.dropna(subset=core)
    log.step(name, "R5_핵심결측", before, d)

    before = d
    d = d[d["area"].between(AREA_MIN, AREA_MAX)]
    log.step(name, "R6_면적이상치", before, d, f"{AREA_MIN}~{AREA_MAX}㎡")

    d = d.copy()
    d["year"] = d["contract_date"].dt.year
    if kind == "trade":
        d["ppsqm"] = d["price"] / d["area"]
        before = d
        d = d[mad_filter(d, "ppsqm", ["gu", "year"])]
        log.step(name, "R7_가격이상치", before, d, f"log ㎡당가 ±{MAD_K}MAD")
    else:
        d["ppsqm"] = d["deposit"] / d["area"]
        keep = ~d["is_jeonse"] | mad_filter(d.assign(ppsqm=d["ppsqm"].where(d["is_jeonse"])),
                                            "ppsqm", ["gu", "year"])
        before = d
        d = d[keep]
        log.step(name, "R7_가격이상치", before, d, f"전세만 log ㎡당보증금 ±{MAD_K}MAD")

    bad_by = (d["build_year"] <= 1900) | (d["build_year"] > d["year"] + 1)
    log.rows.append({"dataset": name, "rule": "R8_건축년도→NaN", "gu": "전체",
                     "n_before": len(d), "n_removed": 0, "n_after": len(d),
                     "note": f"값만 결측 처리 {int(bad_by.sum())}건"})
    d.loc[bad_by, "build_year"] = np.nan
    return d.drop(columns=["ym"])


def missing_report(df, kind):
    cols = [c for c in ["area", "floor", "build_year", "price", "deposit", "term_months",
                        "contract_kind", "pre_deposit", "bldg", "jibun"] if c in df]
    out = []
    for gu, g in df.groupby("gu"):
        for c in cols:
            s = g[c]
            miss = s.isna() | (s.astype(str).str.strip().isin(["", "nan", "미기재"]))
            out.append({"dataset": kind, "gu": gu, "column": c, "n": len(g),
                        "missing_rate": round(miss.mean(), 4)})
    return pd.DataFrame(out)


def load_all(htype: str, kind: str) -> pd.DataFrame:
    frames = []
    api = read_api_dir(dataset_name(htype, kind), kind)
    if len(api):
        frames.append(standardize(api, kind))
    if htype == "rh":
        for f in sorted((RAW / "csv_export").glob(f"*{'매매' if kind == 'trade' else '전월세'}*.csv")):
            frames.append(standardize(read_csv_export(f, kind), kind))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    key = [c for c in ["lawd_cd", "dong", "jibun", "bldg", "area", "floor", "contract_date",
                       "price", "deposit", "monthly_rent"] if c in df]
    first_src = df.groupby(key, dropna=False)["source"].transform("min")
    n_before = len(df)
    df = df[df["source"] == first_src]
    if n_before != len(df):
        print(f"[{kind}] 출처 간 중복 {n_before - len(df):,}행 제거 (API·CSV 동시 수집분)")
    return df


def main():
    INTERIM.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    log = Logger()
    miss = []
    for htype in HOUSING_TYPES:
        for kind in ("trade", "rent"):
            name = dataset_name(htype, kind)
            df = load_all(htype, kind)
            if df.empty:
                print(f"[{name}] 원자료 없음 — 건너뜀")
                continue
            c = clean(df, kind, log, name)
            c.to_csv(INTERIM / f"{name}_clean.csv", index=False, encoding="utf-8-sig")
            miss.append(missing_report(c, name))
            print(f"[{name}] 원자료 {len(df):,} → 정제 {len(c):,}행")
    lg = pd.DataFrame(log.rows)
    lg.to_csv(OUTPUTS / "cleaning_log.csv", index=False, encoding="utf-8-sig")
    if miss:
        pd.concat(miss).to_csv(OUTPUTS / "missing_report.csv", index=False, encoding="utf-8-sig")
    summary = lg[lg.gu != "전체"].groupby(["dataset", "rule"])["n_removed"].sum()
    print(summary.to_string())


if __name__ == "__main__":
    main()
