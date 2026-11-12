import argparse

import pandas as pd

from config import ROOT, EXTERNAL, SEOUL_GU


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default=str(ROOT / "data" / "hug_target.xlsx"))
    ap.add_argument("--out", default="hug_target.csv", help="data/external/ 아래 저장 파일명")
    args = ap.parse_args()

    raw = pd.read_excel(args.xlsx, header=None, usecols=[0, 3, 4, 5],
                        names=["lawd_cd", "accident_count", "accident_amount", "accident_rate"])
    raw["lawd_cd"] = raw["lawd_cd"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    d = raw[raw["lawd_cd"].isin(SEOUL_GU)].copy()

    missing = set(SEOUL_GU) - set(d["lawd_cd"])
    if missing or len(d) != 25:
        raise SystemExit(f"서울 25개 구가 모두 있지 않음: 누락 {sorted(missing)}, 행 {len(d)}")

    for c in ["accident_count", "accident_amount", "accident_rate"]:
        d[c] = pd.to_numeric(d[c], errors="raise")
    d["accident_amount"] = (d["accident_amount"] / 1e8).round(2)
    d["gu"] = d["lawd_cd"].map(SEOUL_GU)
    if (d["accident_rate"] <= 0).any():
        raise SystemExit(f"사고율 0 이하 구: {d.loc[d.accident_rate <= 0, 'gu'].tolist()}")

    out = d[["gu", "lawd_cd", "accident_rate", "accident_count", "accident_amount"]]
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    path = EXTERNAL / args.out
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"저장: {path} (25개 구, 사고율 {out.accident_rate.min()}~{out.accident_rate.max()}%)")


if __name__ == "__main__":
    main()
