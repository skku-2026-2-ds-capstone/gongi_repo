import argparse
import json
import os
import urllib.request

import pandas as pd

from config import EXTERNAL, SEOUL_GU

SRC = EXTERNAL / "lease_order"
KINDS = ["aggregate", "all"]
API_URL = "https://data.iros.go.kr/openapi/cr/rs/selectCrRsRgsCsOpenApi.rest"
API_SEOUL = "900"


def read_kind(kind):
    files = sorted((SRC / kind).glob(f"lease_order_dong_{kind}_*.xls"))
    assert files, f"{SRC / kind} 에 엑셀이 없음"
    d = pd.concat([pd.read_excel(f, header=None, skiprows=2, names=["ym", "sido", "gu", "dong", "n"]) for f in files])
    assert (d.sido == "서울특별시").all() and set(d.gu) <= set(SEOUL_GU.values())
    assert not d.duplicated(["ym", "gu", "dong"]).any(), "같은 동·월이 두 파일에 있음"
    return d.drop(columns="sido").rename(columns={"n": f"n_{kind}"})


def fetch_api(key):
    rows = []
    for a, b in [(2016, 2018), (2019, 2021), (2022, 2024), (2025, 2026)]:
        for cls, name in [("", "n_api_all"), ("02", "n_api_aggregate")]:
            q = (f"{API_URL}?id=0000000079&reqtype=json&key={key}&search_type_api=01"
                 f"&search_start_date_api={a}&search_end_date_api={b}&search_regn1_name_api={API_SEOUL}")
            if cls:
                q += f"&search_real_cls_api={cls}"
            try:
                res = json.load(urllib.request.urlopen(q, timeout=30))["result"]
            except Exception as e:
                raise RuntimeError(str(e).replace(key, "***")) from None
            assert res["head"]["returnCode"] == "APIINFO-0001", res["head"]
            rows += [{"year": int(i["resDate"]), "gu": i["adminRegn2Name"], "col": name, "n": int(i["tot"])}
                     for i in res["items"]["item"]]
    return pd.DataFrame(rows).pivot_table(index=["year", "gu"], columns="col", values="n", fill_value=0).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", action="store_true", help="구×연 API 집계를 받아 엑셀과 대조")
    args = ap.parse_args()

    d = read_kind("aggregate").merge(read_kind("all"), on=["ym", "gu", "dong"], how="outer")
    d = d.fillna(0).astype({"n_aggregate": int, "n_all": int}).sort_values(["ym", "gu", "dong"])
    assert (d.n_aggregate <= d.n_all).mean() > 0.99, "집합건물 수가 전체보다 많은 행이 너무 많음"
    d.to_csv(EXTERNAL / "lease_order_dong.csv", index=False, encoding="utf-8-sig")
    y = d.assign(year=d.ym.str[:4].astype(int)).groupby("year")[["n_aggregate", "n_all"]].sum()
    print(f"{d.ym.min()} ~ {d.ym.max()}, 동 {d.groupby(['gu', 'dong']).ngroups}개\n{y.to_string()}")

    if args.api:
        api = fetch_api(os.environ["OPEN_API_KEY"])
        api.to_csv(EXTERNAL / "lease_order_gu_api.csv", index=False, encoding="utf-8-sig")
        a = api.groupby("year")[["n_api_aggregate", "n_api_all"]].sum()
        print("\n엑셀 ÷ API (서울 합계)\n" + (y / a.values).round(3).to_string())


if __name__ == "__main__":
    main()
