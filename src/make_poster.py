import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import qrcode
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle

from pathlib import Path

from config import OUTPUTS

REPO_ROOT = Path(__file__).resolve().parents[1]

REPO = "https://github.com/skku-2026-2-ds-capstone/gongi_repo"
MAIN, ACCENT, GRAY, LGRAY, BG, OWN = "#1F3A5F", "#D95F02", "#7A7A7A", "#BDBDBD", "#EEF1F5", "#B00020"
LIGHT_A, MID_A, LIGHT_M, MID_M = "#F6C9A0", "#EE9A55", "#B9C6D8", "#7F95B3"
W, H, M, G = 594, 841, 22, 12
MIN_BAR = 0.10


def build(sketch):
    K = 1.0 if sketch else 841 / 297
    pt = lambda x: x * K
    BODY, FIGT, HEAD = pt(7.8), pt(6.8), pt(11.5)
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": FIGT,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRAY,
                         "axes.linewidth": pt(0.8), "xtick.major.width": pt(0.8), "ytick.major.width": pt(0.8),
                         "xtick.major.size": pt(3.5), "ytick.major.size": pt(3.5),
                         "xtick.color": "#333333", "ytick.color": "#333333", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(8.27 * K, 11.69 * K))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    def sub(x, y, w, h):
        return fig.add_axes([x / W, y / H, w / W, h / H])

    def frame(x, y, w, h, num, head, color=MAIN, lw=1.1, owner=None):
        ax.add_patch(Rectangle((x, y), w, h, fc="white", ec=color, lw=pt(lw)))
        ax.text(x + 7, y + h - 7, f"{num}  {head}", fontsize=HEAD if w > 300 else pt(10), fontweight="bold",
                color=color, va="top")
        if sketch and owner:
            ax.text(x + w - 5, y + 3, f"담당: {owner}", fontsize=pt(5.5), color=OWN, ha="right", va="bottom")

    def text(x, y, s, size=7.8, **kw):
        kw.setdefault("color", "#222222")
        ax.text(x, y, s, fontsize=pt(size), va="top", linespacing=1.45, **kw)

    cv = pd.read_csv(OUTPUTS / "dong_lag_ladder.csv")
    cv = cv[cv.condition.str.startswith("기본")].set_index("model")
    ew = pd.read_csv(OUTPUTS / "dong_early_warning.csv")
    ew = ew[ew.condition.str.startswith("기본")].set_index("model")
    sens = pd.read_csv(OUTPUTS / "dong_early_warning_sens.csv")
    dense = sens[sens.check.str.startswith("b") & sens.period.str.startswith("급증 직전")].set_index("model")
    p0_after, p0_before, p0_dense = cv.loc["P0", "spearman"], ew.loc["P0", "spearman"], dense.loc["P0", "spearman"]
    m1_after = cv.loc["M1", "spearman"]
    n_dense = int(dense.loc["P0", "n"])

    if sketch:
        ax.text(M, H - 7, "포스터 스케치 · 공.이. · 레이아웃 B+C 혼합 · A1 594×841mm 비율 · 글자 크기는 A1 인쇄 기준 "
                          "(본문 22pt, 그림 19pt)", fontsize=pt(5.8), color=GRAY, va="top")

    th = 112
    y0 = H - M - th
    ax.add_patch(Rectangle((M, y0), W - 2 * M, th, fc=MAIN, ec=MAIN))
    ax.text(M + 14, y0 + th - 12, "작년 위험이 내년 위험의 가장 강한 신호였다.\n시장 피처는 어느 시기에도 그 값을 넘지 못했다.",
            fontsize=pt(21), fontweight="bold", color="white", va="top", linespacing=1.25)
    ax.text(M + 14, y0 + 30, f"예측 정확도 ρ   작년 값 {p0_after:.2f}  vs  시장 피처 {m1_after:.2f}   ·   "
                             f"급증 직전에도 작년 신청 6건 이상 동은 {p0_dense:.2f}",
            fontsize=pt(11), color="#FFC48A", va="top", fontweight="bold")
    ax.text(M + 14, y0 + 11, "공.이.  김상호 · 박채영 · 정다현 · 이예서   |   성균관대 데이터사이언스캡스톤 2026-2",
            fontsize=pt(7), color="white", va="top")

    h1 = 132
    y1 = y0 - G - h1
    cw = (W - 2 * M - 2 * G) / 3
    x1, x2, x3 = M, M + cw + G, M + 2 * (cw + G)
    frame(x1, y1, cw, h1, "1", "문제와 질문", owner="박채영")
    text(x1 + 8, y1 + h1 - 28, "서울 임차권등기명령\n2022 → 2023년 4배 (14,787건)")
    text(x1 + 8, y1 + h1 - 64, "RQ1  위험 지역은 사고 전에\n         어떤 시장이었나?\n"
                               "RQ2  시장 피처를 더하면\n         위험 지역을 더 잘 고르나?", fontweight="bold", color=MAIN)

    frame(x2, y1, cw, h1, "2", "자료 흐름", owner="김상호")
    flow = [("빌라 실거래 (국토부)", "105.3만 → 정제 100.6만 건"), ("피처 (동 × 연도)", "전세가율 · 가격 변화 · 신축"),
            ("분석 단위", "구 25개 · 동 144개")]
    fy = y1 + h1 - 27
    for i, (a, b) in enumerate(flow):
        ax.add_patch(Rectangle((x2 + 8, fy - 27), cw - 16, 27, fc=BG, ec="none"))
        ax.text(x2 + 13, fy - 4, a, fontsize=pt(7.2), fontweight="bold", va="top", color=MAIN)
        ax.text(x2 + 13, fy - 15, b, fontsize=pt(7.2), va="top")
        if i < 2:
            ax.add_patch(FancyArrowPatch((x2 + cw / 2, fy - 27), (x2 + cw / 2, fy - 34), arrowstyle="-|>",
                                         mutation_scale=pt(7), color=MAIN, lw=pt(1)))
        fy -= 34

    frame(x3, y1, cw, h1, "3", "방법: 모형 사다리", owner="정다현")
    lad = [("P0", "작년 등기명령 비율", ACCENT), ("M0", "전세가율 (기준선)", GRAY),
           ("M1", "+ 가격 · 전세가율 변화", MAIN), ("M2", "+ 신축 비중", MAIN)]
    ly = y1 + h1 - 28
    for k, lab, col in lad:
        ax.add_patch(Rectangle((x3 + 8, ly - 15), 24, 15, fc=col, ec="none"))
        ax.text(x3 + 20, ly - 7.5, k, fontsize=pt(7.2), color="white", ha="center", va="center", fontweight="bold")
        ax.text(x3 + 37, ly - 7.5, lab, fontsize=pt(7.2), va="center")
        ly -= 19
    text(x3 + 8, ly - 1, "구를 통째로 빼는 교차검증\n최소 의미 차이 0.10 사전 고정", size=7.2)

    h2 = 232
    y2 = y1 - G - h2
    frame(M, y2, W - 2 * M, h2, "4", "RQ2 (주 분석): 시장 피처는 작년 값을 넘지 못했다", color=ACCENT, lw=1.6, owner="정다현")

    rows = [("P0", "작년 값만"), ("P0+M1", "작년 값 + 시장 피처"), ("M1", "시장 피처만"), ("M0", "전세가율만")]
    fa = sub(M + 78, y2 + 84, 245, h2 - 132)
    for i, (k, lab) in enumerate(rows):
        y = len(rows) - 1 - i
        p0 = k.startswith("P0")
        dark = ACCENT if p0 else MAIN
        for off, src, col in [(0.27, ew, LIGHT_A if p0 else LIGHT_M), (0.0, dense, MID_A if p0 else MID_M), (-0.27, cv, dark)]:
            v = src.loc[k, "spearman"]
            fa.barh(y + off, v, height=0.25, color=col)
            fa.plot([src.loc[k, "ci_spearman_lo"], src.loc[k, "ci_spearman_hi"]], [y + off] * 2, color="#444444", lw=pt(0.7))
            fa.text(src.loc[k, "ci_spearman_hi"] + 0.02, y + off, f"{v:.2f}", va="center", fontsize=FIGT - pt(0.6),
                    color=dark, fontweight="bold")
    fa.set_yticks(range(len(rows)), [r[1] for r in rows][::-1], fontsize=FIGT + pt(0.4))
    fa.tick_params(axis="y", length=0)
    fa.spines["left"].set_visible(False)
    fa.set_xlim(0, 1.05)
    fa.set_ylim(-0.6, len(rows) - 0.4)
    fa.set_xlabel("예측 정확도 ρ  (예측한 위험 순위가 실제와 맞는 정도, 1 = 완벽)", fontsize=FIGT)
    fa.legend(handles=[Patch(color="#D9D9D9", label=f"급증 직전 전체 (동 {int(ew.loc['P0', 'n'])})"),
                       Patch(color="#9A9A9A", label=f"급증 직전, 작년 6건 이상 (동 {n_dense})"),
                       Patch(color="#555555", label=f"급증 이후 (동 {int(cv.loc['P0', 'n'])})")],
              loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=3, fontsize=FIGT - pt(0.6), frameon=False,
              handlelength=1.0, columnspacing=0.9, handletextpad=0.4)
    text(M + 12, y2 + 38, f"그림 A. 작년 값은 급증 이후 {p0_after:.2f}로 가장 높았다. 급증 직전 {p0_before:.2f}로 약해진 것은\n"
                          f"신청이 드문 동 때문으로, 작년 6건 이상 동은 {p0_dense:.2f}였다. (선 = 95% 구간)",
         size=7.4, fontweight="bold", color=MAIN)

    dn = pd.read_csv(OUTPUTS / "dong_paired_diff.csv").set_index("a")
    fr = [("시장 변화 추가\n고위험 동 골라내기", "M1/linear", "auc", MAIN), ("시장 변화 추가\n위험 순위", "M1/linear", "spearman", MAIN),
          ("신축 비중 추가\n고위험 동 골라내기", "M2/linear", "auc", GRAY), ("신축 비중 추가\n위험 순위", "M2/linear", "spearman", GRAY)]
    bx = M + 400
    fb = sub(bx, y2 + 84, W - M - bx - 12, h2 - 132)
    for i, (lab, key, m, col) in enumerate(fr):
        y = len(fr) - 1 - i
        r = dn.loc[key]
        v = r[f"diff_{m}"]
        fb.barh(y, v, height=0.5, color=col)
        fb.plot([r[f"ci_{m}_lo"], r[f"ci_{m}_hi"]], [y, y], color="#444444", lw=pt(0.8))
        fb.text(max(v, r[f"ci_{m}_hi"]) + 0.015, y, f"{v:+.2f}".replace("-0.00", "0.00"), va="center",
                fontsize=FIGT, color=col, fontweight="bold")
    fb.axvline(0, color="#333333", lw=pt(0.8))
    fb.axvline(MIN_BAR, color=ACCENT, lw=pt(1.4))
    fb.text(MIN_BAR, len(fr) - 0.45, "의미 있는 개선\n기준 +0.10", color=ACCENT, fontsize=FIGT - pt(0.4), ha="left",
            va="bottom", fontweight="bold")
    fb.set_yticks(range(len(fr)), [f[0] for f in fr][::-1], fontsize=FIGT - pt(0.2))
    fb.tick_params(axis="y", length=0)
    fb.spines["left"].set_visible(False)
    fb.set_xlim(-0.12, 0.3)
    fb.set_ylim(-0.6, len(fr) - 0.4)
    fb.set_xlabel("전세가율만 쓴 모형보다 나아진 폭 (동 144개)", fontsize=FIGT)
    text(bx - 70, y2 + 38, "그림 B. 시장 변화를 더한 개선은 +0.09로,\n미리 정한 기준 +0.10에 못 미쳤다. (선 = 95% 구간)",
         size=7.4, fontweight="bold", color=MAIN)

    h3 = 178
    y3 = y2 - G - h3
    frame(M, y3, W - 2 * M, h3, "5", "RQ1 (탐색): 고위험 구는 사고 전부터 신축 빌라가 많았다", owner="이예서")
    g = pd.read_csv(OUTPUTS / "rq1_gu_halfyear.csv")
    halves = sorted(g.half.unique())
    w = g.pivot(index="half", columns="gu", values="new_share").reindex(halves)
    sep = g.drop_duplicates("half").set_index("half")["sep_new_share"].reindex(halves)
    fc = sub(M + 40, y3 + 28, 300, h3 - 62)
    x = np.arange(len(halves))
    for xi in x[(sep == "high>low").to_numpy()]:
        fc.axvspan(xi - 0.5, xi + 0.5, color=BG, lw=0, zorder=0)
    nudge = {"동작구": 0.03, "은평구": -0.02}
    for gname, col in [("강서구", ACCENT), ("중랑구", MID_A), ("은평구", "#4D4D4D"), ("동작구", LGRAY)]:
        fc.plot(x, w[gname], "-", color=col, lw=pt(1.8))
        fc.text(x[-1] + 0.4, w[gname].iloc[-1] + nudge.get(gname, 0), gname[:-1], color=col, fontsize=FIGT,
                va="center", fontweight="bold")
    ticks = [i for i, h in enumerate(halves) if h.endswith("H1") and int(h[:4]) % 2 == 0]
    fc.set_xticks(ticks, [halves[i][:4] for i in ticks])
    fc.set_xlim(-0.5, len(halves) + 1.5)
    fc.set_ylabel("신축 비중", fontsize=FIGT)
    fc.set_title("주황 계열 = 고위험 구 · 회색 계열 = 저위험 구 · 음영 = 두 고위험 구가 모두 높았던 반기",
                 fontsize=FIGT - pt(0.3), loc="left", color="#444444")
    fc.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    tx = M + 380
    text(tx, y3 + h3 - 32, "그림 C. 강서·중랑의 신축 비중은\n16개 반기 중 15개에서\n은평·동작보다 높았다 (음영).",
         size=7.4, fontweight="bold", color=MAIN)
    text(tx, y3 + h3 - 82, "차이는 2017~2020년에 가장 컸다\n(0.23): 2021 고점에 재계약되고\n2023년에 만기를 맞은 세대.\n\n"
                           "구가 2 대 2뿐이라 정확 순열\np ≥ .33. 탐색적 결과다.", size=7.0, color="#444444")

    y4 = M + (0 if sketch else 10)
    h4 = y3 - G - y4
    bw = (W - 2 * M - 2 * G) / 3
    frame(M, y4, bw, h4, "6", "결론", owner="이예서 · 김상호")
    text(M + 8, y4 + h4 - 28, "작년 값이 가장 강한 신호다.\n시장 피처는 '왜 위험한가'를\n설명하는 데 쓴다:\n전세가율↑ · 전셋값↓ · 신축")
    xb = M + bw + G
    frame(xb, y4, bw, h4, "7", "한계 · 다음", owner="이예서")
    text(xb + 8, y4 + h4 - 28, "대리 타깃 (HUG와 ρ 0.61~0.73)\n피처는 빌라 시장만\n2023 타깃에 절차 변화 혼입\n"
                               "다음: 6 대 6 구, 빌라 동 민감도", size=7.4)
    xq = M + 2 * (bw + G)
    frame(xq, y4, bw, h4, "8", "결과물 · 코드", owner="김상호")
    text(xq + 8, y4 + h4 - 28, "재현 파이프라인\n(seed 고정)\nQR = 코드 저장소")
    qs = min(h4 - 30, 60)
    ax.imshow(qrcode.make(REPO).convert("RGB"), extent=(xq + bw - qs - 8, xq + bw - 8, y4 + 10, y4 + 10 + qs), zorder=2)

    if sketch:
        ax.text(M, 6, "읽는 순서 1→8  ·  3초: 제목  ·  30초: 그림 A·B·C 캡션  ·  3분: 상자 1~3, 6~8  ·  담당은 보고서 역할 분담 기준 · "
                      "제목 띠: 박채영 · 최종 레이아웃 확인: 김상호", fontsize=pt(5.5), color=OWN, va="bottom")
    else:
        ax.text(M, 12, "자료: 국토교통부 실거래가 공개 API (공공데이터포털) · 법원 등기정보광장 임차권등기명령 · "
                       "주택도시보증공사 지역별 전세금반환보증 사고현황 (공공데이터포털 15134221)",
                fontsize=pt(5.2), color=GRAY, va="bottom")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sketch", action="store_true", help="12주 스케치 (A4, 담당 표시)")
    args = ap.parse_args()
    out = REPO_ROOT / "materials" / "poster" / ("poster_sketch.pdf" if args.sketch else "poster_draft.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    build(args.sketch).savefig(out)
    print("saved", out)


if __name__ == "__main__":
    main()
