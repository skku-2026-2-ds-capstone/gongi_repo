import argparse
import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import qrcode
from matplotlib import font_manager
from matplotlib.transforms import blended_transform_factory
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

from config import OUTPUTS

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/skku-2026-2-ds-capstone/gongi_repo"

for _f in glob.glob(os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Windows\Fonts\Pretendard-*.otf")) + \
        glob.glob(r"C:\Windows\Fonts\Pretendard-*.otf"):
    font_manager.fontManager.addfont(_f)
FONT = "Pretendard" if any("Pretendard" in f.name for f in font_manager.fontManager.ttflist) else "Malgun Gothic"

NAVY, TEAL, SLATE, INK, BOX, LINE, OWN = "#143D57", "#168C91", "#4A5F6B", "#16242E", "#ECEFF1", "#D8E2E8", "#B00020"
TEAL_L, TEAL_M, NAVY_L, NAVY_M, GRAY = "#A9D8D9", "#5FB3B6", "#C3CDD4", "#7D93A3", "#8B9CA8"
PAGE, SHADOW, CHIP, GRID, TINT = "#F2F5F7", "#DDE5EA", "#2E5B79", "#E7EDF1", "#E6F3F3"
WATERMARK = "#24506D"
W, H, M, G = 594, 841, 16, 9
MIN_BAR = 0.10


def josa(v, kind):
    last = f"{v:.2f}"[-1]
    batchim, rieul = last in "013678", last in "178"
    if kind == "이었다":
        return "이었다" if batchim else "였다"
    if kind == "은":
        return "은" if batchim else "는"
    return "로" if (not batchim or rieul) else "으로"


def house(ax, x0, top, size, K, color=WATERMARK):
    sc = size / 840
    P = lambda x, y: (x0 + (x - 1040) * sc, top - (y - 110) * sc)
    lw = lambda px: px * sc * 72 / 25.4 * K / (841 / 297)
    kw = dict(color=color, solid_capstyle="round", solid_joinstyle="round", zorder=1)
    roof = [P(*p) for p in [(1190, 785), (1190, 565), (1072, 565), (1072, 505), (1515, 135), (1765, 345)]]
    ax.plot(*zip(*roof), lw=lw(24), **kw)
    cx, cy = P(1412, 465)
    ax.add_patch(Circle((cx, cy), 110 * sc, fill=False, ec=color, lw=lw(22), zorder=1))
    ax.plot(*zip(P(1335, 545), P(1268, 612)), lw=lw(46), **kw)
    ax.plot(*zip(P(1412, 405), P(1412, 485)), lw=lw(30), color=color, solid_capstyle="butt", zorder=1)
    ax.add_patch(Circle(P(1412, 520), 12 * sc, fc=color, ec="none", zorder=1))
    for xa, ya, xb in [(1168, 812, 1275), (1376, 662, 1482), (1573, 515, 1680), (1778, 362, 1886)]:
        (ax_, ay), (bx_, by) = P(xa, 938), P(xb, ya)
        ax.add_patch(Rectangle((ax_, ay), bx_ - ax_, by - ay, fc=color, ec="none", zorder=1))


def build(sketch):
    K = 1.0 if sketch else 841 / 297
    pt = lambda x: x * K
    BODY, CAP, FIG_, HEAD_, LAB_ = 8.2, 7.8, 7.4, 13.4, 8.0
    FIGT, HEAD = pt(FIG_), pt(HEAD_)
    plt.rcParams.update({"font.family": FONT, "font.weight": "medium", "axes.unicode_minus": False, "font.size": FIGT,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRAY,
                         "axes.linewidth": pt(0.9), "xtick.major.width": pt(0.9), "ytick.major.width": pt(0.9),
                         "xtick.major.size": pt(3.5), "ytick.major.size": pt(3.5), "axes.facecolor": "none",
                         "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.labelsize": FIGT, "ytick.labelsize": FIGT, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(8.27 * K, 11.69 * K))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    line = lambda s: s * 1.0 * 1.42

    def sub(x, y, w, h):
        return fig.add_axes([x / W, y / H, w / W, h / H])

    def rbox(x, y, w, h, fc=BOX, ec="none", lw=0.0, r=4):
        ax.add_patch(FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r, boxstyle=f"round,pad={r}", fc=fc, ec=ec,
                                    lw=pt(lw), mutation_aspect=1))

    ax.add_patch(Rectangle((0, 0), W, H, fc=PAGE, ec="none", zorder=0))

    def card(x, y, w, h, num, label, head, owner=None, fc="white"):
        rbox(x + 1.0, y - 1.4, w, h, fc=SHADOW, r=5)
        rbox(x, y, w, h, fc=fc, r=5)
        ax.add_patch(Circle((x + 14, y + h - 12), 6.2, fc=TEAL, ec="none", zorder=2))
        ax.text(x + 14, y + h - 12.3, num, fontsize=pt(LAB_), fontweight="bold", color="white", ha="center", va="center",
                zorder=3)
        ax.text(x + 24, y + h - 12.3, label, fontsize=pt(LAB_), fontweight="bold", color=TEAL, va="center")
        ax.text(x + 8, y + h - 24, head, fontsize=HEAD, fontweight="bold", color=NAVY, va="top")
        if sketch and owner:
            ax.text(x + w - 5, y + 3, f"담당: {owner}", fontsize=pt(5.5), color=OWN, ha="right", va="bottom")

    TOP = 47

    def text(x, y, s, size=BODY, **kw):
        kw.setdefault("color", INK)
        ax.text(x, y, s, fontsize=pt(size), va="top", linespacing=1.4, **kw)

    NOTE_ROW, NOTE_GAP, NOTE_HEAD = 12.5, 2.2, 13

    def note_h(n):
        return NOTE_HEAD + n * (NOTE_ROW + NOTE_GAP) + 2

    def note(x, ytop, w, title, rows, key_w):
        ax.add_patch(Rectangle((x, ytop - NOTE_HEAD + 2), 2.4, NOTE_HEAD - 4, fc=TEAL, ec="none", zorder=2))
        ax.text(x + 7, ytop - NOTE_HEAD / 2, title, fontsize=pt(CAP), fontweight="bold", color=NAVY, va="center")
        for i, (k, v) in enumerate(rows):
            yt = ytop - NOTE_HEAD - i * (NOTE_ROW + NOTE_GAP)
            rbox(x, yt - NOTE_ROW, w, NOTE_ROW, r=3)
            ax.text(x + 6, yt - NOTE_ROW / 2, k, fontsize=pt(CAP), fontweight="bold", color=TEAL, va="center")
            ax.text(x + key_w, yt - NOTE_ROW / 2, v, fontsize=pt(CAP), color=INK, va="center")

    def axfix(a):
        for t in a.get_xticklabels() + a.get_yticklabels():
            t.set_fontweight("semibold")
            t.set_color(INK)
        a.xaxis.label.set_fontweight("semibold")
        a.xaxis.label.set_color(SLATE)

    cv = pd.read_csv(OUTPUTS / "dong_lag_ladder.csv")
    cv = cv[cv.condition.str.startswith("기본")].set_index("model")
    ew = pd.read_csv(OUTPUTS / "dong_early_warning.csv")
    ew = ew[ew.condition.str.startswith("기본")].set_index("model")
    sens = pd.read_csv(OUTPUTS / "dong_early_warning_sens.csv")
    dense = sens[sens.check.str.startswith("b") & sens.period.str.startswith("급증 직전")].set_index("model")
    p0_after, p0_before, p0_dense = cv.loc["P0", "spearman"], ew.loc["P0", "spearman"], dense.loc["P0", "spearman"]
    m1_after = cv.loc["M1", "spearman"]
    n_dense = int(dense.loc["P0", "n"])
    gu = pd.read_csv(OUTPUTS / "model_paired_diff.csv")
    gu = gu[gu.comparison.str.startswith("사다리") & (gu.jr == "jr_cell")].set_index("a")
    dn = pd.read_csv(OUTPUTS / "dong_paired_diff.csv").set_index("a")
    fe = pd.read_csv(OUTPUTS / "dong_fixed_effects.csv")
    fe = fe[fe.spec.str.startswith("주")].set_index("feature")
    jr = fe.loc["jr_cell"]

    if sketch:
        ax.text(M, H - 5, "포스터 스케치 · 공.이. · 레이아웃 B+C 혼합 · A1 594×841mm 비율 · 글자 크기는 A1 인쇄 기준 "
                          "(본문 23pt, 그림 21pt)", fontsize=pt(5.8), color=SLATE, va="top")

    th = 136
    y0 = H - M - th
    rbox(M, y0, W - 2 * M, th, fc=NAVY, r=6)
    ax.add_patch(Rectangle((M + 14, y0 + th - 9.5), 30, 1.6, fc=TEAL, ec="none", zorder=2))
    house(ax, W - M - 112, y0 + th - 10, 104, K)
    ax.text(M + 14, y0 + th - 13, "데이터사이언스캡스톤 2026-2  ·  서울 전세보증 사고 위험지역의 시장 특성 분석",
            fontsize=pt(8.0), color=TEAL_L, va="top", fontweight="bold")
    ax.text(M + 14, y0 + th - 27, "작년 위험이 내년 위험의\n가장 강한 신호였다.", fontsize=pt(23), fontweight="bold",
            color="white", va="top", linespacing=1.16)
    ax.text(M + 14, y0 + 52, "시장 피처가 그 값을 넘는다는 근거는 없었다.", fontsize=pt(13.5), fontweight="bold",
            color=TEAL_L, va="top")
    cx = M + 14
    for lab, v, fc, ec, wch in [("작년 값", p0_after, TEAL, TEAL, 88), ("시장 피처", m1_after, CHIP, CHIP, 96),
                                ("급증 직전 · 작년 6건+ 동", p0_dense, "none", TEAL_L, 146)]:
        rbox(cx, y0 + 11, wch, 25, fc=fc, ec=ec, lw=1.4, r=5)
        ax.text(cx + 8, y0 + 23.5, lab, fontsize=pt(7.8), color="white", va="center")
        ax.text(cx + wch - 8, y0 + 23.5, f"{v:.2f}", fontsize=pt(12.5), color="white", va="center", ha="right",
                fontweight="bold")
        cx += wch + 7
    ax.text(cx + 1, y0 + 23.5, "위험 순위 ρ", fontsize=pt(7.2), color=TEAL_L, va="center")
    ax.text(W - M - 14, y0 + 23.5, "4조 공.이.  김상호 · 박채영 · 정다현 · 이예서", fontsize=pt(7.4), color=TEAL_L, va="center",
            ha="right")

    h1 = 152
    y1 = y0 - G - h1
    cw = (W - 2 * M - 2 * G) / 3
    x1, x2, x3 = M, M + cw + G, M + 2 * (cw + G)
    card(x1, y1, cw, h1, "1", "Question", "문제와 질문", owner="박채영")
    text(x1 + 8, y1 + h1 - TOP, "서울 임차권등기명령\n2022 → 2023년 4배", size=BODY)
    for i, (k, q) in enumerate([("RQ1", "위험 지역은 사고 전에\n어떤 시장이었나?"), ("RQ2", "시장 피처를 더하면\n위험 지역을 더 잘 고르나?")]):
        yy = y1 + h1 - TOP - 29 - i * 35
        rbox(x1 + 8, yy - 31, cw - 16, 31, r=3)
        rbox(x1 + 12, yy - 23, 24, 15, fc=TEAL, r=3)
        ax.text(x1 + 24, yy - 15.5, k, fontsize=pt(CAP), fontweight="bold", color="white", va="center", ha="center")
        ax.text(x1 + 41, yy - 15.5, q, fontsize=pt(CAP), color=INK, va="center", linespacing=1.3)

    card(x2, y1, cw, h1, "2", "Data", "자료 흐름", owner="김상호")
    flow = [("빌라 실거래 (국토부)", "105.3만 → 정제 100.6만 건"), ("피처 (동 × 연도)", "전세가율 · 가격 변화 · 신축"),
            ("분석 단위", "구 25개 · 동 144개")]
    fy = y1 + h1 - TOP
    for i, (a, b) in enumerate(flow):
        rbox(x2 + 8, fy - 28, cw - 16, 28, r=3)
        ax.add_patch(Rectangle((x2 + 8, fy - 25), 2.4, 22, fc=TEAL, ec="none", zorder=2))
        ax.text(x2 + 15, fy - 4, a, fontsize=pt(CAP), fontweight="bold", va="top", color=NAVY)
        ax.text(x2 + 15, fy - 16, b, fontsize=pt(CAP), va="top")
        if i < 2:
            ax.add_patch(FancyArrowPatch((x2 + cw / 2, fy - 28.5), (x2 + cw / 2, fy - 34), arrowstyle="-|>",
                                         mutation_scale=pt(8), color=SLATE, lw=pt(1.1)))
        fy -= 34.5

    card(x3, y1, cw, h1, "3", "Method", "방법: 모형 사다리", owner="정다현")
    lad = [("P0", "작년 등기명령 비율", TEAL), ("M0", "전세가율 (기준선)", GRAY),
           ("M1", "+ 가격 · 전세가율 변화", NAVY), ("M2", "+ 신축 비중", NAVY)]
    ly = y1 + h1 - TOP
    for k, lab, col in lad:
        rbox(x3 + 8, ly - 16, 27, 16, fc=col, r=2.5)
        ax.text(x3 + 21.5, ly - 8, k, fontsize=pt(CAP), color="white", ha="center", va="center", fontweight="bold")
        ax.text(x3 + 41, ly - 8, lab, fontsize=pt(CAP), va="center")
        ly -= 19
    text(x3 + 8, ly - 3, "구를 통째로 빼는 교차검증\n최소 의미 차이 0.10 사전 고정", size=CAP, color=SLATE)

    h2 = 230
    y2 = y1 - G - h2
    wb = 224
    xa = M + wb + G
    wa = W - M - xa
    cap_h = note_h(3)
    card(M, y2, wb, h2, "4", "RQ2 판정 · 주 분석 + 확장", "개선 근거 없음", owner="정다현")
    fr = [("위험 순위", gu.loc["M1/linear"], "spearman", NAVY), ("고위험 선별", gu.loc["M1/linear"], "auc", NAVY),
          ("위험 순위", dn.loc["M1/linear"], "spearman", NAVY_M), ("고위험 선별", dn.loc["M1/linear"], "auc", NAVY_M)]
    fb = sub(M + 92, y2 + cap_h + 34, wb - 102, h2 - TOP - cap_h - 42)
    for i, (lab, r, m, col) in enumerate(fr):
        y = len(fr) - 1 - i
        v = r[f"diff_{m}"]
        fb.barh(y, v, height=0.55, color=col)
        fb.plot([r[f"ci_{m}_lo"], r[f"ci_{m}_hi"]], [y, y], color=INK, lw=pt(0.9))
        fb.text(r[f"ci_{m}_hi"] + 0.02, y, f"{v:+.2f}", va="center", fontsize=FIGT, color=col, fontweight="bold")
    fb.grid(axis="x", color=GRID, lw=pt(0.8))
    fb.set_axisbelow(True)
    fb.axvline(0, color=INK, lw=pt(0.9))
    fb.axvline(MIN_BAR, color=TEAL, lw=pt(1.8))
    fb.axhline(1.5, color=LINE, lw=pt(1.0), xmin=-0.6, clip_on=False)
    fb.text(MIN_BAR + 0.02, len(fr) - 0.5, "기준 +0.10", color=TEAL, fontsize=FIGT, ha="left", va="bottom",
            fontweight="bold")
    fb.set_yticks(range(len(fr)), [f[0] for f in fr][::-1])
    tr = blended_transform_factory(fb.transAxes, fb.transData)
    for yc, lab, col in [(2.5, "구 25개\n주 분석", NAVY), (0.5, "동 144개\n확장", NAVY_M)]:
        fb.text(-0.47, yc, lab, transform=tr, ha="right", va="center", fontsize=FIGT, fontweight="bold", color=col,
                linespacing=1.25)
    fb.tick_params(axis="y", length=0)
    fb.spines["left"].set_visible(False)
    fb.set_xlim(-0.3, 0.8)
    fb.set_xticks([0, 0.3, 0.6])
    fb.set_ylim(-0.6, len(fr) - 0.4)
    fb.set_xlabel("전세가율만 쓴 모형보다 나아진 폭", fontsize=FIGT)
    axfix(fb)
    gauc, dm = gu.loc["M1/linear"], dn.loc["M1/linear"]
    note(M + 8, y2 + cap_h + 8, wb - 16, "그림 B. 시장 피처를 더해도 기준 미달",
         [("구 25개", f"AUC {gauc.diff_auc:+.2f}, 기준 바꾸면 유지 안 됨"),
          ("동 144개", f"{dm.diff_spearman:+.2f} · {dm.diff_auc:+.2f}, 기준 0.10 미만"),
          ("판정", "사전 규칙: 개선 근거 없음")], key_w=44)

    card(xa, y2, wa, h2, "5", "사후 · 작년 값과 비교", "작년 값이 가장 강한 신호", owner="정다현")
    rows = [("P0", "작년 값만"), ("P0+M1", "작년 값\n+ 시장 피처"), ("M1", "시장 피처만")]
    fa = sub(xa + 74, y2 + cap_h + 34, wa - 98, h2 - TOP - cap_h - 50)
    for i, (k, lab) in enumerate(rows):
        y = len(rows) - 1 - i
        p0 = k.startswith("P0")
        dark = TEAL if p0 else NAVY
        for off, src, col in [(0.28, ew, TEAL_L if p0 else NAVY_L), (0.0, dense, TEAL_M if p0 else NAVY_M), (-0.28, cv, dark)]:
            v = src.loc[k, "spearman"]
            fa.barh(y + off, v, height=0.27, color=col)
            fa.plot([src.loc[k, "ci_spearman_lo"], src.loc[k, "ci_spearman_hi"]], [y + off] * 2, color=INK, lw=pt(0.8))
            fa.text(src.loc[k, "ci_spearman_hi"] + 0.02, y + off, f"{v:.2f}", va="center", fontsize=FIGT - pt(0.5),
                    color=dark, fontweight="bold")
    fa.set_yticks(range(len(rows)), [r[1] for r in rows][::-1])
    fa.tick_params(axis="y", length=0)
    fa.spines["left"].set_visible(False)
    fa.grid(axis="x", color=GRID, lw=pt(0.8))
    fa.set_axisbelow(True)
    fa.set_xlim(0, 1.08)
    fa.set_xticks([0, 0.5, 1.0])
    fa.set_ylim(-0.6, len(rows) - 0.4)
    fa.set_xlabel("위험 순위를 맞힌 정도 ρ  (1 = 완벽 · 선 = 95% 구간)", fontsize=FIGT)
    leg = fa.legend(handles=[Patch(color="#D3D9DE", label=f"급증 직전 (동 {int(ew.loc['P0', 'n'])})"),
                             Patch(color="#9AA8B3", label=f"직전, 작년 6건+ (동 {n_dense})"),
                             Patch(color="#4F6270", label=f"급증 이후 (동 {int(cv.loc['P0', 'n'])})")],
                    loc="lower left", bbox_to_anchor=(-0.3, 1.0), ncol=3, fontsize=FIGT - pt(0.5), frameon=False,
                    handlelength=0.9, columnspacing=0.8, handletextpad=0.35)
    for t in leg.get_texts():
        t.set_fontweight("semibold")
    axfix(fa)
    m1_before = ew.loc["M1", "spearman"]
    note(xa + 8, y2 + cap_h + 8, wa - 16, "그림 A. 어느 시기든 작년 값이 가장 높았다",
         [("급증 이후", f"작년 값 {p0_after:.2f} vs 시장 피처 {m1_after:.2f}"),
          ("급증 직전", f"작년 값 {p0_before:.2f} · 시장 피처 {m1_before:.2f} (비슷)"),
          ("6건 이상 동", f"직전에도 작년 값 {p0_dense:.2f}: 약세는 잡음일 수 있음")], key_w=62)

    h3 = 134
    y3 = y2 - G - h3
    card(M, y3, W - 2 * M, h3, "6", "RQ1 · 탐색 (4개 구)", "고위험 구는 사고 전부터 신축이 많았다", owner="이예서")
    g = pd.read_csv(OUTPUTS / "rq1_gu_halfyear.csv")
    halves = sorted(g.half.unique())
    w = g.pivot(index="half", columns="gu", values="new_share").reindex(halves)
    sep = g.drop_duplicates("half").set_index("half")["sep_new_share"].reindex(halves)
    fc = sub(M + 26, y3 + 17, 272, h3 - TOP - 27)
    x = np.arange(len(halves))
    for xi in x[(sep == "high>low").to_numpy()]:
        fc.axvspan(xi - 0.5, xi + 0.5, color="#F1F5F7", lw=0, zorder=0)
    nudge = {"강서구": 0.02, "동작구": 0.03, "은평구": -0.04}
    for gname, col in [("강서구", TEAL), ("중랑구", TEAL_M), ("은평구", NAVY), ("동작구", GRAY)]:
        fc.plot(x, w[gname], "-", color=col, lw=pt(2.0))
        fc.text(x[-1] + 0.4, w[gname].iloc[-1] + nudge.get(gname, 0), gname[:-1], color=col, fontsize=FIGT,
                va="center", fontweight="bold")
    ticks = [i for i, h in enumerate(halves) if h.endswith("H1") and int(h[:4]) % 2 == 0]
    fc.set_xticks(ticks, [halves[i][:4] for i in ticks])
    fc.set_xlim(-0.5, len(halves) + 2.2)
    fc.grid(axis="y", color=GRID, lw=pt(0.8))
    fc.set_axisbelow(True)
    fc.set_title("신축 비중 (반기별)", loc="left", fontsize=FIGT, fontweight="semibold", color=SLATE, pad=pt(4))
    fc.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    axfix(fc)
    tx = M + 318
    note(tx, y3 + h3 - TOP + 2, W - M - tx - 8, "그림 C. 강서·중랑이 거의 늘 높았다",
         [("반기", "16개 중 15개에서 높음 (음영)"), ("최대 차이", "2017~2020년, 23%p"),
          ("의미", "2021 재계약 · 2023 만기 세대"), ("한계", "구 2 대 2, 원인은 확인 안 함")], key_w=50)

    y4 = M + (0 if sketch else 8)
    h4 = y3 - G - y4
    wc, wl = 258, 184
    wq = W - 2 * M - wc - wl - 2 * G
    def kv_rows(x, w, rows, key_w, fc, key_color=TEAL):
        top, bottom, gap = y4 + h4 - TOP + 1, y4 + 8, 2.5
        rh = (top - bottom - gap * (len(rows) - 1)) / len(rows)
        for i, (k, v) in enumerate(rows):
            yt = top - i * (rh + gap)
            rbox(x + 8, yt - rh, w - 16, rh, fc=fc, r=3)
            ax.text(x + 14, yt - rh / 2, k, fontsize=pt(CAP), fontweight="bold", color=key_color, va="center")
            ax.text(x + 8 + key_w, yt - rh / 2, v, fontsize=pt(CAP), color=INK, va="center")

    card(M, y4, wc, h4, "7", "Conclusion", "결론: 예측은 작년 값, 설명은 시장 피처", owner="이예서 · 김상호", fc=TINT)
    kv_rows(M, wc, [("예측", "작년 값이 가장 강한 신호"),
                    ("전세가율", f"같은 구 안에서도 위험 ({jr.coef_per_sd:+.2f}, CI {jr.ci_lo:.2f}~{jr.ci_hi:.2f})"),
                    ("전셋값 하락", "주로 구 사이의 차이"),
                    ("신축 비중", "4개 구에서 본 탐색 결과")], key_w=58, fc="white", key_color=NAVY)
    xb = M + wc + G
    card(xb, y4, wl, h4, "8", "Limits · Next", "한계 · 다음", owner="이예서")
    kv_rows(xb, wl, [("타깃", "대리 지표 (ρ 0.61~0.73)"), ("피처", "빌라 시장만"),
                     ("2023", "절차 변경 혼입"), ("14주", "6 대 6 구 · 빌라 동")], key_w=38, fc=BOX)
    xq = xb + wl + G
    card(xq, y4, wq, h4, "9", "Code", "코드", owner="김상호")
    qs = min(h4 - TOP - 6, wq - 16)
    ax.imshow(qrcode.make(REPO, border=1).convert("RGB"),
              extent=(xq + (wq - qs) / 2, xq + (wq + qs) / 2, y4 + 7, y4 + 7 + qs), zorder=3)

    if sketch:
        ax.text(M, 4, "읽는 순서 1→9  ·  3초: 제목  ·  30초: 그림 A·B·C 캡션  ·  3분: 상자 1~3, 7~9  ·  담당은 보고서 역할 분담 기준 · "
                      "제목 띠: 박채영 · 최종 레이아웃 확인: 김상호", fontsize=pt(5.5), color=OWN, va="bottom")
    else:
        ax.text(M, M + 5, "자료: 국토교통부 실거래가 공개 API (공공데이터포털) · 법원 등기정보광장 임차권등기명령 · "
                          "주택도시보증공사 지역별 전세금반환보증 사고현황 (공공데이터포털 15134221)",
                fontsize=pt(5.0), color=SLATE, va="top")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sketch", action="store_true", help="A4 스케치 (담당 표시)")
    args = ap.parse_args()
    out = REPO_ROOT / "materials" / "poster" / ("poster_sketch.pdf" if args.sketch else "poster_draft.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    build(args.sketch).savefig(out)
    print("saved", out)


if __name__ == "__main__":
    main()
