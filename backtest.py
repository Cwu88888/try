"""
标普500 定投策略回测
====================

对比几种"每月投一点钱"的方式，在标普500历史数据上的表现：

  A. 普通定投       每月固定投 50，什么都不管
  B. 跌多了多投     平时投 50；指数比历史高点跌超 20% 时，当月投 100（额外多掏钱）
  C. 留子弹抄底     每月预算同样是 50：平时投 40、存 10 现金；
                    指数跌超 20% 时，把攒的现金一次性全投进去（总预算和 A 完全一样）
  D. 均线择时       每月预算 50：指数在 10 个月均线之上才投（连同之前攒的现金一起投），
                    跌破均线就只存现金不投（经典的"趋势跟踪"策略）

数据：Robert Shiller 的标普500月度数据（含分红），来自
      https://github.com/datasets/s-and-p-500
      价格是月度平均值；分红全部再投资（即总收益）。
      2023-07 之后数据源的分红是空的，用最后已知的股息率补齐。

假设：现金不计利息（对 C、D 略不利）；无交易费用、无税、无汇率；
      每月按当月价格成交。

用法：python3 backtest.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "sp500_shiller_monthly.csv"
OUT = ROOT / "output"

MONTHLY = 50.0          # 每月预算
DRAWDOWN = 0.20         # "跌多了"的阈值：比历史最高点低 20%
MA_MONTHS = 10          # 均线长度（10 个月 ≈ 200 个交易日）


# ---------------------------------------------------------------- 数据
def load_total_return() -> pd.DataFrame:
    d = pd.read_csv(DATA, parse_dates=["Date"])[["Date", "SP500", "Dividend"]]
    d = d[d.Date >= "1900-01-01"].reset_index(drop=True)

    # 分红缺失（为 0）的月份：用最后已知的股息率补齐
    yld = d.Dividend / d.SP500
    yld = yld.where(d.Dividend > 0).ffill()
    d["div_yield"] = yld

    # 总收益指数：价格涨跌 + 每月 1/12 的年化股息再投资
    ret = d.SP500.pct_change().fillna(0) + d.div_yield.shift(1).fillna(0) / 12
    d["tr"] = (1 + ret).cumprod()

    d["peak"] = d.SP500.cummax()
    d["dd"] = d.SP500 / d.peak - 1                       # 距历史高点的回撤
    d["ma"] = d.SP500.rolling(MA_MONTHS).mean()
    return d.set_index("Date")


# ---------------------------------------------------------------- 策略
def run(df: pd.DataFrame, strategy: str):
    """返回 (逐月资产曲线 DataFrame, 现金流列表[负=投入, 最后为正=期末市值])"""
    shares = cash = invested = 0.0
    flows, rows = [], []
    for _, r in df.iterrows():
        price = r.tr
        crash = r.dd <= -DRAWDOWN
        put_in = MONTHLY                      # 本月从口袋里掏的钱
        buy = 0.0                             # 本月实际买入股票的钱

        if strategy == "A":
            buy = MONTHLY
        elif strategy == "B":
            put_in = MONTHLY * 2 if crash else MONTHLY
            buy = put_in
        elif strategy == "C":
            if crash:
                buy = MONTHLY + cash
                cash = 0.0
            else:
                buy = MONTHLY * 0.8
                cash += MONTHLY * 0.2
        elif strategy == "D":
            cash += MONTHLY
            if pd.notna(r.ma) and r.SP500 >= r.ma:
                buy, cash = cash, 0.0

        shares += buy / price
        invested += put_in
        flows.append(-put_in)
        rows.append((r.name, invested, shares * price + cash))

    curve = pd.DataFrame(rows, columns=["Date", "invested", "value"]).set_index("Date")
    flows[-1] += curve.value.iloc[-1]
    return curve, flows


def irr_annual(flows) -> float:
    """按月现金流求年化内部收益率（二分法）。"""
    flows = np.asarray(flows)
    t = np.arange(len(flows))

    def npv(m):
        return np.sum(flows / (1 + m) ** t)

    lo, hi = -0.05, 0.05
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (1 + mid) ** 12 - 1


def max_drawdown(curve: pd.DataFrame) -> float:
    v = curve.value
    return float((v / v.cummax() - 1).min())


COLORS = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green", "D": "tab:red"}

NAMES = {
    "A": "A 普通定投",
    "B": "B 跌多了多投",
    "C": "C 留子弹抄底",
    "D": "D 均线择时",
}


def summarize(df: pd.DataFrame, label: str):
    print(f"\n=== {label}: {df.index[0]:%Y-%m} ~ {df.index[-1]:%Y-%m}"
          f"（{len(df) // 12} 年） ===")
    print(f"{'策略':<10}{'累计投入':>10}{'期末市值':>12}{'翻了几倍':>8}"
          f"{'年化IRR':>9}{'最大回撤':>9}")
    curves = {}
    for k, name in NAMES.items():
        c, f = run(df, k)
        curves[k] = c
        inv, val = c.invested.iloc[-1], c.value.iloc[-1]
        print(f"{name:<10}{inv:>12,.0f}{val:>14,.0f}{val / inv:>10.2f}x"
              f"{irr_annual(f):>10.2%}{max_drawdown(c):>10.1%}")
    return curves


def rolling_windows(d: pd.DataFrame, years=20, start="1950-01-01"):
    """所有可能的 N 年定投窗口（每年起点一个），统计各策略 IRR 跑赢普通定投的比例。"""
    n = years * 12
    sub = d[d.index >= start]
    starts = range(0, len(sub) - n + 1, 12)
    res = {k: [] for k in NAMES}
    for s in starts:
        w = sub.iloc[s: s + n]
        for k in NAMES:
            res[k].append(irr_annual(run(w, k)[1]))
    res = pd.DataFrame(res)
    print(f"\n=== 滚动回测：1950 年以来所有 {years} 年定投窗口（共 {len(res)} 个）===")
    print(f"{'策略':<10}{'IRR中位数':>10}{'最差':>9}{'最好':>9}{'跑赢A的比例':>12}{'平均差距':>10}")
    for k, name in NAMES.items():
        diff = res[k] - res["A"]
        win = (diff > 1e-6).mean() if k != "A" else float("nan")
        win_s = "-" if k == "A" else f"{win:.0%}"
        print(f"{name:<10}{res[k].median():>11.2%}{res[k].min():>10.2%}"
              f"{res[k].max():>10.2%}{win_s:>12}{diff.mean():>+11.2%}")
    return res


# ---------------------------------------------------------------- 主程序
def main():
    OUT.mkdir(exist_ok=True)
    d = load_total_return()
    end = d.index[-1]

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei",
                                       "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    windows = {20: None, 30: None}
    for yrs in windows:
        w = d[d.index > end - pd.DateOffset(years=yrs)]
        windows[yrs] = summarize(w, f"最近 {yrs} 年")

    roll = rolling_windows(d, years=20)

    # ---- 图：最近 20 年资产曲线 + 滚动窗口 IRR 差距
    fig, axes = plt.subplots(2, 1, figsize=(10, 9))
    ax = axes[0]
    for k, c in windows[20].items():
        ax.plot(c.index, c.value, label=NAMES[k], lw=1.6, color=COLORS[k])
    ax.plot(windows[20]["A"].index, windows[20]["A"].invested, "k--", lw=1,
            label="A 累计投入本金")
    ax.set_title("最近 20 年：每月 50 定投标普500（含分红再投资）")
    ax.set_ylabel("账户市值")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    years = [d[d.index >= "1950-01-01"].index[i * 12].year for i in range(len(roll))]
    for k in ["B", "C", "D"]:
        ax.plot(years, (roll[k] - roll["A"]) * 100, marker="o", ms=3, label=NAMES[k],
                color=COLORS[k])
    ax.axhline(0, color="k", lw=1)
    ax.set_title("每个 20 年窗口里，各策略年化收益比普通定投高/低多少")
    ax.set_xlabel("开始定投的年份")
    ax.set_ylabel("年化 IRR 差距（百分点）")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "backtest.png", dpi=120)
    print(f"\n图已保存：{OUT / 'backtest.png'}")


if __name__ == "__main__":
    main()
