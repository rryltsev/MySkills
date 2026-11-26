"""Plot the archived test predictions and subgroup RMSE from compare_models.py."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter
import numpy as np


MODELS = {
    "gp1": ("GPVisc № 1", "#4c5058", "o", (0, (3, 2))),
    "gp_mean": ("Среднее трёх GP", "#0072a3", "^", (0, (6, 2))),
    "curve": ("Модель MYEGA", "#b04b0c", "s", (0, (1, 1.6))),
    "blend": ("GP и MYEGA, 50/50", "#08795b", "D", "-"),
}
SUBSETS = {
    "all": "Весь тест",
    "curated": "Ручная подборка",
    "sciglass": "SciGlass",
    "CMAS_proxy": "Составы по критерию CMAS",
    "slag_in_source_title": "Источники со словом slag¹",
    "Iida_2000": "Iida и соавт., 2000",
}


def render(result_dir):
    with (result_dir / "test_comparison.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    metrics = json.loads((result_dir / "comparison.json").read_text(encoding="utf-8"))["metrics"]
    observed = np.array([float(row["log10_eta"]) for row in rows])
    assert len(rows) == metrics["all"]["gp1"]["n"]
    errors = {name: np.abs([float(row[name]) - target for row, target in zip(rows, observed)]) for name in MODELS}
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
        "axes.spines.right": False, "axes.labelcolor": "#293441",
        "axes.edgecolor": "#8d98a1", "xtick.color": "#424b55", "ytick.color": "#424b55",
        "axes.titleweight": "bold", "axes.titlesize": 14, "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(13.2, 6.3))
    grid = fig.add_gridspec(1, 2, left=0.075, right=0.974, bottom=0.238, top=0.748,
                            width_ratios=(1.12, 1), wspace=0.72)
    cdf_ax = fig.add_subplot(grid[0, 0])
    subset_ax = fig.add_subplot(grid[0, 1])
    fig.text(0.075, 0.948, "Прогноз вязкости на тестовой выборке GPVisc",
             size=21, color="#17293a", weight="bold", va="center")
    handles = [Line2D([], [], label=label, color=color, marker=marker, linestyle=style,
                      lw=2, ms=5) for label, color, marker, style in MODELS.values()]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.074, 0.894),
               ncol=4, frameon=False, handlelength=2.6, columnspacing=2.1, borderaxespad=0)

    cdf_ax.set_title("Доля измерений в пределах ошибки", loc="left", pad=15)
    cdf_ax.set_xscale("symlog", linthresh=0.10, linscale=0.9)
    cdf_ax.set_xlim(0, max(e.max() for e in errors.values()) * 1.04)
    ticks = [n for n in (0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20) if n < cdf_ax.get_xlim()[1]]
    cdf_ax.set_xticks(ticks, [f"{n:g}".replace(".", ",") for n in ticks])
    cdf_ax.xaxis.set_minor_formatter(NullFormatter())
    cdf_ax.set_ylim(0, 101.5)
    cdf_ax.set_yticks(range(0, 101, 20))
    cdf_ax.set_ylabel("Доля тестовых точек, %")
    cdf_ax.set_xlabel("Абсолютная ошибка, dex", labelpad=8)
    cdf_ax.grid(axis="both", color="#dbe1e7", lw=0.7)
    quantiles = 100 * np.arange(1, len(rows) + 1) / len(rows)
    for name, (_, color, marker, style) in MODELS.items():
        sorted_error = np.sort(errors[name])
        cdf_ax.step(np.r_[0, sorted_error], np.r_[0, quantiles], where="post",
                    color=color, linestyle=style, linewidth=1.8, alpha=0.95)

    subset_ax.set_title("Среднеквадратичная ошибка", loc="left", pad=15)
    last_pos = len(SUBSETS) - 1
    subset_ax.axhspan(2.55, last_pos + 0.48, color="#edf5f2", zorder=-2)
    offsets = np.linspace(-0.195, 0.195, len(MODELS))
    group_rows = np.arange(len(SUBSETS))
    for offset, (name, (_, color, marker, _)) in zip(offsets, MODELS.items()):
        rmse = [metrics[k][name]["rmse"] for k in SUBSETS]
        subset_ax.scatter(rmse, group_rows + offset, s=35, color=color, marker=marker, zorder=3)
    subset_ax.set_ylim(last_pos + 0.58, -0.58)
    subset_ax.set_xlim(left=0)
    subset_ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    subset_ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    counts = [f"{metrics[k]['gp1']['n']:,}".replace(",", " ") for k in SUBSETS]
    count_labels = [f"{label}\nn = {n}" for label, n in zip(SUBSETS.values(), counts)]
    subset_ax.set_yticks(group_rows, count_labels)
    subset_ax.tick_params(axis="y", which="both", length=0, pad=9, labelsize=10)
    subset_ax.grid(axis="x", color="#dbe1e7", lw=0.7)
    subset_ax.set_xlabel("RMSE, dex", labelpad=8)
    for pos in (0.5, 1.5, 3.5, 4.5):
        subset_ax.axhline(pos, color="#dbe1e7", lw=0.65, zorder=-1)
    fig.text(0.075, 0.109, "Ошибки рассчитаны для log₁₀(η / Па·с). Подвыборки справа могут пересекаться.",
             color="#384856", fontsize=10)
    fig.text(0.075, 0.071, "Для выборки CMAS сумма CaO, MgO, Al₂O₃ и SiO₂ ≥ 85 масс. %, CaO/SiO₂ по массе от 0,8 до 1,5, 1473–1973 K, P ≤ 0,001 ГПа.",
             color="#384856", fontsize=10)
    fig.text(0.075, 0.033, "¹ Метка по словам slag или slags в библиографической записи источника.",
             color="#384856", fontsize=10)
    fig.savefig(result_dir / "benchmark.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    render(parser.parse_args().results)
