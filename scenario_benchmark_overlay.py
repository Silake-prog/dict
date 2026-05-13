"""
scenario_benchmark_overlay.py
-----------------------------
Overlay published EU-27 H2-demand scenarios (TYNDP, EC, JRC, EHB, Deloitte/CHJU,
Aurora, BP, DNV, McKinsey, IFS, CAN, …) on top of the DemandForge 4-panel stacked
area plot so that each scenario bundle can be read against external benchmarks.

Usage
-----
Drop this file in your notebooks/ folder (same level as how_to_use_demandforge.ipynb)
and call it from a new cell after `all_scenarios` has been built:

    from scenario_benchmark_overlay import plot_sector_stacked_with_benchmarks
    plot_sector_stacked_with_benchmarks(
        all_scenarios, SCENARIO_COLOURS, SECTOR_COLOURS,
        ordered_sectors, _style_ax,
        csv_path="h2_demand_forecasts_2023.csv",   # path to the benchmark CSV
        out_dir=OUT_DIR,
    )

Scientific rigour notes
-----------------------
1. **Scope.**  DemandForge projects six *industrial* H2 end-uses
   (steel, refinery, ammonia, maritime, olefins, e-SAF).  The benchmark
   CSV bundles Industry / Transport / Building / Power.  A strict apples-
   to-apples comparison uses the CSV's `Industry` column ONLY; the
   "Total (full-economy)" envelope is plotted in a secondary shade purely
   as context.
2. **TYNDP highlighting.**  TYNDP 2022 ENTSOG "Distributed" and "Global
   Ambition" are plotted as discrete markers with distinct symbols.
   TYNDP 2024 (Draft Scenarios Report, May 2024 and IGI Report, Jan 2025)
   is not in the CSV; hard-coded EU-27 total H2-demand anchor points from
   the IGI NT+ and the DE/GA supply figures are therefore added as
   supplementary markers.
3. **Envelope construction.**  For each horizon year (2030/2040/2050) the
   literature is summarised by its median and [p25, p75] interquartile
   range (IQR) across *all* published forecasts in the CSV.  IQR is
   preferred over min/max because a few very ambitious scenarios (e.g.
   EC-Hydrogen 2018 at 4818 TWh) skew the envelope and are not
   representative of current policy ambition.
4. **Interpolation.**  Between benchmark horizons the envelope is drawn
   as straight dashed segments rather than shaded because intermediate
   values are NOT part of the published data and would misrepresent the
   source studies as continuous.

Author : S. Brigode, DemandForge, 2025
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------- #
# TYNDP 2024 anchor points (NOT in the CSV; hard-coded from primary sources)  #
# --------------------------------------------------------------------------- #
# Source (authoritative final demand, EU-27, net calorific value):
#   ENTSOG TYNDP 2024 Infrastructure Gaps Identification Report (Jan 2025),
#   Tables 23 & 24 — PCI/PMI hydrogen infrastructure level, reference weather
#   year. NT+ scenario.
#
# Important caveats:
#   * The IGI report's "Europe" aggregate includes the UK. The values below
#     use Table 24 summed over EU-27 member states ONLY, to match the CSV.
#   * NT+ is only modelled at 2030 and 2040; the IGI does NOT provide a
#     2050 horizon. For 2050 TYNDP comparison we therefore rely exclusively
#     on the TYNDP 2022 DE/GA demand figures already in the CSV.
#   * The TYNDP 2024 Draft Scenarios Report §6.4.6 reports 2050 *supply*
#     totals for DE (2,359 TWh) and GA (3,064 TWh), but these include the
#     H2 consumed for P2M/P2L conversion and are NOT directly comparable
#     to final sectoral demand as reported in the CSV. They are therefore
#     intentionally NOT plotted as demand anchors — doing so would overstate
#     the TYNDP 2024 demand projection by 20-30%. If you have the TYNDP 2024
#     Figure 10 Excel download, set `tyndp2024_2050_DE` / `..._GA` below.
#
# References:
#   - Table 23 (Europe total):     2030 = 620 TWh, 2040 = 1,929 TWh
#   - Table 24 (EU-27 only):       2040 = 1,843 TWh  (sum of 27 MS)
#   - Scaled 2030 EU-27:          ~592 TWh          (1,843 / (1,929/620))
# --------------------------------------------------------------------------- #
TYNDP2024_ANCHORS = {
    # (year, scenario_label): total_H2_demand_TWh   |  scope = EU-27
    (2030, "TYNDP 2024 NT+ (IGI)"):   592,    # derived from Table 24 EU-27 scaling
    (2040, "TYNDP 2024 NT+ (IGI)"):  1843,    # Table 24 summed over EU-27 MS
}

# Optional: if the user has the TYNDP 2024 DE/GA 2050 *demand* figures
# (Figure 10 data, Excel download), they can be added here manually.
# Left empty by default because the publicly-documented 2050 numbers are
# supply-side and would mislead the reader.
TYNDP2024_2050_USER_ANCHORS: dict[tuple[int, str], float] = {
    # Example once verified from the Excel:
    # (2050, "TYNDP 2024 Distributed Energy"): 1800,
    # (2050, "TYNDP 2024 Global Ambition"):    2200,
}

# Visual styles for TYNDP series (used for both the CSV 2022 rows and the 2024 anchors)
_TYNDP_STYLES = {
    "ENTSOG - Distributed (2022)":      dict(marker="s", ms=9,  mec="#1d3557", mfc="#a8dadc", lw=0, zorder=6),
    "ENTSOG - Global Ambition (2022)":  dict(marker="D", ms=9,  mec="#1d3557", mfc="#f1faee", lw=0, zorder=6),
    "TYNDP 2024 NT+ (IGI)":             dict(marker="^", ms=10, mec="#000000", mfc="#ffba08", lw=0, zorder=7),
    "TYNDP 2024 Distributed Energy":    dict(marker="s", ms=10, mec="#000000", mfc="#ffba08", lw=0, zorder=7),
    "TYNDP 2024 Global Ambition":       dict(marker="D", ms=10, mec="#000000", mfc="#ffba08", lw=0, zorder=7),
}


# --------------------------------------------------------------------------- #
# 1. Benchmark-CSV parsing                                                    #
# --------------------------------------------------------------------------- #
def _load_benchmarks(csv_path: str | Path) -> pd.DataFrame:
    """Load the EU-27 H2 demand forecast CSV and tidy column names."""
    df = pd.read_csv(csv_path)
    df = df.rename(columns={
        "Demand forecast":            "scenario",
        "TOTAL DEMAND (in TWh/year)": "total",
        "Year":                       "year",
        "Industry":                   "industry",
        "Transport":                  "transport",
        "Building":                   "building",
        "Power":                      "power",
    })
    # Scenarios that only report a single sector (e.g. EC-Baseline 2018 reports
    # only Transport) shouldn't pollute the "industry" envelope.  Drop rows
    # where Industry == 0 AND total > 0 (i.e. study scope is not industrial).
    df = df.copy()
    df["industry_valid"] = ~((df["industry"] == 0) & (df["total"] > 0))
    return df


def _envelope(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return median / p25 / p75 / min / max per horizon year for a column."""
    g = df.groupby("year")[col]
    return pd.DataFrame({
        "median": g.median(),
        "p25":    g.quantile(0.25),
        "p75":    g.quantile(0.75),
        "min":    g.min(),
        "max":    g.max(),
    })


# --------------------------------------------------------------------------- #
# 2. Overlay helper: plot benchmarks on one Axes                              #
# --------------------------------------------------------------------------- #
def _overlay_benchmarks(
    ax: plt.Axes,
    env_industry: pd.DataFrame,
    env_total: pd.DataFrame,
    tyndp_rows: pd.DataFrame,
    show_total_envelope: bool = True,
    show_tyndp2024: bool = True,
) -> list[Line2D]:
    """Draw literature envelope + TYNDP markers on a single Axes.

    Returns the list of proxy legend handles that need to be added.
    """
    handles: list[Line2D] = []

    # --- Industry-only envelope (primary, comparable to DemandForge) --------
    x = env_industry.index.to_numpy(dtype=float)
    ax.fill_between(x, env_industry["p25"], env_industry["p75"],
                    color="#555555", alpha=0.18, zorder=2,
                    label="_Literature IQR (industry)")
    ax.plot(x, env_industry["median"], color="#222222", lw=1.6,
            ls=(0, (4, 2)), zorder=3, label="_Literature median (industry)")

    handles += [
        plt.Rectangle((0, 0), 1, 1, fc="#555555", alpha=0.18,
                      label="Literature IQR — Industry column"),
        Line2D([], [], color="#222222", lw=1.6, ls=(0, (4, 2)),
               label="Literature median — Industry column"),
    ]

    # --- Full-economy envelope (context only) -------------------------------
    if show_total_envelope:
        ax.fill_between(x, env_total["p25"], env_total["p75"],
                        color="#9d4edd", alpha=0.08, zorder=1,
                        label="_Literature IQR (total)")
        ax.plot(x, env_total["median"], color="#9d4edd", lw=1.1,
                ls=":", zorder=2, label="_Literature median (total)")
        handles += [
            plt.Rectangle((0, 0), 1, 1, fc="#9d4edd", alpha=0.15,
                          label="Literature IQR — Total (all sectors)"),
            Line2D([], [], color="#9d4edd", lw=1.1, ls=":",
                   label="Literature median — Total (all sectors)"),
        ]

    # --- Discrete TYNDP 2022 markers (from the CSV) -------------------------
    seen_labels: set[str] = set()
    for _, row in tyndp_rows.iterrows():
        style = _TYNDP_STYLES.get(row["scenario"])
        if style is None:
            continue
        ax.plot(row["year"], row["industry"], **style)
        if row["scenario"] not in seen_labels:
            handles.append(Line2D([], [],
                                  label=f"{row['scenario']} (Industry)",
                                  **style))
            seen_labels.add(row["scenario"])

    # --- TYNDP 2024 anchor points (EU-27, total final H2 demand) ----------
    if show_tyndp2024:
        anchors = {**TYNDP2024_ANCHORS, **TYNDP2024_2050_USER_ANCHORS}
        for (yr, label), value in anchors.items():
            style = _TYNDP_STYLES.get(label, dict(marker="*", ms=11,
                                                  mec="k", mfc="#ffba08", lw=0))
            ax.plot(yr, value, **style)
            if label not in seen_labels:
                handles.append(Line2D([], [], label=f"{label} (EU-27 total)",
                                       **style))
                seen_labels.add(label)

    return handles


# --------------------------------------------------------------------------- #
# 3. Main plotting function                                                   #
# --------------------------------------------------------------------------- #
def plot_sector_stacked_with_benchmarks(
    all_scenarios: dict[str, pd.DataFrame],
    scenario_colours: dict[str, str],
    sector_colours:   dict[str, str],
    ordered_sectors:  list[str],
    style_ax_func,
    *,
    csv_path: str | Path = "h2_demand_forecasts_2023.csv",
    out_dir:  str | Path = ".",
    show_total_envelope: bool = True,
    show_tyndp2024: bool = True,
    filename: str = "sector_stacked_all_bundles_with_benchmarks.png",
) -> plt.Figure:
    """
    Re-draw the 4-panel stacked-area chart of EU-27 H2 demand by sector for
    each DemandForge scenario bundle, AND overlay published scenarios so
    the projections can be read against external benchmarks — with
    special treatment of the TYNDP series.

    Parameters
    ----------
    all_scenarios : dict
        Output of `{b: load_bundle(b) for b in list_bundles()}`.
    scenario_colours, sector_colours, ordered_sectors, style_ax_func :
        Variables / helper from the notebook (pass them in unchanged so the
        aesthetics match the rest of the document).
    csv_path : str | Path
        Path to `h2_demand_forecasts_2023.csv`.
    out_dir : str | Path
        Directory in which to save the figure.
    show_total_envelope : bool
        If True, add a secondary (purple) envelope for the full-economy
        H2 demand (industry + transport + buildings + power). Useful to
        remind the reader that DemandForge scope ≠ full-economy scope.
    show_tyndp2024 : bool
        If True, add TYNDP 2024 NT+ / DE / GA anchor markers (§6.4.6 and
        IGI Table 23 — hard-coded because TYNDP 2024 is not in the CSV).
    filename : str
        Output PNG filename.

    Returns
    -------
    matplotlib.figure.Figure
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Parse benchmark CSV ---------------------------------------------
    df_b = _load_benchmarks(csv_path)
    env_industry = _envelope(df_b[df_b["industry_valid"]], "industry")
    env_total    = _envelope(df_b, "total")
    tyndp_rows   = df_b[df_b["scenario"].isin(_TYNDP_STYLES.keys())]

    # ── 2. Build the 4-panel figure ---------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharey=True)
    legend_handles: list[Line2D] = []

    for ax, (bname, _colour) in zip(axes.flatten(), scenario_colours.items()):
        df_s = all_scenarios[bname]
        pivot = df_s.pivot_table(
            index="year", columns="sector",
            values="h2_demand_mwh_per_yr", aggfunc="sum", fill_value=0,
        ) / 1e6   # MWh -> TWh
        pivot = pivot[[s for s in ordered_sectors if s in pivot.columns]]

        # (a) DemandForge stacked area
        ax.stackplot(
            pivot.index, *[pivot[s] for s in pivot.columns],
            labels=[s.capitalize() for s in pivot.columns],
            colors=[sector_colours[s] for s in pivot.columns],
            alpha=0.85,
        )

        # (b) Benchmark overlay
        h = _overlay_benchmarks(
            ax, env_industry, env_total, tyndp_rows,
            show_total_envelope=show_total_envelope,
            show_tyndp2024=show_tyndp2024,
        )
        if not legend_handles:   # capture once — identical across panels
            legend_handles = h

        # (c) Total-demand line for this DemandForge bundle (for direct
        #     comparison with benchmark Industry envelope)
        total = pivot.sum(axis=1)
        ax.plot(total.index, total.values,
                color="black", lw=2.0, ls="-", zorder=5,
                label="_DemandForge total (industry)")

        # (d) Cosmetics
        style_ax_func(ax, bname, "TWh / yr")
        ax.set_xlim(2019, 2050)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{v:,.0f}"))

    # ── 3. Compose the legend ---------------------------------------------
    # Sector legend (from one of the stackplot Axes)
    sec_handles, sec_labels = axes[0, 0].get_legend_handles_labels()
    # Keep only sector legend entries (exclude the '_'-prefixed private ones)
    sec_items = [(h, l) for h, l in zip(sec_handles, sec_labels)
                 if not l.startswith("_")]
    sec_handles, sec_labels = zip(*sec_items) if sec_items else ([], [])

    # Add black DemandForge-total proxy line
    df_line = Line2D([], [], color="black", lw=2.0, ls="-",
                     label="DemandForge total (sum of 6 industrial sectors)")

    all_handles = list(sec_handles) + [df_line] + legend_handles
    all_labels  = list(sec_labels)  + [df_line.get_label()] \
                  + [h.get_label() for h in legend_handles]

    fig.legend(all_handles, all_labels,
               loc="lower center", ncol=3,
               fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle(
        "EU-27 H₂ demand by sector — DemandForge bundles vs. published scenarios\n"
        "(Industry envelope = strict scope match; Total envelope = full-economy context; "
        "TYNDP 2022 & 2024 anchors highlighted)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[saved] {out_path}")
    return fig


# --------------------------------------------------------------------------- #
# 4. Convenience: horizon-only benchmark summary (single-panel)               #
# --------------------------------------------------------------------------- #
def plot_horizon_comparison(
    all_scenarios: dict[str, pd.DataFrame],
    scenario_colours: dict[str, str],
    *,
    csv_path: str | Path = "h2_demand_forecasts_2023.csv",
    out_dir:  str | Path = ".",
    filename: str = "horizon_benchmark_compare.png",
) -> plt.Figure:
    """
    Companion view: DemandForge total (all 6 sectors) for each bundle at 2030,
    2040, 2050, plotted as thick lines against the full distribution of
    Industry-column benchmarks (boxplot) and TYNDP 2024 anchors (stars).
    """
    out_dir = Path(out_dir)
    df_b = _load_benchmarks(csv_path)
    ind_df = df_b[df_b["industry_valid"]]

    horizons = [2030, 2040, 2050]
    fig, ax = plt.subplots(figsize=(11, 6))

    # Boxplot of the literature Industry column per horizon
    bp_data = [ind_df[ind_df["year"] == y]["industry"].values for y in horizons]
    ax.boxplot(
        bp_data, positions=horizons, widths=3.5,
        patch_artist=True,
        boxprops=dict(facecolor="#eaeaea", color="#444"),
        medianprops=dict(color="#222", lw=1.6),
        whiskerprops=dict(color="#444"),
        capprops=dict(color="#444"),
        flierprops=dict(marker="x", mec="#999", ms=4),
        showmeans=False,
        manage_ticks=False,
        zorder=1,
    )

    # DemandForge totals per bundle
    for bname, colour in scenario_colours.items():
        dfs = all_scenarios[bname]
        totals = (dfs.groupby("year")["h2_demand_mwh_per_yr"].sum() / 1e6)
        y_vals = [totals.get(y, np.nan) for y in horizons]
        ax.plot(horizons, y_vals, "-o", color=colour, lw=2.2, ms=8,
                label=bname, zorder=4)

    # TYNDP 2022 (Industry column) markers
    for scen in ["ENTSOG - Distributed (2022)", "ENTSOG - Global Ambition (2022)"]:
        sub = df_b[df_b["scenario"] == scen]
        ax.plot(sub["year"], sub["industry"],
                **{**_TYNDP_STYLES[scen], "label": f"{scen} (Industry)"})

    # TYNDP 2024 triangles (NT+ final demand, EU-27)
    anchors = {**TYNDP2024_ANCHORS, **TYNDP2024_2050_USER_ANCHORS}
    for (yr, label), value in anchors.items():
        ax.plot(yr, value, **{**_TYNDP_STYLES[label],
                              "label": f"{label} (EU-27 total)"})

    ax.set_xticks(horizons)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("EU-27 H₂ demand (TWh / yr)", fontsize=10)
    ax.set_title(
        "DemandForge vs. benchmark distribution at 2030 / 2040 / 2050\n"
        "Grey box = IQR of published Industry column; whiskers = 1.5·IQR",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    # Dedupe legend
    hs, ls = ax.get_legend_handles_labels()
    seen, hh, ll = set(), [], []
    for h, l in zip(hs, ls):
        if l in seen:
            continue
        seen.add(l)
        hh.append(h); ll.append(l)
    ax.legend(hh, ll, fontsize=8, frameon=False,
              loc="upper left", ncol=1)

    fig.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[saved] {out_path}")
    return fig
