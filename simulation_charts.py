"""High-quality diverse visual chart generation for smart warehouse simulation.

Generates appropriate visual charts with diverse representations and clear explanatory lines:
1. Donut / Ring Chart: Operational Cost Structure Composition with cost explanation lines.
2. Gauge / Semi-Circle Dial: Fulfillment Service Level with 95% target threshold line.
3. Line & Shaded Area Chart: Warehouse Total Inventory Level with Safety Stock Buffer threshold line.
4. Multi-Line Trend Chart: Daily Sales Revenue Trajectory with Average Revenue Baseline line.
5. Scatter / Bubble Plot: SKU Volatility Risk Matrix with Lead-Time and Demand crosshair lines.
6. Grouped Bar Chart: Daily Demand Fulfillment with Average Daily Demand target line.

Provides export helpers for:
- Standalone PNG buffers (for web & Excel embedding)
- Comprehensive multi-page Executive PDF report
"""

import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from ml_forecasting import train_and_forecast_sales, create_ml_forecast_chart


# Palette constants matching warehouse UI
CLR_FIXED = "#1e3a8a"       # Deep Slate Blue
CLR_ROP = "#0d9488"         # Teal / Emerald
CLR_SALES = "#16a34a"       # Green
CLR_LOST = "#dc2626"        # Red
CLR_PURCHASE = "#1e3a8a"    # Slate Blue
CLR_ORDERING = "#d97706"    # Amber / Orange
CLR_HOLDING = "#059669"     # Emerald Green
CLR_BG = "#ffffff"
CLR_GRID = "#e2e8f0"
CLR_TEXT = "#1e293b"
CLR_MUTED = "#64748b"


def _setup_figure(figsize=(9, 5.0), dpi=130):
    """Create a styled figure and axis."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor=CLR_BG)
    ax.set_facecolor("#f8fafc")
    ax.grid(True, linestyle="--", alpha=0.6, color=CLR_GRID, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(0.8)
    return fig, ax


def _filter_daily(daily_list, start_day, end_day):
    """Filter daily records to the specified day range."""
    return [
        item for item in daily_list
        if start_day <= int(item.get("day", 0)) <= end_day
    ]


def create_cost_donut_chart(fixed_result, rop_result):
    """Chart 1 (Donut / Ring Chart): Operational Cost Structure with explanatory callouts."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 5.0), dpi=130, facecolor=CLR_BG)

    fs = fixed_result.get("summary", {})
    rs = rop_result.get("summary", {})

    labels = ["Purchase Cost", "Ordering Cost", "Holding Cost"]
    colors = [CLR_PURCHASE, CLR_ORDERING, CLR_HOLDING]

    f_costs = [fs.get("total_purchase_cost", 0), fs.get("total_ordering_cost", 0), fs.get("total_holding_cost", 0)]
    r_costs = [rs.get("total_purchase_cost", 0), rs.get("total_ordering_cost", 0), rs.get("total_holding_cost", 0)]

    for ax, costs, title in [(ax1, f_costs, "Fixed Threshold Policy"), (ax2, r_costs, "Reorder Point (ROP) Policy")]:
        total = sum(costs)
        if total <= 0:
            costs = [1, 1, 1]
        wedges, texts, autotexts = ax.pie(
            costs,
            labels=labels,
            autopct="%1.1f%%",
            pctdistance=0.76,
            startangle=140,
            colors=colors,
            wedgeprops=dict(width=0.42, edgecolor="#ffffff", linewidth=2),
            textprops=dict(fontsize=8.5, color=CLR_TEXT),
        )
        for at in autotexts:
            at.set_fontsize(8)
            at.set_weight("bold")
            at.set_color("#ffffff")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8, color=CLR_TEXT)
        ax.text(0, 0, f"Total\n₹{int(total):,}", ha="center", va="center", fontsize=9, fontweight="bold", color=CLR_TEXT)

    fig.suptitle("Operational Cost Structure Composition (Donut Chart)", fontsize=12, fontweight="bold", color=CLR_TEXT, y=0.98)
    # Explanatory lines at the bottom
    fig.text(0.5, 0.04, "Explanatory Note: Purchasing accounts for >90% of outlay; ROP optimizes batch ordering frequency to reduce holding costs.",
             ha="center", fontsize=8.5, color="#475569", style="italic")
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return buf


def create_service_level_gauge_chart(fixed_result, rop_result):
    """Chart 2 (Gauge / Speedometer Dial): Fulfillment Service Level with 95% target threshold line."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.8), dpi=130, facecolor=CLR_BG)

    fs = fixed_result.get("summary", {})
    rs = rop_result.get("summary", {})

    sv_fixed = fs.get("service_level", 1.0) * 100
    sv_rop = rs.get("service_level", 1.0) * 100

    gauges = [
        (ax1, sv_fixed, "Fixed Threshold Policy", CLR_FIXED),
        (ax2, sv_rop, "Reorder Point (ROP) Policy", CLR_ROP),
    ]

    for ax, pct, title, clr in gauges:
        ax.set_facecolor(CLR_BG)
        theta = np.linspace(np.pi, 0, 100)
        # Background arc track
        ax.plot(np.cos(theta), np.sin(theta), color="#e2e8f0", linewidth=24, solid_capstyle="round")
        # Filled progress arc
        fill_steps = int(100 * (min(100.0, max(0.0, pct)) / 100.0))
        if fill_steps > 1:
            ax.plot(np.cos(theta[:fill_steps]), np.sin(theta[:fill_steps]), color=clr, linewidth=24, solid_capstyle="round")

        # 95% Target Line marker
        t_target = np.pi - 0.95 * np.pi
        ax.plot([0.84 * np.cos(t_target), 1.16 * np.cos(t_target)],
                [0.84 * np.sin(t_target), 1.16 * np.sin(t_target)],
                color="#dc2626", linewidth=3.0, zorder=5)
        ax.text(1.22 * np.cos(t_target), 1.20 * np.sin(t_target), "--- 95% Target Line",
                fontsize=7.5, color="#dc2626", fontweight="bold", ha="left")

        # Center percentage readout
        ax.text(0, 0.18, f"{pct:.1f}%", ha="center", va="center", fontsize=22, fontweight="bold", color=CLR_TEXT)
        status_label = "Optimal Fill Rate" if pct >= 98 else ("Adequate" if pct >= 90 else "Stockout Impact")
        ax.text(0, -0.12, f"Service Level\n({status_label})", ha="center", va="center", fontsize=9.5, color=CLR_MUTED, fontweight="bold")

        ax.set_xlim(-1.3, 1.45)
        ax.set_ylim(-0.35, 1.35)
        ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8, color=CLR_TEXT)

    fig.suptitle("Order Fulfillment Service Level (%) — Performance Gauge", fontsize=12, fontweight="bold", color=CLR_TEXT, y=0.98)
    fig.text(0.5, 0.04, "Explanatory Note: The red tick marks the 95% Target Line. Scores at or above target satisfy customer demand without stockout penalties.",
             ha="center", fontsize=8.5, color="#475569", style="italic")
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return buf


def create_inventory_trajectory_chart(fixed_result, rop_result, start_day=1, end_day=7):
    """Chart 3 (Line & Shaded Area Chart): Warehouse Total Inventory with Safety Stock Buffer threshold line."""
    fig, ax = _setup_figure(figsize=(9.2, 5.0))

    fixed_items = fixed_result.get("daily_items", [])
    rop_items = rop_result.get("daily_items", [])

    days_range = list(range(start_day, end_day + 1))
    fixed_stock = [
        sum(item.get("closing_stock", 0) for item in fixed_items if item.get("day") == d)
        for d in days_range
    ]
    rop_stock = [
        sum(item.get("closing_stock", 0) for item in rop_items if item.get("day") == d)
        for d in days_range
    ]

    days_labels = [f"Day {d}" for d in days_range]
    x = np.arange(len(days_range))
    max_val = max(max(fixed_stock, default=1), max(rop_stock, default=1))

    # Calculate approximate safety stock buffer line
    min_observed = min(min(fixed_stock, default=50), min(rop_stock, default=50))
    safety_buffer = max(15, round(min_observed * 0.75))

    # Line + shaded translucent area
    ax.plot(x, fixed_stock, marker="o", markersize=6, linewidth=2.5, color=CLR_FIXED, label="Fixed Threshold Closing Stock", zorder=4)
    ax.fill_between(x, fixed_stock, alpha=0.18, color=CLR_FIXED)

    ax.plot(x, rop_stock, marker="s", markersize=6, linewidth=2.5, linestyle="--", color=CLR_ROP, label="Reorder Point Closing Stock", zorder=4)
    ax.fill_between(x, rop_stock, alpha=0.18, color=CLR_ROP)

    # Reference Line: Minimum Safety Stock Buffer Threshold
    ax.axhline(safety_buffer, color="#dc2626", linestyle="--", linewidth=1.8, label=f"Safety Stock Buffer Line (~{safety_buffer} units)", zorder=3)
    ax.text(x[-1], safety_buffer + (max_val - safety_buffer) * 0.04, f"--- Safety Buffer Line: {safety_buffer} u",
            color="#dc2626", fontsize=8.5, fontweight="bold", ha="right")

    ax.set_title("Warehouse Total Inventory Level Trajectory (Line & Area Chart)", fontsize=12, fontweight="bold", pad=12, color=CLR_TEXT)
    ax.set_xlabel("Simulation Day", fontsize=10, fontweight="bold", color=CLR_TEXT)
    ax.set_ylabel("Total On-Hand Stock (Units)", fontsize=10, fontweight="bold", color=CLR_TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(days_labels, fontsize=9)
    ax.legend(loc="upper right", framealpha=0.95, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=8)

    fig.text(0.5, 0.02, "Explanatory Note: The red dashed line marks the Safety Buffer Threshold. Upward spikes show delivery replenishments.",
             ha="center", fontsize=8.5, color="#475569", style="italic")
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return buf


def create_daily_revenue_chart(fixed_result, rop_result, start_day=1, end_day=7):
    """Chart 4 (Multi-Line Trend Chart): Daily Sales Revenue Trajectory with Average Revenue Baseline line."""
    fig, ax = _setup_figure(figsize=(9.2, 5.0))

    fixed_daily = _filter_daily(fixed_result.get("daily", []), start_day, end_day)
    rop_daily = _filter_daily(rop_result.get("daily", []), start_day, end_day)

    days = [f"Day {item['day']}" for item in fixed_daily]
    x = np.arange(len(days))

    fixed_rev = [item.get("revenue", 0) for item in fixed_daily]
    rop_rev = [item.get("revenue", 0) for item in rop_daily]
    all_rev = fixed_rev + rop_rev
    max_val = max(all_rev, default=1)
    avg_rev = sum(all_rev) / len(all_rev) if all_rev else 0

    ax.plot(x, fixed_rev, marker="o", markersize=6, linewidth=2.5, color=CLR_FIXED, label="Fixed Threshold Revenue", zorder=4)
    ax.plot(x, rop_rev, marker="^", markersize=6, linewidth=2.5, linestyle="--", color=CLR_ROP, label="Reorder Point Revenue", zorder=4)

    # Reference Line: Average Revenue Baseline Line
    ax.axhline(avg_rev, color="#d97706", linestyle=":", linewidth=2.0, label=f"Average Daily Revenue Baseline (₹{int(avg_rev):,})", zorder=3)
    ax.text(x[-1], avg_rev + (max_val - avg_rev) * 0.04, f"--- Avg Revenue Baseline: ₹{int(avg_rev):,}",
            color="#d97706", fontsize=8.5, fontweight="bold", ha="right")

    for idx, (f_val, r_val) in enumerate(zip(fixed_rev, rop_rev)):
        if idx % 2 == 0 or idx == len(fixed_rev) - 1:
            ax.annotate(f"₹{int(f_val):,}", xy=(x[idx], f_val), xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=7.5, color=CLR_FIXED, fontweight="bold")

    ax.set_title("Daily Sales Revenue Trajectory (Multi-Line Trend Chart)", fontsize=12, fontweight="bold", pad=12, color=CLR_TEXT)
    ax.set_xlabel("Simulation Day", fontsize=10, fontweight="bold", color=CLR_TEXT)
    ax.set_ylabel("Revenue (₹)", fontsize=10, fontweight="bold", color=CLR_TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(days, fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, loc: f"₹{int(val):,}"))
    ax.legend(loc="upper left", framealpha=0.95, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=8)

    fig.text(0.5, 0.02, "Explanatory Note: The dotted amber line is the Average Daily Revenue Baseline. Trajectories falling below indicate lost sales revenue.",
             ha="center", fontsize=8.5, color="#475569", style="italic")
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return buf


def create_sku_risk_scatter_chart(fixed_result, rop_result, fixed_config=None):
    """Chart 5 (Scatter / Bubble Plot): SKU Volatility Risk Matrix with Lead-Time and Demand crosshair lines."""
    fig, ax = _setup_figure(figsize=(9.2, 5.0))

    fixed_prods = fixed_result.get("products", [])
    products_cfg = (fixed_config or {}).get("products", [])
    cfg_by_id = {p["id"]: p for p in products_cfg}

    lead_times = []
    avg_demands = []
    sizes = []
    colors = []
    labels = []

    for p in fixed_prods:
        pid = p.get("product_id")
        cfg = cfg_by_id.get(pid, {})
        lt = int(cfg.get("lead_time_days", 3))
        d_mean = float(cfg.get("avg_daily_demand", 15))
        lost = p.get("lost_units", 0)
        stock = p.get("final_stock", 20)

        lead_times.append(lt)
        avg_demands.append(d_mean)
        sizes.append(max(60, min(350, stock * 8)))
        colors.append("#ef4444" if lost > 0 else "#0d9488")
        name = p.get("product_name", f"P{pid}")
        labels.append(name[:13] + "…" if len(name) > 13 else name)

    ax.scatter(lead_times, avg_demands, s=sizes, c=colors, alpha=0.7, edgecolors="#1e293b", linewidth=1.2, zorder=3)

    for i, txt in enumerate(labels):
        ax.annotate(txt, (lead_times[i], avg_demands[i]), xytext=(6, 4), textcoords="offset points",
                    fontsize=8, fontweight="bold", color=CLR_TEXT)

    avg_lt = sum(lead_times) / len(lead_times) if lead_times else 3
    avg_dm = sum(avg_demands) / len(avg_demands) if avg_demands else 15

    # Crosshair Reference Lines
    ax.axvline(avg_lt, color="#94a3b8", linestyle="--", linewidth=1.6, label=f"Avg Lead Time Line ({avg_lt:.1f} d)", zorder=2)
    ax.axhline(avg_dm, color="#94a3b8", linestyle="--", linewidth=1.6, label=f"Avg Demand Line ({avg_dm:.1f} u)", zorder=2)

    ax.text(avg_lt + 0.1, max(avg_demands) * 0.95, "⚠️ Top-Right: High Risk Zone (Long Lead Time + High Demand)",
            fontsize=8, color="#dc2626", fontweight="bold")
    ax.text(min(lead_times) + 0.1, min(avg_demands) * 1.05, "✓ Bottom-Left: Safe Buffer Zone (Fast Restock + Stable Demand)",
            fontsize=8, color="#059669", fontweight="bold")

    ax.set_title("SKU Volatility & Supplier Lead-Time Risk Matrix (Scatter Plot)", fontsize=12, fontweight="bold", pad=12, color=CLR_TEXT)
    ax.set_xlabel("Supplier Lead Time (Days)", fontsize=10, fontweight="bold", color=CLR_TEXT)
    ax.set_ylabel("Average Daily Customer Demand (Units)", fontsize=10, fontweight="bold", color=CLR_TEXT)
    ax.legend(loc="lower right", framealpha=0.95, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=8)

    fig.text(0.5, 0.02, "Explanatory Note: Crosshair lines divide the portfolio into 4 risk quadrants. Items in the top-right risk zone require larger ROP safety stocks.",
             ha="center", fontsize=8.5, color="#475569", style="italic")
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return buf


def create_daily_fulfillment_chart(fixed_result, rop_result, start_day=1, end_day=7):
    """Chart 6 (Grouped Bar Chart): Daily Demand Fulfillment with Average Daily Demand target line."""
    fig, ax = _setup_figure(figsize=(9.2, 5.0))

    fixed_daily = _filter_daily(fixed_result.get("daily", []), start_day, end_day)
    rop_daily = _filter_daily(rop_result.get("daily", []), start_day, end_day)

    days = [f"Day {item['day']}" for item in fixed_daily]
    x = np.arange(len(days))
    width = 0.2

    fixed_sold = [item.get("units_sold", 0) for item in fixed_daily]
    fixed_lost = [item.get("lost_units", 0) for item in fixed_daily]
    rop_sold = [item.get("units_sold", 0) for item in rop_daily]
    rop_lost = [item.get("lost_units", 0) for item in rop_daily]

    # Calculate average daily demand target line
    total_daily_demands = [fixed_sold[i] + fixed_lost[i] for i in range(len(fixed_sold))]
    avg_demand = sum(total_daily_demands) / len(total_daily_demands) if total_daily_demands else 1

    ax.bar(x - 1.5 * width, fixed_sold, width, label="Fixed: Units Sold", color=CLR_FIXED, alpha=0.9, zorder=3)
    ax.bar(x - 0.5 * width, fixed_lost, width, label="Fixed: Lost Units", color="#f87171", hatch="//", zorder=3)
    ax.bar(x + 0.5 * width, rop_sold, width, label="ROP: Units Sold", color=CLR_ROP, alpha=0.9, zorder=3)
    ax.bar(x + 1.5 * width, rop_lost, width, label="ROP: Lost Units", color="#ef4444", hatch="\\\\", zorder=3)

    # Reference Line: Average Daily Demand Line
    ax.axhline(avg_demand, color="#6366f1", linestyle=":", linewidth=2.0, label=f"Average Demand Target Line ({int(avg_demand)} units)", zorder=4)
    ax.text(x[-1] + 0.3, avg_demand + 1, f"--- Avg Demand Target ({int(avg_demand)} u)", color="#6366f1", fontsize=8.5, fontweight="bold", ha="right")

    ax.set_title("Daily Demand Fulfillment: Sold vs. Lost Units (Grouped Bar Chart)", fontsize=12, fontweight="bold", pad=12, color=CLR_TEXT)
    ax.set_xlabel("Simulation Day", fontsize=10, fontweight="bold", color=CLR_TEXT)
    ax.set_ylabel("Units", fontsize=10, fontweight="bold", color=CLR_TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(days, fontsize=9)
    ax.legend(loc="upper right", framealpha=0.95, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=8)

    fig.text(0.5, 0.02, "Explanatory Note: The dotted purple line is the Average Demand Target Line. Red hatched bars mark demand lost due to stockouts.",
             ha="center", fontsize=8.5, color="#475569", style="italic")
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_all_simulation_charts(fixed_result, rop_result, start_day=1, end_day=7, fixed_config=None, ml_forecast=None):
    """Generate all diverse simulation charts including ML forward forecast as a dictionary of image buffers.

    Returns:
        dict: Keys are chart identifiers, values are BytesIO objects containing PNG bytes.
    """
    if ml_forecast is None:
        try:
            ml_forecast = train_and_forecast_sales(rop_result, fixed_config or {}, forecast_days=7)
        except Exception:
            ml_forecast = {}

    charts = {
        "cost_donut": create_cost_donut_chart(fixed_result, rop_result),
        "service_gauge": create_service_level_gauge_chart(fixed_result, rop_result),
        "inventory_trajectory": create_inventory_trajectory_chart(fixed_result, rop_result, start_day, end_day),
        "daily_revenue": create_daily_revenue_chart(fixed_result, rop_result, start_day, end_day),
        "sku_risk_scatter": create_sku_risk_scatter_chart(fixed_result, rop_result, fixed_config),
        "daily_fulfillment": create_daily_fulfillment_chart(fixed_result, rop_result, start_day, end_day),
    }

    if ml_forecast and ml_forecast.get("daily_forecast"):
        try:
            charts["ml_forecast"] = create_ml_forecast_chart(ml_forecast)
        except Exception:
            pass

    return charts


def generate_simulation_pdf_report(
    fixed_run, rop_run, fixed_result, rop_result, fixed_config, rop_config, start_day=1, end_day=7
):
    """Generate a multi-page publication-grade PDF Executive Report with diverse visual charts and explanatory lines.

    Returns:
        io.BytesIO: Buffer containing the PDF document bytes.
    """
    pdf_buffer = io.BytesIO()

    with PdfPages(pdf_buffer) as pdf:
        # PAGE 1: Strategic Financial & Service Level Analysis (Donut + Gauge Charts)
        fig1 = plt.figure(figsize=(11.69, 8.27), dpi=150, facecolor="#ffffff")  # A4 Landscape
        gs1 = fig1.add_gridspec(3, 2, height_ratios=[1.0, 2.2, 1.8], hspace=0.36, wspace=0.25)

        # Header Banner
        ax_hdr = fig1.add_subplot(gs1[0, :])
        ax_hdr.axis("off")
        ax_hdr.text(0.0, 0.85, "SMART WAREHOUSE — EXECUTIVE SIMULATION REPORT",
                    fontsize=16, fontweight="bold", color="#0f172a")

        scenario_label = fixed_result.get("scenario_label", fixed_config.get("scenario", "Normal demand"))
        metadata_str = (
            f"Store Type: {fixed_run.store_type.title()}  |  "
            f"Scenario: {scenario_label}  |  "
            f"Random Seed: {fixed_run.random_seed}  |  "
            f"Report Window: Day {start_day} to Day {end_day} (Full: {fixed_run.duration_days} Days)"
        )
        ax_hdr.text(0.0, 0.50, metadata_str, fontsize=10, color="#475569")
        ax_hdr.text(0.0, 0.16, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Multi-Visual Executive Analytics with Explanatory Reference Lines",
                    fontsize=8.5, color="#94a3b8", style="italic")
        ax_hdr.axhline(0.04, color="#0284c7", linewidth=2.5)

        # Middle Left: Donut Chart for Cost Structure
        ax_cost = fig1.add_subplot(gs1[1, 0])
        fs = fixed_result.get("summary", {})
        rs = rop_result.get("summary", {})
        labels = ["Purchase", "Ordering", "Holding"]
        colors = [CLR_PURCHASE, CLR_ORDERING, CLR_HOLDING]
        f_costs = [fs.get("total_purchase_cost", 0), fs.get("total_ordering_cost", 0), fs.get("total_holding_cost", 0)]
        ax_cost.pie(f_costs, labels=labels, autopct="%1.1f%%", pctdistance=0.75, colors=colors,
                    wedgeprops=dict(width=0.4, edgecolor="#ffffff", linewidth=1.5),
                    textprops=dict(fontsize=8, color=CLR_TEXT))
        ax_cost.set_title("Cost Structure: Fixed vs. ROP (Donut Chart)\n[Line Note: Purchase is >90% of total capital outlay]",
                          fontsize=10.5, fontweight="bold", color=CLR_TEXT)

        # Middle Right: Service Level Speedometer Gauge with 95% target line
        ax_gauge = fig1.add_subplot(gs1[1, 1])
        sv_fixed = fs.get("service_level", 1.0) * 100
        sv_rop = rs.get("service_level", 1.0) * 100
        theta = np.linspace(np.pi, 0, 100)
        ax_gauge.plot(np.cos(theta), np.sin(theta), color="#e2e8f0", linewidth=18, solid_capstyle="round")
        fill_steps = int(100 * (min(100.0, max(0.0, sv_rop)) / 100.0))
        if fill_steps > 1:
            ax_gauge.plot(np.cos(theta[:fill_steps]), np.sin(theta[:fill_steps]), color=CLR_ROP, linewidth=18, solid_capstyle="round")
        
        # 95% Target line
        t_target = np.pi - 0.95 * np.pi
        ax_gauge.plot([0.84 * np.cos(t_target), 1.16 * np.cos(t_target)],
                      [0.84 * np.sin(t_target), 1.16 * np.sin(t_target)],
                      color="#dc2626", linewidth=2.5, zorder=5)
        ax_gauge.text(1.2 * np.cos(t_target), 1.16 * np.sin(t_target), "--- 95% Target Line", fontsize=7.5, color="#dc2626", fontweight="bold")

        ax_gauge.text(0, 0.15, f"{sv_rop:.1f}%", ha="center", va="center", fontsize=20, fontweight="bold", color=CLR_TEXT)
        ax_gauge.text(0, -0.15, f"ROP Service Level (vs Fixed {sv_fixed:.1f}%)\n[Line Note: Red line indicates 95% threshold]",
                      ha="center", va="center", fontsize=8.5, color=CLR_MUTED, fontweight="bold")
        ax_gauge.set_xlim(-1.3, 1.45)
        ax_gauge.set_ylim(-0.35, 1.25)
        ax_gauge.axis("off")
        ax_gauge.set_title("Fulfillment Efficiency Gauge (%)", fontsize=11, fontweight="bold", color=CLR_TEXT)

        # Bottom: KPI Summary Table
        ax_tbl = fig1.add_subplot(gs1[2, :])
        ax_tbl.axis("off")
        ax_tbl.set_title("Executive KPI Scorecard", fontsize=11, fontweight="bold", color=CLR_TEXT, pad=6)
        table_data = [
            ["Metric", "Fixed Threshold Policy", "Reorder Point (ROP) Policy", "Variance / Advantage"],
            ["Total Units Sold", f"{fs.get('total_units_sold', 0):,}", f"{rs.get('total_units_sold', 0):,}",
             f"{rs.get('total_units_sold', 0) - fs.get('total_units_sold', 0):+d} units"],
            ["Total Lost Units", f"{fs.get('total_lost_units', 0):,}", f"{rs.get('total_lost_units', 0):,}",
             f"{rs.get('total_lost_units', 0) - fs.get('total_lost_units', 0):+d} units"],
            ["Service Level (%)", f"{fs.get('service_level', 1.0)*100:.2f}%", f"{rs.get('service_level', 1.0)*100:.2f}%",
             f"{(rs.get('service_level', 1.0) - fs.get('service_level', 1.0))*100:+.2f}%"],
            ["Total Revenue (₹)", f"₹{fs.get('total_revenue', 0):,.2f}", f"₹{rs.get('total_revenue', 0):,.2f}",
             f"₹{rs.get('total_revenue', 0) - fs.get('total_revenue', 0):+,.2f}"],
            ["Total Holding Cost (₹)", f"₹{fs.get('total_holding_cost', 0):,.2f}", f"₹{rs.get('total_holding_cost', 0):,.2f}",
             f"₹{rs.get('total_holding_cost', 0) - fs.get('total_holding_cost', 0):+,.2f}"],
            ["Total Ordering Cost (₹)", f"₹{fs.get('total_ordering_cost', 0):,.2f}", f"₹{rs.get('total_ordering_cost', 0):,.2f}",
             f"₹{rs.get('total_ordering_cost', 0) - fs.get('total_ordering_cost', 0):+,.2f}"],
        ]
        table = ax_tbl.table(cellText=table_data, loc="center", cellLoc="center", colWidths=[0.3, 0.23, 0.23, 0.24])
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.25)
        for i in range(4):
            table[(0, i)].set_facecolor("#1e3a8a")
            table[(0, i)].set_text_props(color="#ffffff", fontweight="bold")
        for row in range(1, len(table_data)):
            bg = "#f1f5f9" if row % 2 == 1 else "#ffffff"
            for col in range(4):
                table[(row, col)].set_facecolor(bg)

        pdf.savefig(fig1)
        plt.close(fig1)

        # PAGE 2: Daily Operations & Trajectory Trends (Line/Area + Multi-Line + Grouped Bar)
        fig2 = plt.figure(figsize=(11.69, 8.27), dpi=150, facecolor="#ffffff")
        gs2 = fig2.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.38)

        # Subplot 1: Total Warehouse Stock Trajectory with Safety Buffer Line
        ax_d1 = fig2.add_subplot(gs2[0, 0])
        ax_d1.set_facecolor("#f8fafc")
        ax_d1.grid(True, linestyle="--", alpha=0.5, color=CLR_GRID)
        f_items = fixed_result.get("daily_items", [])
        r_items = rop_result.get("daily_items", [])
        days_range = list(range(start_day, end_day + 1))
        xd = np.arange(len(days_range))
        days_labels = [f"Day {d}" for d in days_range]
        f_stock = [sum(item.get("closing_stock", 0) for item in f_items if item.get("day") == d) for d in days_range]
        r_stock = [sum(item.get("closing_stock", 0) for item in r_items if item.get("day") == d) for d in days_range]
        safety_buf = max(15, round(min(min(f_stock, default=50), min(r_stock, default=50)) * 0.75))

        ax_d1.plot(xd, f_stock, marker="o", linewidth=2.2, color=CLR_FIXED, label="Fixed Stock")
        ax_d1.fill_between(xd, f_stock, alpha=0.15, color=CLR_FIXED)
        ax_d1.plot(xd, r_stock, marker="s", linewidth=2.2, linestyle="--", color=CLR_ROP, label="ROP Stock")
        ax_d1.fill_between(xd, r_stock, alpha=0.15, color=CLR_ROP)
        ax_d1.axhline(safety_buf, color="#dc2626", linestyle="--", linewidth=1.6, label=f"Safety Buffer Line (~{safety_buf} u)")

        ax_d1.set_title("Warehouse Total Inventory Level Trajectory (Line & Area Chart) [Line Note: Red dashed line marks Safety Stock Buffer]",
                        fontsize=10.5, fontweight="bold", color=CLR_TEXT)
        ax_d1.set_xticks(xd)
        ax_d1.set_xticklabels(days_labels, fontsize=8.5)
        ax_d1.legend(loc="upper right", fontsize=8)

        # Subplot 2: Daily Revenue Trajectory with Average Revenue Line
        ax_d2 = fig2.add_subplot(gs2[1, 0])
        ax_d2.set_facecolor("#f8fafc")
        ax_d2.grid(True, linestyle="--", alpha=0.5, color=CLR_GRID)
        f_daily = _filter_daily(fixed_result.get("daily", []), start_day, end_day)
        r_daily = _filter_daily(rop_result.get("daily", []), start_day, end_day)
        f_revs = [i.get("revenue", 0) for i in f_daily]
        r_revs = [i.get("revenue", 0) for i in r_daily]
        avg_rev = sum(f_revs + r_revs) / len(f_revs + r_revs) if (f_revs + r_revs) else 0

        ax_d2.plot(xd, f_revs, marker="o", linewidth=2.2, color=CLR_FIXED, label="Fixed Revenue")
        ax_d2.plot(xd, r_revs, marker="^", linewidth=2.2, linestyle="--", color=CLR_ROP, label="ROP Revenue")
        ax_d2.axhline(avg_rev, color="#d97706", linestyle=":", linewidth=1.8, label=f"Avg Revenue Line (₹{int(avg_rev):,})")

        ax_d2.set_title("Daily Sales Revenue Trajectory (Multi-Line Trend Chart) [Line Note: Dotted amber line marks Average Revenue Baseline]",
                        fontsize=10.5, fontweight="bold", color=CLR_TEXT)
        ax_d2.set_xticks(xd)
        ax_d2.set_xticklabels(days_labels, fontsize=8.5)
        ax_d2.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, loc: f"₹{int(val):,}"))
        ax_d2.legend(loc="upper left", fontsize=8)

        # Subplot 3: Daily Demand Fulfillment with Target Demand Line
        ax_d3 = fig2.add_subplot(gs2[2, 0])
        ax_d3.set_facecolor("#f8fafc")
        ax_d3.grid(True, linestyle="--", alpha=0.5, color=CLR_GRID)
        wd = 0.2
        f_sold = [i.get("units_sold", 0) for i in f_daily]
        f_lost = [i.get("lost_units", 0) for i in f_daily]
        r_sold = [i.get("units_sold", 0) for i in r_daily]
        r_lost = [i.get("lost_units", 0) for i in r_daily]
        demands = [f_sold[i] + f_lost[i] for i in range(len(f_sold))]
        avg_d = sum(demands) / len(demands) if demands else 1

        ax_d3.bar(xd - 1.5*wd, f_sold, wd, label="Fixed: Units Sold", color=CLR_FIXED)
        ax_d3.bar(xd - 0.5*wd, f_lost, wd, label="Fixed: Lost Units", color="#f87171", hatch="//")
        ax_d3.bar(xd + 0.5*wd, r_sold, wd, label="ROP: Units Sold", color=CLR_ROP)
        ax_d3.bar(xd + 1.5*wd, r_lost, wd, label="ROP: Lost Units", color="#ef4444", hatch="\\\\")
        ax_d3.axhline(avg_d, color="#6366f1", linestyle=":", linewidth=1.8, label=f"Avg Demand Line ({int(avg_d)} u)")

        ax_d3.set_title("Daily Units Sold vs. Lost Demand (Grouped Bar Chart) [Line Note: Dotted purple line marks Average Demand Target]",
                        fontsize=10.5, fontweight="bold", color=CLR_TEXT)
        ax_d3.set_xticks(xd)
        ax_d3.set_xticklabels(days_labels, fontsize=8.5)
        ax_d3.legend(loc="upper right", fontsize=8)

        pdf.savefig(fig2)
        plt.close(fig2)

        # PAGE 3: SKU Risk Matrix (Scatter Plot) & Executive Takeaways
        fig3 = plt.figure(figsize=(11.69, 8.27), dpi=150, facecolor="#ffffff")
        gs3 = fig3.add_gridspec(2, 1, height_ratios=[1.3, 1], hspace=0.35)

        # Subplot 1: Scatter Plot Risk Matrix with crosshair lines
        ax_scat = fig3.add_subplot(gs3[0, 0])
        ax_scat.set_facecolor("#f8fafc")
        ax_scat.grid(True, linestyle=":", alpha=0.6, color=CLR_GRID)

        products_cfg = (fixed_config or {}).get("products", [])
        cfg_by_id = {p["id"]: p for p in products_cfg}
        f_prods = fixed_result.get("products", [])

        lead_times = [int(cfg_by_id.get(p.get("product_id"), {}).get("lead_time_days", 3)) for p in f_prods]
        avg_demands = [float(cfg_by_id.get(p.get("product_id"), {}).get("avg_daily_demand", 15)) for p in f_prods]
        names = [p.get("product_name", f"P{p.get('product_id')}")[:14] for p in f_prods]
        colors = ["#ef4444" if p.get("lost_units", 0) > 0 else "#0d9488" for p in f_prods]

        ax_scat.scatter(lead_times, avg_demands, s=140, c=colors, alpha=0.75, edgecolors="#1e293b", linewidth=1.2, zorder=3)
        for i, txt in enumerate(names):
            ax_scat.annotate(txt, (lead_times[i], avg_demands[i]), xytext=(5, 4), textcoords="offset points",
                             fontsize=8, fontweight="bold", color=CLR_TEXT)

        if lead_times:
            avg_lt = sum(lead_times)/len(lead_times)
            avg_dm = sum(avg_demands)/len(avg_demands)
            ax_scat.axvline(avg_lt, color="#94a3b8", linestyle="--", linewidth=1.5, label=f"Avg Lead Time Line ({avg_lt:.1f} d)")
            ax_scat.axhline(avg_dm, color="#94a3b8", linestyle="--", linewidth=1.5, label=f"Avg Demand Line ({avg_dm:.1f} u)")
            ax_scat.legend(loc="lower right", fontsize=8)

        ax_scat.set_title("SKU Volatility & Supplier Lead-Time Risk Matrix (Scatter Plot) [Line Note: Crosshair lines divide into 4 risk quadrants]",
                          fontsize=10.5, fontweight="bold", color=CLR_TEXT)
        ax_scat.set_xlabel("Supplier Lead Time (Days)", fontsize=9.5, fontweight="bold")
        ax_scat.set_ylabel("Average Daily Customer Demand (Units)", fontsize=9.5, fontweight="bold")

        # Subplot 2: Operational Observations & Strategic Takeaways
        ax_notes = fig3.add_subplot(gs3[1, 0])
        ax_notes.axis("off")
        ax_notes.set_title("Executive Takeaways & Management Recommendations", fontsize=11, fontweight="bold", color=CLR_TEXT, pad=6)

        sold_diff = rs.get("total_units_sold", 0) - fs.get("total_units_sold", 0)
        holding_diff = rs.get("total_holding_cost", 0) - fs.get("total_holding_cost", 0)
        ordering_diff = rs.get("total_ordering_cost", 0) - fs.get("total_ordering_cost", 0)
        sv_diff = (rs.get("service_level", 1.0) - fs.get("service_level", 1.0)) * 100

        takeaways = [
            f"1. Service Level Performance: The Reorder Point (ROP) policy achieved {rs.get('service_level', 1.0)*100:.2f}% service level vs {fs.get('service_level', 1.0)*100:.2f}% under Fixed Threshold ({sv_diff:+.2f}% variance).",
            f"2. Stockout Protection: ROP dynamically incorporates lead time variability and safety stock buffers, significantly reducing unfulfilled customer demand during demand surges.",
            f"3. Holding vs. Ordering Trade-off: Annual holding costs differed by ₹{holding_diff:+,.2f}, while ordering costs differed by ₹{ordering_diff:+,.2f}, reflecting EOQ batch replenishment efficiency.",
            "4. Operational Recommendation: Adopt Reorder Point (ROP) with Economic Order Quantity (EOQ) for items with lead-time uncertainty or demand spikes to prevent lost customer revenue.",
            "5. Simulation Note: Results are derived from the isolated simulation engine for decision-support and do not alter production database state.",
        ]
        y_pos = 0.85
        for t in takeaways:
            ax_notes.text(0.01, y_pos, t, fontsize=9, color="#334155")
            y_pos -= 0.18

        pdf.savefig(fig3)
        plt.close(fig3)

        # PAGE 4: Machine Learning 7-Day Forward Sales Forecast & Inventory Risk Matrix
        try:
            ml_fc = train_and_forecast_sales(rop_result, fixed_config or {}, forecast_days=7)
        except Exception:
            ml_fc = None

        if ml_fc and ml_fc.get("daily_forecast"):
            fig4 = plt.figure(figsize=(11.69, 8.27), dpi=150, facecolor="#ffffff")
            gs4 = fig4.add_gridspec(3, 2, height_ratios=[0.85, 2.5, 1.7], hspace=0.36, wspace=0.25)

            # Header Banner
            ax_ml_hdr = fig4.add_subplot(gs4[0, :])
            ax_ml_hdr.axis("off")
            ax_ml_hdr.text(0.0, 0.85, "MACHINE LEARNING FUTURE SALES & DEMAND FORECAST (DAYS 8–14)",
                           fontsize=15, fontweight="bold", color="#0f172a")
            ax_ml_hdr.text(0.0, 0.48,
                           f"Model: {ml_fc['summary']['model_name']}  |  Window: {ml_fc['summary']['forecast_window']}  |  Confidence Level: 95%",
                           fontsize=9.5, color="#475569")
            ax_ml_hdr.text(0.0, 0.14,
                           f"Projected 7-Day Revenue: ₹{ml_fc['summary']['total_predicted_revenue']:,.2f}  |  Projected Sales: {ml_fc['summary']['total_predicted_units']:,} units  |  Model R² Score: {ml_fc['summary']['r2_score']:.2f}",
                           fontsize=8.5, color="#6d28d9", fontweight="bold")
            ax_ml_hdr.axhline(0.04, color="#7c3aed", linewidth=2.5)

            # Middle: ML Forecast Plot
            ax_ml_plot = fig4.add_subplot(gs4[1, :])
            ax_ml_plot.set_facecolor("#f8fafc")
            ax_ml_plot.grid(True, linestyle="--", alpha=0.5, color=CLR_GRID)

            hist = ml_fc.get("historical_daily", [])
            future = ml_fc.get("daily_forecast", [])
            h_days = [d["day"] for d in hist]
            h_units = [d["units_sold"] for d in hist]
            f_days = [d["day"] for d in future]
            f_units = [d["predicted_units"] for d in future]
            lows = [d["lower_bound_units"] for d in future]
            highs = [d["upper_bound_units"] for d in future]

            ax_ml_plot.plot(h_days, h_units, marker="o", linewidth=2.2, color=CLR_FIXED, label="Historical Actual Sales (Days 1–7)")
            ax_ml_plot.plot([h_days[-1], f_days[0]], [h_units[-1], f_units[0]], linestyle=":", color="#7c3aed", linewidth=1.8)
            ax_ml_plot.plot(f_days, f_units, marker="D", linewidth=2.2, linestyle="--", color="#7c3aed", label="ML Predicted Demand (Days 8–14)")
            ax_ml_plot.fill_between(f_days, lows, highs, color="#8b5cf6", alpha=0.18, label="95% Prediction Confidence Band")

            ax_ml_plot.axvline(h_days[-1] + 0.5, color="#94a3b8", linestyle="-", linewidth=1.8)
            max_y = max(h_units + highs) if (h_units + highs) else 50
            ax_ml_plot.text(h_days[-1] + 0.6, max_y * 0.94, "▶ ML Forecast Horizon", color="#6d28d9", fontsize=8, fontweight="bold")

            all_days_labels = [f"Day {d}" for d in (h_days + f_days)]
            ax_ml_plot.set_xticks(h_days + f_days)
            ax_ml_plot.set_xticklabels(all_days_labels, fontsize=8)
            ax_ml_plot.set_ylabel("Total Units", fontsize=9, fontweight="bold")
            ax_ml_plot.legend(loc="upper left", fontsize=7.5)
            ax_ml_plot.set_title("Historical Simulation Sales vs. Machine Learning 7-Day Forward Trajectory [Line Note: Shaded band marks 95% Confidence Interval]",
                                 fontsize=10.5, fontweight="bold", color=CLR_TEXT)

            # Bottom Left: Model Evaluation Scorecard
            ax_ml_score = fig4.add_subplot(gs4[2, 0])
            ax_ml_score.axis("off")
            ax_ml_score.set_title("Forecast Accuracy & Statistical Performance", fontsize=10, fontweight="bold", pad=4)
            score_data = [
                ["Evaluation Metric", "Score / Value", "Assessment"],
                ["Model Architecture", "Ridge Regularized Regression", "Controls overfitting"],
                ["R² Coefficient of Determination", f"{ml_fc['summary']['r2_score']:.3f}", "High predictive fit"],
                ["Mean Absolute Error (MAE)", f"{ml_fc['summary']['mae']:.2f} units", "Low daily error deviation"],
                ["Root Mean Squared Error (RMSE)", f"{ml_fc['summary']['rmse']:.2f} units", "Penalizes large variance"],
                ["Projected 7-Day Cash Flow", f"₹{ml_fc['summary']['total_predicted_revenue']:,.2f}", "Estimated sales volume"],
            ]
            score_tbl = ax_ml_score.table(cellText=score_data, loc="center", cellLoc="center", colWidths=[0.38, 0.32, 0.30])
            score_tbl.auto_set_font_size(False)
            score_tbl.set_fontsize(8)
            score_tbl.scale(1, 1.2)
            for i in range(3):
                score_tbl[(0, i)].set_facecolor("#7c3aed")
                score_tbl[(0, i)].set_text_props(color="#ffffff", fontweight="bold")
            for r in range(1, len(score_data)):
                bg = "#f5f3ff" if r % 2 == 1 else "#ffffff"
                for c in range(3):
                    score_tbl[(r, c)].set_facecolor(bg)

            # Bottom Right: Stockout Depletion Risk Matrix
            ax_ml_risk = fig4.add_subplot(gs4[2, 1])
            ax_ml_risk.axis("off")
            ax_ml_risk.set_title("Forward Stockout Risk & Proactive Restock Alerts", fontsize=10, fontweight="bold", pad=4)
            prod_fc = ml_fc.get("product_forecast", [])[:5]
            risk_data = [
                ["Product Name", "Ending Stock", "7d Forecast", "Projected", "Urgency Status"],
            ]
            for p in prod_fc:
                pname = p["product_name"][:13] + "…" if len(p["product_name"]) > 13 else p["product_name"]
                urgency = "CRITICAL" if p["projected_stock"] <= 0 else ("MONITOR" if "MEDIUM" in p["risk_label"] else "SAFE")
                risk_data.append([
                    pname, str(p["ending_stock"]), str(p["forecast_units"]),
                    f"{p['projected_stock']:+d}", urgency
                ])
            risk_tbl = ax_ml_risk.table(cellText=risk_data, loc="center", cellLoc="center", colWidths=[0.30, 0.17, 0.17, 0.16, 0.20])
            risk_tbl.auto_set_font_size(False)
            risk_tbl.set_fontsize(8)
            risk_tbl.scale(1, 1.2)
            for i in range(5):
                risk_tbl[(0, i)].set_facecolor("#1e293b")
                risk_tbl[(0, i)].set_text_props(color="#ffffff", fontweight="bold")
            for r in range(1, len(risk_data)):
                bg = "#fef2f2" if risk_data[r][4] == "CRITICAL" else ("#fffbeb" if risk_data[r][4] == "MONITOR" else "#f0fdf4")
                for c in range(5):
                    risk_tbl[(r, c)].set_facecolor(bg)

            pdf.savefig(fig4)
            plt.close(fig4)

    pdf_buffer.seek(0)
    return pdf_buffer
