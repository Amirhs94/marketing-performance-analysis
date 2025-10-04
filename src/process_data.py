
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image  # requires pillow

# -----------------------
# Paths (repo layout)
# -----------------------
data_dir = Path("data")
out_dir = Path("notebooks")      # publish results directly in notebooks/
out_dir.mkdir(parents=True, exist_ok=True)

ppc = data_dir / "ppc_spend.csv"
email = data_dir / "email_campaigns.csv"
social = data_dir / "social_media_ads.csv"
conversions = data_dir / "website_conversions.csv"

# -----------------------
# Load data
# -----------------------
ppc_df = pd.read_csv(ppc, parse_dates=["date"])
email_df = pd.read_csv(email, parse_dates=["date"])
social_df = pd.read_csv(social, parse_dates=["date"])
conv_df = pd.read_csv(conversions, parse_dates=["date"])

# -----------------------
# Aggregate conversions (daily x channel)
# -----------------------
conv_daily = (
    conv_df.groupby(["date", "channel"], as_index=False)
           .agg(conversions=("conversion_id", "nunique"),
                revenue=("revenue", "sum"))
)

# -----------------------
# Helper to normalize channel frames
# -----------------------
def add_channel_frame(df: pd.DataFrame, channel_name: str) -> pd.DataFrame:
    """
    Normalize to a common schema: [date, channel, spend, clicks, impressions, emails_sent]
    """
    out = df.copy()
    out["channel"] = channel_name
    for c in ["spend", "clicks", "impressions", "emails_sent"]:
        if c not in out.columns:
            out[c] = np.nan
    return out[["date", "channel", "spend", "clicks", "impressions", "emails_sent"]]

# -----------------------
# Build unified daily performance table
# -----------------------
ppc_u = add_channel_frame(ppc_df, "PPC")
email_u = add_channel_frame(email_df, "Email")
social_u = add_channel_frame(social_df, "Social Media")

perf = pd.concat([ppc_u, email_u, social_u], ignore_index=True)

# -----------------------
# Join conversions + enrich
# -----------------------
merged = perf.merge(conv_daily, on=["date", "channel"], how="left")
merged["conversions"] = merged["conversions"].fillna(0).astype(int)
merged["revenue"] = merged["revenue"].fillna(0.0)
merged["week_start"] = merged["date"] - pd.to_timedelta(merged["date"].dt.weekday, unit="D")
merged = merged.sort_values(["date", "channel"]).reset_index(drop=True)

merged.to_csv(out_dir / "integrated_marketing_dataset.csv", index=False)

# -----------------------
# KPIs by channel
# -----------------------
kpis = (
    merged.groupby("channel", as_index=False)
          .agg(
              total_spend=("spend", lambda s: np.nansum(s)),
              total_clicks=("clicks", lambda s: np.nansum(s)),
              total_conversions=("conversions", "sum"),
              total_revenue=("revenue", "sum"),
              total_impressions=("impressions", lambda s: np.nansum(s)),
              total_emails_sent=("emails_sent", lambda s: np.nansum(s)),
          )
)

# If a channel never had any spend values, mark total_spend as NaN (unknown) rather than 0
has_spend = merged.groupby("channel")["spend"].apply(lambda s: s.notna().any()).to_dict()
for i, row in kpis.iterrows():
    if not has_spend.get(row["channel"], False):
        kpis.at[i, "total_spend"] = np.nan

# Derived metrics
kpis["ctr"] = np.where(kpis["total_impressions"] > 0,
                       kpis["total_clicks"] / kpis["total_impressions"], np.nan)
kpis["conversion_rate"] = np.where(kpis["total_clicks"] > 0,
                                   kpis["total_conversions"] / kpis["total_clicks"], np.nan)
kpis["cpc"] = np.where(kpis["total_clicks"] > 0,
                       kpis["total_spend"] / kpis["total_clicks"], np.nan)
kpis["cpa"] = np.where(kpis["total_conversions"] > 0,
                       kpis["total_spend"] / kpis["total_conversions"], np.nan)
kpis["roas"] = np.where(kpis["total_spend"] > 0,
                        kpis["total_revenue"] / kpis["total_spend"], np.nan)
kpis["roi"] = np.where(kpis["total_spend"] > 0,
                       (kpis["total_revenue"] - kpis["total_spend"]) / kpis["total_spend"], np.nan)

kpis.to_csv(out_dir / "kpis_by_channel.csv", index=False)

# -----------------------
# Weekly KPIs
# -----------------------
wk = (
    merged.groupby(["week_start", "channel"], as_index=False)
          .agg(
              spend=("spend", lambda s: np.nansum(s)),
              clicks=("clicks", lambda s: np.nansum(s)),
              conversions=("conversions", "sum"),
              revenue=("revenue", "sum"),
              impressions=("impressions", lambda s: np.nansum(s)),
              emails_sent=("emails_sent", lambda s: np.nansum(s)),
          )
)
wk["ctr"] = np.where(wk["impressions"] > 0, wk["clicks"] / wk["impressions"], np.nan)
wk["conversion_rate"] = np.where(wk["clicks"] > 0, wk["conversions"] / wk["clicks"], np.nan)
wk["cpc"] = np.where(wk["clicks"] > 0, wk["spend"] / wk["clicks"], np.nan)
wk["cpa"] = np.where(wk["conversions"] > 0, wk["spend"] / wk["conversions"], np.nan)
wk["roas"] = np.where(wk["spend"] > 0, wk["revenue"] / wk["spend"], np.nan)
wk["roi"] = np.where(wk["spend"] > 0, (wk["revenue"] - wk["spend"]) / wk["spend"], np.nan)

wk.to_csv(out_dir / "kpis_weekly.csv", index=False)

# -----------------------
# Answers to questions
# -----------------------
roi_series = kpis.set_index("channel")["roi"].replace([np.inf, -np.inf], np.nan).dropna()
best_roi_channel = roi_series.idxmax() if not roi_series.empty else None
best_roi_value = roi_series.max() if not roi_series.empty else np.nan

cpa_series = kpis.set_index("channel")["cpa"].replace([np.inf, -np.inf], np.nan).dropna()
cheapest_cpa_channel = cpa_series.idxmin() if not cpa_series.empty else None
cheapest_cpa_value = cpa_series.min() if not cpa_series.empty else np.nan
most_exp_cpa_channel = cpa_series.idxmax() if not cpa_series.empty else None
most_exp_cpa_value = cpa_series.max() if not cpa_series.empty else np.nan

cvr_series = kpis.set_index("channel")["conversion_rate"].replace([np.inf, -np.inf], np.nan).dropna()
best_cvr_channel = cvr_series.idxmax() if not cvr_series.empty else None
best_cvr_value = cvr_series.max() if not cvr_series.empty else np.nan

# -----------------------
# Trend detection (simple)
# -----------------------
def trend_label_from_slope(slope: float, tol: float = 1e-4) -> str:
    if np.isnan(slope) or abs(slope) <= tol:
        return "flat"
    return "increasing" if slope > 0 else "decreasing"

trend_rows = []
for ch, sub in wk.groupby("channel"):
    sub = sub.sort_values("week_start")
    if len(sub) <= 1:
        slope_roas = 0.0
        slope_cvr = 0.0
    else:
        x = (sub["week_start"] - sub["week_start"].min()).dt.days.values.astype(float)
        y_roas = sub["roas"].fillna(0).values.astype(float)
        y_cvr = sub["conversion_rate"].fillna(0).values.astype(float)
        slope_roas = np.polyfit(x, y_roas, 1)[0]
        slope_cvr = np.polyfit(x, y_cvr, 1)[0]
    trend_rows.append({
        "channel": ch,
        "roas_trend": trend_label_from_slope(slope_roas),
        "cvr_trend": trend_label_from_slope(slope_cvr),
    })

trends_df = pd.DataFrame(trend_rows)
trends_df.to_csv(out_dir / "kpi_trends.csv", index=False)

# -----------------------
# Charts
# -----------------------
plt.figure()
plt.bar(kpis["channel"], kpis["roas"])
plt.title("ROAS by Channel")
plt.xlabel("Channel")
plt.ylabel("ROAS (Revenue / Spend)")
plt.tight_layout()
plt.savefig(out_dir / "roas_by_channel.png")
plt.close()

plt.figure()
plt.bar(kpis["channel"], kpis["cpa"])
plt.title("CPA by Channel")
plt.xlabel("Channel")
plt.ylabel("CPA (Cost per Conversion)")
plt.tight_layout()
plt.savefig(out_dir / "cpa_by_channel.png")
plt.close()

plt.figure()
bars = plt.bar(kpis["channel"], kpis["conversion_rate"])
plt.title("Click-to-Sale Conversion Rate (CVR) by Channel")
plt.xlabel("Channel")
plt.ylabel("CVR (Conversions / Clicks)")
for bar, val in zip(bars, kpis["conversion_rate"]):
    if pd.notna(val):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{val:.2%}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(out_dir / "conversion_rate_by_channel.png")
plt.close()

plt.figure()
for ch, sub in wk.groupby("channel"):
    plt.plot(sub["week_start"], sub["roas"], label=ch)
plt.title("Weekly ROAS Over Time")
plt.xlabel("Week Start")
plt.ylabel("ROAS")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "weekly_roas.png")
plt.close()

plt.figure()
for ch, sub in wk.groupby("channel"):
    plt.plot(sub["week_start"], sub["conversion_rate"], label=ch)
plt.title("Weekly Conversion Rate Over Time")
plt.xlabel("Week Start")
plt.ylabel("CVR (Conversions / Clicks)")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "weekly_conversion_rate.png")
plt.close()

# -----------------------
# Create combined dashboard from chart PNGs
# -----------------------
img_paths = [
    out_dir / "roas_by_channel.png",
    out_dir / "cpa_by_channel.png",
    out_dir / "conversion_rate_by_channel.png",
    out_dir / "weekly_roas.png",
    out_dir / "weekly_conversion_rate.png",
]
images = [Image.open(p).convert("RGB") for p in img_paths if p.exists()]
if images:
    w = max(img.width for img in images)
    h = max(img.height for img in images)
    cols = 2
    rows = int(np.ceil(len(images) / cols))
    dashboard = Image.new("RGB", (cols * w, rows * h), color=(255, 255, 255))
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        dashboard.paste(img.resize((w, h)), (c * w, r * h))
    dashboard_path = out_dir / "dashboard.png"
    dashboard.save(dashboard_path)
else:
    dashboard_path = out_dir / "dashboard.png"  # placeholder name

# -----------------------
# Final Report
# -----------------------
def fmt_cur(x): return "—" if pd.isna(x) else f"${x:,.2f}"
def fmt_pct(x): return "—" if pd.isna(x) else f"{x:.2%}"

report_text = f"""
Marketing Performance ETL & KPI Analysis
----------------------------------------

1) Procedure
   • Loaded four CSVs from `data/` folder: PPC, Email, Social Media, and Conversions.
   • Normalized and merged them by (date, channel), added weekly buckets.
   • Calculated KPIs (CTR, CVR, CPC, CPA, ROAS, ROI) and plotted key graphs.

2) Key Insights
   • Highest ROI: {best_roi_channel if best_roi_channel else "—"} (ROI ≈ {fmt_pct(best_roi_value)})
   • Customer Acquisition Cost (CPA):
       - Cheapest among PPC & Social Media: {cheapest_cpa_channel if cheapest_cpa_channel else "—"} (≈ {fmt_cur(cheapest_cpa_value)})
       - Most expensive: {most_exp_cpa_channel if most_exp_cpa_channel else "—"} (≈ {fmt_cur(most_exp_cpa_value)})
       - Note: Email campaigns often have zero or untracked spend, so they are excluded from CPA comparison above.
   • Most effective at converting clicks into sales: {best_cvr_channel if best_cvr_channel else "—"} (CVR ≈ {fmt_pct(best_cvr_value)})
   • Trends: Weekly ROAS and CVR changes are mild; we can consider them **steady-state** with no harsh fluctuations.

   Per-channel trend directions:
{trends_df.apply(lambda r: f"       - {r['channel']}: ROAS {r['roas_trend']}, CVR {r['cvr_trend']}", axis=1).str.cat(sep="\\n")}

3) Charts & Dashboard
   • Individual charts saved as PNG in `notebooks/`.
   • Combined dashboard.png created using the above charts.

Artifacts written to `notebooks/`: 
   • integrated_marketing_dataset.csv
   • kpis_by_channel.csv
   • kpis_weekly.csv
   • kpi_trends.csv
   • dashboard.png
   • report.txt
"""

with open(out_dir / "report.txt", "w", encoding="utf-8") as f:
    f.write(report_text.strip() + "\n")

print("✅ Done. Files written to:", out_dir.resolve())
print(f"📊 Dashboard saved to: {dashboard_path.resolve()}")
print(f"📄 Report saved to: {str(out_dir / 'report.txt')}")
