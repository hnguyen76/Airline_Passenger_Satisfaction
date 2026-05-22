from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "passenger_survey_balanced.csv"
DASHBOARD_PATH = ROOT / "dashboard" / "index.html"


THEME_MAP = {
    "Access & Parking": [
        "curbside_dropoff_ease",
        "transport_options_to_airport",
        "parking",
        "parking_facility_quality",
        "parking_space_availability_ease",
        "parking_terminal_access_ease",
        "parking_value_for_money",
    ],
    "Check-in & Airline": [
        "checkin_process",
        "checkin_queue_wait_time",
        "checkin_queue_organization",
        "self_service_kiosk_quantity",
        "checkin_counter_quantity",
        "staff_courtesy",
        "checkin_service_time",
        "ticket_purchase_process",
        "airline_service",
    ],
    "Security & Border": [
        "security_screening_process",
        "security_queue_wait_time",
        "security_queue_organization",
        "security_staff_service",
        "immigration_control",
        "immigration_queue_wait_time",
        "immigration_queue_organization",
        "immigration_staff_service",
        "customs_control",
        "customs_queue_wait_time",
        "customs_queue_organization",
        "customs_staff_service",
        "service_window_quantity",
    ],
    "Commercial Offer": [
        "food_beverage_outlets",
        "food_beverage_outlet_quantity",
        "food_beverage_quality_variety",
        "food_beverage_price_quality",
        "retail_outlets",
        "retail_outlet_quantity",
        "retail_quality_variety",
        "retail_price_quality",
    ],
    "Wayfinding & Comfort": [
        "location_and_movement",
        "signage",
        "flight_information_display_availability",
        "terminal_accessibility",
        "boarding_lounge_comfort",
        "thermal_comfort",
        "acoustic_comfort",
        "seat_availability",
        "reserved_seat_availability",
        "power_outlet_availability",
    ],
    "Digital Experience": [
        "airport_internet",
        "internet_connection_speed",
        "network_access_ease",
    ],
    "Cleanliness & Restrooms": [
        "restrooms",
        "restroom_quantity",
        "restroom_cleanliness",
        "restroom_maintenance",
        "overall_airport_cleanliness",
    ],
    "Baggage & Arrival": [
        "disembarkation_method_rating",
        "baggage_claim_process",
        "baggage_carousel_identification_ease",
        "baggage_claim_time",
        "baggage_integrity",
    ],
}


def clean_label(label: str) -> str:
    return label.replace("_", " ").replace("is applicable", "").strip().title()


def fmt_number(value: float | int) -> str:
    return f"{value:,.0f}"


def fmt_pct(value: float) -> str:
    return f"{value:.1%}"


def fmt_delta(value: float) -> str:
    return f"{value:+.1f} pp"


def detect_rating_columns(df: pd.DataFrame) -> list[str]:
    applicability_cols = [col for col in df.columns if col.endswith("_is_applicable")]
    rating_cols: list[str] = []
    for col in df.select_dtypes(include="number").columns:
        if col == "liked" or col in applicability_cols:
            continue
        values = df[col].dropna()
        if values.empty:
            continue
        if values.between(1, 5).mean() >= 0.95 and values.nunique() <= 5:
            rating_cols.append(col)
    return rating_cols


def benjamini_hochberg(p_values: pd.Series) -> np.ndarray:
    values = p_values.to_numpy(dtype=float)
    adjusted = np.full_like(values, np.nan, dtype=float)
    valid_mask = ~np.isnan(values)
    valid = values[valid_mask]
    if valid.size == 0:
        return adjusted
    order = np.argsort(valid)
    ranked = valid[order]
    total = valid.size
    q_values = ranked * total / np.arange(1, total + 1)
    q_values = np.minimum.accumulate(q_values[::-1])[::-1]
    q_values = np.clip(q_values, 0, 1)
    valid_adjusted = np.empty_like(valid)
    valid_adjusted[order] = q_values
    adjusted[valid_mask] = valid_adjusted
    return adjusted


def cohen_d(group_a: pd.Series, group_b: pd.Series) -> float:
    a = group_a.dropna().to_numpy(dtype=float)
    b = group_b.dropna().to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    if pooled <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) / np.sqrt(pooled))


def cramers_v(contingency_table: pd.DataFrame) -> float:
    chi2, _, _, _ = stats.chi2_contingency(contingency_table)
    total = contingency_table.to_numpy().sum()
    rows, cols = contingency_table.shape
    if total <= 1 or rows < 2 or cols < 2:
        return float("nan")
    phi2 = chi2 / total
    phi2_corr = max(0, phi2 - ((cols - 1) * (rows - 1)) / (total - 1))
    rows_corr = rows - ((rows - 1) ** 2) / (total - 1)
    cols_corr = cols - ((cols - 1) ** 2) / (total - 1)
    denominator = min(cols_corr - 1, rows_corr - 1)
    if denominator <= 0:
        return float("nan")
    return float(np.sqrt(phi2_corr / denominator))


def bar_rows(rows: list[dict], value_key: str, label_key: str, max_value: float | None = None, suffix: str = "") -> str:
    if not rows:
        return ""
    max_metric = max_value if max_value is not None else max(abs(float(row[value_key])) for row in rows) or 1
    parts = []
    for row in rows:
        value = float(row[value_key])
        width = min(100, max(2, abs(value) / max_metric * 100))
        label = html.escape(str(row[label_key]))
        value_text = html.escape(f"{value:.2f}{suffix}")
        parts.append(
            f"""
            <div class="bar-row">
              <div class="bar-label">{label}</div>
              <div class="bar-track"><span style="width:{width:.1f}%"></span></div>
              <div class="bar-value">{value_text}</div>
            </div>
            """
        )
    return "\n".join(parts)


def process_rows(rows: list[dict]) -> str:
    parts = []
    for row in rows:
        width = max(2, min(100, row["liked_rate"] * 100))
        parts.append(
            f"""
            <div class="process-row">
              <div>
                <strong>{html.escape(row["process"])}</strong>
                <span>{fmt_number(row["records"])} records</span>
              </div>
              <div class="process-meter"><span style="width:{width:.1f}%"></span></div>
              <b>{fmt_pct(row["liked_rate"])}</b>
            </div>
            """
        )
    return "\n".join(parts)


def month_line_svg(months: list[dict]) -> str:
    width = 760
    height = 260
    pad_x = 44
    pad_y = 34
    rates = [row["liked_rate"] for row in months]
    low = min(rates) - 0.035
    high = max(rates) + 0.035
    points = []
    for index, row in enumerate(months):
        x = pad_x + index * ((width - pad_x * 2) / (len(months) - 1))
        y = height - pad_y - ((row["liked_rate"] - low) / (high - low)) * (height - pad_y * 2)
        points.append((x, y, row))
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)
    circles = "\n".join(
        f"""<g><circle cx="{x:.1f}" cy="{y:.1f}" r="5"></circle><text x="{x:.1f}" y="{y - 12:.1f}" text-anchor="middle">{fmt_pct(row["liked_rate"])}</text></g>"""
        for x, y, row in points
    )
    labels = "\n".join(
        f"""<text class="axis-label" x="{x:.1f}" y="{height - 8}" text-anchor="middle">{html.escape(row["month"][:3])}</text>"""
        for x, _, row in points
    )
    grid = "\n".join(
        f"""<line class="grid" x1="{pad_x}" x2="{width - pad_x}" y1="{y}" y2="{y}"></line>"""
        for y in np.linspace(pad_y, height - pad_y, 5)
    )
    return f"""
    <svg class="line-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Monthly liked rate trend">
      {grid}
      <polyline points="{polyline}"></polyline>
      {circles}
      {labels}
    </svg>
    """


def table_rows(rows: list[dict]) -> str:
    parts = []
    for rank, row in enumerate(rows, start=1):
        parts.append(
            f"""
            <tr>
              <td>{rank}</td>
              <td>{html.escape(row["attribute"])}</td>
              <td>{row["disliked"]:.2f}</td>
              <td>{row["liked"]:.2f}</td>
              <td>{row["gap"]:+.2f}</td>
              <td>{row["priority_score"]:.2f}</td>
            </tr>
            """
        )
    return "\n".join(parts)


def build_dashboard_data() -> dict:
    df = pd.read_csv(DATA_PATH)
    rating_cols = detect_rating_columns(df)

    process = (
        df.groupby("process")["liked"]
        .agg(records="size", liked_rate="mean")
        .sort_values("liked_rate", ascending=False)
        .reset_index()
        .to_dict("records")
    )

    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    monthly = (
        df.assign(month=pd.Categorical(df["month"], categories=month_order, ordered=True))
        .groupby("month", observed=False)["liked"]
        .agg(records="size", liked_rate="mean")
        .dropna()
        .reset_index()
    )
    monthly["month"] = monthly["month"].astype(str)

    theme_scores = pd.DataFrame(index=df.index)
    for theme, columns in THEME_MAP.items():
        available = [col for col in columns if col in df.columns]
        theme_scores[theme] = df[available].mean(axis=1, skipna=True)
    theme_scores["liked"] = df["liked"]
    theme_summary = theme_scores.groupby("liked").mean().T.rename(columns={0: "disliked", 1: "liked"})
    theme_summary["gap"] = theme_summary["liked"] - theme_summary["disliked"]
    themes = (
        theme_summary.sort_values("gap", ascending=False)
        .reset_index(names="theme")
        .to_dict("records")
    )

    means = df.groupby("liked")[rating_cols].mean().T.rename(columns={0: "disliked", 1: "liked"})
    means["gap"] = means["liked"] - means["disliked"]
    means["priority_score"] = means["gap"] * (5 - means["disliked"])
    priority = (
        means.sort_values("priority_score", ascending=False)
        .head(8)
        .reset_index(names="attribute")
    )
    priority["attribute"] = priority["attribute"].map(clean_label)

    effect_rows = []
    for col in rating_cols:
        disliked = df.loc[df["liked"] == 0, col].dropna()
        liked = df.loc[df["liked"] == 1, col].dropna()
        if len(disliked) < 30 or len(liked) < 30:
            continue
        p_value = stats.ttest_ind(liked, disliked, equal_var=False, nan_policy="omit").pvalue
        effect_rows.append(
            {
                "attribute": clean_label(col),
                "cohens_d": cohen_d(liked, disliked),
                "p_value": p_value,
            }
        )
    effects = pd.DataFrame(effect_rows)
    effects["q_value"] = benjamini_hochberg(effects["p_value"])
    effect_data = (
        effects.sort_values("cohens_d", ascending=False)
        .head(7)[["attribute", "cohens_d", "q_value"]]
        .to_dict("records")
    )

    category_rows = []
    for col in df.select_dtypes(include="str").columns:
        if not 2 <= df[col].nunique(dropna=True) <= 35:
            continue
        temp = df[[col, "liked"]].copy()
        temp[col] = temp[col].fillna("Missing")
        contingency = pd.crosstab(temp[col], temp["liked"])
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            continue
        chi2, p_value, _, _ = stats.chi2_contingency(contingency)
        category_rows.append(
            {
                "feature": clean_label(col),
                "cramers_v": cramers_v(contingency),
                "p_value": p_value,
                "levels": contingency.shape[0],
            }
        )
    categories = pd.DataFrame(category_rows)
    categories["q_value"] = benjamini_hochberg(categories["p_value"])
    category_data = (
        categories.sort_values("cramers_v", ascending=False)
        .head(7)[["feature", "cramers_v", "q_value", "levels"]]
        .to_dict("records")
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]
    )
    x_train, x_test, y_train, y_test = train_test_split(
        df[rating_cols],
        df["liked"].astype(int),
        test_size=0.25,
        random_state=42,
        stratify=df["liked"],
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    boarding_rate = next(row["liked_rate"] for row in process if row["process"] == "Boarding")
    disembark_rate = next(row["liked_rate"] for row in process if row["process"] == "Disembarkation")
    top_priority = priority.iloc[0].to_dict()
    top_theme = themes[0]

    return {
        "records": int(len(df)),
        "columns": int(df.shape[1]),
        "rating_features": len(rating_cols),
        "liked_rate": float(df["liked"].mean()),
        "boarding_rate": float(boarding_rate),
        "disembarkation_rate": float(disembark_rate),
        "process_gap_pp": float((disembark_rate - boarding_rate) * 100),
        "model_accuracy": float(accuracy_score(y_test, predictions)),
        "model_auc": float(roc_auc_score(y_test, probabilities)),
        "top_priority": top_priority,
        "top_theme": top_theme,
        "process": process,
        "monthly": monthly.to_dict("records"),
        "themes": themes,
        "priority": priority.to_dict("records"),
        "effects": effect_data,
        "categories": category_data,
    }


def render_dashboard(data: dict) -> str:
    top_priority = data["top_priority"]
    top_theme = data["top_theme"]
    monthly_svg = month_line_svg(data["monthly"])
    theme_bars = bar_rows(data["themes"], "gap", "theme", suffix="")
    effect_bars = bar_rows(data["effects"], "cohens_d", "attribute", suffix="")
    category_bars = bar_rows(data["categories"], "cramers_v", "feature", suffix="")
    dashboard_json = json.dumps(data, ensure_ascii=True)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Airline Passenger Satisfaction Dashboard</title>
  <style>
    :root {{
      --bg: #070b14;
      --panel: #0d1424;
      --panel-2: #111b2d;
      --line: #23324d;
      --text: #eef5ff;
      --muted: #9fb0c8;
      --soft: #64748b;
      --cyan: #22d3ee;
      --teal: #2dd4bf;
      --amber: #fbbf24;
      --coral: #fb7185;
      --violet: #a78bfa;
      --green: #86efac;
      --shadow: 0 24px 80px rgba(0, 0, 0, 0.34);
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      background:
        radial-gradient(circle at 18% -10%, rgba(34, 211, 238, 0.18), transparent 34rem),
        radial-gradient(circle at 88% 2%, rgba(167, 139, 250, 0.16), transparent 32rem),
        linear-gradient(180deg, #070b14 0%, #0a1020 55%, #070b14 100%);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}

    .shell {{
      width: min(1480px, calc(100% - 36px));
      margin: 0 auto;
      padding: 28px 0 34px;
    }}

    header {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 24px;
      align-items: end;
      padding: 10px 0 22px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.22);
    }}

    .eyebrow {{
      margin: 0 0 10px;
      color: var(--cyan);
      font-size: 0.78rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }}

    h1 {{
      margin: 0;
      max-width: 980px;
      font-size: clamp(2rem, 4.8vw, 4.8rem);
      line-height: 0.98;
      letter-spacing: 0;
    }}

    .subtitle {{
      margin: 16px 0 0;
      max-width: 920px;
      color: var(--muted);
      font-size: clamp(1rem, 1.2vw, 1.18rem);
      line-height: 1.6;
    }}

    .signature {{
      justify-self: end;
      padding: 12px 16px;
      border: 1px solid rgba(34, 211, 238, 0.35);
      background: rgba(13, 20, 36, 0.72);
      border-radius: 8px;
      color: var(--cyan);
      font-weight: 800;
      white-space: nowrap;
    }}

    .story {{
      margin-top: 22px;
      display: grid;
      grid-template-columns: 1.08fr 0.92fr;
      gap: 16px;
      align-items: stretch;
    }}

    .story-lead {{
      min-height: 100%;
      position: relative;
    }}

    .story-lead p {{
      max-width: 900px;
      margin: 14px 0 0;
      color: var(--muted);
      font-size: clamp(1rem, 1.05vw, 1.12rem);
      line-height: 1.65;
    }}

    .story-kicker {{
      display: inline-flex;
      align-items: center;
      gap: 9px;
      margin-bottom: 12px;
      color: var(--amber);
      font-size: 0.78rem;
      font-weight: 900;
      letter-spacing: 0.1em;
      text-transform: uppercase;
    }}

    .story-kicker::before {{
      content: "";
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--amber);
      box-shadow: 0 0 18px rgba(251, 191, 36, 0.58);
    }}

    .story-lead h2 {{
      max-width: 900px;
      font-size: clamp(1.55rem, 2.5vw, 3rem);
      line-height: 1.08;
      text-transform: none;
      letter-spacing: 0;
    }}

    .story-stat-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 22px;
    }}

    .story-stat {{
      padding: 13px;
      border: 1px solid rgba(148, 163, 184, 0.14);
      background: rgba(7, 11, 20, 0.38);
      border-radius: 8px;
    }}

    .story-stat b {{
      display: block;
      font-size: 1.42rem;
      line-height: 1;
      color: var(--text);
      font-variant-numeric: tabular-nums;
    }}

    .story-stat span {{
      display: block;
      margin-top: 7px;
      color: var(--soft);
      font-size: 0.82rem;
      line-height: 1.35;
    }}

    .story-list {{
      display: grid;
      gap: 12px;
      height: 100%;
    }}

    .story-point {{
      padding: 16px;
      border: 1px solid rgba(148, 163, 184, 0.14);
      background: rgba(7, 11, 20, 0.36);
      border-radius: 8px;
    }}

    .story-point strong {{
      display: block;
      margin-bottom: 7px;
      color: #dff7ff;
      font-size: 1rem;
    }}

    .story-point span {{
      display: block;
      color: var(--muted);
      line-height: 1.48;
    }}

    .kpis {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin: 22px 0;
    }}

    .card, .panel {{
      border: 1px solid rgba(148, 163, 184, 0.18);
      background: linear-gradient(180deg, rgba(17, 27, 45, 0.92), rgba(13, 20, 36, 0.94));
      box-shadow: var(--shadow);
      border-radius: 8px;
    }}

    .card {{
      padding: 18px;
      min-height: 126px;
    }}

    .card .label {{
      color: var(--muted);
      font-size: 0.78rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}

    .card .value {{
      margin-top: 14px;
      font-size: clamp(1.8rem, 3.2vw, 3rem);
      line-height: 1;
      font-weight: 900;
    }}

    .card .note {{
      margin-top: 10px;
      color: var(--soft);
      line-height: 1.45;
      font-size: 0.92rem;
    }}

    .grid {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 16px;
      align-items: start;
      margin-top: 16px;
    }}

    .panel {{
      padding: 20px;
      overflow: hidden;
    }}

    .panel h2 {{
      margin: 0;
      font-size: 1.05rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}

    .panel p {{
      color: var(--muted);
      line-height: 1.55;
    }}

    .brief {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}

    .brief-item {{
      border-left: 3px solid var(--cyan);
      padding: 12px 14px;
      background: rgba(7, 11, 20, 0.42);
      border-radius: 6px;
    }}

    .brief-item:nth-child(2) {{ border-left-color: var(--amber); }}
    .brief-item:nth-child(3) {{ border-left-color: var(--teal); }}
    .brief-item b {{ display: block; margin-bottom: 6px; }}
    .brief-item span {{ color: var(--muted); line-height: 1.48; }}

    .donut-wrap {{
      display: grid;
      place-items: center;
      min-height: 300px;
    }}

    .donut {{
      width: min(270px, 76vw);
      aspect-ratio: 1;
      border-radius: 50%;
      display: grid;
      place-items: center;
      background: conic-gradient(var(--teal) 0 {data["liked_rate"] * 100:.2f}%, var(--coral) 0 100%);
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.08), 0 20px 60px rgba(0,0,0,0.32);
    }}

    .donut::before {{
      content: "";
      width: 62%;
      aspect-ratio: 1;
      border-radius: 50%;
      background: var(--panel);
      position: absolute;
    }}

    .donut-inner {{
      position: relative;
      text-align: center;
    }}

    .donut-inner strong {{
      display: block;
      font-size: 3rem;
      line-height: 1;
    }}

    .donut-inner span {{
      color: var(--muted);
      font-weight: 700;
      text-transform: uppercase;
      font-size: 0.76rem;
      letter-spacing: 0.08em;
    }}

    .process-row {{
      display: grid;
      grid-template-columns: 170px 1fr 72px;
      gap: 12px;
      align-items: center;
      padding: 13px 0;
      border-bottom: 1px solid rgba(148, 163, 184, 0.12);
    }}

    .process-row strong, .process-row span {{ display: block; }}
    .process-row span {{ color: var(--soft); font-size: 0.85rem; margin-top: 3px; }}
    .process-row b {{ text-align: right; }}

    .process-meter, .bar-track {{
      height: 10px;
      border-radius: 999px;
      background: rgba(148, 163, 184, 0.13);
      overflow: hidden;
    }}

    .process-meter span, .bar-track span {{
      display: block;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--cyan), var(--teal));
    }}

    .line-chart {{
      width: 100%;
      height: auto;
      margin-top: 8px;
    }}

    .line-chart polyline {{
      fill: none;
      stroke: var(--cyan);
      stroke-width: 4;
      stroke-linecap: round;
      stroke-linejoin: round;
      filter: drop-shadow(0 0 10px rgba(34,211,238,0.35));
    }}

    .line-chart circle {{
      fill: var(--panel);
      stroke: var(--amber);
      stroke-width: 3;
    }}

    .line-chart text {{
      fill: var(--muted);
      font-size: 12px;
      font-weight: 800;
    }}

    .line-chart .axis-label {{ fill: var(--soft); font-size: 12px; }}
    .line-chart .grid {{ stroke: rgba(148,163,184,0.14); stroke-width: 1; }}

    .bar-row {{
      display: grid;
      grid-template-columns: minmax(170px, 260px) 1fr 74px;
      gap: 12px;
      align-items: center;
      padding: 10px 0;
      border-bottom: 1px solid rgba(148, 163, 184, 0.1);
    }}

    .bar-label {{
      color: #dbeafe;
      font-weight: 700;
      line-height: 1.25;
    }}

    .bar-value {{
      color: var(--muted);
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 14px;
      font-size: 0.92rem;
    }}

    th, td {{
      padding: 12px 10px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.13);
      text-align: left;
      vertical-align: top;
    }}

    th {{
      color: var(--muted);
      font-size: 0.74rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}

    td:nth-child(1), td:nth-child(n+3) {{
      font-variant-numeric: tabular-nums;
    }}

    .footer {{
      display: flex;
      justify-content: space-between;
      gap: 18px;
      margin-top: 22px;
      padding: 18px 0 0;
      color: var(--soft);
      border-top: 1px solid rgba(148, 163, 184, 0.18);
    }}

    @media (max-width: 1080px) {{
      header, .grid, .brief, .story {{ grid-template-columns: 1fr; }}
      .signature {{ justify-self: start; }}
      .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}

    @media (max-width: 720px) {{
      .shell {{ width: min(100% - 24px, 1480px); padding-top: 18px; }}
      .kpis {{ grid-template-columns: 1fr; }}
      .story-stat-grid {{ grid-template-columns: 1fr; }}
      .process-row, .bar-row {{ grid-template-columns: 1fr; gap: 8px; }}
      .process-row b, .bar-value {{ text-align: left; }}
      table {{ display: block; overflow-x: auto; white-space: nowrap; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div>
        <p class="eyebrow">CEO and Board Management Dashboard</p>
        <h1>Airline Passenger Satisfaction</h1>
        <p class="subtitle">A board-ready story of how passengers experience the airport journey, where friction shows up, and which service moments most clearly shape satisfaction.</p>
      </div>
      <div class="signature">Created by Hieu Nguyen</div>
    </header>

    <section class="story">
      <article class="panel story-lead">
        <div class="story-kicker">Dataset Story</div>
        <h2>Every survey row is a passenger journey, from access and check-in to comfort, baggage, and final sentiment.</h2>
        <p>
          This dataset captures {fmt_number(data["records"])} passenger feedback records across boarding and disembarkation journeys. It combines satisfaction ratings, service touchpoints, travel context, demographics, and a final <strong>liked</strong> outcome. For leadership, the value is not only knowing whether passengers were satisfied; it is understanding which operational moments create that satisfaction.
        </p>
        <p>
          The analysis turns raw survey responses into a management view: journey themes, service gaps, statistically reliable drivers, and practical improvement priorities. The story is clear: satisfaction is shaped less by one isolated score and more by repeated friction in visible moments such as lounge comfort, power access, parking, and baggage claim.
        </p>
        <div class="story-stat-grid">
          <div class="story-stat">
            <b>{fmt_number(data["records"])}</b>
            <span>passenger survey records analyzed</span>
          </div>
          <div class="story-stat">
            <b>{data["rating_features"]}</b>
            <span>rating attributes converted into journey signals</span>
          </div>
          <div class="story-stat">
            <b>{fmt_pct(data["liked_rate"])}</b>
            <span>balanced liked-rate baseline for fair comparison</span>
          </div>
        </div>
      </article>
      <aside class="panel story-list" aria-label="Dataset insights">
        <div class="story-point">
          <strong>What the dataset is about</strong>
          <span>Passenger satisfaction across the airport service journey: access, check-in, security, border control, commercial services, comfort, cleanliness, digital experience, and baggage.</span>
        </div>
        <div class="story-point">
          <strong>What can be extracted</strong>
          <span>Management can isolate the journey themes and service attributes that separate liked trips from disliked trips, then compare those signals against process, month, and passenger segments.</span>
        </div>
        <div class="story-point">
          <strong>Core insight</strong>
          <span>{html.escape(top_theme["theme"])} is the widest theme-level satisfaction gap, while {html.escape(top_priority["attribute"])} is the strongest near-term improvement priority.</span>
        </div>
        <div class="story-point">
          <strong>Decision implication</strong>
          <span>Use the dashboard as an executive operating lens: protect high-performing journey moments, fund the highest-priority friction points, and track whether improvements close the liked vs. disliked gap.</span>
        </div>
      </aside>
    </section>

    <section class="kpis" aria-label="Executive KPI cards">
      <article class="card">
        <div class="label">Survey Base</div>
        <div class="value">{fmt_number(data["records"])}</div>
        <div class="note">{fmt_number(data["columns"])} fields, {data["rating_features"]} rating attributes</div>
      </article>
      <article class="card">
        <div class="label">Liked Rate</div>
        <div class="value">{fmt_pct(data["liked_rate"])}</div>
        <div class="note">Balanced dataset baseline for driver comparison</div>
      </article>
      <article class="card">
        <div class="label">Process Gap</div>
        <div class="value">{data["process_gap_pp"]:.1f} pp</div>
        <div class="note">Disembarkation ahead of boarding satisfaction</div>
      </article>
      <article class="card">
        <div class="label">Model Signal</div>
        <div class="value">{data["model_auc"]:.3f}</div>
        <div class="note">ROC AUC from rating-only validation model</div>
      </article>
    </section>

    <section class="panel">
      <h2>Board Brief</h2>
      <div class="brief">
        <div class="brief-item">
          <b>Primary management priority</b>
          <span>{html.escape(top_priority["attribute"])} has the highest combined dissatisfaction gap and weak baseline score.</span>
        </div>
        <div class="brief-item">
          <b>Experience theme with greatest separation</b>
          <span>{html.escape(top_theme["theme"])} leads the liked vs. disliked theme gap at {top_theme["gap"]:+.2f} rating points.</span>
        </div>
        <div class="brief-item">
          <b>Operating implication</b>
          <span>Boarding comfort, power access, parking, and arrival baggage should be treated as visible loyalty levers.</span>
        </div>
      </div>
    </section>

    <section class="grid">
      <article class="panel">
        <h2>Monthly Satisfaction Pulse</h2>
        {monthly_svg}
      </article>
      <article class="panel">
        <h2>Overall Sentiment</h2>
        <div class="donut-wrap">
          <div class="donut">
            <div class="donut-inner">
              <strong>{fmt_pct(data["liked_rate"])}</strong>
              <span>liked trips</span>
            </div>
          </div>
        </div>
      </article>
    </section>

    <section class="grid">
      <article class="panel">
        <h2>Process Performance</h2>
        {process_rows(data["process"])}
      </article>
      <article class="panel">
        <h2>Theme Gap</h2>
        {theme_bars}
      </article>
    </section>

    <section class="panel">
      <h2>Management Priority Stack</h2>
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Attribute</th>
            <th>Disliked Avg</th>
            <th>Liked Avg</th>
            <th>Gap</th>
            <th>Priority</th>
          </tr>
        </thead>
        <tbody>
          {table_rows(data["priority"])}
        </tbody>
      </table>
    </section>

    <section class="grid">
      <article class="panel">
        <h2>Statistical Effect Size</h2>
        {effect_bars}
      </article>
      <article class="panel">
        <h2>Categorical Association</h2>
        {category_bars}
      </article>
    </section>

    <footer class="footer">
      <span>Static dashboard generated from passenger_survey_balanced.csv.</span>
      <span>Created by Hieu Nguyen</span>
    </footer>
  </main>
  <script type="application/json" id="dashboard-data">{dashboard_json}</script>
</body>
</html>
"""


def main() -> None:
    data = build_dashboard_data()
    DASHBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    DASHBOARD_PATH.write_text(render_dashboard(data), encoding="utf-8")
    print(f"Dashboard written to {DASHBOARD_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
