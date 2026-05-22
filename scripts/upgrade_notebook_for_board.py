from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "airline_passenger_satisfaction_report.ipynb"


TITLE = """# Airline Passenger Satisfaction Analysis

CEO and Board Management Report

A professional decision-ready report built from `passenger_survey_balanced.csv`. The notebook moves from executive conclusions to statistical validation so leadership can see both the management story and the analytical evidence behind it.

[Open the dark mode dashboard](../dashboard/index.html)

**Created by Hieu Nguyen**
"""


EXECUTIVE_FRAMING = """## 1. Executive Framing

This notebook is structured for senior leadership review. It prioritizes business implications first, then provides the statistical appendix needed to defend the recommendations.

Board-level questions answered:

1. Where is passenger satisfaction strongest and weakest across the journey?
2. Which experience gaps are large enough to deserve management attention?
3. Which signals remain reliable after statistical testing and multiple-testing correction?
4. Which operational priorities should be funded, monitored, or escalated?
"""


SCORECARD_MD = """## 2. CEO & Board Executive Scorecard

This section turns the analysis into a management agenda. The scorecard keeps the focus on decision quality: scale, satisfaction baseline, process gap, largest journey theme gap, and top improvement lever.
"""


SCORECARD_CODE = r'''
board_process = df.groupby("process")["liked"].agg(records="size", liked_rate="mean").sort_values("liked_rate", ascending=False)

board_rating_means = df.groupby("liked")[rating_cols].mean().T.rename(columns={0: "Disliked", 1: "Liked"})
board_rating_means["Gap"] = board_rating_means["Liked"] - board_rating_means["Disliked"]
board_rating_means["Priority Score"] = board_rating_means["Gap"] * (5 - board_rating_means["Disliked"])

board_theme_map = {
    "Access & Parking": [
        "curbside_dropoff_ease", "transport_options_to_airport", "parking", "parking_facility_quality",
        "parking_space_availability_ease", "parking_terminal_access_ease", "parking_value_for_money",
    ],
    "Check-in & Airline": [
        "checkin_process", "checkin_queue_wait_time", "checkin_queue_organization", "self_service_kiosk_quantity",
        "checkin_counter_quantity", "staff_courtesy", "checkin_service_time", "ticket_purchase_process", "airline_service",
    ],
    "Security & Border": [
        "security_screening_process", "security_queue_wait_time", "security_queue_organization", "security_staff_service",
        "immigration_control", "immigration_queue_wait_time", "immigration_queue_organization", "immigration_staff_service",
        "customs_control", "customs_queue_wait_time", "customs_queue_organization", "customs_staff_service", "service_window_quantity",
    ],
    "Commercial Offer": [
        "food_beverage_outlets", "food_beverage_outlet_quantity", "food_beverage_quality_variety",
        "food_beverage_price_quality", "retail_outlets", "retail_outlet_quantity", "retail_quality_variety", "retail_price_quality",
    ],
    "Wayfinding & Comfort": [
        "location_and_movement", "signage", "flight_information_display_availability", "terminal_accessibility",
        "boarding_lounge_comfort", "thermal_comfort", "acoustic_comfort", "seat_availability", "reserved_seat_availability",
        "power_outlet_availability",
    ],
    "Digital Experience": ["airport_internet", "internet_connection_speed", "network_access_ease"],
    "Cleanliness & Restrooms": [
        "restrooms", "restroom_quantity", "restroom_cleanliness", "restroom_maintenance", "overall_airport_cleanliness",
    ],
    "Baggage & Arrival": [
        "disembarkation_method_rating", "baggage_claim_process", "baggage_carousel_identification_ease",
        "baggage_claim_time", "baggage_integrity",
    ],
}

board_theme_scores = pd.DataFrame(index=df.index)
for theme, columns in board_theme_map.items():
    available = [col for col in columns if col in df.columns]
    board_theme_scores[theme] = df[available].mean(axis=1, skipna=True)
board_theme_scores["liked"] = df["liked"]
board_theme_summary = board_theme_scores.groupby("liked").mean().T.rename(columns={0: "Disliked", 1: "Liked"})
board_theme_summary["Gap"] = board_theme_summary["Liked"] - board_theme_summary["Disliked"]

top_priority = board_rating_means.sort_values("Priority Score", ascending=False).iloc[0]
top_theme = board_theme_summary.sort_values("Gap", ascending=False).iloc[0]
boarding_rate = board_process.loc["Boarding", "liked_rate"]
disembarkation_rate = board_process.loc["Disembarkation", "liked_rate"]

scorecard = pd.DataFrame({
    "Executive KPI": [
        "Survey records",
        "Satisfaction baseline",
        "Boarding liked rate",
        "Disembarkation liked rate",
        "Process gap",
        "Largest theme gap",
        "Top management priority",
    ],
    "Result": [
        f"{len(df):,}",
        f"{df['liked'].mean():.1%}",
        f"{boarding_rate:.1%}",
        f"{disembarkation_rate:.1%}",
        f"{(disembarkation_rate - boarding_rate) * 100:.1f} percentage points",
        f"{clean_label(top_theme.name)} ({top_theme['Gap']:+.2f} rating points)",
        f"{clean_label(top_priority.name)} (priority score {top_priority['Priority Score']:.2f})",
    ],
    "Board Interpretation": [
        "Large enough base for stable pattern detection.",
        "Balanced target is ideal for comparing liked vs. disliked experiences.",
        "Boarding is the weaker journey process and should be monitored closely.",
        "Arrival/disembarkation performs slightly better in this survey base.",
        "A modest but visible process gap that can compound across passenger volume.",
        "This journey theme most clearly separates liked from disliked experiences.",
        "This attribute combines a large satisfaction gap with a weak disliked-group baseline.",
    ],
})

management_agenda = (
    board_rating_means.sort_values("Priority Score", ascending=False)
    .head(6)
    .rename(index=clean_label)
    [["Disliked", "Liked", "Gap", "Priority Score"]]
)

display(Markdown("### Board Scorecard"))
display(scorecard)
display(Markdown("### Immediate Management Agenda"))
display(management_agenda)
'''.strip()


def renumber_headings(source: str) -> str:
    replacements = {
        "## 2. Satisfaction At A Glance": "## 3. Satisfaction At A Glance",
        "## 3. Journey Theme Scores": "## 4. Journey Theme Scores",
        "## 4. Service Attribute Drivers": "## 5. Service Attribute Drivers",
        "## 5. Segment Patterns": "## 6. Segment Patterns",
        "## 6. Advanced Statistical Validation": "## 7. Advanced Statistical Validation",
        "## 7. Compact Predictive Check": "## 8. Compact Predictive Check",
        "## 8. Recommendations": "## 9. Recommendations",
        "## 9. Closing Note": "## 10. Closing Note",
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    return source


def main() -> None:
    nb = nbf.read(NOTEBOOK_PATH, as_version=4)

    nb.cells[0].source = TITLE.strip()
    nb.cells[1].source = EXECUTIVE_FRAMING.strip()

    for cell in nb.cells:
        if cell.cell_type == "markdown":
            cell.source = renumber_headings(cell.source)

    has_scorecard = any(
        cell.cell_type == "markdown" and "## 2. CEO & Board Executive Scorecard" in cell.source
        for cell in nb.cells
    )
    if not has_scorecard:
        nb.cells[4:4] = [
            nbf.v4.new_markdown_cell(SCORECARD_MD.strip()),
            nbf.v4.new_code_cell(SCORECARD_CODE),
        ]
    else:
        for index, cell in enumerate(nb.cells):
            if cell.cell_type == "markdown" and "## 2. CEO & Board Executive Scorecard" in cell.source:
                if index + 1 < len(nb.cells) and nb.cells[index + 1].cell_type == "code":
                    nb.cells[index + 1].source = SCORECARD_CODE
                break

    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Upgraded {NOTEBOOK_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
