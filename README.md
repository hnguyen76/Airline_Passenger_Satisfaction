# Airline Passenger Satisfaction Analysis

CEO and Board Management report for exploring airline passenger survey data, identifying experience drivers, and presenting an executive dark mode dashboard.

**Created by Hieu Nguyen**

## Project Structure

```text
.
|-- passenger_survey_balanced.csv
|-- notebooks/
|   `-- airline_passenger_satisfaction_report.ipynb
|-- dashboard/
|   `-- index.html
|-- scripts/
|   |-- build_dashboard.py
|   `-- upgrade_notebook_for_board.py
|-- .github/
|   `-- workflows/deploy-dashboard.yml
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## Setup

Create and activate the virtual environment named `airlinevenv`:

```powershell
python -m venv airlinevenv
.\airlinevenv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name airlinevenv --display-name "Python (airlinevenv)"
```

Open the notebook in VS Code, JupyterLab, or another Jupyter-compatible editor:

```powershell
notebooks/airline_passenger_satisfaction_report.ipynb
```

Regenerate the saved notebook outputs:

```powershell
.\airlinevenv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks\airline_passenger_satisfaction_report.ipynb
```

Build the static dark mode dashboard:

```powershell
.\airlinevenv\Scripts\python.exe scripts\build_dashboard.py
```

Open the dashboard locally:

```powershell
dashboard\index.html
```

## GitHub Pages Dashboard

The repository includes a GitHub Actions workflow at `.github/workflows/deploy-dashboard.yml`. After pushing to `main`, enable GitHub Pages with **Source: GitHub Actions** in the repository settings to publish the `dashboard/` folder.

## Report Highlights

- CEO and board scorecard with management interpretation.
- Passenger satisfaction overview and monthly trend analysis.
- Cleaned and grouped rating dimensions across airport journey stages.
- Dark mode executive dashboard designed for GitHub Pages.
- Beautiful Matplotlib visuals designed to render directly on GitHub.
- Driver analysis showing which service attributes most separate liked vs. disliked trips.
- Advanced statistical validation with confidence intervals, effect sizes, FDR-corrected tests, and Cramer's V.
- A compact predictive model section to validate the strongest satisfaction signals.
- Business recommendations based on observed satisfaction gaps.

## Data

The dataset is stored in `passenger_survey_balanced.csv`. It contains balanced passenger feedback records with `liked` as the target variable.

## Signature

Created by Hieu Nguyen
