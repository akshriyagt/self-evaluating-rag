# Customer Lifetime Value (CLV) & Churn Prediction Pipeline
### Project Overview

---

## 1. Objective

Build an end-to-end analytics pipeline that identifies which customers are likely to churn, estimates how much revenue each customer is worth over their lifetime, and surfaces this through an interactive dashboard — enabling a business to prioritize retention efforts on high-value, high-risk customers.

**Business framing (for resume/portfolio):**
> "Built an end-to-end churn and CLV pipeline that identifies the top 10% of at-risk high-value customers, enabling a targeted retention campaign with an estimated $X revenue impact."

---

## 2. Skills Demonstrated

| Skill Area | What It Shows |
|---|---|
| SQL | Data modeling, schema design, window functions, aggregation |
| Python | Feature engineering, ML modeling, statistical/probabilistic modeling, model explainability |
| Excel | Business-facing scenario/ROI analysis |
| Tableau / Power BI | Dashboard design, stakeholder communication, data storytelling |

---

## 3. Datasets

Pick one depending on whether you want subscription-style or transaction-style data:

- **Telco Customer Churn** (Kaggle) — customer-level data with a churn flag, contract type, tenure, charges. Best for classification-focused churn modeling.
- **Online Retail II** (UCI) — transaction-level data (invoices, quantities, prices, customer IDs). Best for CLV modeling and RFM analysis.

You can also combine ideas from both if you want a richer feature set.

---

## 4. Project Phases

### Phase 1 — Data Modeling (SQL)
- Load raw data into a database (Postgres, SQLite, or DuckDB)
- Design a star schema: `fact_transactions`, `dim_customers`, `dim_date`
- Build queries for:
  - Monthly active customers
  - Cohort retention tables
  - Revenue per customer
  - RFM (Recency, Frequency, Monetary) features

### Phase 2 — Feature Engineering & Modeling (Python)
- Engineer features: tenure, RFM scores, average order value, contract type, support interactions (if available)
- **Churn model:** XGBoost or LightGBM classifier to predict churn probability
- **CLV model:** BG/NBD + Gamma-Gamma model (via the `lifetimes` library) or a regression-based approach
- **Explainability:** SHAP values to explain individual churn risk drivers

### Phase 3 — Business Scenario Analysis (Excel)
- Export model outputs into a "what-if" retention ROI calculator
- Model scenario: *if churn is reduced by X% among high-CLV segments, what is the revenue impact?*

### Phase 4 — Dashboard (Tableau / Power BI)
Three-page dashboard:
1. **Executive Overview** — churn rate trend, revenue at risk, CLV distribution
2. **Segment Drill-Down** — risk broken out by tenure, contract type, region
3. **Customer Lookup** — individual churn probability, top SHAP drivers, CLV estimate

### Phase 5 — Narrative & Packaging
- Write up findings with a clear business impact statement
- Package as a portfolio piece: GitHub repo (SQL + Python code) + dashboard link + 1-page summary

---

## 5. Suggested Timeline (2–3 Weeks)

| Week | Focus |
|---|---|
| Week 1 | Data sourcing, SQL schema, data modeling |
| Week 2 | Feature engineering, churn model, CLV model, SHAP |
| Week 3 | Excel ROI calculator, dashboard build, write-up |

---

## 6. Stretch Goals

- Add a **simulated real-time scoring** component (batch job that re-scores customers weekly)
- Build a **segment-level intervention simulator**: compare "do nothing" vs. "targeted offer" vs. "broad discount" strategies
- Deploy the churn model as a simple API (Flask/FastAPI) for a more full-stack feel

---

## 7. Tools & Libraries

- **Database:** PostgreSQL / SQLite / DuckDB
- **Python:** pandas, scikit-learn, XGBoost/LightGBM, `lifetimes`, SHAP
- **Visualization:** Tableau or Power BI
- **Other:** Excel/Google Sheets for scenario modeling
