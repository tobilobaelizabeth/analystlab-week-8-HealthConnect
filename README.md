# HealthConnect — Week 8 Final Analytics & Decision Support

**Data Analytics Track · AnalystLab Africa Experience Lab**

HealthConnect is a data analytics and decision-support project focused on improving patient appointment attendance and healthcare support using data and AI.

This repository contains the **Week 8 final analytics deliverable**, which converts the validated findings from Weeks 5–7 into an evidence-based decision-support package for final project integration.

The Week 8 analysis recomputes and validates the final KPI register, closes the Data Analytics side of the T6 benchmark handoff, produces 30-day slot-recovery scenarios, generates a final decision-support dashboard, and documents validated findings, cross-track handoffs, limitations, and an executive summary.

---

## Project Overview

Missed appointments are the central operational problem examined in this project.

The final analysis uses **5,000 appointments covering January 2025–June 2026** and evaluates patterns associated with appointment no-shows.

The analysis identifies and validates two primary drivers:

1. **Booking lead time**
2. **Previous no-show history**

These drivers are incorporated into a transparent, rule-based risk tier that can be applied at booking without requiring a predictive model.

The final package also examines the clinic's existing reminder process and develops conservative, assumption-driven scenarios for targeted reminder allocation.

> **Important:** Scenario estimates are not forecasts, and the observed reminder relationship is explicitly treated as **associational rather than causal**. A randomized test is identified as the appropriate next step for establishing causal impact.

---

## Key Findings

### Appointment attendance

* **48.46%** of all appointments end in a no-show.
* **51.15%** of non-cancelled appointments are lost to no-shows.
* The dataset contains approximately **135 no-show slots per month**.

### Booking lead time

No-show rates increase with booking lead time:

| Booking lead time | No-show rate |
| ----------------- | -----------: |
| 0–7 days          |        27.8% |
| 31–60 days        |        60.5% |

The relationship remains present on the chronological holdout:

* Full-data Cramér's V: **0.252**
* Holdout Cramér's V: **0.237**

### Previous no-show history

Previous no-show history is the second validated driver:

* Full-data Cramér's V: **0.125**
* Holdout Cramér's V: **0.101**

The combination of long booking lead time and **2+ previous no-shows** reaches approximately **73% no-show rate**.

### Risk-tier triage

The project uses a transparent four-rule scoring system based on:

* Booking lead time
* Previous no-show history
* Distance to clinic
* Reminder status

The resulting tiers are:

| Tier | Full-data no-show rate | Holdout no-show rate |
| ---- | ---------------------: | -------------------: |
| Low  |                  30.9% |                32.6% |
| High |                 71.94% |               68.42% |

The **68.42% High-tier holdout rate** is used as the conservative benchmark transferred to Data Science for the T6 model comparison.

### Reminder coverage gap

Reminder coverage is not concentrated on the highest-risk bookings:

* High-risk reminder coverage: **38.9%**
* Low-risk reminder coverage: **87.3%**
* **660** long-lead (≥31 days) appointments are unreached by reminders.

This identifies a potential operational opportunity for targeted reminder allocation.

---

## Decision-Support Scenarios

The final package models targeted interventions using the validated driver set.

### R1 — Targeted reminders

Target:

* **1,038** long-lead (≥15 days) appointments currently receiving no reminder.

The measured long-lead reminder gap is **4.22 percentage points**, with `p = 0.0223`.

This relationship is explicitly treated as **associational, not causal**.

### R2 — History-aware confirmations

Target:

* **531** appointments belonging to patients with **2+ previous no-shows**.

Their observed no-show rate is **61.02%**, compared with the overall rate of **48.46%**.

### R1 + R2 target union

The combined target group contains:

* **1,464 appointments**
* **29.3% of appointment volume**
* **57.65% no-show rate**
* **34.8% of all no-shows**

### Scenario range

| Scenario          | Assumption                   | Estimated no-shows prevented / month | Estimated / year |
| ----------------- | ---------------------------- | -----------------------------------: | ---------------: |
| S1 — Conservative | 25% of measured association  |                                 ~0.6 |               ~7 |
| S2 — Central      | 50% of measured association  |                                 ~1.2 |              ~15 |
| S3 — Optimistic   | 100% of measured association |                                 ~2.4 |              ~29 |

**S3 is a ceiling, not a forecast.** The scenarios use assumptions applied to an observed association; they do not establish that reminders cause the estimated reduction.

---

## Validation

Week 7's testing suite contained **20 validation tests**, of which:

* **19/20 passed**
* The primary drivers remained stable on unseen chronological holdout data.
* The rule-based risk tier remained stable across five splits.
* The final KPI register was recomputed against the raw dataset.
* Dashboard values were validated against the underlying data.

The one documented failure concerns the unadjusted reminder test on the smaller holdout:

* Holdout `p = 0.292`
* `n = 1,654`

The propensity-adjusted reminder analysis remained significant:

* Adjusted OR = **0.830**
* `p = 0.006`

The project therefore does **not** treat the reminder relationship as causal.

---

## Repository Structure

A complete Week 8 package contains the following core outputs:

```text
.
├── HealthConnect_Week8_Final_Analytics_Decision_Support.ipynb
├── week8_analysis.py
│
├── HealthConnect_Week8_Final_Dashboard.png
├── HealthConnect_Week8_Final_KPI_Register.csv
├── HealthConnect_Week8_Integration_Readiness.csv
├── HealthConnect_Week8_T6_Benchmark_Closure.csv
├── HealthConnect_Week8_Slot_Recovery_Scenarios.csv
├── HealthConnect_Week8_Targeting_Concentration.csv
├── HealthConnect_Week8_Final_Findings_Recommendations.csv
├── HealthConnect_Week8_CrossTrack_Handoff_Register.csv
├── HealthConnect_Week8_Limitation_Register.csv
│
├── HealthConnect_Week5_Cleaned_Dataset.csv
│
├── Week8_Project_Summary.docx
├── Week8_CrossTrack_Collaboration.docx
└── Week8_Video_Presentation_Script.docx
```

The exact contents of a local checkout may vary depending on which upstream Week 5–7 artefacts are included.

---

## Main Deliverables

### `HealthConnect_Week8_Final_Analytics_Decision_Support.ipynb`

The primary final analytics notebook.

It contains:

* Week 7 → Week 8 transition
* Final integration-readiness assessment
* Final KPI register
* T6 benchmark closure
* Slot-recovery scenarios
* Final decision-support dashboard
* Validated findings and recommendations
* Cross-track handoff register
* Limitation register
* Executive summary

### `week8_analysis.py`

The computational engine supporting the Week 8 notebook.

It:

1. Loads the cleaned Week 5 dataset.
2. Recreates analytical bands and the established risk-tier logic.
3. Recomputes holdout effect sizes.
4. Generates the final KPI register.
5. Produces the T6 benchmark closure.
6. Calculates decision-support scenarios.
7. Produces targeting-concentration outputs.
8. Generates the final dashboard.
9. Writes the final findings, handoff, readiness, and limitation CSVs.
10. Prints the final executive summary.

The engine deliberately treats Week 7 outputs as inputs/evidence rather than silently modifying them.

---

## Data Requirements

The analysis expects the cleaned Week 5 dataset:

```text
HealthConnect_Week5_Cleaned_Dataset.csv
```

The analysis uses fields including:

* `booking_date`
* `appointment_date`
* `appointment_outcome`
* `booking_lead_days`
* `distance_to_clinic_km`
* `previous_no_shows`
* `reminder_sent`

The notebook derives:

* `is_noshow`
* `lead_band`
* `prev_ns_band`
* `risk_score`
* `risk_tier`

The source dataset should remain unchanged so that the Week 8 results remain reproducible.

---

## Risk-Tier Logic

The rule-based score is intentionally transparent.

### Booking lead time

```text
31+ days  → +2
15–30 days → +1
<15 days  → +0
```

### Previous no-shows

```text
2+ previous no-shows → +2
1 previous no-show   → +1
0 previous no-shows  → +0
```

### Distance

```text
20+ km → +1
```

### Reminder

```text
No reminder → +1
```

The resulting score is converted into:

```text
0–1 → Low
2–3 → Medium
4–6 → High
```

This rule was retained from the earlier project stages so that Week 8 validates and integrates an established analytical component rather than introducing an unrelated model.

---

## Running the Analysis

### Requirements

The analysis uses Python with the following main packages:

```text
pandas
numpy
scipy
matplotlib
jupyter
```

Install the dependencies with:

```bash
pip install pandas numpy scipy matplotlib jupyter
```

### Run the computational engine

From the repository directory:

```bash
python week8_analysis.py
```

The script expects the cleaned dataset in the parent directory structure used by the project.

### Run the notebook

Launch Jupyter:

```bash
jupyter notebook
```

Then open:

```text
HealthConnect_Week8_Final_Analytics_Decision_Support.ipynb
```

Run the notebook cells from top to bottom.

---

## Generated Outputs

Running the analysis produces the Week 8 evidence package, including:

| Output                                                   | Purpose                                        |
| -------------------------------------------------------- | ---------------------------------------------- |
| `HealthConnect_Week8_Final_KPI_Register.csv`             | Validated monitoring KPIs                      |
| `HealthConnect_Week8_Integration_Readiness.csv`          | Final integration-readiness matrix             |
| `HealthConnect_Week8_T6_Benchmark_Closure.csv`           | Data Science benchmark handoff                 |
| `HealthConnect_Week8_Slot_Recovery_Scenarios.csv`        | Conservative scenario analysis                 |
| `HealthConnect_Week8_Targeting_Concentration.csv`        | Descriptive targeting analysis                 |
| `HealthConnect_Week8_Final_Findings_Recommendations.csv` | Validated findings mapped to recommendations   |
| `HealthConnect_Week8_CrossTrack_Handoff_Register.csv`    | Cross-track dependencies and handoffs          |
| `HealthConnect_Week8_Limitation_Register.csv`            | Remaining limitations and mitigations          |
| `HealthConnect_Week8_Final_Dashboard.png`                | Six-panel executive decision-support dashboard |

---

## Dashboard

The final dashboard contains six panels:

1. **No-shows by booking lead time**
   Shows the validated lead-time relationship with confidence intervals.

2. **Lead time × previous no-show history**
   Shows the compounding risk pattern.

3. **Risk-tier performance**
   Shows Low, Medium, and High tier no-show rates.

4. **Reminder coverage by risk tier**
   Highlights the operational reminder-coverage gap.

5. **Scenario value**
   Shows the estimated annual no-shows prevented under S1–S3 assumptions.

6. **Final status block**
   Summarises validation, benchmark, KPI, targeting, and scenario results.

---

## Cross-Track Integration

The Week 8 analytics package provides explicit handoffs to other project tracks.

### Data Science

Receives:

* Validated feature specification
* Rule-based risk-tier cohort
* Holdout-validated findings
* **68.42%** conservative benchmark

Data Science is responsible for comparing the candidate model's top-decile precision against this benchmark.

### Project Management

Receives:

* Final insights
* KPI definitions
* Limitations
* Integration-readiness information

Project Management owns the final disposition of the unresolved `waiting_time_minutes` issue.

### ML Engineering

Receives:

* Final KPI definitions
* Validated feature definitions
* Monitoring requirements

### Generative AI

Receives:

* Validated findings
* KPI definitions
* Explicit limitation statements

This supports answering attendance-pattern questions without inventing unsupported clinic facts.

---

## Limitations

The findings should be interpreted within the limitations of the dataset and study design.

### 1. Reminder effect is associational

The reminder analysis is not based on randomized assignment. The adjusted OR of `0.830` does not establish causality.

A randomized channel test is identified as the appropriate next step before making causal claims.

### 2. `waiting_time_minutes` remains unresolved

The variable is excluded from the analysis and Data Science feature set pending Project Management's disposition.

### 3. Limited explained variance

The model's pseudo R² is **0.075**, indicating that the measured variables explain only a limited amount of outcome variation.

The outputs should therefore be used primarily for **ranking and prioritisation**, not individual-level prediction.

### 4. Missing clinical and socioeconomic context

The dataset does not capture factors such as:

* Transport constraints
* Work commitments
* Illness
* Cost of care

### 5. No intervention history

The data does not provide a suitable before/after intervention record, limiting causal evaluation.

### 6. Synthetic training data

The findings are internally consistent within the supplied dataset but have not been validated against real clinic operations.

The project should therefore be treated as a **decision-support framework ready for validation with real operational data**, rather than as evidence of guaranteed real-world impact.

---

## Responsible Interpretation

This project intentionally separates:

**Observed evidence**

from:

**Scenario assumptions**

and:

**Causal claims**

In particular:

* Risk tiers describe observed patterns.
* Scenario estimates are assumption-driven.
* The reminder relationship is associational.
* S3 represents an upper-bound scenario rather than a forecast.
* Real-world intervention impact requires controlled testing.

This distinction is central to the project's evidence-first approach.

---

## Project Status

**Week 8 — Final Analytics / Decision Support**

The Analytics component is considered validated for final integration based on:

* 19/20 Week 7 tests passing
* Holdout confirmation of the primary drivers
* Stable risk-tier performance across validation splits
* Recomputed KPI values matching the established register
* Final dashboard validation
* Completed decision-support scenarios
* Completed cross-track handoff register
* Completed limitation register

The remaining external integration work includes the Data Science T6 model comparison and Project Management's disposition of `ISS-09`.

---

## Reproducibility

The project prioritises reproducibility by:

* Reading from the unchanged cleaned dataset
* Recreating analytical bands programmatically
* Recomputing headline metrics
* Recomputing holdout effect sizes
* Validating KPI values against the source data
* Generating evidence CSVs programmatically
* Generating the dashboard from the same analytical data
* Explicitly recording assumptions and limitations

The computational engine is therefore intended to make the final Week 8 results auditable rather than relying solely on manually entered presentation figures.

---

## License

No license information is specified in the supplied project materials.

If this repository is intended for public distribution, add an appropriate license file before publishing.

---

## Acknowledgements

**HealthConnect Clinic Experience Lab**
**AnalystLab Africa Experience Lab**
**Data Analytics Track**

Project focus:

> Improving Patient Appointment Attendance and Healthcare Support Using Data and AI
