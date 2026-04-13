---
title: "The Health Data Pipeline: Data-Centric Lessons Learned for Data Scientists & MLEs"
date: 2025-11-12
category: "Tutorials"
excerpt: "A data-centric playbook for building robust health data workflows: semantics first, layered architecture, cohort factories, and reproducible analysis."
coverImage: "/assets/images/writing/medium_whatsml.png"
---

## Engineering a Scalable Health Data Workflow

Working as a data scientist or ML engineer on a large-scale health data project is a unique challenge, especially for those without a medical background. You are immediately confronted with deep semantic complexity and business rules that are difficult to grasp. The data was often designed for billing or administration, not research.

This article shares technical lessons learned from navigating this complexity. It is not an exhaustive academic guide, nor is it a tell-all of a specific project. Instead, it is a collection of hard-won lessons and architectural choices that helped us build a robust, reproducible, and effective data workflow.

The goal is to share a data-centric playbook for other technical profiles (DS/MLE) facing this context, focusing on pipeline structure, dynamic cohort management, and scaling analysis.

Although I focus on health data, some of these technical patterns and principles are broadly applicable and can be adapted to any domain facing complex, legacy, or semantically rich data systems.

### The Real Bottleneck: Moving from Technical to Semantic

My first realization was that the primary bottleneck was not technical; it was semantic. Health databases (like the French SNDS) are typically primary-use systems. They are built to run the healthcare system: processing reimbursements, managing billing, and handling hospital administration.

**They were not built for scientific research.**

Consequently, the hardest question is rarely "How do I join 10 tables?" but rather:

> "What does 'diabetic' mean in this database?"
> Is it an ICD-10 diagnostic code? An ATC code for a specific medication? Do we require three occurrences over two years to confirm it?

This single question reveals the true challenge: translating complex clinical rules into code. Defining a comorbidity score (like the Charlson index) or a simple treatment exposure window requires a perfect translation of these rules.

While data volume (tens of millions of rows, hundreds of tables) is an amplifier, it is not the core problem. A simple semantic logic error, when applied to 10 million patients, does not produce a small rounding error. It produces a fundamentally flawed study and dangerous conclusions.

**The Data Dictionary: Our Most Valuable Deliverable**

Given this, diving directly into implementation is reckless. I found that the most valuable deliverable, produced before any heavy ETL, was the living Data Dictionary.

This document is co-created and validated by two key groups:

- Clinicians and researchers who define the need
- Source data experts who know where the data actually lives

This contract maps every business concept (incident patient, comorbidity X) to:

- The exact source tables and columns
- The precise transformation rules (e.g., ICD C18-C21 AND diag_date > surgery_date)
- The choices made to handle ambiguity (e.g., a single code is considered a suspicion and is excluded)

This upfront investment in documentation might feel slow. But it saved weeks of refactoring by ensuring that 80% of the pipeline code was not thrown away after the first validation meeting.

### A Resilient Structure: The Bronze, Silver, Gold Architecture

When facing this semantic complexity, a classic ingestion -> transformation monolith pipeline is brittle. Mixing technical cleaning (nulls, date formats), business logic (calculating diabetes), and analysis aggregation (a final patient-level table) creates a debugging nightmare.

The Bronze/Silver/Gold (B/S/G) architecture is a useful separation of concerns. It turns a monolithic, slow, and opaque recalculation into a series of targeted, faster, and more manageable steps.

**Bronze: The Never Ask Twice Landing Zone**

This is a 1:1, immutable copy of the source data, stored in an efficient format (e.g., Parquet) and partitioned by ingestion date.

- Objective: Data insurance
- Transformation: None, other than basic schema enforcement
- Rationale: In healthcare, data access is a gauntlet of regulation, authorization, and delays. The Bronze layer ensures we never have to ask for the source data again

**Silver: The Single Source of Truth**

This is where the Data Dictionary becomes code. The objective is to create the single, standardized, and enriched source of truth for all analysis (data warehouse).

Transformations:

- Cleaning and standardization: Deduplication, null handling, date standardization
- Mapping: Standardizing disparate coding systems (ICD, ATC, LOINC) to a common vocabulary
- Enrichment and business logic: This is where we join diagnostics, prescriptions, and visits. We create the complex derived variables: comorbidity scores, treatment windows, and the (0/1) flags for pathologies (our diabetic patient)

Key principle: test everything. This layer is worthless without trust. We adopted a simple rule: no data is promoted to Silver without passing data quality tests. These tests are both technical (is the primary key unique?) and semantic (is the prevalence of diabetes plausible, between 5% and 10%?).

**Gold: The Analytics-Ready Data Marts**

The Silver layer is clean but still normalized and complex. A researcher does not want to write 15 joins to get their study population. The objective is to provide analysis-ready tables, aggregated and optimized for recurring research questions.

Example: a `cohort_study_A` table where each row is one patient, and columns are their features (age, sex, comorbidity flags, visit dates).

This layer is what enables speed and reproducibility. Most end-users (researchers, analysts, ML models) will consume data from here.

### Scaling Analysis: The Cohort Factory

The Gold layer is composed of many data marts. Each new scientific study typically requires a new cohort, which means a new Gold table.

You quickly find out that creating these cohorts manually results in 10 nearly-identical-but-subtly-different scripts, creating a maintenance and reproducibility crisis.

The solution is to abstract this process into a Cohort Factory. This is a single, generic codebase that generates any Gold table based on two inputs:

- Access to all Silver tables (the source of truth)
- A configuration file (e.g., YAML) that describes the cohort

Example `config_study_A.yml`:

```yaml
study_name: "study_treatment_X_vs_Y"

# 1. Inclusion/Exclusion criteria (from Silver tables)
inclusion_criteria:
  - { type: "diagnosis", codes: ["C18", "C19"], source: "silver_diagnoses" }
  - { type: "treatment", codes: ["L01X"], dose: "< 25mg", source: "silver_prescriptions" }

exclusion_criteria:
  - { type: "event", code: "Z51.0", source: "silver_acts" }

# 2. Temporal logic (how events interact)
temporal_logic:
  - "treatment after diagnosis"

# 3. Matching logic (if needed)
matching:
  type: "case_control"
  on_criterion: "treatment_X" # The cases
  ratio: 4
  on_variables: ["age_category", "sex"]

seed: 42 # Reproducibility
```

Building this factory was a significant upfront engineering effort. But the payoff was immense:

- Abstraction: We write and test the complex join/matching/temporal logic once
- Reproducibility: The version-controlled YAML file is the exact documentation for the cohort. The seed ensures any random matching is reproducible
- Speed: A new cohort study becomes a 3-hour configuration and execution task

### Standardizing Analysis: From Table 1 to Inference

With a `gold_study_A` table created, the analysis process itself can be standardized.

**Phase 1: The Automated Descriptive Foundation**

This is a go/no-go step executed every time a Gold table is generated. It answers: "Who is in my cohort, and is the data sane?"

This automated script produces two key artifacts:

- The sanity check report: A technical data profile (pandas-profiling style) to catch pipeline errors (e.g., 80% unexpected nulls, wrong data types)
- Table 1 (for researchers): This is the classic, publication-ready descriptive table found in medical journals. It summarizes the population (N (%) for categorical, Mean (SD) for continuous) and stratifies it by the group of interest (e.g., Column 1 = Cases, Column 2 = Controls)

**Phase 2: Statistical Inference**

Once the descriptive foundation is validated, the goal shifts to inference: understanding the relationships between variables and the outcome (e.g., developing the disease) to answer the research question.

This is where the methodology becomes study-specific. Some protocols will require survival analysis (e.g., Cox Proportional-Hazards models), others will use logistic regression (simple, conditional, pooled, etc.).

While the specific model changes, we can establish structuring principles to keep the work reproducible and efficient:

- Encapsulate the analysis: We standardize the analysis by creating a single, configurable pipeline (e.g., a statsmodels pipeline) that encapsulates all preprocessing and modeling. This allows us to iterate on covariates without rewriting the entire analysis script.
- Provide key diagnostic metrics: To facilitate interpretation and comparison between models, we make it a rule to always output diagnostic scores alongside results. This includes model fit metrics like AIC and variable diagnostics like the VIF to check for multicollinearity.

### From Inference to Prediction: Key Lessons in Clinical ML

Sometimes the goal is not inference (explaining a phenomenon) but prediction. This paradigm shift has immediate technical consequences: black-box models (XGBoost, LightGBM) are welcome, and metrics shift from p-values to AUC and F1-scores.

Here are the most critical lessons I learned.

**Lesson 1: Handle Class Imbalance with Care**

Severe class imbalance is the norm in health data (e.g., predicting a rare adverse event). While techniques like SMOTE (oversampling) can sometimes improve a model's F1-score, they come at a high cost.

Oversampling fundamentally breaks the model's probability calibration. By showing the model a fake 50/50 world, its output of 80% probability no longer means 80% of similar patients. I found it far safer to use loss function weighting (penalizing errors on the minority class more), which boosts recall without distorting the input data distribution.

**Lesson 2: Calibration is Non-Negotiable**

An ML model with a 0.90 F1-score can be useless (or even dangerous) in a clinical setting.

A physician does not act on an F1-score. They act on a probability. If the model says this patient has an 80% risk of readmission, the doctor will trigger a heavy intervention. But if, in reality, patients in that 80% bucket only have a 10% risk (the model is uncalibrated), the model is disastrous.

In health, a probability must mean a probability.

I learned to treat the Brier Score as a primary metric, just as important as AUC. For models notorious for over-confidence (like Gradient Boosting and XGB), adding a post-training calibration step (Isotonic Regression) was a mandatory part of the pipeline.

**Lesson 3: Embrace ML Engineering (Even at the Start)**

For a model to be reproducible and trustworthy, every run must be logged. Using tools like MLflow or W&B from day one is not optional. We must (at least) track:

- Parameters: Hyperparameters, feature set, and the random seed for the data split
- Artifacts: The resulting metrics (AUC, Brier), key graphs (calibration plots, confusion matrix), and the serialized model file

### Conclusion: A Framework for Trust

The success of a health data science project is not measured by algorithmic complexity, but by the rigor of its data preparation.

This framework is not just a set of absolute rules, but a collection of guardrails learned from the field:

1. The challenge is semantic: The most expensive errors are not in code, but in a misunderstanding of a clinical term. The Data Dictionary is probably your most valuable asset.
2. Isolate to mitigate: The B/S/G architecture is a risk-management strategy. It separates technical cleaning (Silver) from study-specific creation (Gold), making the system testable and debuggable.
3. Automate for reproducibility: In research, iteration is constant. Automating cohort creation (the factory) and analysis is the only way to scale while ensuring consistency.
4. Prioritize robustness over performance: In clinical ML, a well-calibrated, interpretable (SHAP), and reproducible model is infinitely more valuable than a poorly understood model with a slightly higher F1-score.

Ultimately, this technical framework is about building a foundation of trust: trust with the clinicians who define the logic, and trust in the results that may one day impact patient care.
