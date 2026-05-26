# Diabetes Risk Assessment

A web app that estimates a person's risk of type 2 diabetes from a short
lifestyle and health questionnaire. It returns a **calibrated risk score**, a
**high-risk or low-risk flag**, and the **top factors** driving that individual's result.

**Live app:** https://diabetes-prediction-app-wqd7003.streamlit.app/

> ⚠️ **Not a medical device.** This is an educational screening tool built on
> self-reported survey data. It does not diagnose diabetes and is not a substitute
> for advice from a qualified healthcare professional.

---

## What it does

1. The user answers ~15 questions (age, sex, height/weight, diet, activity, smoking,
   existing conditions, general & mental health).
2. The answers are turned into the 22 engineered features the model expects.
3. A calibrated XGBoost pipeline returns the probability of diabetes.
4. The probability is compared against a tuned decision threshold to produce a
   high/low flag.
5. SHAP attributes the prediction to the user's actual answers, and the UI shows the
   top factors that raised or lowered their risk.

## The model

| | |
|---|---|
| **Data** | [BRFSS 2017](https://www.cdc.gov/brfss/annual_data/annual_2017.html) (CDC Behavioral Risk Factor Surveillance System), ~13.7% positive class |
| **Final model** | XGBoost classifier wrapped in `CalibratedClassifierCV` |
| **Pipeline** | leakage-safe `sklearn` Pipeline, sentinel-code recoding, imputation + missingness flags, scaling, one-hot encoding |
| **Operating point** | decision threshold ≈ `0.135`, tuned on validation data for a screening recall target (sensitivity ≥ 0.80) |
| **Explanations** | per-instance SHAP `TreeExplainer` contributions, aggregated back to human-readable factors |

The full modelling workflow follows CRISP-DM and is documented in
[`diabetes_prediction.ipynb`](diabetes_prediction.ipynb), benchmarking Logistic
Regression, Decision Tree, Random Forest, Gaussian Naïve Bayes, and XGBoost, with
threshold tuning and probability calibration. The trained pipeline, threshold, and
feature order are serialised to `diabetes_model.joblib`.

## Tech stack

- **Backend / inference:** Python, Streamlit, scikit-learn `1.7.1`, XGBoost, SHAP
- **Frontend:** custom Streamlit component built with React 18 + TypeScript +
  Framer Motion (`components/frontend/`), rendered full-bleed inside the Streamlit page

## License
For educational use. Built on public CDC BRFSS 2017 data.