# InsurePredict — Streamlit Insurance Cost Estimator

A Streamlit web app that uses the supplied `model.pkl` Linear Regression model to estimate medical insurance charges.

## Project files

- `app.py` — Streamlit application
- `model.pkl` — trained model supplied for this project
- `insurance.csv` — dataset supplied for this project
- `requirements.txt` — Python dependencies

## Run locally

1. Install Python 3.10–3.12.
2. Open a terminal in this project folder.
3. Create and activate a virtual environment (recommended):

   Windows:
   ```bash
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   ```

4. Install dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Start the app:

   ```bash
   streamlit run app.py
   ```

6. Streamlit will show a local URL, usually `http://localhost:8501`.

## Important model notes

- The saved model is a `sklearn.linear_model.LinearRegression` estimator.
- It expects 11 columns in this exact order:
  `age`, `bmi`, `children`, `sex_female`, `sex_male`, `smoker_no`, `smoker_yes`,
  `region_northeast`, `region_northwest`, `region_southeast`, `region_southwest`.
- The app creates the one-hot encoded inputs to match that order.
- The model was serialized using joblib-compatible format and should be loaded with `joblib.load`, not `pickle.load`.
- The model was trained with scikit-learn 1.6.1, so the requirements pin that version.
- The output is an estimate of insurance charges, not a decision about insurance approval or eligibility.

## Security

Only load pickle/joblib model files from a trusted source. Deserializing an untrusted model file can execute code.
