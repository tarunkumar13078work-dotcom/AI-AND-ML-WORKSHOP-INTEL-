from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ============================================================
# InsurePredict | Insurance Cost Estimator
# ============================================================
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.pkl"
DATA_PATH = BASE_DIR / "insurance.csv"

FEATURE_COLUMNS = [
    "age", "bmi", "children",
    "sex_female", "sex_male",
    "smoker_no", "smoker_yes",
    "region_northeast", "region_northwest",
    "region_southeast", "region_southwest",
]

st.set_page_config(
    page_title="InsurePredict | Cost Estimator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------ Styling ------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --bg: #080d19;
    --panel: #101a2b;
    --panel-2: #142139;
    --line: rgba(151, 177, 220, .15);
    --text: #edf3ff;
    --muted: #94a6c5;
    --blue: #6d8cff;
    --violet: #9b7bff;
    --mint: #64e0c0;
}
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background:
      radial-gradient(ellipse at 80% 0%, rgba(100, 91, 210, .13), transparent 36%),
      radial-gradient(ellipse at 0% 25%, rgba(48, 113, 190, .10), transparent 30%),
      var(--bg);
    color: var(--text);
}
[data-testid="stHeader"] { background: rgba(8,13,25,.72); }
[data-testid="stSidebar"] {
    background: #0c1423;
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] > div { padding-top: 1.4rem; }
.block-container { padding-top: 2.1rem; padding-bottom: 3rem; max-width: 1440px; }
h1, h2, h3 { font-family: 'Manrope', sans-serif; color: var(--text); letter-spacing: -.5px; }
p, label, .stMarkdown { color: var(--text); }
small, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }

.brand {
    display:flex; align-items:center; gap:11px; padding: 8px 4px 22px;
}
.brand-mark {
    width:42px; height:42px; display:flex; align-items:center; justify-content:center;
    border-radius:14px; font-size:21px; color:white;
    background:linear-gradient(135deg,#547cff,#916bff);
    box-shadow:0 8px 24px rgba(103,123,255,.25);
}
.brand-name { font-family:'Manrope'; font-weight:800; font-size:18px; color:#f4f7ff; }
.brand-sub { color:#8497b8; font-size:10px; letter-spacing:1.6px; text-transform:uppercase; margin-top:1px; }
.hero {
    position:relative; overflow:hidden; padding:30px 34px; border-radius:22px;
    border:1px solid rgba(135,161,216,.20);
    background:linear-gradient(112deg,rgba(28,53,91,.96),rgba(28,35,75,.92) 58%,rgba(67,48,111,.85));
    box-shadow:0 18px 50px rgba(0,0,0,.18); margin-bottom:25px;
}
.hero:after {
    content:""; position:absolute; width:270px; height:270px; right:-65px; top:-145px;
    border:1px solid rgba(255,255,255,.12); border-radius:50%;
    box-shadow:0 0 0 28px rgba(255,255,255,.025),0 0 0 58px rgba(255,255,255,.02);
}
.hero-kicker { color:#b8c9f0; text-transform:uppercase; letter-spacing:2px; font-size:11px; font-weight:800; }
.hero-title { color:white; font-family:'Manrope'; font-size:clamp(27px,3vw,39px); font-weight:800; line-height:1.15; margin:10px 0 9px; }
.hero-copy { color:#c0cce4; font-size:14px; max-width:650px; margin:0; line-height:1.7; }
.section-head { display:flex; align-items:center; gap:10px; margin:10px 0 14px; }
.section-dot { width:8px; height:8px; background:var(--mint); border-radius:50%; box-shadow:0 0 14px rgba(100,224,192,.6); }
.section-title { font-family:'Manrope'; color:#edf3ff; font-weight:800; font-size:18px; }
.section-caption { color:#91a3c2; font-size:12px; margin-top:2px; }
.panel {
    background:linear-gradient(145deg,rgba(20,33,55,.96),rgba(14,24,42,.96));
    border:1px solid var(--line); border-radius:18px; padding:21px 22px;
}
.metric-card {
    background:linear-gradient(145deg,#14233b,#101a2c);
    border:1px solid var(--line); border-radius:16px; padding:17px 19px;
    min-height:105px;
}
.metric-label { color:#91a5c7; font-size:10px; font-weight:800; letter-spacing:1.3px; text-transform:uppercase; }
.metric-value { color:#f3f6ff; font-family:'Manrope'; font-size:25px; font-weight:800; margin-top:7px; }
.metric-note { color:#8295b7; font-size:11px; margin-top:2px; }
.result-panel {
    border:1px solid rgba(117,146,255,.35); border-radius:20px; padding:24px;
    background:radial-gradient(circle at 100% 0%,rgba(136,104,255,.19),transparent 40%),
               linear-gradient(130deg,#172d4c,#1c2041);
    box-shadow:0 15px 40px rgba(0,0,0,.18);
}
.result-label { color:#b4c5e8; font-size:11px; text-transform:uppercase; letter-spacing:1.5px; font-weight:800; }
.result-value { font-family:'Manrope'; color:white; font-size:clamp(31px,4vw,45px); font-weight:800; margin:8px 0; }
.result-foot { color:#9fb0d0; font-size:12px; }
div.stButton > button, div.stFormSubmitButton > button {
    border:0; border-radius:11px; min-height:45px; color:white; font-weight:800;
    background:linear-gradient(100deg,#5f83ff,#8b68f5);
    box-shadow:0 8px 22px rgba(93,119,255,.20);
}
div.stButton > button:hover, div.stFormSubmitButton > button:hover {
    color:white; border:1px solid #a8b7ff; filter:brightness(1.08);
}
.stTextInput input, .stNumberInput input, div[data-baseweb="select"] > div {
    background:#0d1728 !important; border-color:#2a3b59 !important; color:#edf3ff !important;
    border-radius:10px !important;
}
[data-testid="stMetric"] {
    background:#111e32; border:1px solid var(--line); padding:14px 16px; border-radius:14px;
}
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
hr { border-color:var(--line); }
div[data-testid="stAlert"] { border-radius:12px; }
.footer { color:#7184a5; font-size:11px; text-align:center; padding-top:18px; }
.sidebar-note { background:#111f34; border:1px solid var(--line); padding:13px; border-radius:12px; color:#9eb0cf; font-size:11px; line-height:1.6; }
</style>
""", unsafe_allow_html=True)


# ------------------------------ Data loading ------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("model.pkl is missing. Put it beside app.py.")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


try:
    model = load_model()
except Exception as exc:
    st.error(f"Unable to load model.pkl: {exc}")
    st.info("Confirm model.pkl is in the same folder as app.py and install requirements.txt.")
    st.stop()

data = load_data()


# ------------------------------ Sidebar ------------------------------
with st.sidebar:
    st.markdown("""
    <div class="brand">
      <div class="brand-mark">🛡</div>
      <div><div class="brand-name">InsurePredict</div>
      <div class="brand-sub">Intelligent cost estimator</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**WORKSPACE**")
    page = st.radio(
        "Navigate",
        ["Cost estimator", "Model insights", "Dataset explorer"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**MODEL STATUS**")
    st.markdown(
        '<div class="sidebar-note">🟢 &nbsp; Model loaded<br>'
        '<b>Linear Regression</b><br>11 encoded input features<br>'
        'Target: insurance charges</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown(
        '<div class="sidebar-note">🔒 Your inputs are processed by this local app. '
        'No user account or database is configured.</div>',
        unsafe_allow_html=True,
    )


# ------------------------------ Shared hero ------------------------------
st.markdown("""
<div class="hero">
  <div class="hero-kicker">INSURANCE ANALYTICS &nbsp; / &nbsp; MACHINE LEARNING</div>
  <div class="hero-title">Make informed estimates.<br>Plan with clarity.</div>
  <p class="hero-copy">A clean, data-driven way to estimate medical insurance charges.
  Enter a few details and let your trained machine learning model calculate an estimate.</p>
</div>
""", unsafe_allow_html=True)


# ------------------------------ Cost estimator ------------------------------
if page == "Cost estimator":
    st.markdown("""
    <div class="section-head"><div class="section-dot"></div>
      <div><div class="section-title">Customer profile</div>
      <div class="section-caption">Enter the details used by the trained prediction model.</div></div>
    </div>
    """, unsafe_allow_html=True)

    form_col, result_col = st.columns([1.08, 0.92], gap="large")

    with form_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        with st.form("prediction_form"):
            st.markdown("#### Personal details")
            p1, p2 = st.columns(2)
            with p1:
                age = st.slider("Age", 18, 100, 30)
                sex = st.selectbox("Sex", ["female", "male"])
            with p2:
                bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
                children = st.number_input("Children / dependents", min_value=0, max_value=10, value=0, step=1)

            st.markdown("---")
            st.markdown("#### Lifestyle & location")
            p3, p4 = st.columns(2)
            with p3:
                smoker = st.selectbox("Smoking status", ["no", "yes"])
            with p4:
                region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

            submitted = st.form_submit_button("✦  Generate cost estimate", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Store the last result so it remains visible after Streamlit reruns.
    if submitted:
        features = {
            "age": float(age),
            "bmi": float(bmi),
            "children": float(children),
            "sex_female": int(sex == "female"),
            "sex_male": int(sex == "male"),
            "smoker_no": int(smoker == "no"),
            "smoker_yes": int(smoker == "yes"),
            "region_northeast": int(region == "northeast"),
            "region_northwest": int(region == "northwest"),
            "region_southeast": int(region == "southeast"),
            "region_southwest": int(region == "southwest"),
        }
        input_frame = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        try:
            st.session_state["last_prediction"] = {
                "value": float(model.predict(input_frame)[0]),
                "inputs": {
                    "Age": f"{age} years",
                    "Sex": sex.title(),
                    "BMI": f"{bmi:.1f}",
                    "Children": str(children),
                    "Smoker": smoker.title(),
                    "Region": region.title(),
                },
            }
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    with result_col:
        st.markdown("""
        <div class="section-head"><div class="section-dot"></div>
          <div><div class="section-title">Prediction overview</div>
          <div class="section-caption">Your estimated insurance charges.</div></div>
        </div>
        """, unsafe_allow_html=True)

        result = st.session_state.get("last_prediction")
        if result:
            amount = result["value"]
            st.markdown(f"""
            <div class="result-panel">
              <div class="result-label">ESTIMATED INSURANCE CHARGES</div>
              <div class="result-value">${amount:,.2f}</div>
              <div class="result-foot">Output generated by your Linear Regression model</div>
            </div>
            """, unsafe_allow_html=True)
            if amount < 0:
                st.warning("The model returned a negative estimate for these inputs. Linear Regression can produce values that are not realistic at the edges of its training data.")
            st.markdown("#### Submitted profile")
            for label, value in result["inputs"].items():
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;padding:8px 2px;'
                    f'border-bottom:1px solid rgba(151,177,220,.12);">'
                    f'<span style="color:#91a5c7;font-size:13px">{label}</span>'
                    f'<b style="color:#edf3ff;font-size:13px">{value}</b></div>',
                    unsafe_allow_html=True,
                )
            if st.button("Clear prediction", use_container_width=True):
                st.session_state.pop("last_prediction", None)
                st.rerun()
        else:
            st.markdown("""
            <div class="result-panel" style="min-height:220px;display:flex;flex-direction:column;justify-content:center;">
              <div style="font-size:32px;margin-bottom:12px">✧</div>
              <div class="result-label">YOUR ESTIMATE WILL APPEAR HERE</div>
              <div style="color:#dce6fa;font-size:17px;font-weight:700;margin:9px 0">Ready when you are</div>
              <div class="result-foot">Complete the customer profile and generate an estimate to view the result.</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")
            st.markdown(
                '<div class="sidebar-note">ⓘ &nbsp; The estimate is based on patterns in the supplied dataset. '
                'It is not an official insurance quote or eligibility decision.</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")
    st.markdown('<div class="footer">INSUREPREDICT &nbsp; • &nbsp; STREAMLIT ML APPLICATION</div>', unsafe_allow_html=True)


# ------------------------------ Model insights ------------------------------
elif page == "Model insights":
    st.markdown("## Model insights")
    st.caption("Technical overview of the model powering this application.")
    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="metric-card"><div class="metric-label">Algorithm</div><div class="metric-value">Linear Regression</div><div class="metric-note">Supervised learning</div></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="metric-card"><div class="metric-label">Input features</div><div class="metric-value">11</div><div class="metric-note">After one-hot encoding</div></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="metric-card"><div class="metric-label">Prediction target</div><div class="metric-value">Charges</div><div class="metric-note">Estimated numeric value</div></div>', unsafe_allow_html=True)

    st.markdown("### How a prediction is generated")
    steps = [
        ("01", "Collect profile details", "Age, BMI, number of children, sex, smoking status and region."),
        ("02", "Encode categories", "Convert sex, smoker and region into the same one-hot encoded columns used during training."),
        ("03", "Run the model", "Pass the 11 features to the saved Linear Regression estimator."),
        ("04", "Display estimate", "Show the model's predicted insurance charges."),
    ]
    for number, title, description in steps:
        st.markdown(
            f'<div class="metric-card" style="margin:10px 0;display:flex;gap:16px;align-items:center;">'
            f'<div style="font-family:Manrope;color:#9c8cff;font-size:22px;font-weight:800">{number}</div>'
            f'<div><div style="font-weight:800;color:#edf3ff">{title}</div>'
            f'<div style="color:#91a5c7;font-size:12px">{description}</div></div></div>',
            unsafe_allow_html=True,
        )

    with st.expander("View the exact feature order expected by the model"):
        st.code("\n".join(FEATURE_COLUMNS), language="text")
    st.warning("This model estimates charges. It does not classify applicants as approved or rejected, and the displayed estimate should not be treated as a real insurer's quote.")


# ------------------------------ Dataset explorer ------------------------------
elif page == "Dataset explorer":
    st.markdown("## Dataset explorer")
    st.caption("Explore the insurance.csv file supplied with your trained model.")
    if data is None:
        st.warning("insurance.csv is not present beside app.py.")
    else:
        total_rows = len(data)
        total_columns = len(data.columns)
        missing = int(data.isna().sum().sum())
        avg_charge = data["charges"].mean() if "charges" in data.columns else 0

        k1, k2, k3, k4 = st.columns(4)
        for col, label, value, note in [
            (k1, "RECORDS", f"{total_rows:,}", "Rows in dataset"),
            (k2, "FEATURES", str(total_columns - 1), "Input columns"),
            (k3, "MISSING VALUES", f"{missing:,}", "Across all columns"),
            (k4, "AVERAGE CHARGE", f"${avg_charge:,.0f}", "Dataset average"),
        ]:
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

        st.markdown("### Charges distribution")
        if "charges" in data.columns:
            histogram = pd.cut(data["charges"], bins=12).value_counts().sort_index()
            chart_data = pd.DataFrame({"Charge range": histogram.index.astype(str), "Records": histogram.values}).set_index("Charge range")
            st.bar_chart(chart_data, height=280)

        st.markdown("### Dataset records")
        search = st.text_input("Filter rows (search across visible values)", placeholder="Try: southeast, female, yes...")
        view = data.copy()
        if search.strip():
            mask = view.astype(str).apply(lambda column: column.str.contains(search, case=False, na=False)).any(axis=1)
            view = view[mask]
        st.caption(f"Showing {min(len(view), 500)} of {len(view):,} matching records.")
        st.dataframe(view.head(500), use_container_width=True, hide_index=True)

        with st.expander("View descriptive statistics"):
            st.dataframe(data.describe(include="all").transpose(), use_container_width=True)


st.markdown('<div class="footer">INSUREPREDICT &nbsp; • &nbsp; EDUCATIONAL MACHINE LEARNING PROJECT</div>', unsafe_allow_html=True)
