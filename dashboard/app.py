"""
AegisChain — Energy Supply Chain War Room
Run with:  streamlit run app.py   (from the dashboard/ folder)
"""

import sys
import os
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "optimizer"))
sys.path.insert(0, str(THIS_DIR.parent / "cv"))
sys.path.insert(0, str(THIS_DIR))

import streamlit as st
import pandas as pd

from model import solve_scenario, SUPPLIERS, CORRIDOR, TRANSIT_DAYS, BASE_RISK
from report import decision_card
from congestion import PORTS
from forecast import forecast_congestion, should_preempt

# Real CV pipeline is optional — if ultralytics/torch aren't available
# (e.g. a resource-constrained deployment), the app still runs fully
# on manual sliders. This keeps the deployed site reliable for judges
# even if the CV dependency is heavy for the hosting environment.
CV_AVAILABLE = True
try:
    from infer_congestion import image_to_congestion_index
except Exception:
    CV_AVAILABLE = False

st.set_page_config(page_title="AegisChain War Room", layout="wide")

st.title("🛢️ AegisChain — Energy Supply Chain War Room")
st.caption("AI Decision Intelligence Platform for Energy Supply Chains — demonstrated on an India–Hormuz scenario")

# ---------------------------------------------------------------------------
# SIDEBAR — scenario controls
# ---------------------------------------------------------------------------
st.sidebar.header("Scenario Controls")

disrupt = st.sidebar.toggle("⚠️ Trigger: Strait of Hormuz closes", value=False)

corridor_capacity_pct = st.sidebar.slider(
    "Hormuz corridor capacity remaining (%)", min_value=0, max_value=100, value=10, step=5,
    disabled=not disrupt,
)

st.sidebar.subheader("JNPT Port Congestion")

cv_mode_options = ["Manual (slider)"]
if CV_AVAILABLE:
    cv_mode_options.append("Live CV (upload satellite image)")
input_mode = st.sidebar.radio("Congestion input source", cv_mode_options, key="input_mode")

jnpt_current = None  # the "now" reading — feeds both the optimizer AND the forecast trend

if input_mode == "Live CV (upload satellite image)":
    uploaded = st.sidebar.file_uploader("Upload a port satellite image", type=["jpg", "jpeg", "png"])
    baseline_count = st.sidebar.number_input(
        "Baseline ship count (normal day)", min_value=1, value=20, step=1,
        help="Typical/median ship count for this port on a normal day — calibrate from historical images."
    )
    if uploaded is not None:
        temp_path = str(THIS_DIR / "_uploaded_temp.jpg")
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        with st.sidebar:
            with st.spinner("Running ship detection..."):
                result = image_to_congestion_index(temp_path, baseline_count)
        jnpt_current = result["congestion_index"]
        st.sidebar.image(uploaded, caption=f"{result['ship_count']} ships detected", use_container_width=True)
        st.sidebar.metric("Computed congestion index", jnpt_current)
    else:
        st.sidebar.info("Upload an image to compute a live congestion index.")
        jnpt_current = 1.0
else:
    jnpt_current = st.sidebar.slider("JNPT congestion multiplier (current)", 1.0, 2.5,
                                       1.35 if disrupt else 1.0, 0.05)

st.sidebar.subheader("Other Ports (manual)")
port_congestion = {}
for port in PORTS:
    if port == "JNPT":
        continue
    default = 1.2 if (port == "Kandla" and disrupt) else 1.0
    mult = st.sidebar.slider(f"{port} congestion multiplier", 1.0, 2.0, default, 0.05, key=port)
    if mult > 1.0:
        port_congestion[port] = mult

if jnpt_current and jnpt_current > 1.0:
    port_congestion["JNPT"] = jnpt_current

# ---------------------------------------------------------------------------
# CONGESTION TREND — now wired to the current JNPT reading above.
# The text box holds PAST readings only; "now" is appended automatically
# from whichever source (slider or live CV) is active above, so moving
# the slider or uploading a new image directly changes the forecast.
# ---------------------------------------------------------------------------
st.sidebar.subheader("JNPT Congestion Trend (for forecast)")
past_input = st.sidebar.text_input(
    "Past readings, comma-separated (not including now)", "1.0, 1.05, 1.1, 1.2"
)

run_button = st.sidebar.button("▶ Run Optimization", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# BUILD SCENARIO KWARGS
# ---------------------------------------------------------------------------
scenario_kwargs = {}
if disrupt:
    scenario_kwargs["disrupted_corridor"] = "Hormuz"
    scenario_kwargs["corridor_capacity_multiplier"] = corridor_capacity_pct / 100.0
if port_congestion:
    scenario_kwargs["port_congestion"] = port_congestion

scenario_label = "hormuz_closure" if disrupt else "current_state"

if run_button or "card" not in st.session_state:
    with st.spinner("Solving optimization..."):
        card = decision_card(scenario_kwargs, scenario_label=scenario_label)
        st.session_state["card"] = card
        st.session_state["scenario_kwargs"] = scenario_kwargs
        st.session_state["scenario_desc"] = (
            f"Strait of Hormuz closes to {corridor_capacity_pct}% capacity, "
            f"port congestion: {port_congestion or 'none'}"
            if disrupt else "Current baseline operations, no disruption"
        )

card = st.session_state["card"]

# ---------------------------------------------------------------------------
# BEFORE / AFTER METRIC CARDS
# ---------------------------------------------------------------------------
st.subheader("Decision Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Expected Delay (days)", card["after"]["expected_delay_days"],
    delta=round(card["after"]["expected_delay_days"] - card["before"]["expected_delay_days"], 1),
    delta_color="inverse",
)
col2.metric(
    "Risk (%)", card["after"]["risk_pct"],
    delta=round(card["after"]["risk_pct"] - card["before"]["risk_pct"], 1),
    delta_color="inverse",
)
col3.metric(
    "Cost (₹Cr/week)", card["after"]["cost_cr_per_week"],
    delta=f"{card['cost_change_pct']:+.1f}%", delta_color="inverse",
)
col4.metric("Confidence", f"{card['confidence_pct']}%")

if card["after"].get("shortfall_MMbbl"):
    st.error(
        f"⚠ Unmet demand: {card['after']['shortfall_MMbbl']} MMbbl/week shortfall — "
        f"operational impact, not just a cost increase."
    )

st.caption(f"Top recommendation: **{card['top_recommendation']}**")

# ---------------------------------------------------------------------------
# ALLOCATION COMPARISON
# ---------------------------------------------------------------------------
st.subheader("Allocation: Baseline vs Scenario")

def alloc_to_df(result, label):
    rows = []
    for (i, j), qty in result["allocation"].items():
        rows.append({"Supplier": i, "Refinery": j, "Volume (MMbbl/wk)": round(qty, 2),
                     "Corridor": CORRIDOR[i], "Scenario": label})
    return pd.DataFrame(rows)

df_before = alloc_to_df(card["baseline_raw"], "Baseline")
df_after = alloc_to_df(card["scenario_raw"], "Scenario")
combined = pd.concat([df_before, df_after], ignore_index=True)

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Baseline**")
    st.dataframe(df_before, hide_index=True, use_container_width=True)
with col_b:
    st.markdown("**Scenario**")
    st.dataframe(df_after, hide_index=True, use_container_width=True)

supplier_totals = combined.groupby(["Supplier", "Scenario"])["Volume (MMbbl/wk)"].sum().unstack(fill_value=0)
st.bar_chart(supplier_totals)

# ---------------------------------------------------------------------------
# CONGESTION FORECAST — wired to slider/CV via jnpt_current appended to history
# ---------------------------------------------------------------------------
st.subheader("Port Congestion Forecast (24–72h)")
try:
    past = [float(x.strip()) for x in past_input.split(",") if x.strip()]
    history = past + [jnpt_current]
    fc = forecast_congestion(history)
    preempt = should_preempt(history)
    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
    fcol1.metric("Current (from control above)", history[-1])
    fcol2.metric("+24h", fc[24])
    fcol3.metric("+48h", fc[48])
    fcol4.metric("+72h", fc[72])
    if preempt:
        st.warning("📈 Forecast crosses congestion threshold within 48h — recommend proactive rerouting now, "
                    "before the port actually clogs.")
    else:
        st.success("Congestion trend stable — no preemptive action needed.")
except ValueError:
    st.error("Enter valid comma-separated numbers for past readings.")

# ---------------------------------------------------------------------------
# LLM POLICY COPILOT
# ---------------------------------------------------------------------------
st.subheader("🤖 Policy Copilot — Why this recommendation?")

if st.button("Explain this decision"):
    try:
        from explain_gemini import explain
        with st.spinner("Generating explanation..."):
            text = explain(card["baseline_raw"], card["scenario_raw"], st.session_state["scenario_desc"])
        st.info(text)
    except Exception as e:
        st.error(f"Couldn't reach the explanation model: {e}\n\n"
                 f"Check that GEMINI_API_KEY is set (in .env locally, or Streamlit Cloud secrets when deployed).")

if not CV_AVAILABLE:
    st.sidebar.caption("ℹ️ Live CV mode unavailable in this environment (ultralytics not installed/loadable).")
