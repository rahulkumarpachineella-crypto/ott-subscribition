import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from data_generator import generate_ott_dataset
from model import train_churn_model, score_users, top_risk_factors_for_user
from retention_engine import recommend_actions

st.set_page_config(page_title="StreamGuard — OTT Churn & Retention AI", page_icon="🎬", layout="wide")

# ---------------------------------------------------------------------------
# Cached data + model (computed once per deployment, then reused)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    return generate_ott_dataset(n_users=6000)

@st.cache_resource
def get_trained_model(df):
    return train_churn_model(df)

@st.cache_data
def get_scored_df(_trained, df):
    return score_users(_trained, df)


raw_df = load_data()
trained = get_trained_model(raw_df)
scored_df = get_scored_df(trained, raw_df)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🎬 StreamGuard")
st.sidebar.caption("AI churn prediction & retention engine for OTT platforms")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview Dashboard", "🔍 User Lookup", "📁 Bulk Predict (Upload CSV)", "🧪 Campaign Simulator", "ℹ️ About / Model"],
)
st.sidebar.divider()
st.sidebar.metric("Model AUC", f"{trained.auc:.3f}")
st.sidebar.metric("Model Accuracy", f"{trained.accuracy:.1%}")

# ---------------------------------------------------------------------------
# Page: Overview Dashboard
# ---------------------------------------------------------------------------
if page == "📊 Overview Dashboard":
    st.title("Churn Risk Overview")

    col1, col2, col3, col4 = st.columns(4)
    total_users = len(scored_df)
    high_risk = (scored_df["risk_segment"] == "High").sum()
    med_risk = (scored_df["risk_segment"] == "Medium").sum()
    revenue_at_risk = scored_df.loc[scored_df["risk_segment"] == "High", "monthly_fee"].sum()

    col1.metric("Total Users", f"{total_users:,}")
    col2.metric("High Risk Users", f"{high_risk:,}", f"{high_risk/total_users:.1%} of base")
    col3.metric("Medium Risk Users", f"{med_risk:,}", f"{med_risk/total_users:.1%} of base")
    col4.metric("Monthly Revenue at Risk", f"${revenue_at_risk:,.0f}", "from High-risk users")

    c1, c2 = st.columns(2)
    with c1:
        seg_counts = scored_df["risk_segment"].value_counts().reindex(["Low", "Medium", "High"])
        fig = px.pie(
            names=seg_counts.index, values=seg_counts.values,
            color=seg_counts.index,
            color_discrete_map={"Low": "#2ecc71", "Medium": "#f1c40f", "High": "#e74c3c"},
            title="Users by Risk Segment", hole=0.45,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        imp = trained.feature_importances.head(8).sort_values()
        fig2 = px.bar(
            x=imp.values, y=[i.replace("_enc", "").replace("_", " ").title() for i in imp.index],
            orientation="h", title="Top Churn Drivers (model feature importance)",
            labels={"x": "Relative importance", "y": ""},
        )
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig3 = px.histogram(
            scored_df, x="churn_probability", nbins=30, color="risk_segment",
            color_discrete_map={"Low": "#2ecc71", "Medium": "#f1c40f", "High": "#e74c3c"},
            title="Distribution of Churn Probability",
        )
        st.plotly_chart(fig3, use_container_width=True)
    with c4:
        by_plan = scored_df.groupby("plan_type")["churn_probability"].mean().sort_values(ascending=False)
        fig4 = px.bar(x=by_plan.index, y=by_plan.values, title="Avg Churn Probability by Plan Type",
                      labels={"x": "Plan", "y": "Avg churn probability"})
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Highest-Risk Users")
    top_risk_tbl = scored_df.sort_values("churn_probability", ascending=False).head(15)[
        ["user_id", "plan_type", "monthly_fee", "days_since_last_login",
         "avg_watch_hours_per_week", "churn_probability", "risk_segment"]
    ]
    st.dataframe(top_risk_tbl, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Page: User Lookup
# ---------------------------------------------------------------------------
elif page == "🔍 User Lookup":
    st.title("Individual User Risk & Retention Plan")
    user_id = st.selectbox("Select a user", scored_df["user_id"].tolist())
    user_row = scored_df[scored_df["user_id"] == user_id].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Churn Probability", f"{user_row['churn_probability']:.1%}")
    c2.metric("Risk Segment", user_row["risk_segment"])
    c3.metric("Plan", f"{user_row['plan_type']} (${user_row['monthly_fee']})")

    st.progress(min(float(user_row["churn_probability"]), 1.0))

    st.subheader("Usage Snapshot")
    snap_cols = st.columns(4)
    snap_cols[0].metric("Days Since Login", int(user_row["days_since_last_login"]))
    snap_cols[1].metric("Weekly Watch Hrs", f"{user_row['avg_watch_hours_per_week']:.1f}")
    snap_cols[2].metric("Completion Rate", f"{user_row['content_completion_rate']:.0%}")
    snap_cols[3].metric("Tenure (months)", int(user_row["subscription_tenure_months"]))

    st.subheader("🎯 Why this user is at risk")
    factors = top_risk_factors_for_user(trained, user_row, top_n=4)
    for f in factors:
        st.markdown(f"- {f}")

    st.subheader("💡 Recommended Retention Actions")
    actions = recommend_actions(user_row, user_row["risk_segment"])
    for a in actions:
        with st.container(border=True):
            st.markdown(f"**{a['action']}**")
            st.caption(f"Why: {a['why']}  \nChannel: {a['channel']}")

# ---------------------------------------------------------------------------
# Page: Bulk Predict
# ---------------------------------------------------------------------------
elif page == "📁 Bulk Predict (Upload CSV)":
    st.title("Bulk Churn Prediction")
    st.write(
        "Upload a CSV with your real user data (same columns as the demo dataset) "
        "to score churn risk and get retention recommendations at scale."
    )
    st.download_button(
        "Download sample CSV template",
        raw_df.drop(columns=["churn"]).head(20).to_csv(index=False),
        file_name="ott_users_template.csv",
    )

    uploaded = st.file_uploader("Upload user data CSV", type=["csv"])
    if uploaded is not None:
        try:
            new_df = pd.read_csv(uploaded)
            required = [c for c in raw_df.columns if c != "churn"]
            missing = [c for c in required if c not in new_df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                result = score_users(trained, new_df)
                st.success(f"Scored {len(result)} users.")
                st.dataframe(
                    result.sort_values("churn_probability", ascending=False),
                    use_container_width=True, hide_index=True,
                )
                st.download_button(
                    "Download scored results",
                    result.to_csv(index=False),
                    file_name="scored_users.csv",
                )
        except Exception as e:
            st.error(f"Could not process file: {e}")

# ---------------------------------------------------------------------------
# Page: Campaign Simulator
# ---------------------------------------------------------------------------
elif page == "🧪 Campaign Simulator":
    st.title("Retention Campaign ROI Simulator")
    st.write("Estimate the revenue impact of targeting at-risk users with a retention campaign.")

    target_segment = st.selectbox("Target segment", ["High", "Medium", "High + Medium"])
    if target_segment == "High + Medium":
        target_df = scored_df[scored_df["risk_segment"].isin(["High", "Medium"])]
    else:
        target_df = scored_df[scored_df["risk_segment"] == target_segment]

    campaign_cost_per_user = st.slider("Campaign cost per targeted user ($)", 0.0, 10.0, 1.5, 0.25)
    expected_retention_lift = st.slider("Expected reduction in churn probability from campaign", 0.0, 0.9, 0.30, 0.05)
    horizon_months = st.slider("Revenue horizon to project (months)", 1, 24, 12)

    n_targeted = len(target_df)
    baseline_expected_churners = (target_df["churn_probability"] * target_df["monthly_fee"]).sum()
    reduced_churn_prob = target_df["churn_probability"] * (1 - expected_retention_lift)
    saved_monthly_revenue = ((target_df["churn_probability"] - reduced_churn_prob) * target_df["monthly_fee"]).sum()
    total_campaign_cost = n_targeted * campaign_cost_per_user
    projected_revenue_saved = saved_monthly_revenue * horizon_months
    roi = (projected_revenue_saved - total_campaign_cost) / total_campaign_cost if total_campaign_cost > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users Targeted", f"{n_targeted:,}")
    c2.metric("Campaign Cost", f"${total_campaign_cost:,.0f}")
    c3.metric(f"Revenue Saved ({horizon_months}mo)", f"${projected_revenue_saved:,.0f}")
    c4.metric("Projected ROI", f"{roi:.0%}")

    months = list(range(1, horizon_months + 1))
    cumulative = [saved_monthly_revenue * m - total_campaign_cost for m in months]
    fig = px.line(x=months, y=cumulative, markers=True,
                  labels={"x": "Month", "y": "Cumulative net revenue impact ($)"},
                  title="Cumulative Net Impact of Retention Campaign")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
else:
    st.title("About StreamGuard")
    st.markdown(
        """
StreamGuard is an end-to-end AI churn prediction and retention system for OTT
(video/audio streaming) platforms.

**How it works**
1. A `RandomForestClassifier` is trained on user behavioral features (watch time,
   login recency, completion rate, support tickets, payment failures, plan
   history, etc.) to predict probability of churn.
2. Each user is placed into a **Low / Medium / High** risk segment.
3. A per-user **explainability layer** ranks which specific factors are driving
   *that user's* risk (not just global feature importance).
4. A **rule-based retention engine** maps those specific drivers to concrete,
   personalized actions — a different playbook for "inactive user" vs.
   "price-sensitive user" vs. "payment-failure user".
5. A **campaign simulator** lets a growth/retention team estimate ROI before
   spending budget on an intervention.

**Model performance on held-out test data**
- ROC-AUC: {auc:.3f}
- Accuracy: {acc:.1%}

Swap in your real user data via the **Bulk Predict** page — no retraining code
needed as long as the columns match the template.
        """.format(auc=trained.auc, acc=trained.accuracy)
    )
