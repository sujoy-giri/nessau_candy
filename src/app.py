import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
import sys
sys.path.append(".")
from geo_utils import haversine

st.set_page_config(
    page_title="Nassau Candy | Factory Optimization", page_icon="🏭", layout="wide"
)

DATA_PATH = "../outputs/cleaned_data.csv"
FACTORY_PATH = "../data/factories.csv"
MODEL_PATH = "../models/random_forest_model.joblib"
REC_PATH = "../outputs/reassignment_recommendations.csv"
CLUSTER_PATH = "../outputs/route_clusters.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    factories = pd.read_csv(FACTORY_PATH)
    recs = pd.read_csv(REC_PATH)
    clusters = pd.read_csv(CLUSTER_PATH)
    return df, factories, recs, clusters

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df, factories, recs, clusters = load_data()
bundle = load_model()
pipeline = bundle["pipeline"]

st.title("🏭 Factory Reallocation & Shipping Optimization")
st.caption("Nassau Candy Distributor — Decision Intelligence Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔧 Factory Optimization Simulator", "⚖️ What-If Scenario Analysis",
     "📋 Recommendation Dashboard", "⚠️ Risk & Impact Panel"]
)


with tab1:
    st.subheader("Simulate lead time across all factories for a product")

    col1, col2, col3 = st.columns(3)
    with col1:
        product = st.selectbox("Product", sorted(df["Product Name"].unique()))

    available_regions = sorted(recs[recs["Product Name"] == product]["Region"].unique())
    available_modes = sorted(recs[recs["Product Name"] == product]["Ship Mode"].unique())

    with col2:
        region = st.selectbox("Destination Region", available_regions)
    with col3:
        ship_mode = st.selectbox("Ship Mode", available_modes)

    prod_row = df[df["Product Name"] == product].iloc[0]
    division = prod_row["Division"]
    current_factory = prod_row["Factory"]
    units = df[df["Product Name"] == product]["Units"].mean()

    region_rows = df[df["Region"] == region]
    dest_lat = region_rows["Dest_Lat"].mean()
    dest_lon = region_rows["Dest_Lon"].mean()


    sim_rows = []
    for _, frow in factories.iterrows():
        dist = haversine(frow["Latitude"], frow["Longitude"], dest_lat, dest_lon)
        X = pd.DataFrame([{
            "Product Name": product, "Factory": frow["Factory"], "Region": region,
            "Ship Mode": ship_mode, "Division": division,
            "Shipping_Distance_KM": dist, "Units": units,
        }])
        pred = pipeline.predict(X)[0]
        margin = df[df["Factory"] == frow["Factory"]]["Profit_Margin"].mean()
        sim_rows.append({
            "Factory": frow["Factory"], "Distance (km)": round(dist, 1),
            "Predicted Lead Time (days)": round(pred, 2),
            "Avg Profit Margin": round(margin, 3),
            "Current": "✅" if frow["Factory"] == current_factory else "",
        })

    sim_df = pd.DataFrame(sim_rows).sort_values("Predicted Lead Time (days)")
    st.dataframe(sim_df, use_container_width=True, hide_index=True)

    fig = px.bar(
        sim_df.sort_values("Predicted Lead Time (days)"),
        x="Factory", y="Predicted Lead Time (days)",
        color="Current",
        title=f"Predicted lead time by factory — {product} → {region} ({ship_mode})",
    )
    st.plotly_chart(fig, use_container_width=True)



with tab2:
    st.subheader("Compare current vs. recommended assignment")

    match = recs[
        (recs["Product Name"] == product)
        & (recs["Region"] == region)
        & (recs["Ship Mode"] == ship_mode)
    ]

    if len(match):
        row = match.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Current Factory", row["Current Factory"])
        c2.metric("Recommended Factory", row["Recommended Factory"],
                   delta=("No change" if row["Action"] == "Keep" else "Reassign"))
        c3.metric("Lead Time Reduction", f"{row['Lead_Time_Reduction_%']}%",
                   delta=f"{row['Current Lead Time'] - row['Recommended Lead Time']:.2f} days")
    else:
        st.warning("No pre-computed scenario for this exact combination yet.")


with tab3:
    st.subheader("Ranked factory reassignment recommendations")

    action_filter = st.multiselect(
        "Action", options=recs["Action"].unique(), default=["Reassign"]
    )
    filtered = recs[recs["Action"].isin(action_filter)] if action_filter else recs

    st.dataframe(
        filtered.sort_values("Lead_Time_Reduction_%", ascending=False),
        use_container_width=True, hide_index=True,
    )


    top_factories = filtered[filtered["Action"] == "Reassign"]["Recommended Factory"].value_counts()
    if len(top_factories):
        fig3 = px.pie(
            values=top_factories.values, names=top_factories.index,
            title="Recommended factory reassignment distribution",
        )
        st.plotly_chart(fig3, use_container_width=True)




with tab4:
    st.subheader("Profit impact alerts & high-risk reassignments")

    high_risk = recs[(recs["Risk"] == "High") & (recs["Action"] == "Reassign")]
    st.metric("High-risk reassignments flagged", len(high_risk))

    if len(high_risk):
        st.dataframe(
            high_risk[["Product Name", "Region", "Current Factory", "Recommended Factory",
                       "Lead_Time_Reduction_%", "Avg_Profit_Margin"]],
            use_container_width=True, hide_index=True,
        )
        st.warning(
            "These combinations show thin profit margins — reassignment could improve "
            "speed but review cost impact before executing."
        )
    else:
        st.success("No high-risk reassignments flagged under current thresholds.")

    st.subheader("Consistently slow routes (clustering output)")
    slow = clusters[clusters["Cluster_Label"] == "Consistently Slow"].sort_values(
        "order_count", ascending=False
    )
    st.dataframe(
        slow[["Region", "Product Name", "Factory", "avg_lead_time", "order_count"]].head(20),
        use_container_width=True, hide_index=True,
    )