import streamlit as st
from model import train, predict

st.set_page_config(page_title="Online Shoppers Purchase Prediction", page_icon="🛒")


@st.cache_resource
def load():
    return train()


model, columns, metrics, months, visitors = load()

st.title("🛒 Online Shopper Purchase Prediction")
st.write("Enter a visitor's session details to predict whether they will make a purchase.")

with st.sidebar:
    st.header("Model performance")
    st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    st.metric("Buyer recall", f"{metrics['buyer_recall']:.1%}")
    st.metric("Buyer precision", f"{metrics['buyer_precision']:.1%}")
    st.caption(
        f"Only {metrics['buyer_share']:.0%} of sessions end in a purchase, "
        "so recall and precision for buyers matter more than accuracy."
    )

st.subheader("Pages visited")
c1, c2 = st.columns(2)
administrative = c1.number_input("Administrative pages", 0, 30, 0)
administrative_duration = c2.number_input("Administrative duration (sec)", 0.0, 5000.0, 0.0)
informational = c1.number_input("Informational pages", 0, 30, 0)
informational_duration = c2.number_input("Informational duration (sec)", 0.0, 5000.0, 0.0)
product_related = c1.number_input("Product pages", 0, 800, 10)
product_related_duration = c2.number_input("Product duration (sec)", 0.0, 70000.0, 300.0)

st.subheader("Session metrics")
c3, c4, c5 = st.columns(3)
bounce = c3.number_input("Bounce rate", 0.0, 0.2, 0.02, step=0.005, format="%.3f")
exit_rate = c4.number_input("Exit rate", 0.0, 0.2, 0.04, step=0.005, format="%.3f")
page_values = c5.number_input("Page value", 0.0, 400.0, 0.0)
special_day = st.slider("Closeness to a special day (0 = far, 1 = very close)", 0.0, 1.0, 0.0, 0.2)

st.subheader("Visitor details")
c6, c7 = st.columns(2)
month = c6.selectbox("Month", months, index=months.index("Nov") if "Nov" in months else 0)
visitor = c7.selectbox("Visitor type", visitors)
operating_systems = c6.number_input("Operating system (code)", 1, 8, 2)
browser = c7.number_input("Browser (code)", 1, 13, 2)
region = c6.number_input("Region (code)", 1, 9, 1)
traffic = c7.number_input("Traffic type (code)", 1, 20, 2)
weekend = st.checkbox("Weekend visit")

if st.button("Predict", type="primary"):
    label, proba = predict(
        model,
        columns,
        {
            "Administrative": administrative,
            "Administrative_Duration": administrative_duration,
            "Informational": informational,
            "Informational_Duration": informational_duration,
            "ProductRelated": product_related,
            "ProductRelated_Duration": product_related_duration,
            "BounceRates": bounce,
            "ExitRates": exit_rate,
            "PageValues": page_values,
            "SpecialDay": special_day,
            "OperatingSystems": operating_systems,
            "Browser": browser,
            "Region": region,
            "TrafficType": traffic,
            "Weekend": weekend,
            "Month": month,
            "VisitorType": visitor,
        },
    )
    if label == 1:
        st.success(f"✅ This user is likely to PURCHASE ({proba:.1%} probability)")
    else:
        st.error(f"❌ This user is unlikely to purchase ({proba:.1%} probability)")
    st.progress(proba)
