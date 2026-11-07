
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

# Page configuration
st.set_page_config(
    page_title="Banking Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("models/fraud_model.pkl")

# Load testing dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/test_transactions.csv")

model = load_model()
df = load_data()

# Separate input features and actual labels
X = df.drop(columns=["Class"])
y = df["Class"]

# Generate predictions for test transactions
@st.cache_data
def evaluate_model(_model, features, labels):
    predictions = _model.predict(features)
    probabilities = _model.predict_proba(features)[:, 1]
    return predictions, probabilities

predictions, probabilities = evaluate_model(model, X, y)

# Dashboard heading
st.title("Banking Fraud Detection and Risk Analysis System")
st.caption("Machine Learning Based Credit Card Fraud Detection")

st.divider()

# Create three dashboard tabs
tab1, tab2, tab3 = st.tabs([
    "Overview",
    "Fraud Detection",
    "Model Performance"
])

# TAB 1 - OVERVIEW
with tab1:
    st.header("Transaction Overview")

    total = len(df)
    fraud = int(y.sum())
    legitimate = total - fraud

    col1, col2, col3 = st.columns(3)

    col1.metric("Test Transactions", f"{total:,}")
    col2.metric("Legitimate Transactions", f"{legitimate:,}")
    col3.metric("Fraudulent Transactions", f"{fraud:,}")

    st.subheader("Transaction Distribution")

    chart_data = pd.DataFrame({
        "Transaction Type": ["Legitimate", "Fraudulent"],
        "Count": [legitimate, fraud]
    })

    st.bar_chart(chart_data.set_index("Transaction Type"))

    st.info(
        "Fraudulent transactions are very rare compared "
        "to legitimate transactions."
    )

# TAB 2 - FRAUD DETECTION
with tab2:
    st.header("Transaction Risk Analysis")

    index = st.number_input(
        "Select Test Transaction Index",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1
    )

    transaction = X.iloc[[index]]
    actual = int(y.iloc[index])

    st.write("Selected Transaction Details")
    st.dataframe(transaction, use_container_width=True)

    if st.button("Analyze Transaction"):
        probability = model.predict_proba(transaction)[0][1]
        prediction = model.predict(transaction)[0]

        st.metric(
            "Estimated Fraud Probability",
            f"{probability * 100:.2f}%"
        )

        if probability < 0.30:
            risk = "LOW"
            st.success("Low Risk Transaction")
        elif probability < 0.70:
            risk = "MEDIUM"
            st.warning("Medium Risk Transaction")
        else:
            risk = "HIGH"
            st.error("High Risk Transaction")

        st.write("Risk Category:", risk)

        st.write(
            "Model Prediction:",
            "Fraud" if prediction == 1 else "Legitimate"
        )

        st.write(
            "Actual Label:",
            "Fraud" if actual == 1 else "Legitimate"
        )

        st.caption(
            "Risk levels are illustrative and are not "
            "validated banking risk ratings."
        )

# TAB 3 - MODEL PERFORMANCE
with tab3:
    st.header("Model Performance Analysis")

    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)
    ap = average_precision_score(y, probabilities)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Fraud Precision", f"{precision:.2%}")
    c2.metric("Fraud Recall", f"{recall:.2%}")
    c3.metric("Fraud F1-Score", f"{f1:.2%}")
    c4.metric("Average Precision", f"{ap:.4f}")

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, predictions, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        ax=ax
    )

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title("Random Forest Confusion Matrix")

    st.pyplot(fig)
    plt.close(fig)

    st.write(
        "The confusion matrix shows correct predictions "
        "and incorrect classifications made by the model."
    )

    
    # Feature Importance Analysis
    st.divider()

    st.subheader("Feature Importance Analysis")

    # Get feature importance values from Random Forest
    importance = model.feature_importances_

    # Create a dataframe containing feature names and scores
    feature_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    })

    # Sort features from highest to lowest importance
    feature_df = feature_df.sort_values(
        by="Importance",
        ascending=False
    )

    # Select the 10 most important features
    top_features = feature_df.head(10)

    # Display feature importance chart
    fig2, ax2 = plt.subplots(figsize=(9, 5))

    sns.barplot(
        data=top_features,
        x="Importance",
        y="Feature",
        color="steelblue",
        ax=ax2
    )

    ax2.set_title("Top 10 Important Features")
    ax2.set_xlabel("Feature Importance Score")
    ax2.set_ylabel("Transaction Features")

    st.pyplot(fig2)
    plt.close(fig2)

    st.write(
        "This chart shows the features that contribute "
        "most to the Random Forest model's predictions."
    )

    st.caption(
        "The features V1-V28 are anonymized. "
        "Importance does not imply causation."
    )

