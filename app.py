import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import time

# Load and cache data
@st.cache_data
def load_data():
    data = pd.read_csv('final-dataset.csv')
    return data

# Function to plot confusion matrix
def plot_confusion_matrix(cm, title='Confusion Matrix'):
    st.subheader(title)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)

# Function to encode categorical variables
def encode_categorical(data):
    # Mapping for 'transaction_type'
    transaction_mapping = {
        'CW2CW': 1, 'AW2CW': 2, 'CW2MW': 3, 'CC2CW': 4,
        'CW2CB': 5, 'CB2CW': 6, 'CW2AW': 7
    }
    data['transaction_type'] = data['transaction_type'].map(transaction_mapping)

    # Mapping for 'sender_account_type'
    sender_mapping = {
        'CUSTOMER': 1, 'AGENT': 2
    }
    data['sender_account_type'] = data['sender_account_type'].map(sender_mapping)

    # Mapping for 'receiver_account_type'
    receiver_mapping = {
        'CUSTOMER': 1, 'AGENT': 2, 'BANK': 3,
        'MERCHANT': 4, 'PARKING_ACCOUNT': 5
    }
    data['receiver_account_type'] = data['receiver_account_type'].map(receiver_mapping)

    return data

# Function to plot transaction type distribution
def plot_transaction_distribution(data):
    st.subheader("Transaction Type Distribution")
    transaction_counts = data['transaction_type'].value_counts().sort_index()
    transaction_labels = ['CW2CW', 'AW2CW', 'CW2MW', 'CC2CW', 'CW2CB', 'CB2CW', 'CW2AW']
    fig = px.bar(
        x=transaction_labels,
        y=transaction_counts.values,
        labels={'x': 'Transaction Type', 'y': 'Count'},
        title='Transaction Type Distribution'
    )
    st.plotly_chart(fig)

# Main App
def main():
    st.title("Anomaly Detection Training App")

    # Load the data
    data = load_data()

    # Encode categorical variables
    data = encode_categorical(data)

    # Features and target
    feature_columns = [
        "amount", "transaction_type", "receiver_credit_amount",
        "sender_debit_amount", "depositor_running_balance",
        "withdrawer_running_balance", "sender_account_type",
        "receiver_account_type"
    ]
    X = data[feature_columns].values
    y = data["is_anomaly"].astype(int).values

    # Split data
    xtrain, xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.10, random_state=42
    )

    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox(
        "Choose a model",
        ("Decision Tree", "Logistic Regression", "Random Forest", "SGD Classifier")
    )

    # Initialize the selected model
    if model_choice == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_choice == "Random Forest":
        model = RandomForestClassifier()
    elif model_choice == "SGD Classifier":
        model = SGDClassifier(loss='log')  # Using logistic regression loss for binary classification

    # Start training button
    if st.button("Start Training"):
        st.info("Training started...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Simulate training with progress bar
        epochs = 100
        for epoch in range(epochs):
            # Simulate some training time
            time.sleep(0.02)  # Adjust as needed for realism

            # Update progress bar
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)

            # Update status text
            status_text.text(f"Training... {int(progress * 100)}%")

        # Actual model training
        with st.spinner('Fitting the model...'):
            model.fit(xtrain, ytrain)

        # Predictions
        ypred = model.predict(xtest)

        # Accuracy
        accuracy = accuracy_score(ytest, ypred)
        st.success(f"{model_choice} Accuracy: {accuracy:.2f}")

        # Confusion Matrix
        cm = confusion_matrix(ytest, ypred)
        plot_confusion_matrix(cm, title=f'{model_choice} Confusion Matrix')

        # Classification Report
        st.subheader("Classification Report:")
        report = classification_report(ytest, ypred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

        # Plot transaction type distribution
        plot_transaction_distribution(data)

        st.balloons()

    # Option to display raw data
    if st.checkbox("Show raw data"):
        st.subheader("Raw Data")
        st.write(data)

if __name__ == "__main__":
    main()
