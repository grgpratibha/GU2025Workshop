# Install required packages if not already installed:
# pip install kagglehub[pandas-datasets] scikit-learn pandas streamlit

import streamlit as st
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load Kaggle dataset
file_path = "Mobile Phone Pricing.csv"

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "khwaishsaxena/mobile-phone-pricing-dataset",
    file_path
)

# Prepare feature and target
X = df.drop('price_range', axis=1)
y = df['price_range']
feature_names = X.columns.tolist()
target_names = sorted(y.unique().astype(str))

def main():
    st.title("📱 Mobile Price Classification Test - ML Models (Streamlit)")

    # Sidebar configuration
    st.sidebar.header("Model Settings")
    test_size = st.sidebar.slider("Test Size (Fraction of Data)", 0.1, 0.5, 0.2, 0.05)
    random_state = st.sidebar.number_input("Random State", value=42, step=1)

    st.sidebar.header("Choose Classifiers")
    use_random_forest = st.sidebar.checkbox("Random Forest", True)
    use_decision_tree = st.sidebar.checkbox("Decision Tree", False)
    use_svm = st.sidebar.checkbox("Support Vector Machine (SVM)", False)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    results = {}

    if use_random_forest:
        rf_model = RandomForestClassifier(random_state=random_state)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        results['Random Forest'] = {
            'Accuracy': accuracy_score(y_test, rf_pred),
            'Report': classification_report(y_test, rf_pred, output_dict=True)
        }

    if use_decision_tree:
        dt_model = DecisionTreeClassifier(random_state=random_state)
        dt_model.fit(X_train, y_train)
        dt_pred = dt_model.predict(X_test)
        results['Decision Tree'] = {
            'Accuracy': accuracy_score(y_test, dt_pred),
            'Report': classification_report(y_test, dt_pred, output_dict=True)
        }

    if use_svm:
        svm_model = SVC(random_state=random_state)
        svm_model.fit(X_train, y_train)
        svm_pred = svm_model.predict(X_test)
        results['SVM'] = {
            'Accuracy': accuracy_score(y_test, svm_pred),
            'Report': classification_report(y_test, svm_pred, output_dict=True)
        }

    # Display dataset
    st.subheader(" Sample Data")
    st.dataframe(df.head())

    # Show results
    st.subheader("Model Results")
    for name, result in results.items():
        st.markdown(f"**{name}**")
        st.write(f"Accuracy: {result['Accuracy']:.2f}")
        st.dataframe(pd.DataFrame(result['Report']).transpose())

    # Predict individual input
    st.subheader("Trying a Prediction Model")
    user_input = []
    for feature in feature_names:
        min_val = float(X[feature].min())
        max_val = float(X[feature].max())
        mean_val = float(X[feature].mean())
        user_val = st.slider(f"{feature}", min_val, max_val, mean_val)
        user_input.append(user_val)

    if st.button("Predict"):
        input_array = [user_input]
        if use_random_forest:
            rf_result = rf_model.predict(input_array)
            st.success(f"Random Forest Prediction: Class {rf_result[0]}")
        if use_decision_tree:
            dt_result = dt_model.predict(input_array)
            st.success(f"Decision Tree Prediction: Class {dt_result[0]}")
        if use_svm:
            svm_result = svm_model.predict(input_array)
            st.success(f"SVM Prediction: Class {svm_result[0]}")

if __name__ == "__main__":
    main()
