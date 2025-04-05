#pip install scikit-learn
#pip install pandas
#pip install streamlit
#To run the code: streamlit run main.py
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load Iris dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

def main():
    st.title("Streamlit Demo for ML Model on Iris Dataset")

    # Sidebar for user input
    st.sidebar.header("Model Configuration")
    test_size = st.sidebar.slider("Test Size (Fraction of Data)", 0.1, 0.5, 0.2, 0.05)
    random_state = st.sidebar.number_input("Random State", value=42, step=1)

    # Model selection
    st.sidebar.header("Select Algorithms")
    use_random_forest = st.sidebar.checkbox("Random Forest", True)
    use_decision_tree = st.sidebar.checkbox("Decision Tree", False)
    use_svm = st.sidebar.checkbox("Support Vector Machine (SVM)", False)

    # Split data
    X = df[data.feature_names]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Results dictionary
    results = {}

    if use_random_forest:
        rf_model = RandomForestClassifier(random_state=random_state)
        rf_model.fit(X_train, y_train)
        rf_y_pred = rf_model.predict(X_test)
        results['Random Forest'] = {
            'Accuracy': accuracy_score(y_test, rf_y_pred),
            'Report': classification_report(y_test, rf_y_pred, target_names=data.target_names, output_dict=True)
        }

    if use_decision_tree:
        dt_model = DecisionTreeClassifier(random_state=random_state)
        dt_model.fit(X_train, y_train)
        dt_y_pred = dt_model.predict(X_test)
        results['Decision Tree'] = {
            'Accuracy': accuracy_score(y_test, dt_y_pred),
            'Report': classification_report(y_test, dt_y_pred, target_names=data.target_names, output_dict=True)
        }

    if use_svm:
        svm_model = SVC(random_state=random_state)
        svm_model.fit(X_train, y_train)
        svm_y_pred = svm_model.predict(X_test)
        results['SVM'] = {
            'Accuracy': accuracy_score(y_test, svm_y_pred),
            'Report': classification_report(y_test, svm_y_pred, target_names=data.target_names, output_dict=True)
        }

    # Display results
    st.write("### Dataset Overview")
    st.dataframe(df.head())

    st.write("### Model Results")
    for model_name, result in results.items():
        st.write(f"#### {model_name}")
        st.write(f"Accuracy: {result['Accuracy']:.2f}")
        st.write("Classification Report:")
        st.dataframe(pd.DataFrame(result['Report']).transpose())

    # User input for prediction
    st.write("### Make a Prediction")
    input_data = []
    st.write("#### Select Feature Values Using Sliders")
    for feature in data.feature_names:
        min_value = float(X[feature].min())
        max_value = float(X[feature].max())
        value = st.slider(f"{feature}", min_value, max_value, float(X[feature].mean()))
        input_data.append(value)

    if st.button("Predict Individually"):
        if use_random_forest:
            rf_prediction = rf_model.predict([input_data])
            st.write(f"Random Forest Prediction: {data.target_names[rf_prediction[0]]}")
        if use_decision_tree:
            dt_prediction = dt_model.predict([input_data])
            st.write(f"Decision Tree Prediction: {data.target_names[dt_prediction[0]]}")
        if use_svm:
            svm_prediction = svm_model.predict([input_data])
            st.write(f"SVM Prediction: {data.target_names[svm_prediction[0]]}")

if __name__ == "__main__":
    main()