import streamlit as st
import random
import numpy as np

label_names = [
    'type_blocker', 'type_regression', 'type_bug',
    'type_documentation', 'type_enhancement',
    'type_task', 'type_dependency_upgrade'
]

st.title("🔍 Multi-Label Software Defect Prediction (Demo Mode)")

text_input = st.text_area("Enter software defect report:")

model_choice = st.selectbox("Select Model", ["Logistic Regression", "SVM", "DNN (Demo)"])

if st.button("Predict"):
    if not text_input.strip():
        st.warning("Please enter a defect report.")
    else:
        y_pred = np.random.randint(0, 2, size=(len(label_names)))
        y_confidence = np.random.uniform(0.5, 1.0, size=(len(label_names)))

        st.subheader("Predicted Labels:")
        for i, label in enumerate(label_names):
            if y_pred[i] == 1:
                st.write(f"✅ {label} — Confidence: {y_confidence[i]:.2f}")
