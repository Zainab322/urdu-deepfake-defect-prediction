#Using task 1 and task 2 models
import streamlit as st
import numpy as np
import joblib
import librosa

# ----- Safe loader -----
def safe_load_model(filename, fallback=None):
    try:
        return joblib.load(filename)
    except:
        return fallback

# ----- Load Models -----
scaler_audio = safe_load_model('audio_scaler.pkl')
svm_audio = safe_load_model('model_svm_audio.pkl')
logreg_audio = safe_load_model('model_logreg_audio.pkl')
perceptron_audio = safe_load_model('model_perceptron_audio.pkl')

# Defect Prediction
vectorizer = safe_load_model('vectorizer.pkl')
logreg_defect = safe_load_model('logreg_model.pkl')
svm_defect = safe_load_model('svm_model.pkl')
perceptron_defect = safe_load_model('perceptron_model.pkl')

st.title("🧠 Deepfake & Defect Detection App")

tab1, tab2 = st.tabs(["🔊 Deepfake Audio Detection", "🐞 Software Defect Prediction"])

# ----- Audio Deepfake Detection -----
with tab1:
    st.header("Upload Audio File")
    audio_file = st.file_uploader("Choose WAV/MP3", type=['wav', 'mp3'])
    model_choice = st.selectbox("Choose Model", ["SVM", "Logistic Regression", "Perceptron"])

    if audio_file is not None:
        try:
            audio, sr = librosa.load(audio_file, sr=22050)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfcc = mfcc[:, :50] if mfcc.shape[1] > 50 else np.pad(mfcc, ((0, 0), (0, 50 - mfcc.shape[1])), mode='constant')
            feature = mfcc.flatten().reshape(1, -1)
            feature_scaled = scaler_audio.transform(feature) if scaler_audio else feature

            if model_choice == "SVM":
                if svm_audio:
                    prob = svm_audio.predict_proba(feature_scaled)[0][1]
                else:
                    st.error("SVM model not found.")
                    prob = 0.0
            elif model_choice == "Logistic Regression":
                if logreg_audio:
                    prob = logreg_audio.predict_proba(feature_scaled)[0][1]
                else:
                    st.error("Logistic Regression model not found.")
                    prob = 0.0
            elif model_choice == "Perceptron":
                if perceptron_audio:
                    prob = perceptron_audio.predict_proba(feature_scaled)[0][1]
                else:
                    st.error("Perceptron model not found.")
                    prob = 0.0

            prediction = "🟢 Bonafide" if prob < 0.5 else "🔴 Deepfake"
            st.success(f"Prediction: {prediction}")
            st.write(f"Confidence: {prob:.2f}")
        except Exception as e:
            st.error(f"Audio processing failed: {str(e)}")

# ----- Defect Prediction -----
with tab2:
    st.header("Input Defect Description or Features")
    text_input = st.text_area("Enter text or vector features")

    model_choice_defect = st.selectbox("Choose Model", ["Logistic Regression", "SVM", "Perceptron"], key="defect")

    if st.button("Predict Defect Labels"):
        if text_input.strip() == "":
            st.warning("Enter some text.")
        elif vectorizer is None:
            st.error("Vectorizer not loaded.")
        else:
            try:
                X_vect = vectorizer.transform([text_input])

                if model_choice_defect == "Logistic Regression" and logreg_defect:
                    pred = logreg_defect.predict(X_vect)[0]
                    scores = logreg_defect.predict_proba(X_vect)
                elif model_choice_defect == "SVM" and svm_defect:
                    pred = svm_defect.predict(X_vect)[0]
                    scores = svm_defect.decision_function(X_vect)
                elif model_choice_defect == "Perceptron" and perceptron_defect:
                    pred = perceptron_defect.predict(X_vect)[0]
                    scores = perceptron_defect.decision_function(X_vect)
                else:
                    st.error("Selected model is not available.")
                    pred = []
                    scores = []

                if len(pred):
                    st.subheader("Predicted Labels:")
                    for i, val in enumerate(pred):
                        if val == 1:
                            conf = scores[i] if isinstance(scores, (list, np.ndarray)) else 1.0
                            st.write(f"✅ Label {i} (Confidence: {conf:.2f})")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
