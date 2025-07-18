# Speech Emotion Recognition Using MFCC and Machine Learning

This project performs **Speech Emotion Recognition (SER)** by extracting **MFCC (Mel Frequency Cepstral Coefficients)** features from audio files and applying various **Machine Learning algorithms** to classify emotions from speech data.

## 📌 Project Title

**Comparative Analysis of ML Algorithms for Speech Emotion Recognition Using MFCC Features**

---

## 🎯 Objective

To develop a speech emotion recognition system and **compare the performance** of multiple machine learning models to determine the most effective approach for classifying emotions from audio speech using MFCC features.

---

## 📂 Dataset

- **Source:** [RAVDESS Dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)  
- **Content:** Speech audio files of 24 actors in 8 emotional states:
  - Neutral
  - Calm
  - Happy
  - Sad
  - Angry
  - Fearful
  - Disgust
  - Surprised

---

## 🔍 Feature Extraction

- **MFCC (Mel Frequency Cepstral Coefficients):**
  - Extracted 40 MFCC features from each `.wav` audio file.
  - Used `librosa` for audio processing.

---

## 🤖 Machine Learning Models Used

| Model                | Accuracy |
|---------------------|----------|
| MLPClassifier        | 76.60%   |
| Logistic Regression  | ~69.59%  |
| K-Nearest Neighbors  | ~61.40%  |
| Random Forest        | **78.94%** |
| SVM (Support Vector Machine) | **78.94%** |
| XGBoost              | ~71.34%  |

---

## ✅ Best Performing Models

- **Random Forest**
- **SVM**

Both models achieved an accuracy of **78.94%** on the test set.

---

## 🧪 Testing

You can test the model with a single audio file:

```python
test_file = "path_to_audio.wav"
features = extract_mfcc(test_file).reshape(1, -1)
pred = model.predict(features)
print("Predicted Emotion:", label_encoder.inverse_transform(pred)[0])
