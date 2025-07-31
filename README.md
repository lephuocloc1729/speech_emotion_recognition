# Speech Emotion Recognition with CNN-BiLSTM

This project implements a robust Speech Emotion Recognition (SER) pipeline using a hybrid CNN-BiLSTM architecture trained on multiple public datasets. It leverages feature extraction, data augmentation, and gender-aware label engineering to build a generalizable emotion classifier.

## 📌 Project Goals

- Build a generalizable SER model that performs well across speakers, genders, and datasets.
- Combine multiple SER datasets to enrich diversity and robustness.
- Apply meaningful data augmentation to simulate real-world conditions.
- Evaluate and compare multiple CNN-BiLSTM architectures.

## 🗂️ Datasets Used

We used four publicly available datasets, harmonized into 7 emotions:

| Dataset | Speakers      | Emotions                                                        | Notes                                 |
| ------- | ------------- | --------------------------------------------------------------- | ------------------------------------- |
| RAVDESS | 24 (12M, 12F) | Neutral, Calm\*, Happy, Sad, Angry, Fearful, Disgust, Surprised | Clean studio recordings               |
| TESS    | 2F            | Angry, Disgust, Fear, Happy, Neutral, Sad, Pleasant Surprise\*  | Canadian English                      |
| SAVEE   | 4M            | Angry, Disgust, Fear, Happy, Neutral, Sad, Surprised            | British English                       |
| CREMA-D | 91 (48M, 43F) | Angry, Disgust, Fear, Happy, Neutral, Sad                       | Crowd-sourced labels, diverse accents |

_Note_: Calm → Neutral; Pleasant Surprise → Surprise

## 🎧 Feature Extraction

Audio files were resampled to 22050 Hz, trimmed/padded to 3 seconds, and converted to mono.

We extracted the following features using torchaudio:

- MFCC (n=40)
- Mel Spectrogram (dB)
- RMS Energy
- Zero-Crossing Rate (ZCR)

## ⚙️ Data Augmentation

To improve generalization, each audio sample underwent 6 augmentations:

- Additive White Gaussian Noise
- Pitch Shifting (±2 semitones)
- Time Shifting (±20% offset)
- Speed Variation (0.8x, 1.25x)

Each augmented version was treated as a new sample, expanding dataset size 6×.

## 🧠 Model Architecture

We tested 3 models built with PyTorch:

| Model   | CNN Layers | Bi-LSTM Layers | Target           | Accuracy |
| ------- | ---------- | -------------- | ---------------- | -------- |
| Model A | 4          | 1              | Gender + Emotion | 85.5%    |
| Model B | 5          | 1              | Gender + Emotion | 71.7%    |
| Model C | 4          | 1              | Emotion only     | 68.7%    |

## 🧪 Training Setup

- **Optimizer**: Adam + ReduceLROnPlateau
- **Loss**: CrossEntropy
- **Regularization**: L2 (1e-5), Dropout (0.3)
- **Epochs**: 200 with EarlyStopping (patience = 5)
- **Batch size**: 32
- **Evaluation metric**: Accuracy

## 📈 Results & Evaluation

- Model A (gender-emotion target) significantly outperformed the others.
- Confusion matrix analysis shows improved classification for similar emotions.
- Augmentation and label engineering improved speaker invariance.

## 📊 Key Takeaways

- Gender-aware labels are crucial for generalization in SER.
- Combining datasets improves robustness across recording conditions and accents.
- A lightweight CNN-BiLSTM model is sufficient to reach 85%+ accuracy.

## 🚀 Future Work

- Explore transformer-based architectures (e.g., Wav2Vec, Whisper).
- Add multilingual datasets and spontaneous speech.
- Enable real-time inference deployment.

## 📁 Folder Structure

```
.
├── data/              # Raw and processed datasets
├── models/            # Saved model checkpoints
├── notebooks/         # Jupyter notebooks for EDA and prototyping
├── scripts/           # Feature extraction, training, and evaluation scripts
├── results/           # Confusion matrices, classification reports
├── slides/            # Final presentation (PDF/Canva)
├── report/            # LaTeX report
└── README.md          # Project overview
```

## ⚠️ Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Key packages:

- torch, torchaudio
- librosa
- numpy, pandas
- matplotlib, seaborn

## 👥 Team Members

- Lộc (Leader, modeling, presentation)
- Tiến (Training, report writing)
- Huy (Training, report writing)
- An (Presentation design)
- Đức (Dataset exploration)
- Minh (Preprocessing & augmentation)
- Nhân (Evaluation & visualizations)
