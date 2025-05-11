# IndicTTS Deepfake Challenge – Detecting AI-Generated Speech Across 16 Indian Languages

## 📄 Competition Link

[IndicTTS Deepfake Challenge on Kaggle](https://www.kaggle.com/competitions/indic-tts-deepfake-challenge/overview)

## 🏆 Overview

Develop machine learning models to distinguish between real human speech and AI-synthesized audio across 16 Indian languages. Models should output the probability that a given audio clip is AI-generated (Text-to-Speech).

* **Start:** March 8, 2025
* **Close:** March 11, 2025
* **Metric:** ROC-AUC (Receiver Operating Characteristic – Area Under Curve)

## 📂 Dataset Description

The dataset contains **33,737 audio samples** (\~63 hours) spanning **16 languages**.

| Split | Samples | Labels        | Languages        |
| ----- | ------: | :------------ | :--------------- |
| Train |  31,102 | `is_tts` 0/1  | All 16 languages |
| Test  |   2,635 | `is_tts = -1` | All 16 languages |

### Columns (Train)

* `id` (int): Unique sample identifier
* `audio` (Audio): Raw waveform file
* `text` (string): Transcription
* `language` (string): Language code/name
* `is_tts` (int): 1 if AI-generated, 0 if real

### Languages

Assamese, Bengali, Bodo, Dogri, Kannada, Malayalam, Marathi, Sanskrit, Nepali, English, Telugu, Hindi, Odia, Manipuri, Gujarati, Tamil.

### Accessing the Dataset

Load directly via Hugging Face `datasets`:

```python
from datasets import load_dataset

dataset = load_dataset("SherryT997/IndicTTS-Deepfake-Challenge-Data")
train_ds = dataset["train"]  # with labels
test_ds  = dataset["test"]   # is_tts = -1
```

Or use the Kaggle dataset mirror.

## 🛠 Environment & Dependencies

* **Python 3.8+**
* **NumPy**, **Pandas**
* **Librosa**, **SoundFile**, **Torchaudio**
* **Hugging Face Transformers & Datasets**
* **PyTorch** (or **TensorFlow/Keras**)
* **scikit-learn**
* **Matplotlib**, **Seaborn**
* **tqdm**

Install via:

```bash
pip install numpy pandas librosa soundfile torchaudio datasets transformers torch scikit-learn matplotlib seaborn tqdm
```

## 🔍 Exploratory Data Analysis (EDA)

1. **Class Balance:** count real vs. TTS samples per language.
2. **Duration Distribution:** histogram of audio lengths.
3. **Spectrogram Samples:** visualize Mel-spectrograms of real vs. synthetic.
4. **Language-wise Variation:** compare feature distributions across languages.

## 🏗 Model Architectures & Approaches

### 1. Baseline: MFCC + Logistic Regression

* **Feature Extraction:** compute 40-dimensional MFCCs (mean & std over time)
* **Classifier:** scikit-learn `LogisticRegression`
* **Pros:** fast, interpretable
* **Cons:** limited capacity for complex patterns

### 2. CNN on Mel-Spectrograms

* **Preprocessing:** convert waveform to 128×T Mel-spectrograms
* **Backbone:** ResNet18 (modified first conv for 1-channel input)
* **Head:** global average pooling + dense sigmoid output
* **Training:**

  * Loss: `BCEWithLogitsLoss()`
  * Optimizer: AdamW (lr=3e-4)
  * Scheduler: Cosine LR decay
  * Augmentations: time/frequency masking (SpecAugment)

### 3. Transformer: Wav2Vec2 Fine-Tuning

* **Pretrained Model:** `facebook/wav2vec2-base-960h`
* **Modification:** add a classification head (pooler + dense)
* **Training:**

  * Freeze feature extractor first 5 epochs, then unfreeze
  * Loss: `BCEWithLogitsLoss()`
  * Optimizer: AdamW (lr=1e-5)
  * Mixed-precision with `torch.cuda.amp`

## 🎓 Training Pipeline

1. **Dataset Class:** custom PyTorch `Dataset` reading waveform and labels.
2. **Transforms & Collate:** padding/truncating, feature extraction on the fly.
3. **DataLoader:** balanced sampling or stratified k-fold splits.
4. **Training Loop:** forward, compute loss, backward, optimization step.
5. **Validation:** compute ROC-AUC per epoch on held-out split.
6. **Checkpointing:** save best model by highest validation ROC-AUC.

## 📊 Evaluation

* **Primary Metric:** ROC-AUC (macro and per-language breakdown)
* **Secondary:** Precision-Recall AUC, confusion matrix at optimal threshold

## 🚀 Inference & Submission

1. **Load Best Weights**
2. **Preprocess Test Audio:** same steps as training.
3. **Predict Probabilities:** `sigmoid(logits)` for each `id`.
4. **Generate CSV:** two columns `id,is_tts` with probabilities.

```bash
python submission.py --model wav2vec2 --weights best.pt \
    --input_csv test_metadata.csv --output submission.csv
```

## 💡 Skills & Techniques Demonstrated

* **Audio Processing:** feature engineering (MFCC, spectrograms)
* **Deep Learning:** CNNs, Transformer fine-tuning
* **Hugging Face Ecosystem:** `datasets`, `transformers`
* **Evaluation:** ROC-AUC, stratified validation
* **Performance Optimization:** mixed-precision, learning rate scheduling
* **Cross-Language Generalization:** handling 16 diverse Indian languages

## 🏃 How to Reproduce

1. Clone this repo and navigate to `notebooks/`.
2. Install dependencies.
3. Load dataset via Hugging Face or Kaggle.
4. Run EDA notebook: `eda_indictts_deepfake.ipynb`.
5. Train baseline & advanced models: `train_mfcc_lr.ipynb`, `train_cnn_spec.ipynb`, `train_wav2vec2.ipynb`.
6. Execute `submission.py` to produce `submission.csv`.

---

**Secure the future of speech authenticity!**
