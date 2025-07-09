# Neural Machine Translation (English → Spanish) using Seq2Seq in Keras

This repository contains an implementation of a **Neural Machine Translation (NMT)** model that translates English sentences into Spanish using a **Sequence-to-Sequence (Seq2Seq)** architecture with **LSTM** networks in Keras.

---

## 🧠 Model Architecture

- **Encoder**: 
  - Embedding layer
  - LSTM layer (returns final states only)
  
- **Decoder**:
  - Embedding layer
  - LSTM layer (takes encoder states as initial state)
  - Dense layer with softmax activation for token prediction

---

## 📁 Dataset

The model is trained on the `spa.txt` dataset which consists of English-Spanish sentence pairs.

**Preprocessing Steps**:
- Lowercasing all text
- Removing non-alphabetical characters
- Appending special tokens: `"START_"` and `"_END"` to the Spanish targets

---

## 🧪 Features

- Custom **data generator** to efficiently feed padded sequences in batches
- Handles **masking** for variable-length inputs
- Uses `ReduceLROnPlateau` and `ModelCheckpoint` callbacks for better training
- Saves best model (`nmt_eng2spa.h5`) and final weights (`nmt_weights_last.h5`)

---

## 📊 Training Configuration

- **Optimizer**: RMSprop
- **Loss**: Categorical Crossentropy
- **Batch Size**: 128
- **Epochs**: 50
- **Latent Dim**: 256

---

## 🚀 How to Run

### 1. Clone the Repo

```bash
git clone https://github.com/amoghagain/Neural_machine_translation_seq2seq.git
cd Neural_machine_translation_seq2seq
