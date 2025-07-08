# 🎼 Symbolic Music Generation – CSE 253 Project

This repository contains experiments in symbolic music generation using deep learning and probabilistic models, developed for the Machine Learning for Music course (CSE 253, Spring 2025).

---

## 📌 Project Overview

We explored two generation pipelines:

### 1. **Unconditioned Generation**  
Symbolic melody and chord generation from scratch:
- **Chord Modeling:** 2nd-order Markov Chain
- **Melody Modeling:** LSTM-based RNN
- **Dataset:** MAESTRO 2018 (MIDI)

### 2. **Conditioned Generation**  
Structured generation conditioned on melody, chord, and bass lines:
- **Tokenizer:** REMI tokenization
- **Model:** MiniTransformer (decoder-only)
- **Feature Modeling:** MLP regressors for duration, velocity, and tempo
- **Dataset:** Nottingham Folk Dataset (ABC notation → MIDI)

---

## ⚙️ Key Features

### 🔹 symbolic_unconditioned.ipynb
- MIDI parsing, chord estimation, melody extraction
- Sequence generation with:
  - Markov Chain for chord progression
  - LSTM RNN for melody (PyTorch)
- Combined generation into new MIDI files
- Visualization of note/chord distributions

### 🔹 symbolic_conditioned.ipynb
- ABC-to-MIDI pipeline and data augmentation
- REMI token construction and vocabulary
- Transformer model trained to predict symbolic tokens
- Feature-based expressive decoding (duration, velocity, tempo)
- Structured section control: intro → climax → resolution
- Audio rendering with FluidSynth

---

## 🎯 Objective

Can we build musically coherent, structured compositions using symbolic models and small datasets?  
We aim to balance interpretability and musical quality with efficient, modular architectures.

---

## 🛠️ Requirements

- Python 3.x
- Jupyter Notebook
- PyTorch, scikit-learn, music21, pretty_midi, numpy, matplotlib
- Audio rendering: `fluidsynth`, `FluidR3_GM.sf2`

---

## ▶️ Usage

1. Clone this repo.
2. Open either notebook:
   - `symbolic_unconditioned.ipynb` for MAESTRO-based generation
   - `symbolic_conditioned.ipynb` for REMI-tokenized Nottingham modeling
3. Follow the steps to install dependencies and run the code.
4. MIDI/audio output will be saved to the working directory.

---

## ✍️ Authors

Rachel Wang, Juhak Lee, Slater Mutunga, Vincent Tu  
Contributions: data processing, model training, symbolic generation, evaluation, visualization
