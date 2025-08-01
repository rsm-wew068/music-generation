# CSE253 Machine Learning for Music

The course covers fundamental audio signal processing, machine learning for music analysis, and advanced music generation techniques.

## 🎵 Sample Outputs

**Listen to generated music from the final project:**
- [🎼 Conditioned Generation (Structured)](./final_project/symbolic_conditioned.mid) - Multi-instrument folk music with intro/climax/resolution
- [🎹 Unconditioned Generation (Basic)](./final_project/symbolic_unconditioned.mid) - Melody and chord progression generation

*Note: MIDI files can be played with any MIDI-compatible software or converted to audio using tools like FluidSynth*

## 📚 Course Structure

### **Homework Assignments** (Fundamentals)

#### **Homework 1: Audio Signal Processing & MIDI Classification**
**Location:** `homework1/homework1_stub.ipynb`

**Topics Covered:**
- **Audio Signal Processing Fundamentals**
  - Note-to-frequency conversion (A4=440Hz reference)
  - Audio effects: fade-out, delay, mixing, concatenation
  - Waveform generation: sawtooth waves using harmonic summation
- **MIDI Binary Classification**
  - Feature extraction from MIDI files
  - Machine learning classification techniques
  - Audio signal analysis and processing

**Dataset:** Custom MIDI files for binary classification task
**Technologies:** NumPy, SciPy, librosa, MIDI processing libraries

---

#### **Homework 2: Audio Feature Extraction & Analysis**
**Location:** `homework2/homework2_stub.ipynb`

**Topics Covered:**
- **Audio Feature Engineering**
  - Mel-frequency cepstral coefficients (MFCCs)
  - Spectral features and audio analysis
  - Feature extraction pipelines
- **Audio Classification**
  - Machine learning models for audio classification
  - Feature-based audio analysis

**Dataset:** Audio files with extracted MFCC features for classification
**Technologies:** librosa, scikit-learn, audio processing libraries

---

#### **Homework 3: Symbolic Music Generation with Markov Chains**
**Location:** `homework3/homework3_stub.ipynb`

**Topics Covered:**
- **Markov Chain Music Generation**
  - Probability-based sequence generation
  - Musical pattern learning
  - Symbolic music representation
- **REMI Tokenization**
  - MIDI tokenization using REMI method
  - Position, Pitch, Velocity, Duration tokens
  - Structured music representation

**Dataset:** MIDI files for training Markov chain transition probabilities
**Technologies:** miditok, Markov chains, probability theory

---

#### **Homework 4: Diffusion Models for Audio Generation**
**Location:** `homework4/sat-253/homework4_stub.ipynb`

**Topics Covered:**
- **Diffusion Models**
  - Stable Audio Tools (SAT) framework
  - Audio generation using diffusion processes
  - Conditional audio generation
- **Advanced Audio Generation**
  - Neural audio synthesis
  - Diffusion-based music creation

**Dataset:** Audio samples for training diffusion models (Stable Audio Tools framework)
**Technologies:** Stable Audio Tools, PyTorch, diffusion models

---

#### **Homework 5: Advanced Music Generation**
**Location:** `homework5/homework5 stub.ipynb`

**Topics Covered:**
- **Advanced Music Generation Techniques**
  - Neural music synthesis
  - Complex music generation pipelines
  - State-of-the-art music AI models

**Dataset:** Audio embeddings and playlist data for advanced generation tasks
**Technologies:** Advanced ML frameworks, music generation libraries

---

### **Major Projects**

#### **Assignment 1: Music Machine Learning Tasks**
**Location:** `assignment1/`

**Three Core Tasks:**

1. **Task 1: Composer Classification** (`task1_composer_classification/`)
   - **Goal**: Classify classical composers from MIDI files
   - **Dataset**: MIDI files from 8 classical composers (Beethoven, Chopin, Bach, Liszt, Schubert, Haydn, Mozart, Schumann)
   - **Approach**: XGBoost/LightGBM with MIDI feature extraction
   - **Features**: Pitch/duration/velocity stats, interval bigrams, chord roots
   - **Performance**: ~57% accuracy (target: 70%)
   - **Challenge**: Distribution shift between train/test sets

2. **Task 2: Next Sequence Prediction** (`task2_next_sequence_prediction/`)
   - **Goal**: Predict whether one MIDI segment follows another
   - **Dataset**: MIDI sequence pairs for training/testing sequence relationships
   - **Approach**: MLPClassifier and XGBoost with feature engineering
   - **Features**: MIDI statistics, pitch overlap, key/tempo matching
   - **Performance**: ~85% accuracy (successful)
   - **Success**: Robust cross-validation and feature engineering

3. **Task 3: Music Tagging (Audio)** (`task3_audio_classification/`)
   - **Goal**: Multi-label genre classification from audio files
   - **Dataset**: Audio files with 10 genre tags (rock, jazz, pop, electronic, country, blues, oldies, dance, punk, chill)
   - **Approach**: 3-layer CNN with mel-spectrograms
   - **Features**: MelSpectrogram + AmplitudeToDB via torchaudio
   - **Performance**: ~30-35% mAP (improved over baseline)
   - **Challenge**: Limited model capacity vs. test distribution

**Key Files:**
- `assignment1.py`: Main implementation with three model classes
- `baseline.py`: Baseline implementations for comparison
- `writeup.txt`: Comprehensive analysis and lessons learned
- `predictions*.json`: Generated predictions for each task

**Technologies:** PyTorch, XGBoost, LightGBM, torchaudio, music21, scikit-learn

---

#### **Final Project: Symbolic Music Generation**
**Location:** `final_project/`

**Two Main Tasks:**

1. **Task 1: Unconditioned Symbolic Generation** (`symbolic_unconditioned.ipynb`)

We use the MAESTRO 2018 dataset, a collection of high-quality MIDI piano performances, to generate unconditioned symbolic music. The pipeline separates the harmony and melody generation to maintain musical structure:
  - **Chord Generation**: A 2nd-order Markov Chain is trained from scratch to generate plausible chord progressions. Chords are extracted per 1-second interval and represented as pitch classes.
  - **Melody Generation**: An LSTM-based RNN is trained to predict the next note in an 8-note melody sequence. Melodies are parsed from the highest pitch above C3 per segment.
  - **Model Training**: Both models are trained from scratch using the MAESTRO data — there is no fine-tuning or transfer learning involved.
  - **Evaluation**: Outputs are assessed based on temporal and harmonic alignment (4 melody notes per chord), melodic continuity, and structural coherence.
  - **MIDI Output**: Final sequences are merged into playable MIDI files using pretty_midi.

2. **Task 2: Conditioned Symbolic Generation** (`symbolic_conditioned.ipynb`)

Using the Nottingham folk music corpus (in ABC format), we construct a symbolic generation pipeline that learns multi-instrument music (melody, chords, bass) using REMI tokenization and a Transformer decoder:
  - **Data Augmentation**: Sequences are transposed ±1 and ±2 semitones to expand the training set ~5×.
  - **REMI Tokenization**: Melody, chord, and bass lines are interleaved into structured REMI-style tokens (e.g., Bar_0, Position_3, Track_Melody, Note_C5, …).
  - **Training**:
  - Stage 1 (Baseline): The Transformer is trained on a small 2k-sample subset for 10 epochs.
  - Stage 2 (Fine-Tuning): The model is then fine-tuned on the full dataset with a validation split, learning rate scheduler, and gradient clipping over 30 epochs.
  - **Structured Generation**: Final music is generated in three sections: intro, climax (pitch-shifted +5, thinned), and resolution (pitch-shifted −4), forming a musical arc.
  - **Expressive Decoding**:
  - Trained MLP regressors predict:
  - Note duration
  - MIDI velocity (dynamics)
  - Articulation (legato/staccato)
  - Tempo (BPM per bar)
  - **Evaluation**: Evaluation was conducted qualitatively via structured music playback and validation loss monitoring. The model’s ability to generate coherent multi-instrumental music was assessed by enforcing structured phrases and expressive performance (duration, velocity, articulation, tempo). While we log validation loss during training, no quantitative or user-study-based evaluation was performed.
  - **Output**: The generated symbolic sequences are rendered into expressive, human-like MIDI and WAV files using music21 and fluidsynth.

#### 👩🏻‍💻 My Contributions

The final project was a 4-person team effort; I led the symbolic conditioned generation and Transformer modeling.

---

## 🎯 Learning Progression

### **Fundamentals → Advanced**
1. **Homework 1-2**: Audio signal processing basics
2. **Homework 3**: Symbolic music generation fundamentals
3. **Homework 4-5**: Advanced AI music generation
4. **Assignment 1**: Applied ML for music analysis
5. **Final Project**: End-to-end music generation systems

### **Skills Developed**
- **Audio Processing**: Signal analysis, feature extraction, effects
- **Machine Learning**: Classification, regression, sequence modeling
- **Music Theory**: MIDI processing, chord analysis, musical structure
- **Deep Learning**: CNNs, RNNs, Transformers for music
- **AI Generation**: Diffusion models, neural synthesis, symbolic generation

## 🛠️ Technology Stack

### **Core Libraries**
- **PyTorch**: Deep learning framework
- **librosa**: Audio processing and analysis
- **music21**: Music theory and analysis
- **scikit-learn**: Machine learning algorithms
- **numpy/scipy**: Numerical computing
- **torchaudio**: Audio processing for PyTorch

### **Specialized Tools**
- **miditok**: MIDI tokenization (REMI)
- **pretty_midi**: MIDI file processing
- **Stable Audio Tools**: Diffusion-based audio generation
- **FluidSynth**: MIDI to audio conversion

## 📊 Performance Summary

| Assignment | Task | My Performance | Baseline | Improvement | Status |
|------------|------|----------------|----------|-------------|---------|
| Assignment 1 | Composer Classification | ~57% accuracy | ~45% accuracy | +12% | ⚠️ Below target (70%) |
| Assignment 1 | Sequence Prediction | ~85% accuracy | ~70% accuracy | +15% | ✅ Successful |
| Assignment 1 | Audio Classification | ~30-35% mAP | ~27% mAP | +3-8% | ✅ Improved over baseline |
| Final Project | Conditioned Generation | Structured output | Basic output | Significant | ✅ Complete |
| Final Project | Unconditioned Generation | Basic output | Random output | Significant | ✅ Complete |

## 🎵 Key Insights

### **Assignment 1 Lessons**
- **Feature engineering** is crucial for symbolic music tasks
- **Distribution shift** between train/test sets is a major challenge
- **Classical ML** (XGBoost) can outperform deep learning for structured data
- **Cross-validation** is essential for robust evaluation

### **Final Project Lessons**
- **REMI tokenization** provides excellent structure for music generation
- **Multi-instrument modeling** creates more realistic compositions
- **Expressive control** (duration, velocity, articulation) enhances quality
- **Structured generation** (intro/climax/resolution) improves musical coherence

## 📝 Notes

- **Assignment 1** focuses on **music analysis and classification**
- **Final Project** focuses on **music generation and composition**
- **Homework series** provides foundational knowledge

This repository represents a complete journey through modern music AI, from basic audio processing to advanced generative systems.
