# %%
# Probably more imports than are really necessary...
import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from tqdm import tqdm
import librosa
import numpy as np
import miditoolkit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, average_precision_score, accuracy_score
import random

# %% [markdown]
# ## Metrics

# %%
def accuracy1(groundtruth, predictions):
    correct = 0
    for k in groundtruth:
        if not (k in predictions):
            print("Missing " + str(k) + " from predictions")
            return 0
        if predictions[k] == groundtruth[k]:
            correct += 1
    return correct / len(groundtruth)

# %%
def accuracy2(groundtruth, predictions):
    correct = 0
    for k in groundtruth:
        if not (k in predictions):
            print("Missing " + str(k) + " from predictions")
            return 0
        if predictions[k] == groundtruth[k]:
            correct += 1
    return correct / len(groundtruth)

# %%
TAGS = ['rock', 'oldies', 'jazz', 'pop', 'dance',  'blues',  'punk', 'chill', 'electronic', 'country']

# %%
def accuracy3(groundtruth, predictions):
    preds, targets = [], []
    for k in groundtruth:
        if not (k in predictions):
            print("Missing " + str(k) + " from predictions")
            return 0
        prediction = [1 if tag in predictions[k] else 0 for tag in TAGS]
        target = [1 if tag in groundtruth[k] else 0 for tag in TAGS]
        preds.append(prediction)
        targets.append(target)
    
    mAP = average_precision_score(targets, preds, average='macro')
    return mAP

# %% [markdown]
# ## Task 1: Composer classification

# %%
dataroot1 = "student_files/task1_composer_classification"

# %%
# XGBoost model
# accuracy = 0.55 
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import miditoolkit
import numpy as np
import os

class model1():
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            eval_metric='mlogloss',
            random_state=42
        )
        self.label_encoder = LabelEncoder()

    def features(self, path):
        clean_path = path.replace("midis/", "")
        midi_path = os.path.join(dataroot1, "midis", clean_path)

        try:
            midi_obj = miditoolkit.midi.parser.MidiFile(midi_path)
            notes = [note for inst in midi_obj.instruments for note in inst.notes]
        except:
            return [0.0] * 9

        if len(notes) == 0:
            return [0.0] * 9

        pitches = [note.pitch for note in notes]
        durations = [note.end - note.start for note in notes]
        velocities = [note.velocity for note in notes]

        intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]
        mean_interval = np.mean(intervals) if intervals else 0
        std_interval = np.std(intervals) if intervals else 0

        return [
            np.mean(pitches),
            np.std(pitches),
            np.min(pitches),
            np.max(pitches),
            np.mean(durations),
            np.std(durations),
            np.mean(velocities),
            len(notes),
            mean_interval,
            std_interval
        ]

    def train(self, path):
        with open(path, 'r') as f:
            train_json = eval(f.read())

        X_train = [self.features(k) for k in train_json]
        y_train = [train_json[k] for k in train_json]
        self.label_encoder.fit(y_train)
        y_encoded = self.label_encoder.transform(y_train)

        self.model.fit(X_train, y_encoded)

    def predict(self, path, outpath=None):
        d = eval(open(path, 'r').read())
        predictions = {}

        for k in d:
            x = self.features(k)
            pred_encoded = self.model.predict([x])[0]
            pred = self.label_encoder.inverse_transform([pred_encoded])[0]
            predictions[k] = str(pred)

        if outpath:
            with open(outpath, "w") as z:
                z.write(str(predictions) + '\n')

        return predictions

# %% [markdown]
# ## Task 2: Sequence prediction

# %%
dataroot2 = "student_files/task2_next_sequence_prediction"

# %%
# MLP no Pytorch
# accuracy = 0.78
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import miditoolkit
import numpy as np
import os

class model2():
    def __init__(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=1e-4,
            early_stopping=True,
            max_iter=500,
            random_state=42
        )
        self.scaler = StandardScaler()

    def extract_features(self, path):
        path = os.path.join(dataroot2, path)
        try:
            midi = miditoolkit.MidiFile(path)
            notes = [note for inst in midi.instruments for note in inst.notes]
        except:
            return [0.0] * 5

        if len(notes) == 0:
            return [0.0] * 5

        pitches = [note.pitch for note in notes]
        durations = [note.end - note.start for note in notes]
        velocities = [note.velocity for note in notes]
        intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]

        return [
            np.mean(pitches),
            np.std(pitches),
            np.mean(durations),
            np.mean(velocities),
            np.std(intervals) if intervals else 0.0
        ]

    def combine_features(self, f1, f2):
        x1 = self.extract_features(f1)
        x2 = self.extract_features(f2)
        diff = np.abs(np.array(x1) - np.array(x2)).tolist()
        return x1 + x2 + diff  # 15 features total

    def train(self, path):
        data = eval(open(path).read())
        X = [self.combine_features(f1, f2) for (f1, f2) in data]
        y = [data[(f1, f2)] for (f1, f2) in data]

        X = self.scaler.fit_transform(X)  # amplitude adjustment
        self.model.fit(X, y)

    def predict(self, path, outpath=None):
        data = eval(open(path).read())
        X = [self.combine_features(f1, f2) for (f1, f2) in data]
        X = self.scaler.transform(X)

        keys = list(data)
        preds = self.model.predict(X)
        predictions = {k: bool(p) for k, p in zip(keys, preds)}

        if outpath:
            with open(outpath, 'w') as f:
                f.write(str(predictions) + "\n")
        return predictions

# %% [markdown]
# ## Task 3: Audio classification

# %%
# Some constants (you can change any of these if useful)
SAMPLE_RATE = 16000
N_MELS = 64
N_CLASSES = 10
AUDIO_DURATION = 10 # seconds
BATCH_SIZE = 32

# %%
dataroot3 = "student_files/task3_audio_classification"

# %% [markdown]
# ## Run everything...

# %%
def run1():
    model = model1()
    model.train(dataroot1 + "/train.json")
    train_preds = model.predict(dataroot1 + "/train.json")
    test_preds = model.predict(dataroot1 + "/test.json", "predictions1.json")
    train_labels = eval(open(dataroot1 + "/train.json").read())
    acc1 = accuracy1(train_labels, train_preds)
    print("Task 1 training accuracy = " + str(acc1))

# %%
def run2():
    model = model2()
    model.train(dataroot2 + "/train.json")
    train_preds = model.predict(dataroot2 + "/train.json")
    test_preds = model.predict(dataroot2 + "/test.json", "predictions2.json")
    
    train_labels = eval(open(dataroot2 + "/train.json").read())
    acc2 = accuracy2(train_labels, train_preds)
    print("Task 2 training accuracy = " + str(acc2))

# %%
# def run3():
#     loaders = Loaders(dataroot3 + "/train.json", dataroot3 + "/test.json")
#     model = CNNClassifier()
#     pipeline = Pipeline(model, 1e-4)
    
#     pipeline.train(loaders.loaderTrain, loaders.loaderValid, 5)
#     train_preds, train_mAP = pipeline.evaluate(loaders.loaderTrain, 0.5)
#     valid_preds, valid_mAP = pipeline.evaluate(loaders.loaderValid, 0.5)
#     test_preds, _ = pipeline.evaluate(loaders.loaderTest, 0.5, "predictions3.json")
    
#     all_train = eval(open(dataroot3 + "/train.json").read())
#     for k in valid_preds:
#         # We split our training set into train+valid
#         # so need to remove validation instances from the training set for evaluation
#         all_train.pop(k)
#     acc3 = accuracy3(all_train, train_preds)
#     print("Task 3 training mAP = " + str(acc3))

# %%
run1()

# %%
run2()

# %%
# run3()

# %%



