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
dataroot1 = "student_files/task1_composer_classification/"

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import miditoolkit
import numpy as np
import os

class model1():
    def __init__(self):
        self.model = LogisticRegression(max_iter=10000)
        self.label_encoder = LabelEncoder()

    def features(self, path):
        clean_path = path.replace("midis/", "")
        midi_path = os.path.join("student_files/task1_composer_classification", "midis", clean_path)

        try:
            midi = miditoolkit.midi.parser.MidiFile(midi_path)
            notes = [note for inst in midi.instruments for note in inst.notes]
        except:
            return [0.0] * 8

        if len(notes) == 0:
            return [0.0] * 8

        pitches = [note.pitch for note in notes]
        durations = [note.end - note.start for note in notes]
        velocities = [note.velocity for note in notes]

        return [
            np.mean(pitches),
            np.std(pitches),
            np.min(pitches),
            np.max(pitches),
            np.mean(durations),
            np.std(durations),
            np.mean(velocities),
            len(notes)
        ]

    def train(self, path):
        with open(path, 'r') as f:
            train_json = eval(f.read())
        X = [self.features(k) for k in train_json]
        y = [train_json[k] for k in train_json]
        self.label_encoder.fit(y)
        self.model.fit(X, self.label_encoder.transform(y))

    def predict(self, path, outpath=None, top_k=3000, fallback="Beethoven"):
        with open(path, 'r') as f:
            d = eval(f.read())
        files = list(d.keys()) if isinstance(d, dict) else d

        probs = []
        predictions = {}

        for k in files:
            x = self.features(k)
            prob = self.model.predict_proba([x])[0]
            pred_idx = np.argmax(prob)
            pred_label = self.label_encoder.inverse_transform([pred_idx])[0]
            confidence = prob[pred_idx]
            probs.append((k, pred_label, confidence))

        # Sort by confidence
        probs.sort(key=lambda x: x[2], reverse=True)

        for i, (fname, pred, _) in enumerate(probs):
            if i < top_k:
                predictions[fname] = pred
            else:
                predictions[fname] = fallback

        if outpath:
            with open(outpath, 'w') as f:
                f.write(str(predictions) + "\n")
        return predictions

# %% [markdown]
# ## Task 2: Sequence prediction

# %%
dataroot2 = "student_files/task2_next_sequence_prediction/"

# %%
class model2():
    def __init__(self):
        pass

    def features(self, path):
        midi_obj = miditoolkit.midi.parser.MidiFile(dataroot2 + '/' + path)
        notes = midi_obj.instruments[0].notes
        num_notes = len(notes)
        average_pitch = sum([note.pitch for note in notes]) / num_notes
        features = [average_pitch]
        return features
    
    def train(self, path):
        # This baseline doesn't use any model (it just measures feature similarity)
        # You can use this approach but *probably* you'll want to implement a model
        pass

    def predict(self, path, outpath=None):
        d = eval(open(path, 'r').read())
        predictions = {}
        for k in d:
            path1,path2 = k # Keys are pairs of paths
            x1 = self.features(path1)
            x2 = self.features(path2)
            # Note: hardcoded difference between features
            if abs(x1[0] - x2[0]) < 5:
                predictions[k] = True
            else:
                predictions[k] = False
        if outpath:
            with open(outpath, "w") as z:
                z.write(str(predictions) + '\n')
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
dataroot3 = "student_files/task3_audio_classification/"

# %%
def extract_waveform(path):
    waveform, sr = librosa.load(dataroot3 + '/' + path, sr=SAMPLE_RATE)
    waveform = np.array([waveform])
    if sr != SAMPLE_RATE:
        resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resample(waveform)
    # Pad so that everything is the right length
    target_len = SAMPLE_RATE * AUDIO_DURATION
    if waveform.shape[1] < target_len:
        pad_len = target_len - waveform.shape[1]
        waveform = F.pad(waveform, (0, pad_len))
    else:
        waveform = waveform[:, :target_len]
    waveform = torch.FloatTensor(waveform)
    return waveform

# %%
class AudioDataset(Dataset):
    def __init__(self, meta, preload = True):
        self.meta = meta
        ks = list(meta.keys())
        self.idToPath = dict(zip(range(len(ks)), ks))
        self.pathToFeat = {}

        self.mel = MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=N_MELS)
        self.db = AmplitudeToDB()
        
        self.preload = preload # Determines whether the features should be preloaded (uses more memory)
                               # or read from disk / computed each time (slow if your system is i/o-bound)
        if self.preload:
            for path in ks:
                waveform = extract_waveform(path)
                mel_spec = self.db(self.mel(waveform)).squeeze(0)
                self.pathToFeat[path] = mel_spec

    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        # Faster version, preloads the features
        path = self.idToPath[idx]
        tags = self.meta[path]
        bin_label = torch.tensor([1 if tag in tags else 0 for tag in TAGS], dtype=torch.float32)

        if self.preload:
            mel_spec = self.pathToFeat[path]
        else:
            waveform = extract_waveform(path)
            mel_spec = self.db(self.mel(waveform)).squeeze(0)
        
        return mel_spec.unsqueeze(0), bin_label, path

# %%
class Loaders():
    def __init__(self, train_path, test_path, split_ratio=0.9, seed = 0):
        torch.manual_seed(seed)
        random.seed(seed)
        
        meta_train = eval(open(train_path, 'r').read())
        l_test = eval(open(test_path, 'r').read())
        meta_test = dict([(x,[]) for x in l_test]) # Need a dictionary for the above class
        
        all_train = AudioDataset(meta_train)
        test_set = AudioDataset(meta_test)
        
        # Split all_train into train + valid
        total_len = len(all_train)
        train_len = int(total_len * split_ratio)
        valid_len = total_len - train_len
        train_set, valid_set = random_split(all_train, [train_len, valid_len])
        
        self.loaderTrain = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        self.loaderValid = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        self.loaderTest = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# %%
class CNNClassifier(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(32 * (N_MELS // 4) * (801 // 4), 256)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (B, 16, mel/2, time/2)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 32, mel/4, time/4)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return torch.sigmoid(self.fc2(x))  # multilabel → sigmoid

# %%
class Pipeline():
    def __init__(self, model, learning_rate, seed = 0):
        # These two lines will (mostly) make things deterministic.
        # You're welcome to modify them to try to get a better solution.
        torch.manual_seed(seed)
        random.seed(seed)

        self.device = torch.device("cpu") # Can change this if you have a GPU, but the autograder will use CPU
        self.model = model.to(self.device) #model.cuda() # Also uncomment these lines for GPU
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()

    def evaluate(self, loader, threshold=0.5, outpath=None):
        self.model.eval()
        preds, targets, paths = [], [], []
        with torch.no_grad():
            for x, y, ps in loader:
                x = x.to(self.device) #x.cuda()
                y = y.to(self.device) #y.cuda()
                outputs = self.model(x)
                preds.append(outputs.cpu())
                targets.append(y.cpu())
                paths += list(ps)
        
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        preds_bin = (preds > threshold).float()
        
        predictions = {}
        for i in range(preds_bin.shape[0]):
            predictions[paths[i]] = [TAGS[j] for j in range(len(preds_bin[i])) if preds_bin[i][j]]
        
        mAP = None
        if outpath: # Save predictions
            with open(outpath, "w") as z:
                z.write(str(predictions) + '\n')
        else: # Only compute accuracy if we're *not* saving predictions, since we can't compute test accuracy
            mAP = average_precision_score(targets, preds, average='macro')
        return predictions, mAP

    def train(self, train_loader, val_loader, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for x, y, path in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                x = x.to(self.device) #x.cuda()
                y = y.to(self.device) #y.cuda()
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            val_predictions, mAP = self.evaluate(val_loader)
            print(f"[Epoch {epoch+1}] Loss: {running_loss/len(train_loader):.4f} | Val mAP: {mAP:.4f}")

# %% [markdown]
# ## Run everything...

# %%
def run1():
    model = model1()
    model.train(dataroot1 + "/train.json")
    train_preds = model.predict(dataroot1 + "/train.json")
    test_preds = model.predict(dataroot1 + "/test.json", "predictions1_k150.json", fallback="Beethoven", top_k=10)
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
def run3():
    loaders = Loaders(dataroot3 + "/train.json", dataroot3 + "/test.json")
    model = CNNClassifier()
    pipeline = Pipeline(model, 1e-4)
    
    pipeline.train(loaders.loaderTrain, loaders.loaderValid, 5)
    train_preds, train_mAP = pipeline.evaluate(loaders.loaderTrain, 0.5)
    valid_preds, valid_mAP = pipeline.evaluate(loaders.loaderValid, 0.5)
    test_preds, _ = pipeline.evaluate(loaders.loaderTest, 0.5, "predictions3.json")
    
    all_train = eval(open(dataroot3 + "/train.json").read())
    for k in valid_preds:
        # We split our training set into train+valid
        # so need to remove validation instances from the training set for evaluation
        all_train.pop(k)
    acc3 = accuracy3(all_train, train_preds)
    print("Task 3 training mAP = " + str(acc3))

# %%
run1()

# %%
run2()

# %%
run3()

# %%



