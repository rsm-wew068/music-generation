# %% [markdown]
# ## Homework 3: Symbolic Music Generation Using Markov Chains

# %% [markdown]
# **Before starting the homework:**
# 
# Please run `pip install miditok` to install the [MiDiTok](https://github.com/Natooz/MidiTok) package, which simplifies MIDI file processing by making note and beat extraction more straightforward.
# 
# You’re also welcome to experiment with other MIDI processing libraries such as [mido](https://github.com/mido/mido), [pretty_midi](https://github.com/craffel/pretty-midi) and [miditoolkit](https://github.com/YatingMusic/miditoolkit). However, with these libraries, you’ll need to handle MIDI quantization yourself, for example, converting note-on/note-off events into beat positions and durations.

# %%
# # run this command to install MiDiTok
# #!pip install miditok
# !pip install midiutil

# %%
# import required packages
import random
from glob import glob
from collections import defaultdict

import numpy as np
from numpy.random import choice

from symusic import Score
from miditok import REMI, TokenizerConfig
from midiutil import MIDIFile

# %%
# You can change the random seed but try to keep your results deterministic!
# If I need to make changes to the autograder it'll require rerunning your code,
# so it should ideally generate the same results each time.
random.seed(42)

# %% [markdown]
# ### Load music dataset
# We will use a subset of the [PDMX dataset](https://zenodo.org/records/14984509). 
# 
# Please find the link in the homework spec.
# 
# All pieces are monophonic music (i.e. one melody line) in 4/4 time signature.

# %%
midi_files = glob('PDMX_subset/*.mid')
len(midi_files)

# %% [markdown]
# ### Train a tokenizer with the REMI method in MidiTok

# %%
config = TokenizerConfig(num_velocities=1, use_chords=False, use_programs=False)
tokenizer = REMI(config)
tokenizer.train(vocab_size=1000, files_paths=midi_files)

# %% [markdown]
# ### Use the trained tokenizer to get tokens for each midi file
# In REMI representation, each note will be represented with four tokens: `Position, Pitch, Velocity, Duration`, e.g. `('Position_28', 'Pitch_74', 'Velocity_127', 'Duration_0.4.8')`; a `Bar_None` token indicates the beginning of a new bar.

# %%
# e.g.:
midi = Score(midi_files[0])
tokens = tokenizer(midi)[0].tokens
tokens[:10]

# %% [markdown]
# 1. Write a function to extract note pitch events from a midi file; and another extract all note pitch events from the dataset and output a dictionary that maps note pitch events to the number of times they occur in the files. (e.g. {60: 120, 61: 58, …}).
# 
# `note_extraction()`
# - **Input**: a midi file
# 
# - **Output**: a list of note pitch events (e.g. [60, 62, 61, ...])
# 
# `note_frequency()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: a dictionary that maps note pitch events to the number of times they occur, e.g {60: 120, 61: 58, …}

# %%
def note_extraction(midi_file):
    try:
        midi = Score(midi_file)
        if not midi.tracks:
            return []
        notes = midi.tracks[0].notes  # get notes from the first (and only) track
        return [note.pitch for note in notes]
    except Exception as e:
        print(f"Error reading {midi_file}: {e}")
        return []

# %%
def note_frequency(midi_files):
    # Q1b: Your code goes here
    freq = {}
    for midi_file in midi_files:
        notes = note_extraction(midi_file)
        for pitch in notes:
            freq[pitch] = freq.get(pitch, 0) + 1
    return freq

# %% [markdown]
# 2. Write a function to normalize the above dictionary to produce probability scores (e.g. {60: 0.13, 61: 0.065, …})
# 
# `note_unigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: a dictionary that maps note pitch events to probabilities, e.g. {60: 0.13, 61: 0.06, …}

# %%
def note_unigram_probability(midi_files):
    note_counts = note_frequency(midi_files)
    total = sum(note_counts.values())
    unigramProbabilities = {}
    
    for pitch, count in note_counts.items():
        unigramProbabilities[pitch] = count / total if total > 0 else 0.0
    
    return unigramProbabilities

# %% [markdown]
# 3. Generate a table of pairwise probabilities containing p(next_note | previous_note) values for the dataset; write a function that randomly generates the next note based on the previous note based on this distribution.
# 
# `note_bigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `bigramTransitions`: key: previous_note, value: a list of next_note, e.g. {60:[62, 64, ..], 62:[60, 64, ..], ...} (i.e., this is a list of every other note that occured after note 60, every note that occured after note 62, etc.)
# 
#   - `bigramTransitionProbabilities`: key:previous_note, value: a list of probabilities for next_note in the same order of `bigramTransitions`, e.g. {60:[0.3, 0.4, ..], 62:[0.2, 0.1, ..], ...} (i.e., you are converting the values above to probabilities)
# 
# `sample_next_note()`
# - **Input**: a note
# 
# - **Output**: next note sampled from pairwise probabilities

# %%
def note_bigram_probability(midi_files):
    bigramTransitions = defaultdict(list)
    bigramTransitionProbabilities = defaultdict(list)

    # Q3a: Your code goes here
    # Build bigram transitions: previous_note -> list of next_notes
    for midi_file in midi_files:
        notes = note_extraction(midi_file)
        for i in range(1, len(notes)):
            prev_note = notes[i-1]
            next_note = notes[i]
            bigramTransitions[prev_note].append(next_note)

    # Now convert to probabilities
    for prev_note, next_notes in bigramTransitions.items():
        # Count occurrences of each next_note
        next_note_counts = defaultdict(int)
        for n in next_notes:
            next_note_counts[n] += 1
        total = sum(next_note_counts.values())
        # Store unique next_notes and their probabilities in the same order
        unique_next_notes = list(next_note_counts.keys())
        probs = [next_note_counts[n] / total for n in unique_next_notes]
        bigramTransitions[prev_note] = unique_next_notes
        bigramTransitionProbabilities[prev_note] = probs

    return bigramTransitions, bigramTransitionProbabilities

# %%
def sample_next_note(note):
    # Q3b: Your code goes here
    # Use the bigram model to sample the next note given the previous note
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)
    if note in bigramTransitions and bigramTransitions[note]:
        next_notes = bigramTransitions[note]
        probs = bigramTransitionProbabilities[note]
        return choice(next_notes, p=probs)
    else:
        # fallback: sample from unigram if no bigram available
        unigramProbabilities = note_unigram_probability(midi_files)
        notes = list(unigramProbabilities.keys())
        probs = list(unigramProbabilities.values())
        return choice(notes, p=probs)

# %% [markdown]
# 4. Write a function to calculate the perplexity of your model on a midi file.
# 
#     The perplexity of a model is defined as 
# 
#     $\quad \text{exp}(-\frac{1}{N} \sum_{i=1}^N \text{log}(p(w_i|w_{i-1})))$
# 
#     where $p(w_1|w_0) = p(w_1)$, $p(w_i|w_{i-1}) (i>1)$ refers to the pairwise probability p(next_note | previous_note).
# 
# `note_bigram_perplexity()`
# - **Input**: a midi file
# 
# - **Output**: perplexity value

# %%
def note_bigram_perplexity(midi_file):
    # Compute perplexity of the bigram model on a midi file
    notes = note_extraction(midi_file)
    N = len(notes)
    if N == 0:
        return float('inf')
    unigramProbabilities = note_unigram_probability(midi_files)
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)
    log_prob_sum = 0.0

    for i, note in enumerate(notes):
        if i == 0:
            # Use unigram probability for the first note
            prob = unigramProbabilities.get(note, 1e-12)
        else:
            prev_note = notes[i-1]
            next_notes = bigramTransitions.get(prev_note, [])
            probs = bigramTransitionProbabilities.get(prev_note, [])
            if note in next_notes:
                idx = next_notes.index(note)
                prob = probs[idx]
            else:
                # fallback to unigram probability if unseen bigram
                prob = unigramProbabilities.get(note, 1e-12)
        # Avoid log(0)
        prob = max(prob, 1e-12)
        log_prob_sum += np.log(prob)

    perplexity = np.exp(-log_prob_sum / N)
    return perplexity

# %% [markdown]
# 5. Implement a second-order Markov chain, i.e., one which estimates p(next_note | next_previous_note, previous_note); write a function to compute the perplexity of this new model on a midi file. 
# 
#     The perplexity of this model is defined as 
# 
#     $\quad \text{exp}(-\frac{1}{N} \sum_{i=1}^N \text{log}(p(w_i|w_{i-2}, w_{i-1})))$
# 
#     where $p(w_1|w_{-1}, w_0) = p(w_1)$, $p(w_2|w_0, w_1) = p(w_2|w_1)$, $p(w_i|w_{i-2}, w_{i-1}) (i>2)$ refers to the probability p(next_note | next_previous_note, previous_note).
# 
# 
# `note_trigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `trigramTransitions`: key - (next_previous_note, previous_note), value - a list of next_note, e.g. {(60, 62):[64, 66, ..], (60, 64):[60, 64, ..], ...}
# 
#   - `trigramTransitionProbabilities`: key: (next_previous_note, previous_note), value: a list of probabilities for next_note in the same order of `trigramTransitions`, e.g. {(60, 62):[0.2, 0.2, ..], (60, 64):[0.4, 0.1, ..], ...}
# 
# `note_trigram_perplexity()`
# - **Input**: a midi file
# 
# - **Output**: perplexity value

# %%
def note_trigram_probability(midi_files):
    trigramTransitions = defaultdict(list)
    trigramTransitionProbabilities = defaultdict(list)
    
    # Q5a: Your code goes here
    # Build trigram transitions: (prev_prev_note, prev_note) -> list of next_notes
    for midi_file in midi_files:
        notes = note_extraction(midi_file)
        for i in range(2, len(notes)):
            prev_prev_note = notes[i-2]
            prev_note = notes[i-1]
            next_note = notes[i]
            trigramTransitions[(prev_prev_note, prev_note)].append(next_note)

    # Now convert to probabilities
    for prev_notes, next_notes in trigramTransitions.items():
        # Count occurrences of each next_note
        next_note_counts = defaultdict(int)
        for n in next_notes:
            next_note_counts[n] += 1
        total = sum(next_note_counts.values())
        unique_next_notes = list(next_note_counts.keys())
        probs = [next_note_counts[n] / total for n in unique_next_notes]
        trigramTransitions[prev_notes] = unique_next_notes
        trigramTransitionProbabilities[prev_notes] = probs

    return trigramTransitions, trigramTransitionProbabilities

# %%
def note_trigram_perplexity(midi_file):
    """
    Compute the perplexity of the trigram model on a midi file.
    """
    notes = note_extraction(midi_file)
    N = len(notes)
    if N == 0:
        return float('inf')
    unigramProbabilities = note_unigram_probability(midi_files)
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)
    trigramTransitions, trigramTransitionProbabilities = note_trigram_probability(midi_files)
    log_prob_sum = 0.0

    for i, note in enumerate(notes):
        if i == 0:
            # Use unigram probability for the first note
            prob = unigramProbabilities.get(note, 1e-12)
        elif i == 1:
            # Use bigram probability for the second note
            prev_note = notes[i-1]
            next_notes = bigramTransitions.get(prev_note, [])
            probs = bigramTransitionProbabilities.get(prev_note, [])
            if note in next_notes:
                idx = next_notes.index(note)
                prob = probs[idx]
            else:
                prob = unigramProbabilities.get(note, 1e-12)
        else:
            prev_prev_note = notes[i-2]
            prev_note = notes[i-1]
            key = (prev_prev_note, prev_note)
            next_notes = trigramTransitions.get(key, [])
            probs = trigramTransitionProbabilities.get(key, [])
            if note in next_notes:
                idx = next_notes.index(note)
                prob = probs[idx]
            else:
                # fallback to bigram, then unigram
                next_notes_bi = bigramTransitions.get(prev_note, [])
                probs_bi = bigramTransitionProbabilities.get(prev_note, [])
                if note in next_notes_bi:
                    idx = next_notes_bi.index(note)
                    prob = probs_bi[idx]
                else:
                    prob = unigramProbabilities.get(note, 1e-12)
        prob = max(prob, 1e-12)
        log_prob_sum += np.log(prob)

    perplexity = np.exp(-log_prob_sum / N)
    return perplexity

# %% [markdown]
# 6. Our model currently doesn’t have any knowledge of beats. Write a function that extracts beat lengths and outputs a list of [(beat position; beat length)] values.
# 
#     Recall that each note will be encoded as `Position, Pitch, Velocity, Duration` using REMI. Please keep the `Position` value for beat position, and convert `Duration` to beat length using provided lookup table `duration2length` (see below).
# 
#     For example, for a note represented by four tokens `('Position_24', 'Pitch_72', 'Velocity_127', 'Duration_0.4.8')`, the extracted (beat position; beat length) value is `(24, 4)`.
# 
#     As a result, we will obtain a list like [(0,8),(8,16),(24,4),(28,4),(0,4)...], where the next beat position is the previous beat position + the beat length. As we divide each bar into 32 positions by default, when reaching the end of a bar (i.e. 28 + 4 = 32 in the case of (28, 4)), the beat position reset to 0.

# %%
duration2length = {
    '0.2.8': 2,  # sixteenth note, 0.25 beat in 4/4 time signature
    '0.4.8': 4,  # eighth note, 0.5 beat in 4/4 time signature
    '1.0.8': 8,  # quarter note, 1 beat in 4/4 time signature
    '2.0.8': 16, # half note, 2 beats in 4/4 time signature
    '4.0.4': 32, # whole note, 4 beats in 4/4 time signature
}

# %% [markdown]
# `beat_extraction()`
# - **Input**: a midi file
# 
# - **Output**: a list of (beat position; beat length) values

# %%
def beat_extraction(midi_file):
    # Q6: Your code goes here
    # Use tokenizer to get tokens for the midi file
    tokens = tokenizer(Score(midi_file))[0].tokens
    beat_info = []
    i = 0
    while i < len(tokens):
        if tokens[i].startswith('Position_'):
            # Extract beat position
            beat_pos = int(tokens[i].split('_')[1])
            # Find the corresponding duration token
            # The next three tokens should be Pitch, Velocity, Duration
            if i + 3 < len(tokens) and tokens[i+3].startswith('Duration_'):
                duration_str = tokens[i+3].split('_')[1]
                beat_length = duration2length.get(duration_str, None)
                if beat_length is not None:
                    beat_info.append((beat_pos, beat_length))
            i += 4
        else:
            i += 1
    return beat_info

# %% [markdown]
# 7. Implement a Markov chain that computes p(beat_length | previous_beat_length) based on the above function.
# 
# `beat_bigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `bigramBeatTransitions`: key: previous_beat_length, value: a list of beat_length, e.g. {4:[8, 2, ..], 8:[8, 4, ..], ...}
# 
#   - `bigramBeatTransitionProbabilities`: key - previous_beat_length, value - a list of probabilities for beat_length in the same order of `bigramBeatTransitions`, e.g. {4:[0.3, 0.2, ..], 8:[0.4, 0.4, ..], ...}

# %%
def beat_bigram_probability(midi_files):
    bigramBeatTransitions = defaultdict(list)
    bigramBeatTransitionProbabilities = defaultdict(list)
    
    # Q7: Your code goes here
    # Build bigram transitions: previous_beat_length -> list of next beat_length
    for midi_file in midi_files:
        beat_info = beat_extraction(midi_file)
        for i in range(1, len(beat_info)):
            prev_beat_length = beat_info[i-1][1]
            next_beat_length = beat_info[i][1]
            bigramBeatTransitions[prev_beat_length].append(next_beat_length)
    
    # Now convert to probabilities
    for prev_beat_length, next_beat_lengths in bigramBeatTransitions.items():
        next_beat_counts = defaultdict(int)
        for bl in next_beat_lengths:
            next_beat_counts[bl] += 1
        total = sum(next_beat_counts.values())
        unique_next_beat_lengths = list(next_beat_counts.keys())
        probs = [next_beat_counts[bl] / total for bl in unique_next_beat_lengths]
        bigramBeatTransitions[prev_beat_length] = unique_next_beat_lengths
        bigramBeatTransitionProbabilities[prev_beat_length] = probs

    return bigramBeatTransitions, bigramBeatTransitionProbabilities

# %% [markdown]
# 8. Implement a function to compute p(beat length | beat position), and compute the perplexity of your models from Q7 and Q8. For both models, we only consider the probabilities of predicting the sequence of **beat lengths**.
# 
# `beat_pos_bigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `bigramBeatPosTransitions`: key - beat_position, value - a list of beat_length
# 
#   - `bigramBeatPosTransitionProbabilities`: key - beat_position, value - a list of probabilities for beat_length in the same order of `bigramBeatPosTransitions`
# 
# `beat_bigram_perplexity()`
# - **Input**: a midi file
# 
# - **Output**: two perplexity values correspond to the models in Q7 and Q8, respectively

# %%
def beat_pos_bigram_probability(midi_files):
    bigramBeatPosTransitions = defaultdict(list)
    bigramBeatPosTransitionProbabilities = defaultdict(list)
    
    # Q8a: Your code goes here
    # Build bigram transitions: beat_position -> list of beat_length
    for midi_file in midi_files:
        beat_info = beat_extraction(midi_file)
        for pos, length in beat_info:
            bigramBeatPosTransitions[pos].append(length)
    
    # Now convert to probabilities
    for pos, lengths in bigramBeatPosTransitions.items():
        length_counts = defaultdict(int)
        for l in lengths:
            length_counts[l] += 1
        total = sum(length_counts.values())
        unique_lengths = list(length_counts.keys())
        probs = [length_counts[l] / total for l in unique_lengths]
        bigramBeatPosTransitions[pos] = unique_lengths
        bigramBeatPosTransitionProbabilities[pos] = probs

    return bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities

# %%
def beat_bigram_perplexity(midi_file):
    bigramBeatTransitions, bigramBeatTransitionProbabilities = beat_bigram_probability(midi_files)
    bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities = beat_pos_bigram_probability(midi_files)
    # Q8b: Your code goes here
    # Hint: one more probability function needs to be computed

    beat_info = beat_extraction(midi_file)
    N = len(beat_info)
    if N == 0:
        return float('inf'), float('inf')

    # Perplexity for Q7: p(beat_length_i | beat_length_{i-1})
    log_prob_sum_Q7 = 0.0
    for i in range(N):
        if i == 0:
            # For the first beat, use the marginal probability of the beat length
            # Compute marginal probability from all beat lengths in the dataset
            all_lengths = []
            for midi_f in midi_files:
                all_lengths.extend([l for _, l in beat_extraction(midi_f)])
            total = len(all_lengths)
            count = all_lengths.count(beat_info[i][1])
            prob = count / total if total > 0 else 1e-12
        else:
            prev_beat_length = beat_info[i-1][1]
            next_beat_length = beat_info[i][1]
            next_lengths = bigramBeatTransitions.get(prev_beat_length, [])
            probs = bigramBeatTransitionProbabilities.get(prev_beat_length, [])
            if next_beat_length in next_lengths:
                idx = next_lengths.index(next_beat_length)
                prob = probs[idx]
            else:
                # fallback to marginal probability
                all_lengths = []
                for midi_f in midi_files:
                    all_lengths.extend([l for _, l in beat_extraction(midi_f)])
                total = len(all_lengths)
                count = all_lengths.count(next_beat_length)
                prob = count / total if total > 0 else 1e-12
        prob = max(prob, 1e-12)
        log_prob_sum_Q7 += np.log(prob)
    perplexity_Q7 = np.exp(-log_prob_sum_Q7 / N)

    # Perplexity for Q8: p(beat_length_i | beat_position_i)
    log_prob_sum_Q8 = 0.0
    for i in range(N):
        pos = beat_info[i][0]
        length = beat_info[i][1]
        next_lengths = bigramBeatPosTransitions.get(pos, [])
        probs = bigramBeatPosTransitionProbabilities.get(pos, [])
        if length in next_lengths:
            idx = next_lengths.index(length)
            prob = probs[idx]
        else:
            # fallback to marginal probability
            all_lengths = []
            for midi_f in midi_files:
                all_lengths.extend([l for _, l in beat_extraction(midi_f)])
            total = len(all_lengths)
            count = all_lengths.count(length)
            prob = count / total if total > 0 else 1e-12
        prob = max(prob, 1e-12)
        log_prob_sum_Q8 += np.log(prob)
    perplexity_Q8 = np.exp(-log_prob_sum_Q8 / N)

    return perplexity_Q7, perplexity_Q8

# %% [markdown]
# 9. Implement a Markov chain that computes p(beat_length | previous_beat_length, beat_position), and report its perplexity. 
# 
# `beat_trigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `trigramBeatTransitions`: key: (previous_beat_length, beat_position), value: a list of beat_length
# 
#   - `trigramBeatTransitionProbabilities`: key: (previous_beat_length, beat_position), value: a list of probabilities for beat_length in the same order of `trigramBeatTransitions`
# 
# `beat_trigram_perplexity()`
# - **Input**: a midi file
# 
# - **Output**: perplexity value

# %%
def beat_trigram_probability(midi_files):
    trigramBeatTransitions = defaultdict(list)
    trigramBeatTransitionProbabilities = defaultdict(list)

    # Q9a: Your code goes here
    # Build trigram transitions: (previous_beat_length, beat_position) -> list of next beat_length
    for midi_file in midi_files:
        beat_info = beat_extraction(midi_file)
        for i in range(1, len(beat_info)):
            prev_beat_length = beat_info[i-1][1]
            curr_beat_position = beat_info[i][0]
            next_beat_length = beat_info[i][1]
            trigramBeatTransitions[(prev_beat_length, curr_beat_position)].append(next_beat_length)

    # Now convert to probabilities
    for key, next_lengths in trigramBeatTransitions.items():
        next_length_counts = defaultdict(int)
        for l in next_lengths:
            next_length_counts[l] += 1
        total = sum(next_length_counts.values())
        unique_next_lengths = list(next_length_counts.keys())
        probs = [next_length_counts[l] / total for l in unique_next_lengths]
        trigramBeatTransitions[key] = unique_next_lengths
        trigramBeatTransitionProbabilities[key] = probs

    return trigramBeatTransitions, trigramBeatTransitionProbabilities

# %%
def beat_trigram_perplexity(midi_file):
    """
    Compute the perplexity of the trigram beat model on a midi file.
    """
    trigramBeatTransitions, trigramBeatTransitionProbabilities = beat_trigram_probability(midi_files)
    beat_info = beat_extraction(midi_file)
    N = len(beat_info)
    if N == 0:
        return float('inf')

    # Compute marginal probability for fallback
    all_lengths = []
    for midi_f in midi_files:
        all_lengths.extend([l for _, l in beat_extraction(midi_f)])
    total = len(all_lengths)

    log_prob_sum = 0.0
    for i in range(N):
        if i == 0:
            # Use marginal probability for the first beat
            count = all_lengths.count(beat_info[i][1])
            prob = count / total if total > 0 else 1e-12
        else:
            prev_beat_length = beat_info[i-1][1]
            curr_beat_position = beat_info[i][0]
            next_beat_length = beat_info[i][1]
            key = (prev_beat_length, curr_beat_position)
            next_lengths = trigramBeatTransitions.get(key, [])
            probs = trigramBeatTransitionProbabilities.get(key, [])
            if next_beat_length in next_lengths:
                idx = next_lengths.index(next_beat_length)
                prob = probs[idx]
            else:
                # fallback to marginal probability
                count = all_lengths.count(next_beat_length)
                prob = count / total if total > 0 else 1e-12
        prob = max(prob, 1e-12)
        log_prob_sum += np.log(prob)
    perplexity = np.exp(-log_prob_sum / N)
    return perplexity

# %% [markdown]
# 10. Use the model from Q5 to generate N notes, and the model from Q8 to generate beat lengths for each note. Save the generated music as a midi file (see code from workbook1) as q10.mid. Remember to reset the beat position to 0 when reaching the end of a bar.
# 
# `music_generate`
# - **Input**: target length, e.g. 500
# 
# - **Output**: a midi file q10.mid
# 
# Note: the duration of one beat in MIDIUtil is 1, while in MidiTok is 8. Divide beat length by 8 if you use methods in MIDIUtil to save midi files.

# %%
def music_generate(length):
    # Get probability models
    unigramProbabilities = note_unigram_probability(midi_files)
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)
    trigramTransitions, trigramTransitionProbabilities = note_trigram_probability(midi_files)
    beat_pos_transitions, beat_pos_probs = beat_pos_bigram_probability(midi_files)

    # Sample notes using trigram model
    sampled_notes = []
    # Start with two notes sampled from unigram and bigram
    notes_list = list(unigramProbabilities.keys())
    notes_probs = list(unigramProbabilities.values())
    first_note = choice(notes_list, p=notes_probs)
    sampled_notes.append(first_note)
    # For the second note, use bigram
    next_notes = bigramTransitions.get(first_note, [])
    next_probs = bigramTransitionProbabilities.get(first_note, [])
    if next_notes:
        second_note = choice(next_notes, p=next_probs)
    else:
        second_note = choice(notes_list, p=notes_probs)
    sampled_notes.append(second_note)
    # Now sample using trigram
    for i in range(2, length):
        prev_prev = sampled_notes[i-2]
        prev = sampled_notes[i-1]
        key = (prev_prev, prev)
        next_notes_tri = trigramTransitions.get(key, [])
        next_probs_tri = trigramTransitionProbabilities.get(key, [])
        if next_notes_tri:
            next_note = choice(next_notes_tri, p=next_probs_tri)
        else:
            # fallback to bigram
            next_notes_bi = bigramTransitions.get(prev, [])
            next_probs_bi = bigramTransitionProbabilities.get(prev, [])
            if next_notes_bi:
                next_note = choice(next_notes_bi, p=next_probs_bi)
            else:
                next_note = choice(notes_list, p=notes_probs)
        sampled_notes.append(next_note)

    # Sample beat lengths using beat position model (Q8)
    sampled_beats = []
    beat_position = 0
    for i in range(length):
        lengths = beat_pos_transitions.get(beat_position, [])
        probs = beat_pos_probs.get(beat_position, [])
        if lengths:
            beat_length = choice(lengths, p=probs)
        else:
            # fallback: sample from all beat lengths in dataset
            all_lengths = []
            for midi_f in midi_files:
                all_lengths.extend([l for _, l in beat_extraction(midi_f)])
            if all_lengths:
                beat_length = choice(all_lengths)
            else:
                beat_length = 8  # default to quarter note
        sampled_beats.append((beat_position, beat_length))
        beat_position += beat_length
        if beat_position >= 32:
            beat_position = 0

    # Save the generated music as a midi file
    midi_out = MIDIFile(1)
    track = 0
    time = 0  # in beats
    channel = 0
    volume = 100
    for i in range(length):
        pitch = sampled_notes[i]
        beat_pos, beat_length = sampled_beats[i]
        duration = beat_length / 8  # MIDIUtil: 1 = one beat
        midi_out.addNote(track, channel, pitch, time, duration, volume)
        time += duration
    with open("q10.mid", "wb") as outf:
        midi_out.writeFile(outf)



