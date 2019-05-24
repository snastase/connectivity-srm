import numpy as np
from copy import deepcopy
import gensim
from nltk.corpus import stopwords

story_name = 'forgot'

# Load dictionary of input filenames
with open('metadata.json') as f:
    metadata = json.load(f)

# Load in pre-trained word2vec embeddings (slow)
vectors_fn = '~/Work/GoogleNews-vectors-negative300.bin'
embeddings = gensim.models.KeyedVectors.load_word2vec_format(
    vectors_fn, binary=True)
n_dims = 300

# Non-word annotations and stop words for exclusion
exclude = ['sp', '{LG}', '{NS}', '{LG}', '{OH}',
           '{CL}', '{PS}', '{GP}']
stop_list = stopwords.words('english') + ['']

# TR for fMRI acquisitions
duration = metadata[story_name]['stimulus_duration']
TR = 1.5
n_TRs = np.ceil(duration / TR).astype(int)
data_trims = metadata[story_name]['data_trims']
assert (data_trims[0] + n_TRs + data_trims[1] ==
        metadata[story_name]['n_TRs'])

# Sampling rate for forced-alignment algorithm
hz = 512

# Unpack output of forced-alignment algorithm
input_fn = metadata[story_name]['timestamps']
with open(input_fn) as f:
    lines = [line.strip().split(',')[:3] for line in f.readlines()]
    
# Reorganize the transcript and exclude some words
transcript = []
for word, onset, offset in lines:
    if word in exclude:
        continue
    onset, offset = (np.round(float(onset) / hz, decimals=3),
                     np.round(float(offset) / hz, decimals=3))
    transcript.append([onset, offset, word])

# Save this standardized transcript for sharing
with open(f'transcripts/{story_name}_words.tsv', 'w') as f:
    for line in transcript:
        f.write('\t'.join([str(item) for item in line]) + '\n')

# Downsample transcript to TRs
transcript_copy = deepcopy(transcript)
tr_transcript, tr_words, crossing_words = [], [], []
for tr in np.arange(n_TRs):
    update_tr = False
    tr_start, tr_end = tr * TR, tr * TR + TR
    if len(crossing_words) > 0:
        tr_words.extend(crossing_words)
    crossing_words = []
    while not update_tr:
        if len(transcript_copy) > 0:
            onset, offset, word = transcript_copy.pop(0)
        else:
            onset = np.inf
        if tr_start < onset < tr_end:
            if word not in stop_list:
                tr_words.append(word)
                if tr_end < offset < tr_end + TR:
                    crossing_words.append(word)
        else:
            transcript_copy.insert(0, [onset, offset, word])
            tr_transcript.append(tr_words)
            tr_words = []
            update_tr = True
assert len(tr_transcript) == n_TRs

# Compile word embeddings for each TR
tr_embeddings = []
skipped_words = []
for tr_words in tr_transcript:
    if len(tr_words) > 1:
        tr_embedding = []
        for word in tr_words:
            try:
                vector = embeddings.get_vector(word)
            except KeyError:
                skipped_words.append(word)
                vector = np.full(n_dims, np.nan)
                continue
            tr_embedding.append(vector)
        if len(tr_embedding) == 1:
            tr_embedding = tr_embedding[0]
        else:
            tr_embedding = np.mean(tr_embedding, axis=0)        
    else:
        tr_embedding = np.full(n_dims, np.nan)
    tr_embeddings.append(tr_embedding)
tr_embeddings = np.vstack(tr_embeddings)

# Save embedding array
np.save(f'transcripts/{story_name}_word2vec.npy', tr_embeddings)
