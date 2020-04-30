import json
from glob import glob
from os.path import basename, exists, join
import numpy as np
import pandas as pd

# Pre-seed our metadata dictionary with some info
metadata = {'pieman':
              {'stimulus_duration': 450,
               'n_TRs': 300,
               'data_trims': [10, 8]},
            'prettymouth':
              {'stimulus_duration': 712,
               'n_TRs': 475,
               'data_trims': [14, 10]},
            'milkyway':
              {'stimulus_duration': 292,
               'n_TRs': 297,
               'data_trims': [14, 10]},
            'slumlordreach':
              {'n_TRs': 1205,
               'data_trims': [20, 8]},
            'notthefall':
              {'n_TRs': 400,
               'data_trims': [2, 8]},
            '21styear':
              {'n_TRs': 2249,
               'data_trims': [14, 10]},
            'bronx (PNI)':
              {'stimulus_duration': 374,
               'n_TRs': 390,
               'data_trims': [8, 8]},
            'pieman (PNI)':
              {'stimulus_duration': 278,
               'n_TRs': 294,
               'data_trims': [8, 8]},
            'black':
              {'stimulus_duration': 800,
               'n_TRs': 550,
               'timestamps': 'transcripts/black_timestamps.txt',
               'model': 'transcripts/black_word2vec.npy',
               'model_trims': [0, 0],
               'data_trims': [8, 8]},
            'forgot':
              {'stimulus_duration': 837,
               'n_TRs': 574,
               'timestamps': 'transcripts/forgot_timestamps.txt',
               'model': 'transcripts/forgot_word2vec.npy',
               'model_trims': [0, 0],
               'data_trims': [8, 8]}}

# Loop through stories (with dumb hack to not double-count pieman)
stories = [('pieman_', 'pieman'), 'prettymouth',
           'milkyway', 'slumlordreach',
           'notthefall', '21styear',
           ('bronx', 'bronx (PNI)'),
           ('piemanpni', 'pieman (PNI)'),
            'black', 'forgot']
subjects = {}
for story in stories:
    if type(story) != str:
        substory, story = story
    else:
        substory = story
    data_fns = glob(join('data',
                         f'sub-*_task-{substory}*fsaverage6*.tproject.gii'))
    subjects[story] = []
    for data_fn in data_fns:
        split_fn = basename(data_fn).split('_')
        sub = split_fn[0]
        subjects[story].append(sub)
        hemi = split_fn[-1].split('hemi-')[1][:2]

        if 'data' not in metadata[story]:
            metadata[story]['data'] = {}
        if sub not in metadata[story]['data']:
            metadata[story]['data'][sub] = {'lh': {},
                                            'rh': {}}
        metadata[story]['data'][sub][hemi] = data_fn

# Save the resulting dictionary to json
with open(join('data', 'metadata.json'), 'w') as f:
    json.dump(metadata, f, sort_keys=True, indent=2)


# Summarize subjects and sessions
with open(join('data', 'metadata.json')) as f:
    metadata = json.load(f)
    
# Subjects and stories
stories = ['pieman', 'prettymouth', 'milkyway', 'slumlordreach',
           'notthefall', '21styear', 'pieman (PNI)', 'bronx (PNI)',
           'black', 'forgot']

subjects = {}
for story in stories:
    for subject in metadata[story]['data'].keys():
        if subject not in subjects:
            subjects[subject] = [story]
        else:
            subjects[subject].append(story)

counts = []
for subject in subjects:
    counts.append(len(subjects[subject]))
unique = np.unique(counts, return_counts=True)

proportions = [c / sum(unique[1]) for c in unique[1]]


# Plot histogram of subjects across stories
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
bars = ax1.bar(unique[0], unique[1], color='gray')
for bar in bars:
    n = bar.get_height()
    ax1.annotate(f'{n}', xy=(bar.get_x() + bar.get_width() / 2, n),
                 xytext=(0,2), textcoords="offset points",
                 ha='center', va='bottom')
ax1.set_ylim(0, 95)
ax1.set_xlabel("number of stories")
ax1.set_ylabel("number of subjects")
ax1.set_title("Histogram of subjects\nparticipating in multiple stories")

histomat = np.zeros((len(stories), len(subjects)))
for y, subject in enumerate(subjects):
    for story in subjects[subject]:
        x = stories.index(story)
        histomat[x, y] = 1
ax2 = sns.heatmap(histomat, cmap=['.25', '.75'], cbar=False,
            xticklabels=10, yticklabels=stories)
ax2.tick_params(axis=u'both', which=u'both',length=0)
ax2.set_title("Subject participation\nacross stories")
ax2.set_xlabel("subject ID")
light = mpatches.Patch(color='.75', label='participant')
dark = mpatches.Patch(color='.25', label='not participant')
legend = ax2.legend(handles=[light, dark], loc='lower left',
                   framealpha=1)
plt.tight_layout()
plt.savefig('figures/subject_histogram.png',
            bbox_inches='tight', dpi=300, transparent=True)