import json
from glob import glob
from os.path import basename, exists, join

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