import json
from glob import glob
from os.path import basename, exists, join

metadata = {'black':
             {'stimulus_duration': 800,
              'n_TRs': 550,
              'timestamps': 'transcripts/black_timestamps.txt',
              'model': 'transcripts/black_word2vec.npy',
              'model_trims': [0, 0],
              'data':
                {'sub-02':
                  {'lh': 
                    'data/sub-02_task-black.fsaverage6.lh.tproject.gii',
                   'rh':
                    'data/sub-02_task-black.fsaverage6.rh.tproject.gii'},
                 'sub-15':
                  {'lh':
                    'data/sub-15_task-black.fsaverage6.lh.tproject.gii',
                   'rh':
                    'data/sub-15_task-black.fsaverage6.rh.tproject.gii'}},
              'data_trims': [8, 8]},
            'forgot':
              {'stimulus_duration': 837,
               'n_TRs': 574,
               'timestamps': 'transcripts/forgot_timestamps.txt',
               'model': 'transcripts/forgot_word2vec.npy',
               'model_trims': [0, 0],
               'data':
                 {'sub-02':
                   {'lh':
                     'data/sub-02_task-forgot.fsaverage6.lh.tproject.gii',
                    'rh':
                     'data/sub-02_task-forgot.fsaverage6.rh.tproject.gii'},
                  'sub-15':
                   {'lh':
                     'data/sub-15_task-forgot.fsaverage6.lh.tproject.gii',
                    'rh':
                     'data/sub-15_task-forgot.fsaverage6.rh.tproject.gii'}},
               'data_trims': [8, 8]}}

stories = ['black', 'forgot']
for story in stories:
    data_fns = glob(join('data',
                         f'sub-*_task-{story}.fsaverage6.*.tproject.gii'))
    for data_fn in data_fns:
        split_fn = basename(data_fn).split('_')
        sub = split_fn[0]
        hemi = split_fn[-1].split('.')[-3]

        if not story in metadata:
            metadata[story] = {'data': {}}
        if sub not in metadata[story]['data']:
            metadata[story]['data'][sub] = {'lh': {},
                                            'rh': {}}
        metadata[story]['data'][sub][hemi] = data_fn

with open('metadata.json', 'w') as f:
    json.dump(metadata, f, sort_keys=True, indent=2)
