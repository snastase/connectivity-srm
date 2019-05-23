import json

metadata = {'black':
             {'stimulus_duration': 837,
              'n_TRs': 574,
              'timestamps': 'transcripts/black_timestamps.txt',
              'model': 'transcripts/black_word2vec.npy',
              'model_trims': [0, 0],
              'data':
                {'sub-02':
                  'data/sub-02_task-black.fsaverage6.lh.tproject.gii',
                 'sub-15':
                  'data/sub-15_task-black.fsaverage6.lh.tproject.gii'},
              'data_trims': [8, 8]},
            'forgot':
              {'stimulus_duration': 800,
               'n_TRs': 550,
               'timestamps': 'transcripts/forgot_timestamps.txt',
               'model': 'transcripts/forgot_word2vec.npy',
               'model_trims': [0, 0],
               'data':
                 {'sub-02':
                   'data/sub-02_task-forgot.fsaverage6.lh.tproject.gii',
                  'sub-15':
                   'data/sub-15_task-forgot.fsaverage6.lh.tproject.gii'},
               'data_trims': [8, 8]}}

with open('metadata.json', 'w') as f:
    json.dump(metadata, f, sort_keys=True, indent=2)
