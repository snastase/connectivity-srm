import json

input_fns = {'black':
                 {'model': 'transcripts/black_word2vec_embeddings.npy',
                  'model_trims': [0, 0],
                  'data': {'sub-02': 'data/sub-02_task-black.fsaverage6.lh.tproject.gii',
                           'sub-15': 'data/sub-15_task-black.fsaverage6.lh.tproject.gii'},
                  'data_trims': [8, 8]},
             'forgot':
                  {'model': 'transcripts/forgot_word2vec_embeddings.npy',
                   'model_trims': [0, 0],
                   'data': {'sub-02': 'data/sub-02_task-forgot.fsaverage6.lh.tproject.gii',
                            'sub-15': 'data/sub-15_task-forgot.fsaverage6.lh.tproject.gii'},
                   'data_trims': [8, 8]}}

with open('input_fns.json', 'w') as f:
    json.dump(input_fns, f, sort_keys=True, indent=2)