import json
from os.path import exists, join
import numpy as np
from scipy.stats import zscore
from gifti_io import read_gifti


# Function for delaying model embedding by several TRs
def delay_model(model, delays=[2, 3, 4, 5]):

    # Horizontally stack semantic vectors at varying delays
    delayed = []
    for delay in delays:
        delayed.append(np.vstack((np.full((delay,
                                           model.shape[1]),
                                          np.nan),
                                  model[:-delay])))
    delayed = np.hstack(delayed)

    # Zero out any NaNs in delayed embedding
    delayed[np.isnan(delayed)] = 0

    return delayed


# Split models into train and test halves
def split_models(metadata, stories=None, subjects=None, half=1,
                 delays=[2, 3, 4, 5], zscore_model=False):

    # By default grab all stories in metadata
    stories = check_keys(metadata, keys=stories)

    assert half in [1, 2]

    model_splits = {}
    for story in stories:

        # Get model
        model = np.load(metadata[story]['model'])

        # Trim model
        model_trims = metadata[story]['model_trims']
        model = model[model_trims[0]:(-model_trims[1] or None), :]

        # Find the midpoint of model and split
        midpoint = model.shape[0] // 2

        if half == 1:
            half_model = model[:midpoint]
        elif half == 2:
            half_model = model[midpoint:]

        # Optionally z-score model features
        if zscore_model:
            half_model = zscore(half_model, axis=0)

        # Horizontally stack delayed replicates of model
        half_model = delay_model(half_model, delays=delays)

        model_splits[story] = half_model

        print(f"Loaded model for story '{story}' half {half}")

    return model_splits


# Split into first and second half for train/test and save
def split_data(metadata, stories=None, subjects=None,
               hemisphere=None, zscore_data=True,
               mask=None, roi=None, save_files=True):

    # By default grab all stories in metadata
    stories = check_keys(metadata, keys=stories)

    # Loop through stories
    for story in stories:

        # By default just grab all subjects in metadata
        subject_list = check_keys(metadata[story]['data'],
                                  keys=subjects, subkey=story)

        # Use data trims to find midpoint
        data_trims = metadata[story]['data_trims']
        n_TRs = metadata[story]['n_TRs']
        midpoint = (n_TRs - data_trims[0] - data_trims[1]) // 2

        # Loop through subjects and split data
        for subject in subject_list:

            # By default grab both hemispheres
            hemis = check_keys(metadata[story]['data'][subject],
                               keys=hemisphere)

            # One or both hemispheres 
            for hemi in hemis:

                # Load in data from GIfTI
                data_fn = metadata[story]['data'][subject][hemi]
                surf_data = read_gifti(data_fn)

                # Optionally mask
                if mask and roi:
                    surf_data = surf_data[:, mask[hemi]]

                # Trim data
                assert surf_data.shape[0] == n_TRs, ("TR mismatch! "
                    f"Expected {n_TRs}, but got {surf_data.shape[0]}")
                surf_data = surf_data[data_trims[0]:(
                    -data_trims[1] or None), :]

                half1_data = surf_data[:midpoint, :]
                half2_data = surf_data[midpoint:, :]

                if zscore_data:
                    half1_data = zscore(half1_data, axis=0)
                    half2_data = zscore(half2_data, axis=0)

                if save_files:
                    if mask and roi:
                        half1_fn = (f'data/{subject}_task-{story}_'
                                    f'half-1_{roi}_{hemi}.npy')
                        half2_fn = (f'data/{subject}_task-{story}_'
                                    f'half-2_{roi}_{hemi}.npy')

                    else:
                        half1_fn = (f'data/{subject}_task-{story}_'
                                    f'half-1_{hemi}.npy')
                        half2_fn = (f'data/{subject}_task-{story}_'
                                    f'half-2_{hemi}.npy')

                    np.save(half1_fn, half1_data)
                    np.save(half2_fn, half2_data)

            print(f"Saved split-half data for subject '{subject}' "
                  f"and story '{story}'")


# Load preexisting split data
def load_split_data(metadata, stories=None, subjects=None,
                    hemisphere=None, mask=None, half=1, prefix=None,
                    verbose=False):

    # By default grab all stories in metadata
    stories = check_keys(metadata, keys=stories)

    # Check half assignment
    assert half in [1, 2]

    # Loop through stories
    data_splits = {}
    for story in stories:

        data_splits[story] = {}

        # By default just grab all subjects in metadata
        subject_list = check_keys(metadata[story]['data'],
                                  keys=subjects, subkey=story)

        # Loop through subjects and split data
        for subject in subject_list:

            data_splits[story][subject] = {}

            # By default grab both hemispheres
            hemis = check_keys(metadata[story]['data'][subject],
                               keys=hemisphere)

            # One or both hemispheres                
            for hemi in hemis:

                if prefix:
                    half_fn = (f'data/{subject}_task-{story}_'
                               f'half-{half}_{prefix}_{hemi}.npy')
                else:
                    half_fn = (f'data/{subject}_task-{story}_'
                               f'half-{half}_{hemi}.npy')

                if not exists(half_fn):
                    print(f"Couldn't find {half_fn}!!!")
                    pass

                # Load files
                half_data = np.load(half_fn, allow_pickle=True)

                if mask:
                    half_data = half_data[:, mask[hemi]]

                data_splits[story][subject][hemi] = half_data

            if verbose:
                print(f"Loaded subject '{subject}' data "
                      f"for story '{story}' half {half}")

    return data_splits


# Convenience function to check dictionary keys and assume defaults
def check_keys(data, keys=None, subkey=None):

    # By default grab all available keys
    if type(keys) == dict and subkey:
        keys = keys[subkey]
    if not keys:
        keys = data.keys()
    elif type(keys) == str:
        assert keys in data.keys()
        keys = [keys]
    elif type(keys) == list:
        for key in keys:
            assert key in data.keys()
        keys = keys
    else:
        raise KeyError(f"Unrecognized keys: {keys}")

    return list(keys)


# Name guard for when we actually want to split all data
if __name__ == '__main__':

    # Load dictionary of input filenames and parameters
    with open(join('data', 'metadata.json')) as f:
        metadata = json.load(f)

    # Split whole-brain data, no ROIs
    stories = ['pieman', 'prettymouth', 'milkyway',
               'slumlordreach', 'notthefall', '21styear',
               'pieman (PNI)', 'bronx (PNI)', 'black', 'forgot']
    
    split_data(metadata, stories=stories, subjects=None,
               hemisphere=None, zscore_data=True,
               save_files=True)

    # Split data into ROIs   
    rois = ['EAC', 'AAC', 'TPOJ', 'PMC']
    for roi in rois:
        mask_lh = np.load(f'data/{roi}_mask_lh.npy').astype(bool)
        mask_rh = np.load(f'data/{roi}_mask_rh.npy').astype(bool)
        mask = {'lh': mask_lh, 'rh': mask_rh}

        split_data(metadata, stories=stories, subjects=None,
                   hemisphere=None, zscore_data=True,
                   mask=mask, roi=f'{roi}_noSRM', save_files=True)
