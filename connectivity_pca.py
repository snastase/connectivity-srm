import json
from os.path import exists, join
from time import time
import numpy as np
from scipy.stats import zscore
from sklearn.decomposition import PCA
from gifti_io import read_gifti
from split_stories import check_keys, load_split_data
from target_isfc import (parcel_means, parcel_srm,
                         vertex_isc, target_isfc)


# Fit PCA (on connectivity matrices)
def pca_fit(target_fcs, stories=None, subjects=None,
            hemisphere=None, k=360, n_iter=10,
            half=1, save_prefix=None):

    # By default grab all stories
    stories = check_keys(target_fcs, keys=stories)

    # Recompile FCs accounting for repeat subjects
    subject_fcs = {}
    for story in stories:

        # By default just grab all subjects
        subject_list = check_keys(target_fcs[story], keys=subjects,
                                  subkey=story)

        for subject in subject_list:

            # For simplicity we just assume same hemis across stories/subjects
            hemis = check_keys(target_fcs[story][subject],
                               keys=hemisphere)

            for hemi in hemis:
                
                # If subject is not already there, make new dict for them
                if subject not in subject_fcs:
                    subject_fcs[subject] = {}
                    
                # If hemispheres aren't in there, add them
                if hemi not in subject_fcs[subject]:
                    subject_fcs[subject][hemi] = []
                    
                # Finally, make list of connectivity matrices per subject
                subject_fcs[subject][hemi].append(
                        target_fcs[story][subject][hemi])

    # Stack FCs in connectivity space (for all subjects across stories!)
    all_subjects = list(subject_fcs.keys())
    for subject in all_subjects:
        for hemi in hemis:

            # If more than one connectivity per subject, take average
            if len(subject_fcs[subject][hemi]) > 1:
                subject_fcs[subject][hemi] = np.mean(subject_fcs[subject][hemi],
                                                     axis=0)
            else:
                subject_fcs[subject][hemi] = subject_fcs[subject][hemi][0]

    # Convert FCs to list for PCA (grab the shared space too)
    transforms = {}
    for hemi in hemis:
        
        # Declare PCA for this hemi
        pca = PCA(n_components=k)

        subject_stack = []
        for subject in all_subjects:
            subject_stack.append(subject_fcs[subject][hemi].T)
        subject_stack = np.vstack(subject_stack)

        # Fit PCA
        start = time()
        pca.fit(subject_stack)
        print(f"Finished fitting PCA after {time() - start:.1f} seconds")
        
        transforms[hemi] = pca

    if save_prefix:
        np.save(f'data/half-{half}_{save_prefix}_pca.npy', transforms)
        
    return transforms


# Apply learned SRM projections to data
def pca_transform(data, transforms, half=1, stories=None,
                  subjects=None, hemisphere=None,
                  zscore_transformed=True, save_prefix=None):

    # By default grab all stories
    stories = check_keys(data, keys=stories)

    data_transformed = {}
    for story in stories:

        data_transformed[story] = {}

        # By default just grab all subjects
        subject_list = check_keys(data[story], keys=subjects,
                                  subkey=story)

        for subject in subject_list:

            data_transformed[story][subject] = {}
            hemis = check_keys(data[story][subject],
                               keys=hemisphere)

            for hemi in hemis:

                transformed = transforms[hemi].transform(data[story][subject][hemi])

                # Optionally z-score transformed output data
                if zscore_transformed:
                    transformed = zscore(transformed, axis=0)

                data_transformed[story][subject][hemi] = transformed

                if save_prefix:
                    save_fn = (f'data/{subject}_task-{story}_'
                               f'half-{half}_{save_prefix}_{hemi}.npy')
                    np.save(save_fn, transformed)

    return data_transformed


# Fit connectivity SRM and transform both training and test data
def connectivity_pca(train_data, test_data, targets, target_fc=target_isfc,
                     train_half=1, test_half=2, stories=None, subjects=None,
                     hemisphere=None, save_prefix=None, **kwargs):

    # Compute ISFCs with targets (save/load to save time)
    if save_prefix:
        target_fcs_fn = join('data', (f'half-{train_half}_' + 
                             '_'.join(save_prefix.split('_')[:2]) + '_isfcs.npy'))

        if exists(target_fcs_fn):
            target_fcs = np.load(target_fcs_fn, allow_pickle=True).item()
            print(f"Loaded pre-existing ISFCs {target_fcs_fn}")
        else:
            target_fcs = target_fc(train_data, targets, stories=stories,
                                   subjects=subjects, hemisphere=hemisphere)
            np.save(target_fcs_fn, target_fcs)
            print(f"Saved ISFCs as {target_fcs_fn}")
    else:
        target_fcs = target_fc(train_data, targets, stories=stories,
                               subjects=subjects, hemisphere=hemisphere)

    # Fit SRM on connectivities and get transformation matrices
    transforms = pca_fit(target_fcs, stories=stories,
                         hemisphere=hemisphere, half=train_half,
                         save_prefix=save_prefix, **kwargs)

    # Apply transformations to training data
    train_transformed = pca_transform(train_data, transforms,
                                      half=train_half,
                                      stories=stories,
                                      subjects=subjects,
                                      hemisphere=hemisphere,
                                      save_prefix=save_prefix + '-train')
    
    print("Finished applying cPCA transformations to training data")

    # Apply transformations to test data
    test_transformed = pca_transform(test_data, transforms,
                                     half=test_half,
                                     stories=stories,
                                     subjects=subjects,
                                     hemisphere=hemisphere,
                                     save_prefix=save_prefix + '-test')
    
    print("Finished applying cPCA transformations to test data")

    return train_transformed, test_transformed


# Fit connectivity SRM and transform both training and test data
def temporal_pca(train_data, test_data, train_half=1, test_half=2,
                 stories=None, subjects=None,
                 hemisphere=None, save_prefix=None, **kwargs):
    
    # Transpose time-series training data
    train_data_t = {}
    
    # By default grab all stories
    stories = check_keys(train_data, keys=stories)
    for story in stories:
        train_data_t[story] = {}

        # By default just grab all subjects
        subject_list = check_keys(train_data[story], keys=subjects,
                                  subkey=story)
        for subject in subject_list:
            train_data_t[story][subject] = {}

            # For simplicity we just assume same hemis across stories/subjects
            hemis = check_keys(train_data[story][subject],
                               keys=hemisphere)
            for hemi in hemis:
                train_data_t[story][subject][hemi] = train_data[story][subject][hemi].T

    # Fit SRM on connectivities and get transformation matrices
    transforms = pca_fit(train_data_t, stories=stories,
                         hemisphere=hemisphere, **kwargs)

    # Apply transformations to training data
    train_transformed = pca_transform(train_data, transforms,
                                      half=train_half,
                                      stories=stories,
                                      subjects=subjects,
                                      hemisphere=hemisphere,
                                      save_prefix=save_prefix + '-train')
    
    print("Finished applying tPCA transformations to training data")

    # Apply transformations to test data
    test_transformed = pca_transform(test_data, transforms,
                                     half=test_half,
                                     stories=stories,
                                     subjects=subjects,
                                     hemisphere=hemisphere,
                                     save_prefix=save_prefix + '-test')
    
    print("Finished applying tPCA transformations to test data")

    return train_transformed, test_transformed


# Name guard for when we want to actually compute cSRM
if __name__ == '__main__':

    # Load dictionary of input filenames and parameters
    with open('data/metadata.json') as f:
        metadata = json.load(f)

    # Subjects and stories
    stories = ['pieman', 'prettymouth', 'milkyway',
               'slumlordreach', 'notthefall', '21styear',
               'pieman (PNI)', 'bronx (PNI)', 'black', 'forgot']

    # Load the surface parcellation
    atlas = {'lh': read_gifti('data/MMP_fsaverage6.lh.gii')[0],
             'rh': read_gifti('data/MMP_fsaverage6.rh.gii')[0]}
    parcel_labels = {'lh': np.unique(atlas['lh'])[1:],
                     'rh': np.unique(atlas['rh'])[1:]}

    # Select story halves for training and test
    train_half, test_half = 1, 2
    
    # Load in first-half training surface data
    target_data = load_split_data(metadata, stories=stories,
                                  subjects=None,
                                  half=train_half)
    
    # Select the type of connectivity targets
    target_types = ['parcel-mean', 'parcel-srm', 'vertex-isc']
    target_type = target_types[0]

    # Compute targets as parcel mean time-series
    if target_type == 'parcel-mean':
        targets = parcel_means(target_data, atlas, parcel_labels=parcel_labels,
                               stories=stories, subjects=None)

    # Compute targets using parcelwise cSRM
    elif target_type == 'parcel-srm':
        targets = parcel_srm(target_data, atlas, k=3,
                             parcel_labels=parcel_labels,
                             stories=stories, subjects=subjects)
        
    # Compute targets based on vertex-wise ISCs
    elif target_type == 'vertex-isc':
        threshold = .2
        targets = vertex_isc(target_data, threshold=threshold, stories=stories,
                             subjects=subjects, half=half, save_iscs=True)
        target_type = target_type + '_thresh-{threshold}'
    
    # Save targets for re-use (may also be costly to re-compute)
    #np.save(f'data/targets_half-{train_half}_{target_type}.npy', targets)

    # Load in ROI masks for both hemispheres
    roi = 'EAC'
    mask_lh = np.load(f'data/{roi}_mask_lh.npy',
                      allow_pickle=True).astype(bool)
    mask_rh = np.load(f'data/{roi}_mask_rh.npy',
                      allow_pickle=True).astype(bool)
    mask = {'lh': mask_lh, 'rh': mask_rh}

    # Re-load in first-half training surface data with ROI mask
    train_data = load_split_data(metadata, stories=stories,
                                 subjects=None,
                                 mask=mask, half=1)

    # Re-load in first-half test surface data with ROI mask
    test_data = load_split_data(metadata, stories=stories,
                                subjects=None,
                                mask=mask, half=2)

    # Apply connectivity PCA stacked across all stories
    n_iter = 10
    #for k in [10, 50, 100, 300]:
    for k in [200]:
        train_transformed, test_transformed = connectivity_pca(
                                                train_data, test_data, targets,
                                                target_fc=target_isfc,
                                                train_half=1, test_half=2,
                                                stories=stories, subjects=None,
                                                save_prefix=f'{roi}_{target_type}_k-{k}_cPCA',
                                                k=k)
        print(f"Finished cPCA (k = {k}) in {roi} for all stories")
        
    # Apply connectivity PCA per story
    n_iter = 10
    #for roi in ['EAC', 'TPOJ', 'PMC']:
    for roi in ['AAC']:

        # Load in ROI masks for both hemispheres
        mask_lh = np.load(f'data/{roi}_mask_lh.npy',
                          allow_pickle=True).astype(bool)
        mask_rh = np.load(f'data/{roi}_mask_rh.npy',
                          allow_pickle=True).astype(bool)
        mask = {'lh': mask_lh, 'rh': mask_rh}

        # Re-load in first-half training surface data with ROI mask
        train_data = load_split_data(metadata, stories=stories,
                                     subjects=None,
                                     mask=mask, half=1)

        # Re-load in first-half test surface data with ROI mask
        test_data = load_split_data(metadata, stories=stories,
                                    subjects=None,
                                    mask=mask, half=2)

        #for k in [10, 50, 100, 300]:
        for k in [100]:
            for story in ['pieman', 'prettymouth', 'milkyway',
                          'slumlordreach', 'notthefall', '21styear',
                          'pieman (PNI)', 'bronx (PNI)', 'black', 'forgot']:
                train_transformed, test_transformed = connectivity_pca(
                                                        train_data, test_data, targets,
                                                        target_fc=target_isfc,
                                                        train_half=1, test_half=2,
                                                        stories=[story], subjects=None,
                                                        save_prefix=f'{roi}_{target_type}_k-{k}_cPCA-{story}',
                                                        k=k)
                print(f"Finished cSRM (k = {k}) in {roi} for {story}")

    # Apply temporal SRM per story
    n_iter = 10
    #for k in [10, 50, 100, 300]:
    for k in [100]:
        for story in ['pieman', 'prettymouth', 'milkyway',
                      'slumlordreach', 'notthefall', '21styear',
                      'pieman (PNI)', 'bronx (PNI)', 'black', 'forgot']:
            train_transformed, test_transformed = temporal_pca(
                                                    train_data, test_data,
                                                    train_half=1, test_half=2,
                                                    stories=[story], subjects=None,
                                                    save_prefix=f'{roi}_k-{k}_tPCA-{story}',
                                                    k=k)
            print(f"Finished tSRM (k = {k}) in {roi} for {story}")