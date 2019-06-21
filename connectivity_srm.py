import json
from time import time
import numpy as np
from scipy.stats import zscore
from brainiak.funcalign.srm import SRM
from gifti_io import read_gifti
from split_stories import check_keys, load_split_data
from target_isfc import parcel_srm, target_isfc


# Fit SRM on connectivity matrices
def srm_fit(target_fcs, stories=None, subjects=None,
            hemisphere=None, k=360, n_iter=10):

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
                if subject not in subject_fcs:
                    subject_fcs[subject] = {}
                if hemi not in subject_fcs[subject]:
                    subject_fcs[subject][hemi] = []
                subject_fcs[subject][hemi].append(
                        target_fcs[story][subject][hemi])

    # Stack FCs across stories in connectivity space
    for subject in subject_list:
        for hemi in hemis:

            if len(subject_fcs[subject][hemi]) > 1:
                subject_fcs[subject][hemi] = np.mean(subject_fcs[subject][hemi],
                                                     axis=0)
            else:
                subject_fcs[subject][hemi] = subject_fcs[subject][hemi][0]

    # Convert FCs to list for SRM
    transforms = {}
    for hemi in hemis:

        # Declare SRM for this hemi
        srm = SRM(n_iter=n_iter, features=k)

        subject_ids, subject_stack = [], []
        for subject in subject_list:
            subject_ids.append(subject)
            subject_stack.append(subject_fcs[subject][hemi])
            if subject not in transforms:
                transforms[subject] = {}

        # Train SRM and apply
        start = time()
        srm.fit(subject_stack)
        print(f"Finished fitting SRM after {time() - start:.1f} seconds")

        for subject_id, transform in zip(subject_ids, srm.w_):
            transforms[subject_id][hemi] = transform

    return transforms


# Apply learned SRM projections to data
def srm_transform(data, transforms, half=1, stories=None,
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

                transform = transforms[subject][hemi]
                transformed = data[story][subject][hemi].dot(transform)

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
def connectivity_srm(train_data, test_data, targets, target_fc=target_isfc,
                     train_half=1, test_half=2, stories=None, subjects=None,
                     hemisphere=None, save_prefix=None, **kwargs):

    # Compute ISFCs with targets
    target_fcs = target_fc(train_data, targets, stories=stories,
                           subjects=subjects, hemisphere=hemisphere)

    # Fit SRM on connectivities and get transformation matrices
    transforms = srm_fit(target_fcs, hemisphere=hemisphere, **kwargs)

    # Apply transformations to training data
    train_transformed = srm_transform(train_data, transforms,
                                      half=train_half,
                                      stories=stories,
                                      subjects=subjects,
                                      hemisphere=hemisphere,
                                      save_prefix=save_prefix + '-train')
    
    print("Finished applying cSRM transformations to training data")

    # Apply transformations to test data
    test_transformed = srm_transform(test_data, transforms,
                                     half=test_half,
                                     stories=stories,
                                     subjects=subjects,
                                     hemisphere=hemisphere,
                                     save_prefix=save_prefix + '-test')
    
    print("Finished applying cSRM transformations to test data")

    return train_transformed, test_transformed


# Name guard for when we want to actually compute cSRM
if __name__ == '__main__':

    # Load dictionary of input filenames and parameters
    with open('metadata.json') as f:
        metadata = json.load(f)

    # Subjects and stories
    stories = ['black', 'forgot']
    exclude = [6, 7, 9, 11, 12, 13, 26, 27, 28, 33]
    subject_list = [f'sub-{i:02}' for i in range(1, 49)
                    if i not in exclude]
    subjects = {story: subject_list for story in stories}

    # Load the surface parcellation
    atlas = {'lh': read_gifti('data/MMP_fsaverage6.lh.gii')[0],
             'rh': read_gifti('data/MMP_fsaverage6.rh.gii')[0]}
    parcel_labels = {'lh': np.unique(atlas['lh'])[1:],
                     'rh': np.unique(atlas['rh'])[1:]}

    # Select story halves for training and test
    training_half, test_half = 1, 2
    
    # Load in first-half training surface data
    target_data = load_split_data(metadata, stories=stories,
                                  subjects=subjects,
                                  half=training_half)
    
    # Select the type of connectivity targets
    target_types = ['parcel-mean', 'parcel-srm', 'vertex-isc']
    target_type = target_types[2]

    # Compute targets as parcel mean time-series
    if target_type == 'parcel-mean':
        targets = parcel_means(target_data, atlas, parcel_labels=parcel_labels,
                               stories=stories, subjects=subjects)

    # Compute targets using parcelwise cSRM
    elif target_type == 'parcel-srm':
        targets = parcel_srm(target_data, atlas, k=3,
                             parcel_labels=parcel_labels,
                             stories=stories, subjects=subjects)
        
    # Compute targets based on vertex-wise ISCs
    elif target_type == 'vertex-isc'
        threshold = .2
        targets = vertex_isc(target_data, threshold=threshold, stories=stories,
                             subjects=subjects, half=half, save_iscs=True)
        target_type = target_type + '_thresh-{threshold}'
    
    # Save targets for re-use (may also be costly to re-compute)
    np.save(f'data/targets_half-{half}_{target_type}.npy', targets)

    # Load in ROI masks for both hemispheres
    roi = 'PMC'
    mask_lh = np.load(f'data/{roi}_mask_lh.npy').astype(bool)
    mask_rh = np.load(f'data/{roi}_mask_rh.npy').astype(bool)
    mask = {'lh': mask_lh, 'rh': mask_rh}

    # Re-load in first-half training surface data with ROI mask
    train_data = load_split_data(metadata, stories=stories,
                                 subjects=subjects,
                                 mask=mask, half=1)

    # Re-load in first-half test surface data with ROI mask
    test_data = load_split_data(metadata, stories=stories,
                                subjects=subjects,
                                mask=mask, half=2)

    # Apply connectivity SRM
    n_iter = 100
    for k in [10, 50, 100, 300]:
        train_transformed, test_transformed = connectivity_srm(
                                                train_data, test_data, targets,
                                                target_fc=target_isfc,
                                                train_half=1, test_half=2,
                                                stories=stories, subjects=subjects,
                                                save_prefix=f'{roi}_vertex-isc_k-{k}_cSRM',
                                                k=k, n_iter=n_iter)
