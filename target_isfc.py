import json
import numpy as np
from scipy.stats import zscore
from brainiak.isc import isfc
from split_stories import load_split_data
from gifti_io import read_gifti

# Load dictionary of input filenames
with open('metadata.json') as f:
    metadata = json.load(f)

stories = ['black', 'forgot']
subject_list = ['sub-01', 'sub-02', 'sub-10', 'sub-15']
subjects = {story: subject_list for story in stories}
half = 1
hemi = 'lh'
roi = 'AAC'

# Load in ROI masks for both hemispheres
mask_lh = np.load(f'data/{roi}_mask_lh.npy').astype(bool)
mask_rh = np.load(f'data/{roi}_mask_rh.npy').astype(bool)
mask = {'lh': mask_lh, 'rh': mask_rh}

# Load the surface parcellation
n_parcels = 360
atlas = np.hstack((read_gifti('data/MMP_fsaverage6.lh.gii'),
                   read_gifti('data/MMP_fsaverage6.rh.gii')
                   + 1000))[0]
parcel_labels = np.unique(atlas)
parcel_labels = parcel_labels[~np.logical_or(parcel_labels == 0,
                                             parcel_labels == 1000)]
assert len(parcel_labels) == n_parcels


# Compute means for all parcels
def parcel_means(data, atlas, parcel_labels=None,
                 stories=None, subjects=None, hemi='lh'):
    
    # By default grab all stories
    if not stories:
        stories = data.keys()

    parcels = {}
    for story in stories:

        parcels[story] = {}

        # By default just grab all subjects
        if not subjects:
            subject_list = data[story]['data'].keys()
        else:
            subject_list = subjects[story]

        # Stack left and right hemispheres for each subject
        for subject in subject_list:

            # Horizontally stack surface data
            train_stack = np.hstack((train_data[story][subject]['lh'],
                                     train_data[story][subject]['rh']))

            # Compute mean time series per parcel
            parcel_tss = []
            for parcel_label in parcel_labels:

                # Get mean for this parcel
                parcel_ts = np.mean(train_stack[:, atlas == parcel_label],
                                    axis=1)

                # Expand dimension for easier stacking
                parcel_tss.append(np.expand_dims(parcel_ts, 1))

            # Stack parcel means
            parcel_tss = np.hstack(parcel_tss)
            assert parcel_tss.shape[1] == n_parcels

            parcels[story][subject] = parcel_tss

        print(f"Finished computing parcel means for '{story}'")
    
    return parcels


# Compute ISFC between ROI voxels and parcel means
def target_isfc(data, targets, stories=None, subjects=None,
                hemi='lh', zscore_isfcs=True):
    
    # By default grab all stories
    if not stories:
        stories = data.keys()
    
    target_isfcs = {}
    for story in stories:

        target_isfcs[story] = {}
        
        # By default just grab all subjects
        if not subjects:
            subject_list = data[story]['data'].keys()
        else:
            subject_list = subjects[story]
        
        # Grab ROI data and targets
        data_stack = np.dstack(([data[story][subject][hemi]
                                 for subject in subject_list]))
        target_stack = np.dstack(([targets[story][subject]
                                   for subject in subject_list]))

        # Compute ISFCs between ROI and targets
        isfcs = isfc(data_stack, targets=target_stack)
        
        # Optionally z-score across targets
        if zscore_isfcs:
            isfcs = zscore(np.nan_to_num(isfcs), axis=2)
        
        for s, subject in enumerate(subject_list):
            target_isfcs[story][subject] = {}
            target_isfcs[story][subject][hemi] = isfcs[s]
            
        print(f"Finished computing target ISFCs for story '{story}'")
            
    return target_isfcs


# Load in first-half training surface data
train_data = load_split_data(metadata, stories=stories,
                             subjects=subjects,
                             half=half)

# Compute targets
targets = parcel_means(train_data, atlas, parcel_labels=parcel_labels,
                       stories=stories, subjects=subjects, hemi=hemi)

# Re-load in first-half training surface data with ROI mask
train_data = load_split_data(metadata, stories=stories,
                             subjects=subjects,
                             mask=mask, half=half)

# Compute ISFCs with targets
target_isfcs = target_isfc(train_data, targets, stories=stories,
                           subjects=subjects, hemi=hemi)