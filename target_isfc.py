import json
import numpy as np
from os.path import join
from split_stories import load_split_data
from gifti_io import read_gifti, write_gifti

# Load dictionary of input filenames
with open('metadata.json') as f:
    metadata = json.load(f)

stories = ['black', 'forgot']
subject_list = ['sub-01', 'sub-02', 'sub-10', 'sub-15']
subjects = {story: subject_list for story in stories}
half = 1
    
# Load nonzero vertices mask (excludes medial wall)
mask_lh = np.load('data/cortex_mask_lh.npy').astype(bool)
mask_rh = np.load('data/cortex_mask_rh.npy').astype(bool)
mask = {'lh': mask_lh, 'rh': mask_rh}

# Load the surface parcellation
n_parcels = 360
atlas = np.hstack((read_gifti('data/MMP_fsaverage6.lh.gii')[0, mask_lh],
                   read_gifti('data/MMP_fsaverage6.rh.gii')[0, mask_rh]
                   + 1000))
parcel_labels = np.unique(atlas)
parcel_labels = parcel_labels[~np.logical_or(parcel_labels == 0,
                                             parcel_labels == 1000)]
assert len(parcel_labels) == n_parcels


# Load in first-half training surface data
train_data = load_split_data(metadata, stories=stories,
                             subjects=subjects,
                             mask=mask, half=half)

# Compute means for all parcels
parcel_means = {}
for story in stories:
    
    parcel_means[story] = {}
    
    # By default just grab all subjects in metadata
    if not subjects:
        subject_list = metadata[story]['data'].keys()
    else:
        subject_list = subjects[story]
        
    # Stack left and right hemispheres for each subject
    for subject in subject_list:
                
        # Horizontally stack surface data
        train_stack = np.hstack((train_data[story][subject]['lh'],
                                 train_data[story][subject]['rh']))
    
        # Compute mean time series per parcel
        parcels = []
        for parcel_label in parcel_labels:

            # Get mean for this parcel
            parcel = np.mean(train_stack[:, atlas == parcel_label],
                             axis=1)

            # Expand dimension for easier stacking
            parcels.append(np.expand_dims(parcel, 1))

        # Stack parcel means
        parcels = np.hstack(parcels)
        assert parcels.shape[1] == n_parcels
        
        parcel_means[story][subject] = parcels

    print(f"Finished computing parcel means for {story}")
