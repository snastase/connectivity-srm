import json
import numpy as np
from scipy.stats import zscore
from brainiak.isc import isfc
from split_stories import check_keys


# Compute means for all parcels
def parcel_means(data, atlas, parcel_labels=None,
                 stories=None, subjects=None):
    
    # By default grab all stories
    stories = check_keys(data, keys=stories)

    parcels = {}
    for story in stories:

        parcels[story] = {}

        # By default just grab all subjects
        subject_list = check_keys(data[story], keys=subjects,
                                  subkey=story)

        # Stack left and right hemispheres for each subject
        for subject in subject_list:

            # Horizontally stack surface data
            data_stack = np.hstack((data[story][subject]['lh'],
                                    data[story][subject]['rh']))

            # Compute mean time series per parcel
            parcel_tss = []
            for parcel_label in parcel_labels:

                # Get mean for this parcel
                parcel_ts = np.mean(data_stack[:, atlas == parcel_label],
                                    axis=1)

                # Expand dimension for easier stacking
                parcel_tss.append(np.expand_dims(parcel_ts, 1))

            # Stack parcel means
            parcel_tss = np.hstack(parcel_tss)
            assert parcel_tss.shape[1] == len(parcel_labels)

            parcels[story][subject] = parcel_tss

        print(f"Finished computing parcel means for '{story}'")
    
    return parcels


# Compute ISFC between ROI voxels and parcel means
def target_isfc(data, targets, stories=None, subjects=None,
                hemisphere=None, zscore_isfcs=True):
    
    # By default grab all stories
    stories = check_keys(data, keys=stories)
    
    target_isfcs = {}
    for story in stories:

        target_isfcs[story] = {}
        
        # By default just grab all subjects
        subject_list = check_keys(data[story], keys=subjects,
                                  subkey=story)
        
        for subject in subject_list:
            target_isfcs[story][subject] = {}
        
        # Stack targets in third dimension
        target_stack = np.dstack(([targets[story][subject]
                           for subject in subject_list]))
        
        # By default grab both hemispheres
        hemis = check_keys(data[story][subject_list[0]],
                           keys=hemisphere)
        
        # Get for specified hemisphere(s)
        for hemi in hemis:
            
            # Grab ROI data and targets
            data_stack = np.dstack(([data[story][subject][hemi]
                                     for subject in subject_list]))

            # Compute ISFCs between ROI and targets
            isfcs = isfc(data_stack, targets=target_stack)

            # Optionally z-score across targets
            if zscore_isfcs:
                isfcs = zscore(np.nan_to_num(isfcs), axis=2)

            for s, subject in enumerate(subject_list):
                target_isfcs[story][subject][hemi] = isfcs[s]
            
        print(f"Finished computing target ISFCs for story '{story}'")
            
    return target_isfcs
