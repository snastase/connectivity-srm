import json
import numpy as np
from scipy.stats import zscore
from brainiak.isc import isfc
from brainiak.funcalign.srm import SRM
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

            # Loop through both hemispheres
            hemi_stack = []
            for hemi in ['lh', 'rh']:
                
                # Grab mean parcel time series for this hemisphere
                parcel_tss = []
                for parcel_label in parcel_labels[hemi]:
                    
                    # Get mean for this parcel
                    parcel_ts = np.mean(data[story][subject][hemi][:,
                                            atlas[hemi] == parcel_label],
                                        axis=1)
                    
                    # Expand dimension for easier stacking
                    parcel_tss.append(np.expand_dims(parcel_ts, 1))
                    
                # Stack parcel means
                parcel_tss = np.hstack(parcel_tss)
                hemi_stack.append(parcel_tss)
                
            # Stack hemispheres
            hemi_stack = np.hstack(hemi_stack)
            assert hemi_stack.shape[1] == (len(parcel_labels['lh']) +
                                           len(parcel_labels['rh']))
            
            parcels[story][subject] = hemi_stack

        print(f"Finished computing parcel means for '{story}'")
    
    return parcels


# Compute means for all parcels
def parcel_srm(data, atlas, k=3, parcel_labels=None,
               stories=None, subjects=None):
    
    # By default grab all stories
    stories = check_keys(data, keys=stories)
    
    # Firsts compute mean time-series for all target parcels
    targets = parcel_means(data, atlas,
                           parcel_labels=parcel_labels,
                           stories=stories, subjects=subjects)
    
    # Compute ISFCs with targets for all vertices
    target_fcs = target_isfc(data, targets, stories=stories,
                             subjects=subjects)

    parcels = {}
    for story in stories:
        
        parcels[story] = {}
             
        # By default just grab all subjects
        subject_list = check_keys(data[story], keys=subjects,
                                  subkey=story)
        
        # Loop through both hemispheres
        hemi_stack = []
        for hemi in ['lh', 'rh']:
            
            # Loop through parcels
            parcel_tss = []
            for parcel_label in parcel_labels[hemi]:
                
                # Resort parcel FCs into list of subject parcels
                fc_stack = []
                ts_stack = []
                for subject in subject_list:
                
                    # Grab the connectivities for this parcel
                    parcel_fcs = target_fcs[story][subject][hemi][
                                            atlas[hemi] == parcel_label, :]
                    fc_stack.append(parcel_fcs)
                    
                    ts_stack.append(data[story][subject][hemi][:,
                                            atlas[hemi] == parcel_label])

                # Set up fresh SRM
                srm = SRM(features=k)
                
                # Train SRM on parcel connectivities
                srm.fit(np.nan_to_num(fc_stack))
                
                # Apply transformations to time series
                transformed_stack = [ts.dot(w) for ts, w
                                     in zip(ts_stack, srm.w_)]
                transformed_stack = np.dstack(transformed_stack)
                parcel_tss.append(transformed_stack)
                print(f"Finished SRM for {hemi} parcel "
                      f"{parcel_label} in '{story}'")
                
            # Stack parcel means
            parcel_tss = np.hstack(parcel_tss)
            hemi_stack.append(parcel_tss)
                
        # Stack hemispheres
        hemi_stack = np.hstack(hemi_stack)
        assert hemi_stack.shape[1] == (len(parcel_labels['lh']) +
                                       len(parcel_labels['rh'])) * k
        assert hemi_stack.shape[2] == len(subject_list)
                
        # Unstack subjects
        hemi_stack = np.dsplit(hemi_stack, hemi_stack.shape[2])
        for subject, ts in zip(subject_list, hemi_stack):
            parcels[story][subject] = np.squeeze(ts)
                
        print(f"Finished applying cSRM to parcels for '{story}'")
        
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
