import json
from os.path import exists
import numpy as np
from brainiak.isc import isc
from split_stories import check_keys, load_split_data


# Function to split time series data into segements
def time_segmentation(data, segment_length, average=False):

    # Get integer number of segments
    n_TRs = data.shape[0]
    n_segments = n_TRs // segment_length
    modulo = n_TRs % n_segments

    # If even division, split and optionally average
    if modulo == 0:
        if average:
            segments = np.vstack([np.mean(segment, axis=0) for segment in
                                  np.array_split(data, n_segments,
                                       axis=0)])            
        else:
            segments = np.vstack([np.ravel(segment) for segment in
                                  np.array_split(data, n_segments,
                                       axis=0)])

    # If modulo, trim modulo off end before splitting
    else:
        if average:
            segments = np.vstack([np.mean(segment, axis=0) for segment in
                      np.array_split(data[:-modulo, ...], n_segments,
                           axis=0)])
        else:
            segments = np.vstack([np.ravel(segment) for segment in
                                  np.array_split(data[:-modulo, ...], n_segments,
                                       axis=0)])
    return segments


# Compute intersubject time-segment pattern correlations
def time_segment_correlation(data, segment_length, average=False):

    n_TRs, n_voxels, n_subjects = data.shape
    n_segments = n_TRs // segment_length

    # For each subject, get correlation with average of others
    correlations = []
    for i_subject in np.arange(n_subjects):

        # Time series for one subject
        subject = data[..., i_subject]

        # Compute average time series of others
        others = np.dstack([data[..., i_other]
                            for i_other in np.arange(n_subjects)
                            if i_other != i_subject])
        assert others.shape[2] == n_subjects - 1
        others = np.nanmean(others, axis=2)

        # Perform time series segmentation
        subject_segments = time_segmentation(subject,
                                             segment_length,
                                             average=average)
        others_segments = time_segmentation(others,
                                            segment_length,
                                            average=average)

        assert (n_segments == subject_segments.shape[0] ==
                others_segments.shape[0])

        # Compute pairwise cross-subject correlations
        correlations.append(np.corrcoef(subject_segments,
                                        others_segments)[:n_segments,
                                                         n_segments:])

    return np.dstack(correlations)


# Classify time segments based on correlation
def correlation_classification(correlations):
    accuracies = []
    n_segments = correlations.shape[0]

    # For each time point compare diagonal correlation with off-diaogonal
    for correlation in np.moveaxis(correlations, 2, 0):
        n_hits = 0
        for i, row in enumerate(correlation):
            diagonal = row[i]
            if diagonal > np.amax([row[off] for off in
                                   np.arange(len(row)) if
                                   off != i]):
                n_hits += 1
        accuracies.append(n_hits / n_segments)

    accuracies = np.array(accuracies)
    chance = 1/n_segments

    return accuracies, chance


# Convenience function stack subjects into array
def stack_subjects(data, subjects=None, hemisphere='lh'):

    # By default just grab all subjects
    subject_list = check_keys(data, keys=subjects)

    subject_stack = np.dstack([data[subject][hemisphere] for
                               subject in subject_list])

    assert subject_stack.shape[2] == len(subject_list)

    return subject_stack


# Name guard for when we actually want to split all daata
if __name__ == '__main__':

    # Load dictionary of input filenames and parameters
    with open('metadata.json') as f:
        metadata = json.load(f)

    # Create story and subject lists
    stories = ['black', 'forgot']
    exclude = [6, 7, 9, 11, 12, 13, 26, 27, 28, 33]
    subject_list = [f'sub-{i:02}' for i in range(1, 49)
                    if i not in exclude]
    subjects = {story: subject_list for story in stories}

    # Parameters for time-segment classification
    segment_length = 10
    average = False

    # Set ROIs, spaces, and hemispheres
    rois = ['EAC', 'AAC', 'TPOJ', 'PMC']    
    prefixes = [('no SRM', 'noSRM'),
                ('cSRM (k = 300)', 'k-300_cSRM-test'),
                ('cSRM (k = 100)', 'k-100_cSRM-test'),
                ('cSRM (k = 50)', 'k-50_cSRM-test'),
                ('cSRM (k = 10)', 'k-10_cSRM-test')]
    hemis = ['lh', 'rh']

    # Load in results file if it already exists
    results_fn = f'data/ts_classification_st{segment_length}_results.npy'
    if exists(results_fn):
        results = np.load(results_fn).item()
    else:
        results = {}

    # Loop through keys without replacing existing ones
    for story in stories:
        if story not in results:
            results[story] = {}

        for roi in rois:
            if roi not in results[story]:
                results[story][roi] = {}

            for prefix in prefixes:
                if prefix[0] not in results[story][roi]:
                    results[story][roi][prefix[0]] = {}

                for hemi in hemis:
                    if hemi not in results[story][roi][prefix[0]]:
                        results[story][roi][prefix[0]][hemi] = {}

                        # Load in either raw data with mask or SRM ROI data
                        data = load_split_data(metadata, stories=story,
                                               subjects=subjects,
                                               hemisphere=hemi,
                                               half=2, prefix=f'{roi}_' + prefix[1])

                        # Depth-stack subjects
                        subject_stack = stack_subjects(data[story],
                                                       subjects=subjects[story],
                                                       hemisphere=hemi)

                        # Compute paired time-segment correlations
                        correlations = time_segment_correlation(subject_stack,
                                                                segment_length,
                                                                average=average)

                        # Classify time segments based on correlations
                        accuracies, chance = correlation_classification(correlations)

                        results[story][roi][prefix[0]][hemi] = accuracies
                        print("Finished computing time-segment "
                              f"classification for {story}, "
                              f"{roi}, {prefix[0]}, {hemi}")

    np.save(results_fn, results)


    # Compute temporal and spatial intersubject coorrelations
    isc_type = 'temporal'

    results_fn = f'data/{isc_type}_isc_no-mean_results.npy'
    if exists(results_fn):
        results = np.load(results_fn).item()
    else:
        results = {}

    # Loop through keys without replacing existing ones
    for story in stories:
        if story not in results:
            results[story] = {}

        for roi in rois:
            if roi not in results[story]:
                results[story][roi] = {}

            for prefix in prefixes:
                if prefix[0] not in results[story][roi]:
                    results[story][roi][prefix[0]] = {}

                for hemi in hemis:
                    if hemi not in results[story][roi][prefix[0]]:
                        results[story][roi][prefix[0]][hemi] = {}

                    # Load in either raw data with mask or SRM ROI data
                    data = load_split_data(metadata, stories=story,
                                           subjects=subjects,
                                           hemisphere=hemi,
                                           half=2, prefix=f'{roi}_' + prefix[1])

                    # Depth-stack subjects
                    subject_stack = stack_subjects(data[story],
                                                   subjects=subjects[story],
                                                   hemisphere=hemi)

                    # Compute paired time-segment correlations
                    if isc_type == 'temporal':
                        iscs = isc(subject_stack)
                    elif isc_type == 'spatial':
                        iscs = np.mean(isc(np.moveaxis(subject_stack, 1, 0)), axis=0)

                    results[story][roi][prefix[0]][hemi] = iscs
                    print(f"Finished computing ISCs for {story}, "
                          f"{roi}, {prefix[0]}, {hemi}")

    np.save(results_fn, results)
