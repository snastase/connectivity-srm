import numpy as np
import matplotlib.pyplot as plt


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
        
    return {'accuracies': np.array(accuracies), 'chance': 1/n_segments}
