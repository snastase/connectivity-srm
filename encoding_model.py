import json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from scipy.stats import rankdata, zscore
from scipy.spatial.distance import cdist
from utils import array_correlation, read_gifti, write_gifti

# Load dictionary of input filenames
with open('metadata.json') as f:
    metadata = json.load(f)

# Keep track of some basic variables
n_vertices = 40962
model_ndim = 300
delays = [2, 3, 4, 5]

rois = ['EAC', 'AAC', 'TPOJ', 'PCC']
roi = 'PCC'
hemi = 'lh'

stories = ['black', 'forgot']
exclude = [7, 11, 12, 13, 26, 27]
subject_list = [f'sub-{i+1:02}' for i in range(48)
            if i not in exclude]
subject_list = ['sub-01', 'sub-02', 'sub-10', 'sub-15']
subjects = {story: subject_list for story in stories}

# Load mask in fsaverage6 space
mask = np.load(f'data/{roi}_mask_{hemi}.npy').astype(bool)


# Function for delaying embedding by several TRs
def delay_embedding(embedding, delays=[2, 3, 4, 5],
                    zscore_features=False):
    
    # Horizontally stack semantic vectors at varying delays
    delayed = []
    for delay in delays:
        delayed.append(np.vstack((np.full((delay,
                                           embedding.shape[1]),
                                          np.nan),
                                  embedding[:-delay])))
    delayed = np.hstack(delayed)
    
    # Zero out any NaNs in delayed embedding
    delayed[np.isnan(delayed)] = 0
    
    # Optionally z-score semantic features
    if zscore_features:
        delayed = zscore(delayed, axis=0)
    
    return delayed


# Load surface data and trim
def load_inputs(metadata, subjects=None,
                stories=None, hemi=None, mask=None):
    
    if not stories:
        stories = metadata.keys()
    
    inputs = {}
    for story in stories:

        inputs[story] = {}
        
        # Load, trim, and delay embedding
        embedding = np.load(metadata[story]['model'])
        model_trims = metadata[story]['model_trims']
        embedding = embedding[model_trims[0]:(-model_trims[1] or None), :]
        model = delay_embedding(embedding)
        inputs[story]['model'] = model
        print(f'Loaded model for story "{story}"')
        
        # Get data for all or some subjects
        inputs[story]['data'] = {}
        
        if not subjects:
            subject_list = metadata[story]['data'].keys()
        else:
            subject_list = subjects[story]

        n_subjects = 0
        for subject in subject_list:
            
            data_fn = metadata[story]['data'][subject][hemi]
            surf_data = read_gifti(data_fn)

            # Trim data
            data_trims = metadata[story]['data_trims']
            surf_data = surf_data[data_trims[0]:(
                -data_trims[1] or None), :]

            # Check that model and data have matching length
            if not model.shape[0] == surf_data.shape[0]:
                raise ValueError(f"Model shape {model.shape} does not "
                                 f"match data shape {surf_data.shape} "
                                 f'for story "{story}"')

            # Optionally apply mask
            if isinstance(mask, np.ndarray):
                surf_data = surf_data[:, mask]

            inputs[story]['data'][subject] = surf_data
            
            n_subjects +=1
    
        print(f'Loaded data for {n_subjects} subjects for story "{story}"')

    return inputs

inputs = load_inputs(metadata, subjects=subjects,
                     stories=stories, hemi=hemi, mask=mask)


# Split into train and test
def story_split(inputs, zscore_data=True, zscore_model=False):
    
    model_splits, data_splits = {}, {}
    for story in inputs:
        
        # Get model for story and split / process
        model = inputs[story]['model']
        
        midpoint = model.shape[0] // 2

        first_model = model[:midpoint]
        second_model = model[midpoint:]

        if zscore_model:
            first_model = zscore(first_model, axis=0)
            second_model = zscore(second_model, axis=0)
            
        model_splits[story] = [(first_model, second_model),
                               (second_model, first_model)]
        
        # Get fMRI data per story and subject
        data_splits[story] = {}
        for subject in inputs[story]['data']:
            
            data = inputs[story]['data'][subject]
            
            assert data.shape[0] == model.shape[0]

            first_data = data[:midpoint]
            second_data = data[midpoint:]

            if zscore_data:
                first_data = zscore(first_data, axis=0)
                second_data = zscore(second_data, axis=0)

            data_splits[story][subject] = [(first_data, second_data),
                                           (second_data, first_data)]

    return model_splits, data_splits


model_splits, data_splits = story_split(inputs)


# Function for selecting and aggregating subjects
def aggregate_subjects(data, subject_list, aggregation='average',
                       cv_fold=0, train_test=0):
    
    # Allow for easy single subject input
    if type(subject_list) == str:
        subject = subject_list
        data = data[subject][cv_fold][train_test]
        
    else:
        if len(subject_list) == 1:
            data = data[subject_list[0]][cv_fold][train_test]
        else:
            n_subjects = len(subject_list)
            assert n_subjects > 1

            # Check subjects are in the data
            for subject in subject_list:
                assert subject in data.keys()

            # Compile test subjects
            data_list = []
            for subject in subject_list:
                data_list.append(data[subject][cv_fold][train_test])

            # Average time data across subjects
            if aggregation == 'average' and n_subjects > 1:
                data = np.mean(data_list, axis=0)

            elif aggregation == 'concatenate' and n_subjects > 1:
                data = np.vstack(data_list)
    
    return data

train_story, test_story = 'forgot', 'forgot'
train_subjects = ['sub-01', 'sub-02', 'sub-10']
test_subjects = ['sub-15']
cv_fold = 0
aggregation = 'average'

if aggregation == 'average':
    train_model = model_splits[train_story][cv_fold][0]
    test_model = model_splits[test_story][cv_fold][1]
elif aggregation == 'concatenation':
    train_model = np.tile(model_splits[train_story][cv_fold][0],
                          (len(train_subjects), 1))
    test_model = np.tile(model_splits[test_story][cv_fold][1],
                         (len(test_subjects), 1))

train_data = aggregate_subjects(data_splits[train_story],
                                train_subjects,
                                aggregation=aggregation,
                                cv_fold=cv_fold,
                                train_test=0)
                                
test_data = aggregate_subjects(data_splits[test_story],
                               test_subjects,
                               cv_fold=cv_fold,
                               train_test=1)

# Make custom correlation scorer
correlation_scorer = make_scorer(array_correlation)


# Declare ridge regression model
alpha = 10.
ridge = Ridge(alpha=alpha, fit_intercept=True, normalize=False,
              copy_X=True, tol=0.001, solver='auto')

# Fit training data
ridge.fit(train_model, train_data)

# Get coefficients of trained model
coefficients = ridge.coef_

# Use trained model to predict response for test data
predicted_data = ridge.predict(test_model)

# Compute correlation between predicted and test response
performance = array_correlation(predicted_data,
                                test_data)

# Collapse coefficients across delays for decoding
collapse_coef = np.mean(np.split(ridge.coef_, len(delays),
                                axis=1), axis=0)
collapse_test_model = np.mean(np.split(test_model, len(delays),
                                axis=1), axis=0)

# Decoding via dot product between test samples and coefficients
predicted_model = test_data.dot(collapse_coef)


# Function to compute correlation-based rank accuracy
def rank_accuracy(predicted_model, test_model, mean=True):
    n_predictions = predicted_model.shape[0]

    # Get correlations between pairs
    correlations = 1 - cdist(predicted_model, test_model,
                             'correlation')

    # Get rank of matching prediction for each
    ranks = []
    for index in np.arange(n_predictions):
        ranks.append(rankdata(correlations[index])[index])

    # Normalize ranks by number of choices
    ranks = np.array(ranks) / n_predictions

    if mean:
        ranks = np.mean(ranks)

    return ranks

accuracy = rank_accuracy(predicted_model, collapse_test_model)

print(np.mean(performance), np.amax(performance))
print(accuracy)


# Fill results back in cortical mask and write GIfTI
results = np.zeros(n_vertices)
results[mask] = performance

template_fn = metadata['black']['data']['sub-02']

write_gifti(results, 'black_performance_test-xs_lh.gii', template_fn)


# Function to run grid search over alphas across voxels
def grid_search(train_model, train_data, alphas, scorer, n_splits=10):

    # Get number of voxels
    n_voxels = train_model.shape[1]

    # Set up ridge regression
    ridge = Ridge(fit_intercept=True, normalize=False,
                  copy_X=True, tol=0.001)

    # Set up grid search
    grid = GridSearchCV(ridge, {'alpha': alphas}, iid=False,
                        scoring=scorer,
                        cv=KFold(n_splits=n_splits), refit=False,
                        return_train_score=False)

    # Loop through voxels
    best_alphas, best_scores, all_scores = [], [], []
    for voxel in np.arange(n_voxels):

        # Perform grid search over alphas for voxel
        grid.fit(train_model, train_data[:, voxel]);

        best_alphas.append(grid.best_params_['alpha'])
        best_scores.append(grid.best_score_)
        
        # Get all scores across folds
        split_scores = []
        for split in np.arange(n_splits):
            split_score = grid.cv_results_[f'split{split}_test_score']
            split_scores.append(split_score)
        all_scores.append(np.mean(split_scores, axis=0))

    best_alphas = np.array(best_alphas)
    best_scores = np.array(best_scores)
    all_scores = np.column_stack(all_scores)
    
    assert (best_alphas.shape[0] == best_scores.shape[0]
            == all_scores.shape[1] == n_voxels)

    return best_alphas, best_scores, all_scores


alphas = np.logspace(-1, 3, num=10)

best_alphas, best_scores, all_scores = grid_search(train_model,
                                                   train_data[:, :100],
                                                   alphas,
                                                   correlation_scorer)
