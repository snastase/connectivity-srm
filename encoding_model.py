import json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from scipy.stats import rankdata, zscore
from scipy.spatial.distance import cdist
from gifti_io import read_gifti, write_gifti
from split_stories import check_keys, load_split_data, split_models
from brainiak.utils.utils import array_correlation

# Load dictionary of input filenames and parameters
with open('metadata.json') as f:
    metadata = json.load(f)

# Keep track of some basic variables
n_vertices = 40962
model_ndim = 300
delays = [2, 3, 4, 5]

stories = ['black', 'forgot']
exclude = [6, 7, 9, 11, 12, 13, 26, 27, 28, 33]
subject_list = [f'sub-{i+1:02}' for i in range(48)
                if i not in exclude]
#subject_list = ['sub-01', 'sub-02', 'sub-10', 'sub-15']
subjects = {story: subject_list for story in stories}


# Function for selecting and aggregating subjects
def aggregate_subjects(data, model, subject_list,
                       hemi='lh',  aggregation='average'):
    
    # Broadcast model if concatenating subjects
    if aggregation == 'concatenate':
        model = np.tile(model, (len(subject_list), 1))
    
    # Allow for easy single subject input
    if type(subject_list) == str:
        subject = subject_list
        data = data[subject][hemi]
        
    else:
        if len(subject_list) == 1:
            data = data[subject_list[0]][hemi]
        else:
            n_subjects = len(subject_list)
            assert n_subjects > 1

            # Check subjects are in the data
            for subject in subject_list:
                assert subject in data.keys()

            # Compile test subjects
            data_list = []
            for subject in subject_list:
                data_list.append(data[subject][hemi])

            # Average time data across subjects
            if aggregation == 'average' and n_subjects > 1:
                data = np.mean(data_list, axis=0)

            elif aggregation == 'concatenate' and n_subjects > 1:
                data = np.vstack(data_list)
                
    if len(model) != len(data):
        raise ValueError("Model and data have mismatching shape! "
                         f"model: {model.shape}, data: {data.shape}")
    
    return data, model

train_story, test_story = 'forgot', 'forgot'
train_subjects = ['sub-02', 'sub-10']
train_subjects = [s for s in subject_list if s is not 'sub-15']
test_subjects = ['sub-15']
hemisphere = 'lh'
aggregation = 'average'

# Load masks for both hemispheres
rois = ['EAC', 'AAC', 'TPOJ', 'PCC', 'cortex']
roi = 'TPOJ'
mask_lh = np.load(f'data/{roi}_mask_lh.npy').astype(bool)
mask_rh = np.load(f'data/{roi}_mask_rh.npy').astype(bool)
mask = {'lh': mask_lh, 'rh': mask_rh}

# Split models and load in data splits
train_model = split_models(metadata, stories=stories,
                           subjects=subjects, half=1,
                           delays=delays)
test_model = split_models(metadata, stories=stories,
                          subjects=subjects, half=2,
                          delays=delays)
    
# Load in split data for train and test
train_data = load_split_data(metadata, stories=stories,
                             subjects=subjects, hemisphere=hemisphere,
                             mask=mask, half=1)
test_data = load_split_data(metadata, stories=stories,
                            subjects=subjects, hemisphere=hemisphere,
                            mask=mask, half=2)

# Load in split cSRM data for train and test
train_data = load_split_data(metadata, stories=stories,
                             subjects=subjects, hemisphere=hemisphere,
                             half=1, prefix=f'{roi}_cSRM-train')
test_data = load_split_data(metadata, stories=stories,
                            subjects=subjects, hemisphere=hemisphere,
                            half=2, prefix=f'{roi}_cSRM-test')
    
# Aggregate data and model across subjects
train_data, train_model = aggregate_subjects(train_data[train_story],
                                             train_model[train_story],
                                             train_subjects,
                                             hemi=hemi,
                                             aggregation=aggregation)
test_data, test_model = aggregate_subjects(test_data[test_story],
                                           test_model[test_story],
                                           test_subjects,
                                           hemi=hemi,
                                           aggregation=aggregation)

# Make custom correlation scorer
correlation_scorer = make_scorer(array_correlation)


# Declare ridge regression model
alpha = 100.
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


###

sns.distplot(mean_r, hist=False, kde_kws={"shade": True})
sns.distplot(performance, hist=False, kde_kws={"shade": True})


# Fill results back in cortical mask and write GIfTI
result_map = np.zeros(n_vertices)
result_map[mask] = performance

template_fn = metadata['black']['data']['sub-02']

write_gifti(result_map, 'black_performance_test-xs_lh.gii', template_fn)


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
