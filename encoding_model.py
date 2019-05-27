import json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from scipy.stats import rankdata, zscore
from scipy.spatial.distance import cdist
from gifti_io import read_gifti, write_gifti
from brainiak.utils.utils import array_correlation

# Load dictionary of input filenames
with open('metadata.json') as f:
    metadata = json.load(f)

# Keep track of some basic variables
n_vertices = 40962
model_ndim = 300
delays = [2, 3, 4, 5]

rois = ['EAC', 'AAC', 'TPOJ', 'PCC', 'cortex']
roi = 'AAC'
hemi = 'lh'

stories = ['black', 'forgot']
exclude = [7, 11, 12, 13, 26, 27]
subject_list = [f'sub-{i+1:02}' for i in range(48)
                if i not in exclude]
subject_list = ['sub-01', 'sub-02', 'sub-10', 'sub-15']
subjects = {story: subject_list for story in stories}

# Load mask in fsaverage6 space
mask = np.load(f'data/{roi}_mask_{hemi}.npy').astype(bool)


# Split models and load in data splits
model_splits = split_models(metadata, stories=stories,
                            subjects=subjects, delays=delays)

data_splits = load_split_data(metadata, stories=stories,
                              subjects=subjects, hemi='lh',
                              mask=mask)


# Function for selecting and aggregating subjects
def aggregate_subjects(data, subject_list, hemi='lh',
                       aggregation='average', cv_fold=0,
                       train_test=0):
    
    # Allow for easy single subject input
    if type(subject_list) == str:
        subject = subject_list
        data = data[subject][hemi][cv_fold][train_test]
        
    else:
        if len(subject_list) == 1:
            data = data[subject_list[0]][hemi][cv_fold][train_test]
        else:
            n_subjects = len(subject_list)
            assert n_subjects > 1

            # Check subjects are in the data
            for subject in subject_list:
                assert subject in data.keys()

            # Compile test subjects
            data_list = []
            for subject in subject_list:
                data_list.append(data[subject][hemi][cv_fold][train_test])

            # Average time data across subjects
            if aggregation == 'average' and n_subjects > 1:
                data = np.mean(data_list, axis=0)

            elif aggregation == 'concatenate' and n_subjects > 1:
                data = np.vstack(data_list)
    
    return data

train_story, test_story = 'forgot', 'forgot'
train_subjects = ['sub-01', 'sub-02', 'sub-10']
train_subjects = ['sub-15']
test_subjects = ['sub-15']
cv_fold = 0
aggregation = 'concatenate'

if aggregation == 'average':
    train_model = model_splits[train_story][cv_fold][0]
    test_model = model_splits[test_story][cv_fold][1]
elif aggregation == 'concatenate':
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
