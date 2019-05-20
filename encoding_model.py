import numpy as np
import nibabel as nib
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from scipy.stats import pearsonr, zscore
import matplotlib.pyplot as plt
from utils import array_correlation, read_gifti, write_gifti

story_name = 'black'
n_TRs = 534
n_vertices = 40962

data_fns = ['data/sub-02_task-black.fsaverage6.lh.tproject.gii',
            'data/sub-10_task-black.fsaverage6.lh.tproject.gii',
            'data/sub-15_task-black.fsaverage6.lh.tproject.gii']
data_fn = data_fns[-1]

embedding = np.load(f'transcripts/{story_name}_word2vec_embeddings.npy')


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

model = delay_embedding(embedding)


# Load surface data and trim
def load_data(data_fn, trims=None):

    surf_data = read_gifti(data_fn)

    surf_data = surf_data[trims[0]:-trims[1]]

    # Get medial wall vertices
    medial_wall = np.all(surf_data == 0, axis=0)
    cortical_mask = np.where(~medial_wall)[0]
    data = surf_data[:, ~medial_wall]
    
    return data, cortical_mask

data, mask = load_data(data_fn, trims=[8, 8])


# Split into train and test
def split_half(model, data, zscore_data=True, zscore_model=False):

    assert data.shape[0] == model.shape[0]

    midpoint = data.shape[0] // 2

    first_model = model[:midpoint]
    second_model = model[midpoint:]

    if zscore_model:
        first_model = zscore(first_model, axis=0)
        second_model = zscore(second_model, axis=0)

    first_data = data[:midpoint]
    second_data = data[midpoint:]

    if zscore_data:
        first_data = zscore(first_data, axis=0)
        second_data = zscore(second_data, axis=0)

    model_splits = [(first_model, second_model),
                    (second_model, first_model)]
    data_splits = [(first_data, second_data),
                   (second_data, first_data)]

    return model_splits, data_splits


model_splits, data_splits = split_half(model, data)
train_model, test_model = model_splits[0]
train_data, test_data = data_splits[0]


# Make custom correlation scorer
correlation_scorer = make_scorer(array_correlation)


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


# Declare ridge regression model
alpha = 10
ridge = Ridge(alpha=alpha, fit_intercept=True, normalize=False,
              copy_X=True, tol=0.001, solver='auto')

# Fit training data
ridge.fit(train_model, train_data)

# Use trained model to predict response for test data
test_prediction = ridge.predict(test_model)

# Compute correlation between predicted and test response
performance = array_correlation(test_prediction,
                                test_data)

results = np.zeros(n_vertices)
results[mask] = performance

write_gifti(results, 'black_performance_alpha10_lh.gii', data_fn)
