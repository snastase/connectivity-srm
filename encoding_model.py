import json
from os.path import exists
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from scipy.stats import rankdata, zscore
from scipy.spatial.distance import cdist
from gifti_io import read_gifti, write_gifti
from split_stories import check_keys, load_split_data, split_models
from brainiak.utils.utils import array_correlation


# Function for selecting and aggregating subjects
def aggregate_subjects(datasets, models, story_list, subject_list,
                       hemi='lh',  aggregation='average'):
    
    # Allow for easy single subject input
    if type(story_list) == str:
        story_list = [story_list]

    # Loop through requested stories
    data_stack, model_stack = [], []
    for story in story_list:

        # Broadcast model if concatenating subjects
        if aggregation == 'concatenate':
            model = np.tile(models[story], (len(subject_list), 1))
        else:
            model = models[story]

        # Allow for easy single subject input
        if type(subject_list) == str:
            subject = subject_list
            data = datasets[story][subject][hemi]

        else:
            if len(subject_list) == 1:
                data = datasets[story][subject_list[0]][hemi]
            else:
                n_subjects = len(subject_list)
                assert n_subjects > 1

                # Check subjects are in the data
                for subject in subject_list:
                    assert subject in datasets[story].keys()

                # Compile test subjects
                data_list = []
                for subject in subject_list:
                    data_list.append(datasets[story][subject][hemi])

                # Average time data across subjects
                if aggregation == 'average' and n_subjects > 1:
                    data = np.mean(data_list, axis=0)

                elif aggregation == 'concatenate' and n_subjects > 1:
                    data = np.vstack(data_list)
                    
        data_stack.append(data)
        model_stack.append(model)
        
    # Stack for multiple stories
    data_stack = np.vstack(data_stack)
    model_stack = np.vstack(model_stack)

    if len(model_stack) != len(data_stack):
        raise ValueError("Model and data have mismatching shape! "
                         f"model: {model_stack.shape}, data: {data_stack.shape}")

    return data_stack, model_stack


# Function to compute correlation-based rank accuracy
def rank_accuracy(predicted_model, test_model, mean=True):
    n_predictions = test_model.shape[0]

    # Get correlations between pairs
    correlations = 1 - cdist(predicted_model, test_model,
                             'correlation')

    # Get rank of matching prediction for each
    ranks = []
    for index in np.arange(n_predictions):
        ranks.append(rankdata(correlations[index])[index])

    # Normalize ranks by number of choices
    ranks = (np.array(ranks) - 1) / (n_predictions - 1)

    if mean:
        ranks = np.mean(ranks)

    return ranks


# Function to run grid search over alphas across voxels
def grid_search(train_model, train_data, alphas, scorer, n_splits=10):

    # Get number of voxels
    n_voxels = train_data.shape[1]

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


# Name guard for actually running encoding model analysis
if __name__ == '__main__':

    # Load dictionary of input filenames and parameters
    with open('data/metadata.json') as f:
        metadata = json.load(f)

    # Create story and subject lists
    stories = ['black', 'forgot']
    
    # Set ROIs, spaces, and hemispheres
    rois = ['EAC', 'AAC', 'TPOJ', 'PMC']    
    prefixes = [('no SRM', 'noSRM', 'noSRM'),
                ('no SRM (average)', 'noSRM', 'noSRM'),
                ('no SRM (within-subject)', 'noSRM', 'noSRM'),
                ('cPCA (k = 100)', 'parcel-mean_k-100_cPCA-train', 'parcel-mean_k-100_cPCA-test'),
                ('cPCA (k = 50)', 'parcel-mean_k-50_cPCA-train', 'parcel-mean_k-50_cPCA-test'),
                ('cPCA (k = 10)', 'parcel-mean_k-10_cPCA-train', 'parcel-mean_k-10_cPCA-test'),
                ('cSRM (k = 100)', 'parcel-mean_k-100_cSRM-train', 'parcel-mean_k-100_cSRM-test'),
                ('cSRM (k = 50)', 'parcel-mean_k-50_cSRM-train', 'parcel-mean_k-50_cSRM-test'),
                ('cSRM (k = 10)', 'parcel-mean_k-10_cSRM-train', 'parcel-mean_k-10_cSRM-test')]
    stories = ['black', 'forgot']
    hemis = ['lh', 'rh']

    # Set some parameters for encoding model
    delays = [2, 3, 4, 5]
    aggregation = 'average'
    story_train = 'all'
    alpha = 100

    # Make custom correlation scorer
    correlation_scorer = make_scorer(array_correlation)

    # Populate results file if it already exists
    results_fn = f'data/encoding_{story_train}-story_avg_inv_results.npy'
    if exists(results_fn):
        results = np.load(results_fn, allow_pickle=True).item()
    else:
        results = {}

    # Loop through keys without replacing existing ones
    for story in stories:
        if story not in results:
            results[story] = {}

        if story_train == 'within':
            train_stories, test_stories = story, story
        elif story_train == 'across':
            test_stories = story
            train_stories = [st for st in stories if st is not test_story]
        elif story_train == 'all':
            test_stories = story
            train_stories = stories

        # By default just grab all subjects
        subject_list = check_keys(metadata[story]['data'])

        # Split models and load in data splits
        train_model_dict = split_models(metadata, stories=stories,
                                        subjects=None, half=1,
                                        delays=delays)
        test_model_dict = split_models(metadata, stories=stories,
                                       subjects=None, half=2,
                                       delays=delays)

        for roi in rois:
            if roi not in results[story]:
                results[story][roi] = {}

            for prefix in prefixes:
                if prefix[0] not in results[story][roi]:
                    results[story][roi][prefix[0]] = {}

                # Load in split cSRM data for train and test
                train_dict = load_split_data(metadata, stories=stories,
                                             subjects=None, hemisphere=hemis,
                                             half=1, prefix=f'{roi}_' + prefix[1])
                test_dict = load_split_data(metadata, stories=stories,
                                            subjects=None, hemisphere=hemis,
                                            half=2, prefix=f'{roi}_' + prefix[2])

                for s in range(len(subject_list)):

                    test_subjects = [subject_list[s]]
                    test_subject = test_subjects[0]

                    # Use leave-one-subject-out cross-validation (unless within-subject)
                    if prefix[0] == 'no SRM (within-subject)':
                        train_subjects = test_subjects
                    else:
                        train_subjects = [sub for sub in subject_list
                                          if sub is not test_subjects[0]]

                    if test_subject not in results[story][roi][prefix[0]]:
                        results[story][roi][prefix[0]][test_subject] = {}

                    for hemi in hemis:
                        if hemi not in results[story][roi][prefix[0]][test_subject]:
                            results[story][roi][prefix[0]][test_subject][hemi] = {}

                            # Aggregate data and model across subjects
                            train_data, train_model = aggregate_subjects(train_dict,
                                                                         train_model_dict,
                                                                         train_stories,
                                                                         train_subjects,
                                                                         hemi=hemi,
                                                                         aggregation=aggregation)
                            test_data, test_model = aggregate_subjects(test_dict,
                                                                       test_model_dict,
                                                                       test_stories,
                                                                       test_subjects,
                                                                       hemi=hemi,
                                                                       aggregation=aggregation)

                            # Get the regional average as well
                            if prefix[0] == 'no SRM (average)':
                                train_data = np.expand_dims(np.mean(train_data,
                                                                    axis=1), 1)
                                test_data = np.expand_dims(np.mean(test_data,
                                                                    axis=1), 1)
                                
                            # Declare ridge regression model
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
                            
                            results[story][roi][prefix[0]][test_subject][hemi]['encoding'] = performance

                            print(f"Finished forwarding encoding analysis for "
                                  f"{story}, {roi}, {prefix[0]}, {test_subjects}, "
                                  f"performance = {np.mean(performance):.4f}")

                            # Decoding via dot product between test samples and coefficients
                            if prefix[0] != 'no SRM (average)':
                                
                                # Collapse coefficients across delays for decoding
                                collapse_coef = np.mean(np.split(ridge.coef_, len(delays),
                                                                 axis=1), axis=0)
                                collapse_test_model = np.mean(np.split(test_model, len(delays),
                                                                axis=1), axis=0)
                                
                                predicted_model = Ridge(alpha=alpha,
                                                        fit_intercept=False).fit(collapse_coef,
                                                                                 test_data.T).coef_
                                raise

                                accuracy = rank_accuracy(predicted_model, collapse_test_model)

                                results[story][roi][prefix[0]][test_subject][hemi]['decoding'] = accuracy
                                
                                print(f"Finished decoding analysis for "
                                      f"{story}, {roi}, {prefix[0]}, {test_subjects}, "
                                      f"accuracy = {accuracy:.4f} ({rank})")
                                accuracies.append(accuracy)

    np.save(results_fn, results)
