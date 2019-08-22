import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import colors


# Plot time-segment classification results
segment_length = 10
chance = {'pieman': segment_length/141,
          'prettymouth': segment_length/226,
          'milkyway': segment_length/137,
          'slumlordreach': segment_length/589,
          'notthefall': segment_length/195,
          '21styear': segment_length/1113,
          'pieman (PNI)': segment_length/139,
          'bronx (PNI)': segment_length/187,
          'black': segment_length/267,
          'forgot': segment_length/279}

# Load exisiting results file
results = np.load(f'data/ts_classification_st{segment_length}_results.npy',
                  allow_pickle=True).item()

# Plot results across all stories for one ROI
melted = {'story': [], 'ROI': [], 'space': [],
          'hemisphere': [], 'accuracy': []}
stories = ['pieman', 'prettymouth', 'milkyway',
           'slumlordreach', 'notthefall', '21styear',
           'pieman (PNI)', 'bronx (PNI)', 'black', 'forgot']
for story in stories:
    for roi in ['AAC']:
        for prefix in ['no SRM',
                       'tSRM (k = 100)',
                       'cSRMw (k = 100)', 
                       'cSRM (k = 100)',
                       'cSRM (k = 100; left-out stories)']:
            if prefix not in results[story][roi]:
                continue
            elif (prefix == 'cSRM (k = 100; left-out stories)' and
                  story in ['pieman (PNI)', 'bronx (PNI)']):
                continue
            for hemi in results[story][roi][prefix].keys():
                for subject in results[story][roi][prefix][hemi]:
                    if prefix == 'tSRM (k = 100)':
                        new_prefix = 'tSRM (within-story)' 
                    elif prefix == 'no SRM (average)':
                        new_prefix = 'no SRM (regional-average)' 
                    elif prefix == 'cSRMw (k = 100)':
                        new_prefix = 'cSRM (within-story)'
                    elif prefix == 'cSRM (k = 100)':
                        new_prefix = 'cSRM (across-story)'
                    elif prefix == 'cSRM (k = 100; left-out stories)':
                        new_prefix = 'cSRM (left-out story)'
                    elif prefix == 'cPCA (k = 100)':
                        new_prefix = 'PCA'
                    else:
                        new_prefix = prefix
                    melted['story'].append(story)
                    melted['ROI'].append(roi)
                    melted['space'].append(new_prefix)
                    melted['hemisphere'].append(hemi)
                    melted['accuracy'].append(subject)

results_df = pd.DataFrame(melted)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

sns.set(style='white', font_scale=1.25)
g = sns.catplot(x='story', y='accuracy', hue='space', aspect=2.4, zorder=0,
                kind='bar', data=results_df[results_df['space'] != 'no SRM (regional-average)'], ax=ax1,
                palette=([colors.to_rgba('.7'), 
                          colors.to_rgba('.5'),
                          'lightsalmon', 'crimson', 'mediumpurple']))
g.set_xticklabels(rotation=90, ha='center', y=0.02)
ax1 = g.facet_axis(0, 0)
for i, story in enumerate(stories):
    ax.hlines(chance[story], i -.4, i + .4, colors='w', linestyle='--')
    
plt.savefig('figures/AAC_ts_classification_all-stories_st10.png',
            bbox_inches='tight', dpi=300, transparent=True)

g = sns.catplot(x='space', y='accuracy', aspect=.7, zorder=0,
                kind='bar', data=results_df[results_df['space'] != 'cSRM (left-out story)'],
                                            palette=([colors.to_rgba('.7'), 
                          colors.to_rgba('.5'),
                          'lightsalmon', 'crimson', 'mediumpurple']))
g.set_xticklabels([])
g.ax.set_xlabel("all stories")



# Load exisiting results file
results = np.load(f'data/ts_classification_st{segment_length}_results.npy',
                  allow_pickle=True).item()

plasma_palette = sns.color_palette("plasma_r", 3)
palette = [colors.to_rgba('.7')] + sns.color_palette("YlGnBu", 3) + plasma_palette

# Plot cSRM results for each ROI
melted = {'story': [], 'ROI': [], 'space': [],
          'hemisphere': [], 'accuracy': []}
for story in ['black', 'forgot']:
    for roi in ['EAC', 'AAC', 'TPOJ', 'PMC']:
        for prefix in ['no SRM',
                       'cPCA (k = 100)',
                       'cPCA (k = 50)',
                       'cPCA (k = 10)',
                       'cSRM (k = 100)', 
                       'cSRM (k = 50)',
                       'cSRM (k = 10)']:
            if prefix[:4] == 'cPCA':
                new_prefix = prefix.replace('cPCA', 'PCA')            
            else:
                new_prefix = prefix
            for hemi in results[story][roi][prefix].keys():
                for subject in results[story][roi][prefix][hemi]:
                    melted['story'].append(story)
                    melted['ROI'].append(roi)
                    melted['space'].append(new_prefix)
                    melted['hemisphere'].append(hemi)
                    melted['accuracy'].append(subject)

results_df = pd.DataFrame(melted)

sns.set(style='white', font_scale=1.25)
g = sns.catplot(x='story', y='accuracy', hue='space', aspect=.65,
                col='ROI', kind='bar', data=results_df,
                palette=palette)
g.set_xticklabels(rotation=90, ha='center', y=0.02)
for col in range(4):
    ax = g.facet_axis(0, col)
    for i, story in enumerate(['black', 'forgot']):
        ax.hlines(chance[story], i -.4, i + .4, colors='w', linestyle='--')

plt.savefig('figures/ts_classification_all-ROIs_st10.png',
            bbox_inches='tight', dpi=300, transparent=True)


# Plot temporal (time-series) ISC results
results = np.load('data/temporal_isc_pca_results.npy',
                  allow_pickle=True).item()           

rois = ['EAC', 'AAC', 'TPOJ', 'PMC']

melted_s = {'story': [], 'ROI': [], 'space': [],
            'hemisphere': [], 'temporal ISC': []}
melted_f = {'story': [], 'ROI': [], 'space': [],
            'hemisphere': [], 'temporal ISC': []}
for story in ['black', 'forgot']:
    for roi in results[story].keys():
        for prefix in ['no SRM',
                       'no SRM (average)',
                       'cPCA (k = 100)',
                       'cPCA (k = 50)',
                       'cPCA (k = 10)',
                       'cSRM (k = 100)', 
                       'cSRM (k = 50)',
                       'cSRM (k = 10)']:
            if prefix[:4] == 'cPCA':
                new_prefix = prefix.replace('cPCA', 'PCA')
            else:
                new_prefix = prefix
            for hemi in results[story][roi][prefix].keys():
                for subject in results[story][roi][prefix][hemi]:
                    melted_s['story'].append(story)
                    melted_s['ROI'].append(roi)
                    melted_s['space'].append(new_prefix)
                    melted_s['hemisphere'].append(hemi)
                    melted_s['temporal ISC'].append(np.mean(subject))
                for f in np.arange(results[story][roi][prefix][hemi].shape[1]):
                    melted_f['story'].append(story)
                    melted_f['ROI'].append(roi)
                    melted_f['space'].append(new_prefix)
                    melted_f['hemisphere'].append(hemi)
                    melted_f['temporal ISC'].append(np.mean(
                        results[story][roi][prefix][hemi][:, f]))

results_s_df = pd.DataFrame(melted_s)
results_f_df = pd.DataFrame(melted_f)

plasma_palette = sns.color_palette("plasma_r", 3)
palette = [colors.to_rgba('.7')] + sns.color_palette("YlGnBu", 3) + plasma_palette

sns.set(style='white', font_scale=1.25)
g = sns.catplot(x='story', y='temporal ISC', hue='space',
                kind='strip', jitter=True, dodge=True,
                col='ROI', data=results_f_df[results_f_df['space'] != 'no SRM (average)'], aspect=.8,
                alpha=.75, zorder=0,
                palette=palette)

for col, roi in enumerate(rois):
    ax = g.facet_axis(0, col)
    avgs = results_f_df[(results_f_df['space'] == 'no SRM (average)') &
                        (results_f_df['ROI'] == roi)]['temporal ISC'].values
    ax.scatter([-.35, -.35, .65, .65], avgs, color='.45')

plt.savefig('figures/new_temporal_isc.png',
            bbox_inches='tight', dpi=300, transparent=True)


# Plot proportion of temporal ISCs exceeding threshold
results = np.load('data/temporal_isc_pca_results.npy',
                  allow_pickle=True).item()           

threshold = .1
y_label = 'proportion $\it{r}$ > .1'

rois = ['EAC', 'AAC', 'TPOJ', 'PMC']

melted = {'story': [], 'ROI': [], 'space': [],
            'hemisphere': [], y_label: []}
for story in ['black', 'forgot']:
    for roi in results[story].keys():
        #for prefix in ['no SRM',
        #               'no SRM (average)',
        #               'cPCA (k = 100)',
        #               'cSRM (k = 100)', 
        #               'cPCA (k = 50)',
        #               'cSRM (k = 50)',
        #               'cPCA (k = 10)',
        #               'cSRM (k = 10)']:
        for prefix in ['no SRM',
                       'no SRM (average)',
                       'cPCA (k = 100)',
                       'cPCA (k = 50)',
                       'cPCA (k = 10)',
                       'cSRM (k = 100)', 
                       'cSRM (k = 50)',
                       'cSRM (k = 10)']:
            if prefix[:4] == 'cPCA':
                new_prefix = prefix.replace('cPCA', 'PCA')
            else:
                new_prefix = prefix
            for hemi in results[story][roi][prefix].keys():
                n_features = results[story][roi][prefix][hemi].shape[1]
                n_subjects = results[story][roi][prefix][hemi].shape[0]
                for s in np.arange(n_subjects):
                    n_exceeding = 0
                    for f in np.arange(n_features):
                        if results[story][roi][prefix][hemi][s, f] > threshold:
                            n_exceeding += 1
                    proportion = n_exceeding / n_features
                    melted['story'].append(story)
                    melted['ROI'].append(roi)
                    melted['space'].append(new_prefix)
                    melted['hemisphere'].append(hemi)
                    melted[y_label].append(proportion)

results_df = pd.DataFrame(melted)
sns.set(style='white', font_scale=1.25)
g = sns.catplot(x='story', y=y_label,
                hue='space', kind='bar',
                col='ROI', data=results_df[(results_df['space'] != 'no SRM (average)') &
                                           (results_df['space'] != 'no SRM')],
                aspect=.8,
                palette=palette[1:])

plt.savefig('figures/new_temporal_isc_proportion.png',
            bbox_inches='tight', dpi=300, transparent=True)


# Plot absolute number of temporal ISCs exceeding threshold
results = np.load('data/temporal_isc_pca_results.npy',
                  allow_pickle=True).item()           

threshold = .1
y_label = 'orthogonal features $\it{r}$ > .1'

rois = ['EAC', 'AAC', 'TPOJ', 'PMC']

melted = {'story': [], 'ROI': [], 'space': [],
            'hemisphere': [], y_label: []}
for story in ['black', 'forgot']:
    for roi in results[story].keys():
        for prefix in ['cPCA (k = 100)',
                       'cPCA (k = 50)',
                       'cPCA (k = 10)',
                       'cSRM (k = 100)', 
                       'cSRM (k = 50)',
                       'cSRM (k = 10)']:
            if prefix[:4] == 'cPCA':
                new_prefix = prefix.replace('cPCA', 'PCA')
            else:
                new_prefix = prefix
            for hemi in results[story][roi][prefix].keys():
                n_features = results[story][roi][prefix][hemi].shape[1]
                n_subjects = results[story][roi][prefix][hemi].shape[0]
                for s in np.arange(n_subjects):
                    n_exceeding = 0
                    for f in np.arange(n_features):
                        if results[story][roi][prefix][hemi][s, f] > threshold:
                            n_exceeding += 1
                    melted['story'].append(story)
                    melted['ROI'].append(roi)
                    melted['space'].append(new_prefix)
                    melted['hemisphere'].append(hemi)
                    melted[y_label].append(n_exceeding)

results_df = pd.DataFrame(melted)
sns.set(style='white', font_scale=1.25)
g = sns.catplot(x='story', y=y_label, sharey=False,
                hue='space', kind='bar',
                col='ROI', data=results_df,
                palette=palette[2:], aspect=.7799)
g.set(ylim=(0, 100))

plt.savefig('figures/new_temporal_isc_absolute.png',
            bbox_inches='tight', dpi=300, transparent=True)


# Plot spatial (pattern) ISC results
results = np.load('data/spatial_isc_pca_results.npy',
                  allow_pickle=True).item()   

melted = {'story': [], 'ROI': [], 'space': [],
          'hemisphere': [], 'spatial ISC': []}
for story in results.keys():
    for roi in results[story].keys():
        for prefix in ['no SRM',
                       'cPCA (k = 100)',
                       'cPCA (k = 50)',
                       'cPCA (k = 10)',
                       'cSRM (k = 100)', 
                       'cSRM (k = 50)',
                       'cSRM (k = 10)']:
            if prefix[:4] == 'cPCA':
                new_prefix = prefix.replace('cPCA', 'PCA')
            else:
                new_prefix = prefix
            for hemi in results[story][roi][prefix].keys():
                for subject in results[story][roi][prefix][hemi]:
                    melted['story'].append(story)
                    melted['ROI'].append(roi)
                    melted['space'].append(new_prefix)
                    melted['hemisphere'].append(hemi)
                    melted['spatial ISC'].append(np.mean(subject))

results_df = pd.DataFrame(melted)
sns.set(style='white', font_scale=1.25)
g = sns.catplot(x='story', y='spatial ISC', hue='space',
                kind='bar', col='ROI', data=results_df,
                aspect=.7,
                palette=palette)

plt.savefig('figures/new_spatial_isc.png',
            bbox_inches='tight', dpi=300, transparent=True)


# Compare transformationns and shared matrices
with open('data/metadata.json') as f:
    metadata = json.load(f)

stories = ['pieman', 'prettymouth', 'milkyway',
           'slumlordreach', 'notthefall', '21styear',
           'pieman (PNI)', 'bronx (PNI)', 'black', 'forgot']

# Load in transformation matrices for across-/within-story cSRMs
w_across = np.load('data/half-1_AAC_parcel-mean_k-100_cSRM_w.npy',
                   allow_pickle=True).item()
w_within = {}
for story in metadata:
    w_within[story] = np.load(f'data/half-1_AAC_parcel-mean_k-100_cSRM-{story}_w.npy',
                              allow_pickle=True).item()

# Load in ISFCs from second (test) half
isfcs = np.load('data/half-2_AAC_parcel-mean_isfcs.npy',
                allow_pickle=True).item()

# Transform and get mean ISFCs across subjects
across, within = [], []
for story in isfcs.keys():
    story_across, story_within = [], []
    for subject in isfcs[story].keys():
        try:
            isfc = isfcs[story][subject][hemi].T
            story_across.append(isfc.dot(w_across[subject][hemi]))
            story_within.append(isfc.dot(w_within[story][subject][hemi]))
            print(f"Finished ISFC projections for {subject} ({story})")
        except KeyError:
            print(f"No transformation for {subject} ({story}) -- skipping!!!")
    across.append(np.ravel(np.mean(story_across, axis=0)))
    within.append(np.ravel(np.mean(story_within, axis=0)))
    print(f"Finished transforming ISFCs for {story}")

r_across = np.corrcoef(np.vstack(across))
r_within = np.corrcoef(np.vstack(within))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 16))
sns.heatmap(r_across, vmin=0, vmax=1, cmap='magma', ax=ax1,
            square=True, xticklabels=stories, yticklabels=stories,
            cbar_kws={'fraction': 0.046, 'pad': 0.04})
ax1.set_title("across-story cSRM")
ax1.collections[0].colorbar.ax.tick_params(length=0)
ax1.collections[0].colorbar.set_label('correlation', labelpad=12, rotation=270)
sns.heatmap(r_within, vmin=0, vmax=1, cmap='magma', ax=ax2,
            square=True, xticklabels=stories, yticklabels=False,
            cbar_kws={'fraction': 0.046, 'pad': 0.04})
ax2.set_title("within-story cSRM")
ax2.collections[0].colorbar.ax.tick_params(length=0)
ax2.collections[0].colorbar.set_label('correlation', labelpad=12, rotation=270)
sns.heatmap(r_across - r_within, cmap='coolwarm', vmin=-.2, vmax=.2, ax=ax3,
            square=True, xticklabels=stories, yticklabels=False,
            cbar_kws={'fraction': 0.046, 'pad': 0.04})
ax3.set_title("difference (across – within)")
ax3.collections[0].colorbar.ax.tick_params(length=0)
ax3.collections[0].colorbar.set_ticks([-.2, -.1, 0, .1, .2])
ax3.collections[0].colorbar.set_label('correlation', labelpad=12, rotation=270)
plt.tight_layout()
plt.savefig('figures/within-vs-across_corrmats.png',
            bbox_inches='tight', dpi=300, transparent=True)

r_both = np.corrcoef(np.vstack(across), np.vstack(within))
n_stories = len(stories)
r_both_off = r_both[:n_stories, n_stories:]
superordinate = [''] * 20
superordinate[5] = 'across'
superordinate[15] = 'within'
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
sns.heatmap(r_both, cmap='magma', vmin=0, vmax=1, ax=ax1,
            square=True, xticklabels=superordinate, yticklabels=stories*2,
            cbar_kws={'fraction': 0.046, 'pad': 0.04})
ax1.collections[0].colorbar.ax.tick_params(length=0)
ax1.collections[0].colorbar.set_label('correlation', labelpad=12, rotation=270)
ax1.xaxis.set_tick_params(length=0)
sns.heatmap(r_both_off, cmap='magma', vmin=0, vmax=1, ax=ax2,
            square=True, xticklabels=stories, yticklabels=False,
            cbar_kws={'fraction': 0.046, 'pad': 0.04})
ax2.collections[0].colorbar.ax.tick_params(length=0)
ax2.collections[0].colorbar.set_label('correlation', labelpad=12, rotation=270)
plt.tight_layout()
plt.savefig('figures/within-across_corrmat.png',
            bbox_inches='tight', dpi=300, transparent=True)
np.mean(np.diag(r_both_off))
np.std(np.diag(r_both_off))
np.amin(np.diag(r_both_off))
np.amax(np.diag(r_both_off))


# Compare subject time series (within-story) across shared spaces
w_across = np.load('data/half-1_AAC_parcel-mean_k-100_cSRM_w.npy',
                   allow_pickle=True).item()
w_within = {}
for story in metadata:
    w_within[story] = np.load(f'data/half-1_AAC_parcel-mean_k-100_cSRM-{story}_w.npy',
                              allow_pickle=True).item()

shared_ts = {'story': [], 'hemisphere': [], 'spatiotemporal ISC': [], 'cSRM space': []}
across_res, within_res = {}, {}
for story in stories:
    across_res[story], within_res[story] = {}, {}
    for hemi in ['lh', 'rh']:
        across_trans, within_trans = [], []
        for subject in w_within[story].keys():
            try:
                test_data = np.load(f'data/{subject}_task-{story}_'
                                    f'half-2_AAC_noSRM_{hemi}.npy')
                
                # Project test data into shared spaces
                across_trans.append(test_data.dot(w_across[subject][hemi]))
                within_trans.append(test_data.dot(w_within[story][subject][hemi]))
                
            except FileNotFoundError:
                print(f"No test data found for {subject} ({story}) -- skipping!!!")
            
            print(f"Finished time series projections for {subject} ({story}, {hemi})")

        # Compute spatiotemporal ISCs
        across_isc = isc([np.ravel(a) for a in across_trans])
        within_isc = isc([np.ravel(a) for a in within_trans])
        across_res[story][hemi] = across_isc
        within_res[story][hemi] = within_isc
        print(f"Finished computing spatiotemporal ISCs for {story} ({hemi})")
              
        for across, within in zip(across_isc, within_isc):
            for sp in [('across-story', across), ('within-story', within)]:
                shared_ts['story'].append(story)
                shared_ts['hemisphere'].append(hemi)
                shared_ts['cSRM space'].append(sp[0])
                shared_ts['spatiotemporal ISC'].append(sp[1][0])

    print(f"Finished comparing time-series for {story}")
    
results_df = pd.DataFrame(shared_ts)
sns.set(style='white', font_scale=1.25)
g = sns.catplot(x='story', y='spatiotemporal ISC', hue='cSRM space',
                kind='bar', data=results_df, palette=['.40', '.70'],
                aspect=1.2, ci=95)
g.set_xticklabels(rotation=90, ha='center', y=0.02)
plt.savefig('figures/within-vs-across_subject_ts_iscs.png',
            bbox_inches='tight', dpi=300, transparent=True)


# Non-parametric t-test
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
signed_ranks = {}
for story in stories:
    diffs = np.mean((across_res[story]['lh'] - within_res[story]['lh'],
                     across_res[story]['rh'] - within_res[story]['rh']), axis=0)
    signed_ranks[story] = wilcoxon(np.ravel(diffs))[1]
p_values = [signed_ranks[s] for s in signed_ranks]  
        
    
# Compare subject ISFCs (within-story) across shared spaces
isfcs = np.load('data/half-2_AAC_parcel-mean_isfcs.npy',
                allow_pickle=True).item()

hemi = 'lh'
across, within = {}, {}
for story in stories:
    across[story], within[story] = [], []
    for subject in w_within[story].keys():
        try:
            isfc = isfcs[story][subject][hemi].T
            across[story].append(test_data.dot(w_across[subject][hemi]))
            within[story].append(test_data.dot(w_within[story][subject][hemi]))
            print(f"Finished ISFC projections for {subject} ({story})")
        except FileNotFoundError:
            print(f"No test data found for {subject} ({story}) -- skipping!!!")
    print(f"Finished transforming ISFCs for {story}")
    
    
# Plot forward encoding model results
story_train='all'

results = np.load(f'data/encoding_{story_train}-story_avg_inv_results.npy',
                  allow_pickle=True).item()

melted_s = {'story': [], 'ROI': [], 'space': [],
            'hemisphere': [], 'correlation': []}
melted_f = {'story': [], 'ROI': [], 'space': [],
            'hemisphere': [], 'correlation': []}

prefixes = ['no SRM',
            'no SRM (average)',
            'no SRM (within-subject)',
            'cPCA (k = 100)',
            'cPCA (k = 50)',
            'cPCA (k = 10)',
            'cSRM (k = 100)', 
            'cSRM (k = 50)',
            'cSRM (k = 10)']

for story in results.keys():
    for roi in results[story].keys():
        for prefix in prefixes:
            if prefix[:4] == 'cPCA':
                new_prefix = prefix.replace('cPCA', 'PCA')            
            else:
                new_prefix = prefix
            for subject in results[story][roi][prefix].keys():
                for hemi in results[story][roi][prefix][subject].keys():
                    melted_s['story'].append(story)
                    melted_s['ROI'].append(roi)
                    melted_s['space'].append(new_prefix)
                    melted_s['hemisphere'].append(hemi)
                    melted_s['correlation'].append(
                        np.mean(results[story][roi][prefix][subject][hemi]['encoding']))
            subject_stack = {'lh': [], 'rh': []}
            for subject in results[story][roi][prefix].keys():
                for hemi in results[story][roi][prefix][subject].keys():
                    subject_stack[hemi].append(
                        results[story][roi][prefix][subject][hemi]['encoding'])
            for hemi in results[story][roi][prefix][subject].keys():
                subject_stack[hemi] = np.mean(subject_stack[hemi], axis=0)
                for feature in subject_stack[hemi]:
                    melted_f['story'].append(story)
                    melted_f['ROI'].append(roi)
                    melted_f['space'].append(new_prefix)
                    melted_f['hemisphere'].append(hemi)
                    melted_f['correlation'].append(feature)

results_s_df = pd.DataFrame(melted_s)
results_f_df = pd.DataFrame(melted_f)

plasma_palette = sns.color_palette("plasma_r", 3)
palette = [colors.to_rgba('.7'), colors.to_rgba('.55')] + sns.color_palette("YlGnBu", 3) + plasma_palette

sns.set(style='white', font_scale=1.25)
g = sns.catplot(x='story', y='correlation', hue='space',
                kind='strip', jitter=True, dodge=True,
                col='ROI',
                data=results_f_df[results_f_df['space'] !=
                                             'no SRM (average)'],
                aspect=.8,
                alpha=.75, zorder=0,
                palette=palette)
for col, roi in enumerate(rois):
    ax = g.facet_axis(0, col)
    avgs = results_f_df[(results_f_df['space'] == 'no SRM (average)') &
                        (results_f_df['ROI'] == roi)]['correlation'].values
    ax.scatter([-.355, -.355, .645, .645], avgs, color='.45')
g.set(ylim=((-0.099, 0.42)))

plt.savefig(f'figures/new_encoding_{story_train}-story_avg.png',
            bbox_inches='tight', dpi=300, transparent=True)


# Plot proportion of vertices exceeding performance threshold
story_train='all'
threshold = .1
y_label = 'proportion $\it{r}$ > .1'

results = np.load(f'data/encoding_{story_train}-story_avg_inv_results.npy',
                  allow_pickle=True).item()

melted = {'story': [], 'ROI': [], 'space': [],
            'hemisphere': [], y_label: []}

rois = ['EAC', 'AAC', 'TPOJ', 'PMC']

melted = {'story': [], 'ROI': [], 'space': [],
            'hemisphere': [], y_label: []}
for story in ['black', 'forgot']:
    for roi in results[story].keys():
        for prefix in ['no SRM',
                       'no SRM (average)',
                       'no SRM (within-subject)',
                       'cPCA (k = 100)',
                       'cPCA (k = 50)',
                       'cPCA (k = 10)',
                       'cSRM (k = 100)', 
                       'cSRM (k = 50)',
                       'cSRM (k = 10)']:
            if prefix[:4] == 'cPCA':
                new_prefix = prefix.replace('cPCA', 'PCA')
            else:
                new_prefix = prefix
            subject_stack = {'lh': [], 'rh': []}
            for subject in results[story][roi][prefix].keys():
                for hemi in results[story][roi][prefix][subject].keys():
                    subject_stack[hemi].append(
                        results[story][roi][prefix][subject][hemi]['encoding'])
            for hemi in ['lh', 'rh']:
                for subj in subject_stack[hemi]:
                    n_features = len(subj)
                    n_exceeding = 0
                    for feat in subj:
                        if feat > threshold:
                            n_exceeding += 1      
                    proportion = n_exceeding / n_features
                    melted['story'].append(story)
                    melted['ROI'].append(roi)
                    melted['space'].append(new_prefix)
                    melted['hemisphere'].append(hemi)
                    melted[y_label].append(proportion)
                    
results_df = pd.DataFrame(melted)
sns.set(style='white', font_scale=1.25)
g = sns.catplot(x='story', y=y_label,
                hue='space', kind='bar',
                col='ROI', data=results_df[(results_df['space'] != 'no SRM (average)') &
                                           (results_df['space'] != 'no SRM') &
                                           (results_df['space'] != 'no SRM (within-subject)')],
                aspect=.8,
                palette=palette[1:])

plt.savefig(f'figures/new_encoding_{story_train}-story_avg_proportion.png',
            bbox_inches='tight', dpi=300, transparent=True)


# Plot proportion of vertices exceeding performance threshold
story_train='all'
threshold = .1
y_label = 'orthogonal features $\it{r}$ > .1'

results = np.load(f'data/encoding_{story_train}-story_avg_inv_results.npy',
                  allow_pickle=True).item()

melted = {'story': [], 'ROI': [], 'space': [],
            'hemisphere': [], y_label: []}

rois = ['EAC', 'AAC', 'TPOJ', 'PMC']

melted = {'story': [], 'ROI': [], 'space': [],
            'hemisphere': [], y_label: []}
for story in ['black', 'forgot']:
    for roi in results[story].keys():
        for prefix in ['cPCA (k = 100)',
                       'cPCA (k = 50)',
                       'cPCA (k = 10)',
                       'cSRM (k = 100)', 
                       'cSRM (k = 50)',
                       'cSRM (k = 10)']:
            if prefix[:4] == 'cPCA':
                new_prefix = prefix.replace('cPCA', 'PCA')
            else:
                new_prefix = prefix
            subject_stack = {'lh': [], 'rh': []}
            for subject in results[story][roi][prefix].keys():
                for hemi in results[story][roi][prefix][subject].keys():
                    subject_stack[hemi].append(
                        results[story][roi][prefix][subject][hemi]['encoding'])
            for hemi in ['lh', 'rh']:
                for subj in subject_stack[hemi]:
                    n_features = len(subj)
                    n_exceeding = 0
                    for feat in subj:
                        if feat > threshold:
                            n_exceeding += 1      
                    melted['story'].append(story)
                    melted['ROI'].append(roi)
                    melted['space'].append(new_prefix)
                    melted['hemisphere'].append(hemi)
                    melted[y_label].append(n_exceeding)
                    
results_df = pd.DataFrame(melted)
sns.set(style='white', font_scale=1.25)
g = sns.catplot(x='story', y=y_label,
                hue='space', kind='bar',
                col='ROI', data=results_df,
                aspect=.797,
                palette=palette[2:])
g.set(ylim=(0, 50))

plt.savefig(f'figures/new_encoding_{story_train}-story_avg_absolute.png',
            bbox_inches='tight', dpi=300, transparent=True)


# Plot model-based decoding results
story_train = 'all'

results = np.load(f'data/encoding_{story_train}-story_avg_inv_results.npy',
                  allow_pickle=True).item()

prefixes = ['no SRM',
            'no SRM (within-subject)',
            'cPCA (k = 100)',
            'cPCA (k = 50)',
            'cPCA (k = 10)',
            'cSRM (k = 100)', 
            'cSRM (k = 50)',
            'cSRM (k = 10)']

melted = {'story': [], 'ROI': [], 'space': [],
          'hemisphere': [], 'rank accuracy': []}
for story in results.keys():
    for roi in results[story].keys():
        for prefix in prefixes:
            if prefix[:4] == 'cPCA':
                new_prefix = prefix.replace('cPCA', 'PCA')            
            else:
                new_prefix = prefix
            for subject in results[story][roi][prefix].keys():
                for hemi in results[story][roi][prefix][subject].keys():
                    melted['story'].append(story)
                    melted['ROI'].append(roi)
                    melted['space'].append(new_prefix)
                    melted['hemisphere'].append(hemi)
                    melted['rank accuracy'].append(
                        results[story][roi][prefix][subject][hemi]['decoding'])

results_df = pd.DataFrame(melted)

plasma_palette = sns.color_palette("plasma_r", 3)
palette = [colors.to_rgba('.7'), colors.to_rgba('.55')] + sns.color_palette("YlGnBu", 3) + plasma_palette

sns.set(style='white', font_scale=1.25)
g = sns.catplot(x='story', y='rank accuracy', hue='space',
                col='ROI', kind='bar', data=results_df, aspect=.75,
                palette=palette)
for col in range(4):
    ax = g.facet_axis(0, col)
    for i, story in enumerate(['black', 'forgot']):
        ax.hlines(.5, i -.4, i + .4, colors='k', linestyle='--', zorder=0)
        ax.hlines(.5, i -.4, i + .4, colors='w', linestyle='--')
g.set(ylim=(0.485, .58))

plt.savefig(f'figures/new_decoding_{story_train}-story_avg.png',
            transparent=True, bbox_inches='tight', dpi=300)
