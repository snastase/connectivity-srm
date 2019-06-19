import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import colors


# Plot time-segment classification results
segment_length = 10
chance = {'black': segment_length/267, 'forgot': segment_length/279}

# Load exisiting results file
results = np.load(f'data/ts_classification_st{segment_length}_results.npy').item()

melted = {'story': [], 'ROI': [], 'space': [],
          'hemisphere': [], 'accuracy': []}
for story in results.keys():
    for roi in results[story].keys():
        for prefix in results[story][roi].keys():
            for hemi in results[story][roi][prefix].keys():
                for subject in results[story][roi][prefix][hemi]:
                    melted['story'].append(story)
                    melted['ROI'].append(roi)
                    melted['space'].append(prefix)
                    melted['hemisphere'].append(hemi)
                    melted['accuracy'].append(subject)

results_df = pd.DataFrame(melted)

sns.set(style='white', font_scale=1.5)
g = sns.catplot(x='story', y='accuracy', hue='space',
                col='ROI', kind='bar', data=results_df, aspect=.65,
                palette=([colors.to_rgba('.7')] +
                          sns.color_palette("plasma_r", 4)))
for col in range(4):
    ax = g.facet_axis(0, col)
    for i, story in enumerate(chance):
        ax.hlines(chance[story], i -.4, i + .4, colors='w', linestyle='--')

plt.savefig('figures/ts_classification_st10.png',
            bbox_inches='tight', dpi=300, transparent=True)


# Plot temporal (time-series) ISC results
results = np.load('data/temporal_isc_results.npy').item()           

melted_s = {'story': [], 'ROI': [], 'space': [],
            'hemisphere': [], 'temporal ISC': []}
melted_f = {'story': [], 'ROI': [], 'space': [],
            'hemisphere': [], 'temporal ISC': []}
for story in results.keys():
    for roi in results[story].keys():
        for prefix in results[story][roi].keys():
            for hemi in results[story][roi][prefix].keys():
                for subject in results[story][roi][prefix][hemi]:
                    melted_s['story'].append(story)
                    melted_s['ROI'].append(roi)
                    melted_s['space'].append(prefix)
                    melted_s['hemisphere'].append(hemi)
                    melted_s['temporal ISC'].append(np.mean(subject))
                for f in np.arange(results[story][roi][prefix][hemi].shape[1]):
                    melted_f['story'].append(story)
                    melted_f['ROI'].append(roi)
                    melted_f['space'].append(prefix)
                    melted_f['hemisphere'].append(hemi)
                    melted_f['temporal ISC'].append(np.mean(
                        results[story][roi][prefix][hemi][:, f]))

results_s_df = pd.DataFrame(melted_s)
results_f_df = pd.DataFrame(melted_f)

g = sns.catplot(x='story', y='temporal ISC', hue='space',
                kind='strip', jitter=True, dodge=True,
                col='ROI', data=results_f_df, aspect=.6,
                alpha=.2, zorder=0, legend=False,
                palette=([colors.to_rgba('.7')] +
                         sns.color_palette("plasma_r", 4)))
for col in range(4):
    ax = g.facet_axis(0, col)
    sns.pointplot(x='story', y='temporal ISC', hue='space', ci=99,
                  join=False, dodge=.65, data=results_s_df,
                  aspect=.6, legend=False, markers='_', color='k')
    ax.legend_.remove()

plt.savefig('figures/temporal_isc.png',
            bbox_inches='tight', dpi=300, transparent=True)


# Plot spatial (pattern) ISC results
results = np.load('data/spatial_isc_subject-mean_results.npy').item()   

melted = {'story': [], 'ROI': [], 'space': [],
          'hemisphere': [], 'spatial ISC': []}
for story in results.keys():
    for roi in results[story].keys():
        for prefix in results[story][roi].keys():
            for hemi in results[story][roi][prefix].keys():
                for subject in results[story][roi][prefix][hemi]:
                    melted['story'].append(story)
                    melted['ROI'].append(roi)
                    melted['space'].append(prefix)
                    melted['hemisphere'].append(hemi)
                    melted['spatial ISC'].append(subject)

results_df = pd.DataFrame(melted)
g = sns.catplot(x='story', y='spatial ISC', hue='space',
                kind='bar', col='ROI', data=results_df,
                aspect=.75, legend=False,
                palette=([colors.to_rgba('.7')] +
                          sns.color_palette("plasma_r", 4)))

plt.savefig('figures/spatial_isc.png',
            bbox_inches='tight', dpi=300, transparent=True)


# Plot forward encoding model results
results = np.load('data/encoding_within-story_avg_results.npy').item()

melted_s = {'story': [], 'ROI': [], 'space': [],
            'hemisphere': [], 'correlation': []}
melted_f = {'story': [], 'ROI': [], 'space': [],
            'hemisphere': [], 'correlation': []}
for story in results.keys():
    for roi in results[story].keys():
        for prefix in results[story][roi].keys():
            for subject in results[story][roi][prefix].keys():
                for hemi in results[story][roi][prefix][subject].keys():
                    melted_s['story'].append(story)
                    melted_s['ROI'].append(roi)
                    melted_s['space'].append(prefix)
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
                    melted_f['space'].append(prefix)
                    melted_f['hemisphere'].append(hemi)
                    melted_f['correlation'].append(feature)

results_s_df = pd.DataFrame(melted_s)
results_f_df = pd.DataFrame(melted_f)

sns.set(style='white', font_scale=1.3)
g = sns.catplot(x='story', y='correlation', hue='space',
                kind='strip', jitter=True, dodge=True,
                col='ROI', data=results_f_df, aspect=.6,
                alpha=.2, zorder=0, legend=False,
                palette=([colors.to_rgba('.7')] +
                          sns.color_palette("plasma_r", 4)))
for col in range(4):
    ax = g.facet_axis(0, col)
    sns.pointplot(x='story', y='correlation', hue='space', ci=99,
                  join=False, dodge=.65, data=results_s_df,
                  aspect=.6, legend=False, markers='_', color='k')
    ax.legend_.remove()

plt.savefig('figures/encoding_within-story_avg.png',
            bbox_inches='tight', dpi=300, transparent=True)


# Plot model-based decoding results
results = np.load('data/encoding_within-story_avg_results.npy').item()

melted = {'story': [], 'ROI': [], 'space': [],
          'hemisphere': [], 'rank accuracy': []}
for story in results.keys():
    for roi in results[story].keys():
        for prefix in results[story][roi].keys():
            for subject in results[story][roi][prefix].keys():
                for hemi in results[story][roi][prefix][subject].keys():
                    melted['story'].append(story)
                    melted['ROI'].append(roi)
                    melted['space'].append(prefix)
                    melted['hemisphere'].append(hemi)
                    melted['rank accuracy'].append(
                        results[story][roi][prefix][subject][hemi]['decoding'])

results_df = pd.DataFrame(melted)

sns.set(style='white')
g = sns.catplot(x='story', y='rank accuracy', hue='space',
                col='ROI', kind='bar', data=results_df, aspect=.6,
                palette=([colors.to_rgba('.7')] +
                          sns.color_palette("plasma_r", 4)))
g.set(ylim=(0.5, None))

plt.savefig('figures/decoding_within-story_avg.png',
            transparent=True, bbox_inches='tight', dpi=300)
