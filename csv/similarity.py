import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipystat

similarity_filename = 'similarity.csv'
checkpoint_maps = [ 'imagenet', 'miniimagenet_with_in', 'stylized_miniimagenet_with_in' ]
names = ['original_val_loader', 'stylized_val_loader', 'original_train_loader', 'stylized_train_loader']
stats = [ 'mean', 'std' ]

id_labels = [
    'model',
    'dataset-split',
    'stat'
]

activation_labels = list(range(0, 512))

labels = id_labels + activation_labels

similarity = pd.DataFrame(columns=labels)

for checkpoint_map in checkpoint_maps:
    for name in names:
        for stat in stats:
            csv_filename = '{}-{}-{}.csv'.format(checkpoint_map, name, stat)
            csv_file = pd.read_csv(csv_filename, header=None)
            mean = csv_file.mean(axis=0).to_dict()
            mean['model'] = checkpoint_map
            mean['dataset-split'] = name
            mean['stat'] = stat
            similarity = similarity.append(mean, ignore_index=True)

similarity = similarity.sort_values(by=id_labels)

similarity.to_csv(similarity_filename)

# summarize

stats = pd.read_csv(similarity_filename)

dataset_types = [ 'original', 'stylized' ]
loader_types = [ 'val', 'train' ]
loader_string_format = '{}_{}_loader'

for loader_type in loader_types:
    # print(loader_type)
    i = 1
    plt.figure(figsize=(20, 8))
    loader_type_stats = pd.DataFrame()
    in_original_mean = None
    in_original_std = None
    min_original_mean = None
    min_original_std = None
    smin_original_mean = None
    smin_original_std = None
    for checkpoint_map in checkpoint_maps:
        # print(checkpoint_map)
        original_loader = loader_string_format.format('original', loader_type)
        stylized_loader = loader_string_format.format('stylized', loader_type)
        original_stats = stats[(stats.model == checkpoint_map) & (stats['dataset-split'] == original_loader)][list(map(str, activation_labels))].transpose()
        stylized_stats = stats[(stats.model == checkpoint_map) & (stats['dataset-split'] == stylized_loader)][list(map(str, activation_labels))].transpose()
        original_mean_column = original_stats.columns[0]
        original_std_column = original_stats.columns[1]
        stylized_mean_column = stylized_stats.columns[0]
        stylized_std_column = stylized_stats.columns[1]
        original_mean = original_stats[original_mean_column]
        stylized_mean = stylized_stats[stylized_mean_column]
        original_std = original_stats[original_std_column]
        stylized_std = stylized_stats[stylized_std_column]
        original_row = pd.DataFrame([checkpoint_map, 'original', original_mean.mean(), original_std.mean()]).transpose()
        stylized_row = pd.DataFrame([checkpoint_map, 'stylized', stylized_mean.mean(), stylized_std.mean()]).transpose()
        original_row.columns = ['model', 'dataset', 'mean', 'std']
        stylized_row.columns = ['model', 'dataset', 'mean', 'std']
        loader_type_stats = pd.concat([loader_type_stats, original_row, stylized_row])
        if checkpoint_map == 'imagenet':
            in_original_mean = original_mean
            in_original_std = original_std
        elif checkpoint_map == 'miniimagenet_with_in':
            min_original_mean = original_mean
            min_original_std = original_std
        elif checkpoint_map == 'stylized_miniimagenet_with_in':
            smin_original_mean = original_mean
            smin_original_std = original_std
        plt.subplot(2, 3, i)
        hist = original_mean.plot.hist(bins=10, alpha=0.5)
        hist = stylized_mean.plot.hist(bins=10, alpha=0.5)
        plt.title('{} {} mean'.format(loader_type, checkpoint_map))
        plt.subplot(2, 3, i+3)
        hist = original_std.plot.hist(bins=10, alpha=0.5)
        hist = stylized_std.plot.hist(bins=10, alpha=0.5)
        plt.title('{} {} std'.format(loader_type, checkpoint_map))
        i+=1
    print('imagenet vs miniimagenet - mean')
    print(scipystat.ttest_ind(in_original_mean, min_original_mean))
    print('imagenet vs miniimagenet - std')
    print(scipystat.ttest_ind(in_original_mean, smin_original_mean))
    print('imagenet vs stylized miniimagenet - mean')
    print(scipystat.ttest_ind(in_original_std, min_original_std))
    print('imagenet vs stylized miniimagenet - std')
    print(scipystat.ttest_ind(in_original_std, smin_original_std))
    plt.savefig('{}-similarity-plots.png'.format(loader_type))
    loader_type_stats.to_csv('{}-similarity-summary.csv'.format(loader_type), float_format='%.f')
