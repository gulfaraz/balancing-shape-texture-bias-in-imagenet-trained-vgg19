import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipystat

similarity_filename = 'similarity.csv'
checkpoint_maps = [ 'imagenet', 'imagenet200_with_in', 'stylized_imagenet200_with_in' ]
names = ['nonstylized_val_loader', 'stylized_val_loader', 'nonstylized_train_loader', 'stylized_train_loader']
stats = [ 'mean', 'std' ]

id_labels = ['model', 'dataset-split', 'stat']

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

dataset_types = [ 'nonstylized', 'stylized' ]
loader_types = [ 'val', 'train' ]
loader_string_format = '{}_{}_loader'

for loader_type in loader_types:
    # print(loader_type)
    i = 1
    plt.figure(figsize=(20, 8))
    loader_type_stats = pd.DataFrame()
    in_nonstylized_mean = None
    in_nonstylized_std = None
    in200_nonstylized_mean = None
    in200_nonstylized_std = None
    in200_stylized_mean = None
    in200_stylized_std = None
    for checkpoint_map in checkpoint_maps:
        # print(checkpoint_map)
        nonstylized_loader = loader_string_format.format('nonstylized', loader_type)
        stylized_loader = loader_string_format.format('stylized', loader_type)
        nonstylized_stats = stats[(stats.model == checkpoint_map) & (stats['dataset-split'] == nonstylized_loader)][list(map(str, activation_labels))].transpose()
        stylized_stats = stats[(stats.model == checkpoint_map) & (stats['dataset-split'] == stylized_loader)][list(map(str, activation_labels))].transpose()
        nonstylized_mean_column = nonstylized_stats.columns[0]
        nonstylized_std_column = nonstylized_stats.columns[1]
        stylized_mean_column = stylized_stats.columns[0]
        stylized_std_column = stylized_stats.columns[1]
        nonstylized_mean = nonstylized_stats[nonstylized_mean_column]
        stylized_mean = stylized_stats[stylized_mean_column]
        nonstylized_std = nonstylized_stats[nonstylized_std_column]
        stylized_std = stylized_stats[stylized_std_column]
        nonstylized_row = pd.DataFrame([checkpoint_map, 'nonstylized', nonstylized_mean.mean(), nonstylized_std.mean()]).transpose()
        stylized_row = pd.DataFrame([checkpoint_map, 'stylized', stylized_mean.mean(), stylized_std.mean()]).transpose()
        nonstylized_row.columns = ['model', 'dataset', 'mean', 'std']
        stylized_row.columns = ['model', 'dataset', 'mean', 'std']
        loader_type_stats = pd.concat([loader_type_stats, nonstylized_row, stylized_row])
        if checkpoint_map == 'imagenet':
            in_nonstylized_mean = nonstylized_mean
            in_nonstylized_std = nonstylized_std
        elif checkpoint_map == 'imagenet200_with_in':
            in200_nonstylized_mean = nonstylized_mean
            in200_nonstylized_std = nonstylized_std
        elif checkpoint_map == 'stylized_imagenet200_with_in':
            in200_stylized_mean = nonstylized_mean
            in200_stylized_std = nonstylized_std
        plt.subplot(2, 3, i)
        hist = nonstylized_mean.plot.hist(bins=10, alpha=0.5)
        hist = stylized_mean.plot.hist(bins=10, alpha=0.5)
        plt.title('{} {} mean'.format(loader_type, checkpoint_map))
        plt.subplot(2, 3, i+3)
        hist = nonstylized_std.plot.hist(bins=10, alpha=0.5)
        hist = stylized_std.plot.hist(bins=10, alpha=0.5)
        plt.title('{} {} std'.format(loader_type, checkpoint_map))
        i+=1
    print('imagenet vs imagenet200 - mean')
    print(scipystat.ttest_ind(in_nonstylized_mean, in200_nonstylized_mean))
    print('imagenet vs imagenet200 - std')
    print(scipystat.ttest_ind(in_nonstylized_mean, in200_stylized_mean))
    print('imagenet vs stylized imagenet200 - mean')
    print(scipystat.ttest_ind(in_nonstylized_std, in200_nonstylized_std))
    print('imagenet vs stylized imagenet200 - std')
    print(scipystat.ttest_ind(in_nonstylized_std, in200_stylized_std))
    plt.savefig('{}-similarity-plots.png'.format(loader_type))
    loader_type_stats.to_csv('{}-similarity-summary.csv'.format(loader_type), float_format='%.f')
