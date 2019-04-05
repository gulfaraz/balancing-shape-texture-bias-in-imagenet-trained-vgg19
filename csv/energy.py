import pandas as pd

energy_filename = 'energy.csv'
datasets = [ 'original', 'stylized' ]
image_transforms = [ 'raw', 'blur', 'bilateral' ]
energy_types = [ 'absolute', 'actual' ]
energy_set_labels = [
    'dataset_index',
    'dataset_class',
    'transformation',
    'L5E5', 'L5R5', 'E5S5',
    'S5S5', 'R5R5', 'E5E5',
    'L5S5', 'E5R5', 'S5R5'
]

energy_set = [
    'L5E5', 'L5R5', 'E5S5',
    'S5S5', 'R5R5', 'E5E5',
    'L5S5', 'E5R5', 'S5R5'
]

stats_labels = [
    'dataset',
    'transformation',
    'energy_type',
    'stat',
    'L5E5', 'L5R5', 'E5S5',
    'S5S5', 'R5R5', 'E5E5',
    'L5S5', 'E5R5', 'S5R5'
]

stats = pd.DataFrame(columns=stats_labels)

for dataset in datasets:
	for image_transform in image_transforms:
		csv_filename = '{}-{}.csv'.format(dataset, image_transform)
		csv_file = pd.read_csv(csv_filename, header=None, names=energy_set_labels)
		for energy_type in energy_types:
			filterType = '{}-{}'.format(image_transform, energy_type)
			filtered_csv = csv_file[csv_file.transformation == filterType]
			mean = filtered_csv[energy_set].mean(axis=0).to_dict()
			std = filtered_csv[energy_set].std(axis=0).to_dict()
			mean['stat'] = 'mean'
			std['stat'] = 'std'
			mean['energy_type'] = std['energy_type'] = energy_type
			mean['transformation'] = std['transformation'] = image_transform
			mean['dataset'] = std['dataset'] = dataset
			stats = stats.append(mean, ignore_index=True)
			stats = stats.append(std, ignore_index=True)

stats = stats.sort_values(by=['energy_type', 'transformation', 'stat', 'dataset'])

stats.to_csv(energy_filename)

# summarize

stats = pd.read_csv(energy_filename)

for energy_type in energy_types:
    energy_type_stats = pd.DataFrame()
    for image_transform in image_transforms:
        print('{} {}'.format(energy_type, image_transform))
        filtered_stats = stats[(stats.energy_type == energy_type) & (stats.transformation == image_transform)][energy_set].transpose()
        diff_stats = filtered_stats.diff(axis=1)
        diff_mean = diff_stats[diff_stats.columns[1]]
        diff_std = diff_stats[diff_stats.columns[3]]
        original_mean = filtered_stats[filtered_stats.columns[0]]
        original_std = filtered_stats[filtered_stats.columns[2]]
        changed_mean = 100 * diff_mean / original_mean
        changed_std = 100 * diff_std / original_std
        stat = pd.concat([changed_mean, changed_std], axis=1)
        stat.columns = ['{}-mean'.format(image_transform), '{}-std'.format(image_transform)]
        energy_type_stats = pd.concat([energy_type_stats, stat], axis=1)
    energy_type_stats.to_csv('{}-energy-summary.csv'.format(energy_type), float_format='%.f')

