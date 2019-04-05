import pandas as pd

checkpoint_maps = [ 'imagenet', 'miniimagenet_with_in', 'stylized_miniimagenet_with_in' ]
names = ['original_val_loader', 'stylized_val_loader', 'original_train_loader', 'stylized_train_loader']
stats = [ 'mean', 'std' ]

id_labels = [
    'model',
    'dataset-split',
    'stat'
]

labels = id_labels + list(range(0, 512))

# print(labels)

similarity = pd.DataFrame(columns=labels)

for checkpoint_map in checkpoint_maps:
    for name in names:
        for stat in stats:
            csv_filename = '{}-{}-{}.csv'.format(checkpoint_map, name, stat)
            # print(csv_filename)
            csv_file = pd.read_csv(csv_filename, header=None)
            mean = csv_file.mean(axis=0).to_dict()
            mean['model'] = checkpoint_map
            mean['dataset-split'] = name
            mean['stat'] = stat
            similarity = similarity.append(mean, ignore_index=True)

similarity = similarity.sort_values(by=id_labels)
print(similarity)
similarity.to_csv('similarity.csv')
