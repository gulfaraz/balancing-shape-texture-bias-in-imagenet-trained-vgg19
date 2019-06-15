import numpy as np
import torch
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm

dataset = torch.rand(2000, 128)
colors = cm.rainbow(np.linspace(0, 1, 2000))

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(dataset)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

print(tsne_results)
x = tsne_results[:, 0]
y = tsne_results[:, 1]
plt.scatter(x, y, color=colors)
plt.savefig('manifold.png')
