number_of_epochs = 50
anneal_start = 30
anneal_width = number_of_epochs - anneal_start

max_beta = 0.9
min_beta = 0.0

beta = min_beta

for i in range(1, number_of_epochs+1):
    print(i, beta)
    if i > anneal_start:
        beta += (max_beta - beta) / (anneal_width * 0.4)
