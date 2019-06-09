import torch

class InstanceNormBatchSwap(torch.nn.Module):
    def __init__(self, n_neurons, affine=False, eps=1e-5):
        super(InstanceNormBatchSwap, self).__init__()
        self.n_neurons = n_neurons
        self.eps = eps
        assert affine == False, 'affine parameters not implemented'

    def forward(self, input):
        assert input.shape[1] == self.n_neurons, "Input has incorrect shape"

        if not self.training:
            return input

        temp = input.view(input.size(0), input.size(1), -1)
        mean = temp.mean(2, keepdim=True).unsqueeze(-1)
        std = temp.std(2, keepdim=True).unsqueeze(-1)
        den = torch.sqrt(std.pow(2) + self.eps)
        output = (input - mean)/den
        indices = torch.randperm(input.size(0))
        output = output * std.index_select(0, indices) + mean.index_select(0, indices)

        return output
    
    def __repr__(self):
        return 'InstanceNormBatchSwap({}, eps={})'.format(self.n_neurons, self.eps)


class InstanceNormSimilarity(torch.nn.Module):
    def __init__(self, n_neurons, affine=False, eps=1e-5, filename=None):
        super(InstanceNormSimilarity, self).__init__()
        self.n_neurons = n_neurons
        self.eps = eps
        self.filename = filename
        assert affine == False, 'affine parameters not implemented'

    def forward(self, input):
        assert input.shape[1] == self.n_neurons, "Input has incorrect shape"
        torch.set_printoptions(profile='full')
        mean_file = open('{}-mean.csv'.format(self.filename), 'a', input.size(0))
        std_file = open('{}-std.csv'.format(self.filename), 'a', input.size(0))
        temp = input.view(input.size(0), input.size(1), -1)
        mean = temp.mean(2)
        std = temp.std(2)
        for line in mean.cpu().numpy():
            mean_file.write('{}\n'.format(', '.join(map(str, line))))
        for line in std.cpu().numpy():
            std_file.write('{}\n'.format(', '.join(map(str, line))))
        # print('mean {} std {}'.format(mean.shape, std.shape))
        # print('mean {} std {}'.format(mean, std))
        mean_file.close()
        std_file.close()
        torch.set_printoptions(profile='default')

        return input
    
    def __repr__(self):
        return 'InstanceNormSimilarity({}, eps={})'.format(self.n_neurons, self.eps)
