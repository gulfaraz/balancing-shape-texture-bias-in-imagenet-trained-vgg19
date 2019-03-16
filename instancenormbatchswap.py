import torch

class InstanceNormBatchSwap(torch.nn.Module):
    def __init__(self, n_neurons, affine=False, eps=1e-5):
        super(InstanceNormBatchSwap, self).__init__()
        self.n_neurons = n_neurons
        self.eps = eps
        assert affine == False, 'affine parameters not implemented'

    def forward(self, input):
        assert input.shape[1] == self.n_neurons, "Input has incorrect shape"

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

