import torch

def reluSquared(x: Tensor):
    x = torch.relu(x)
    return x.square()

relu(x) = max(0, x)

#key insight of this improvement is that ReLU just return the number if its positive and negative numbers become 0
#by squaring this output be increase the difference between the scores ie 0.1 becomes 0.01 and 0.9 becomes 0.81
#ReLU introduces non linearity like with any other activation function which helps models learn