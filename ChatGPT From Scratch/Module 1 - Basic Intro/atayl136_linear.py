import torch

'''
Complete this class by instantiating parameters called "self.weight" and "self.bias", and
use them to complete the forward() method. You do not need to worry about backpropogation.
'''
class CustomLinear(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        # -------------------------------
        # Create learnable parameters
        # -------------------------------
        # Weight matrix shape: (output_size, input_size)
        # Each row corresponds to one "neuron" / output dimension.
        # We initialize with a uniform distribution in a small range
        # to keep activations stable at the start.
        limit = (1.0 / input_size) ** 0.5
        self.weight = torch.nn.Parameter(
            torch.empty(output_size, input_size).uniform_(-limit, limit)
        )

        # Bias vector shape: (output_size,)
        # One bias term per output dimension.
        self.bias = torch.nn.Parameter(
            torch.empty(output_size).uniform_(-limit, limit)
        )

    def forward(self, x):
        '''
        x is a tensor containing a batch of vectors, size (B, input_size).

        Steps:
        1. Multiply the input by the weight matrix:
             (B, input_size) @ (input_size, output_size)^T -> (B, output_size)
        2. Add the bias vector (across the batch).
        3. Return the resulting (B, output_size) tensor.
        '''
        return x @ self.weight.t() + self.bias
