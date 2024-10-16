import torch

class Network(torch.nn.Module):

    def __init__(self, input_dimension, output_dimension, hidden_size=256, output_activation=torch.nn.Identity()):
        super(Network, self).__init__()
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=hidden_size)
        self.layer_2 = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.output_layer = torch.nn.Linear(in_features=hidden_size, out_features=output_dimension)
        self.output_activation = output_activation

    def forward(self, inpt):
        layer_1_output = torch.nn.functional.relu(self.layer_1(inpt))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_activation(self.output_layer(layer_2_output))
        return output
