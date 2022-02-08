# -*- coding: utf-8 -*-
from torch import nn

# Target
# I/P = 784, H1_layer = 128, H2_Layer = 64, O/P Layer = 10

input_size = 784
hidden_1_layer = 128
hidden_2_layer = 64
output_layer = 10

model = nn.Sequential(nn.Linear(input_size, hidden_1_layer), 
                      nn.ReLU(),
                      nn.Linear(hidden_1_layer, hidden_2_layer),
                      nn.ReLU(),
                      nn.Linear(hidden_2_layer, output_layer),
                      nn.Softmax(dim=1))

print(model)

