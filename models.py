"""
Simple models for the app.
Take two inputs and return an output
"""
import torch
from torch import nn

# import ligntning
import lightning as L


class SimpleApp(L.LightningModule):
    """
    Simple app to test torch serve
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128 * 2, output_size)

    def forward(self, inp1, inp2):

        x1 = self.fc1(inp1)
        x2 = self.fc1(inp2)

        x = torch.cat([x1, x2], dim=1)
        x = self.fc2(x)

        return x


model = SimpleApp(10, 1)

example_input1 = torch.rand(32, 10)
example_input2 = torch.rand(32, 10)

# we save those two inputs to use them later (numpy format)
import numpy as np

np.save("example_input1.npy", example_input1.detach().numpy(), allow_pickle=False)
np.save("example_input2.npy", example_input2.detach().numpy(), allow_pickle=False)

# Export the model
scripted_model = model.to_torchscript(
    "model.pt", example_inputs=[example_input1, example_input2], method="trace"
)
