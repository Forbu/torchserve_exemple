"""
torchserve handler for pytorch model
"""
import os
import numpy as np
from io import BytesIO

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Any
from ts.torch_handler.base_handler import BaseHandler


class SimpleHandler(BaseHandler):
    """
    Class for handling the model
    """

    

    def preprocess(self, data):
        """
        Function to preprocess the data
        The data have two arrays inp1, inp2
        """
        print(data)

        # preprocess the data
        inp1 = data[0].get("inp1")
        inp2 = data[0].get("inp2")

        # convert the data to numpy
        inp1 = bytes(inp1)
        inp1 = BytesIO(inp1)
        inp1 = np.load(inp1)

        # same for inp2
        inp2 = bytes(inp2)
        inp2 = BytesIO(inp2)
        inp2 = np.load(inp2)

        print(inp1.shape)
        print(inp2.shape)

        return {"inp1": inp1, "inp2": inp2}

    def inference(self, data):
        """
        Model inference
        """

        # Get the data
        inp1 = data["inp1"]
        inp2 = data["inp2"]

        # Convert the data to tensor
        inp1 = torch.tensor(inp1)
        inp2 = torch.tensor(inp2)

        # Get the output
        output = self.model(inp1, inp2)

        return output

    def handle(self, data, context):
        """
        handle the data
        """
        data = self.preprocess(data)
        output = self.inference(data)
        return self.postprocess(output)

    def postprocess(self, data):
        """
        Function to postprocess the data
        """
        # Convert the data to numpy
        data = data.detach().numpy()

        return data
