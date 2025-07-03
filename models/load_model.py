import torch
from torch import nn
import os

class DeepMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        layer_sizes = [
            1024,
            896,
            768,
            640,
            512,
            384, 384,
            256, 256, 256,
            192, 192,
            128,
            96,
            64,
            32,
            16,
            8,
            1
        ]
        layers = []
        prev_dim = input_dim
        for h in layer_sizes[:-1]:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, layer_sizes[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

input_dim = 25
model_path = os.path.join(os.path.dirname(__file__), "main_model.pth")
loaded_model = DeepMLP(input_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model.load_state_dict(torch.load(model_path, map_location=device))
loaded_model.to(device)
loaded_model.eval()  

print("Model loaded successfully.")
