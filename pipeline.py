import torch
import torch.nn as nn
import numpy as np
import joblib
from features.transform import transform_input

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        layer_sizes = [
            1024, 896, 768, 640, 512, 384, 384,
            256, 256, 256, 192, 192, 128, 96,
            64, 32, 16, 8, 1
        ]
        layers = []
        prev_dim = 25
        for h in layer_sizes[:-1]:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, layer_sizes[-1]))
        self.net = nn.Sequential(*layers)  # <- use self.net here

    def forward(self, x):
        return self.net(x)



class NewsToPricePipeline:
    def __init__(self):
        # Load scalers
        self.scaler_x = joblib.load("data/scaler_x.pkl")
        self.scaler_y = joblib.load("data/scaler_y.pkl")

        # Load model
        self.model = MLP()
        self.model.load_state_dict(torch.load("models/main_model.pth", map_location=torch.device("cpu")))
        self.model.eval()

    def predict(self, judul, isi, tanggal, horizon):
        # Step 1: Extract features from raw input
        features = transform_input(judul, isi, tanggal, horizon)

        # Step 2: Arrange features in correct order
        feature_vector = np.array([features[k] for k in self.scaler_x.feature_names_in_], dtype=np.float32)

        # Step 3: Normalize input
        x_scaled = self.scaler_x.transform([feature_vector])
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

        # Step 4: Predict and inverse transform
        with torch.no_grad():
            y_scaled = self.model(x_tensor).numpy()
        y_pred = self.scaler_y.inverse_transform(y_scaled)[0][0]
        return y_pred

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


