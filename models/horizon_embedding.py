import numpy as np
import random
import torch
import torch.nn as nn

SEED = 37
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

horizon_values = torch.tensor([1, 2, 3, 4, 5, 6, 7]) - 1

embedding_dim = 4
horizon_embedding = nn.Embedding(num_embeddings=7, embedding_dim=embedding_dim)

embedded_horizon = horizon_embedding(horizon_values)
torch.save(horizon_embedding.state_dict(), "data/horizon_embedding_model.pt")
print(embedded_horizon)