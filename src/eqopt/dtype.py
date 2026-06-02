import torch


DEFAULT_TYPE = torch.float64
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")