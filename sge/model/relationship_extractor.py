import torch
import torch.nn as nn
from einops import rearrange


class RelationshipFeatureExtractor(nn.Module):
    def __init__(self, in_dim, out_dim=256, num_layers=4):
        super().__init__()

        self.in_layer = nn.Linear(in_dim, out_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=out_dim, nhead=4)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, features: torch.Tensor):
        """
        Parameters
        ----------
        features: torch.Tensor
            The input features. The shape is (N, dim).

        Returns
        -------
        relationship_features: torch.Tensor
            The extracted relationship features. The shape is (N, dim).
        """

        assert features.ndim == 2
        x = self.in_layer(features)  # (N, dim)
        x = x[None, :, :]  # (1, N, dim)
        x = self.encoder(x)
        relationship_features = x[0]
        
        return relationship_features


class RelationshipExtractor(nn.Module):
    def __init__(self, n_objcat, n_relcat, in_dim=256):
        super().__init__()

        self.obj_classifier = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, n_objcat)
        )

        self.rel_s_proj = nn.Linear(in_dim, in_dim)
        self.rel_o_proj = nn.Linear(in_dim, in_dim)
        self.rel_classifier = nn.Sequential(
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, n_relcat)
        )

    def forward(self, features: torch.Tensor):
        object_classes = self.obj_classifier(features)
        relationship_classes = self.rel_classifier(
            self.rel_s_proj(features)[:, None, :] + \
            self.rel_o_proj(features)[None, :, :]
        )

        return object_classes, relationship_classes