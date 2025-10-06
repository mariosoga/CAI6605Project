import timm
from torch import nn


class MultiTaskEffNet(nn.Module):
    def __init__(self, model_name: str, num_s: int, num_d: int, dropout: float = 0.0, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        feat_dim = self.backbone.num_features
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.species_head = nn.Linear(feat_dim, num_s)
        self.health_head = nn.Linear(feat_dim, 2)
        self.disease_head = nn.Linear(feat_dim, num_d)

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.dropout(feats)
        species_logits = self.species_head(feats)
        health_logits = self.health_head(feats)
        disease_logits = self.disease_head(feats)
        return species_logits, health_logits, disease_logits
