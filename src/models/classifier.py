"""ST-GCN encoder + linear classification head for standard cross-entropy training."""
import logging
import torch
import torch.nn as nn
from src.models.stgcn import STGCNEncoder, build_stgcn_encoder

logger = logging.getLogger(__name__)


class STGCNClassifier(nn.Module):
    def __init__(
        self,
        encoder: STGCNEncoder,
        num_classes: int,
        dropout: float = 0.3,
        aux_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.aux_loss_weight = aux_loss_weight
        self.head = nn.Sequential(
            nn.Linear(encoder.embedding_dim, encoder.embedding_dim),
            nn.LayerNorm(encoder.embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder.embedding_dim, num_classes),
        )

        # Auxiliary heads for each branch (body, left hand, right hand)
        if aux_loss_weight > 0:
            branch_dim = encoder.projection[0].in_features // 3
            self.aux_heads = nn.ModuleList([
                nn.Linear(branch_dim, num_classes)
                for _ in range(3)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, F) -> (B, num_classes) logits"""
        emb = self.encoder(x)
        return self.head(emb)

    def forward_with_aux(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass returning both main logits and auxiliary branch logits.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, T, F)``.

        Returns
        -------
        tuple[torch.Tensor, list[torch.Tensor]]
            ``(logits, aux_logits_list)`` where ``logits`` has shape
            ``(B, num_classes)`` and ``aux_logits_list`` contains one
            ``(B, num_classes)`` tensor per branch. The list is empty when
            auxiliary heads are not configured.
        """
        embedding, branch_feats = self.encoder.forward_with_branches(x)
        logits = self.head(embedding)
        if self.aux_loss_weight > 0 and hasattr(self, "aux_heads"):
            aux_logits = [head(feat) for head, feat in zip(self.aux_heads, branch_feats)]
            return logits, aux_logits
        return logits, []

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """API compatibility with PrototypicalNetwork.classify()"""
        return self.forward(x)


def build_classifier(cfg) -> STGCNClassifier:
    encoder = build_stgcn_encoder(cfg)
    model = STGCNClassifier(
        encoder=encoder,
        num_classes=cfg.num_classes,
        dropout=getattr(cfg, "head_dropout", 0.3),
        aux_loss_weight=getattr(cfg, "aux_loss_weight", 0.0),
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("STGCNClassifier: %.2fM trainable parameters", total_params / 1e6)
    return model
