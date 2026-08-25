from dataclasses import dataclass

import torch


@dataclass
class AnchorChangeEvidence:
    """Reserved interface for later automatic change detection."""

    visible_count: torch.Tensor
    geometry_score: torch.Tensor = None
    feature_score: torch.Tensor = None
    photometric_score: torch.Tensor = None
    positive_votes: torch.Tensor = None
    negative_votes: torch.Tensor = None
    confidence: torch.Tensor = None
