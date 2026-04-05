"""
Core ML models for Virtual Try-On:

    BodyReconstructionModel         — regresses SMPL shape + pose from an image.
    ConditionalGarmentDrapingModel  — FiLM-conditioned GNN for garment draping.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ===========================================================================
# BodyReconstructionModel
# ===========================================================================

class BodyReconstructionModel(nn.Module):
    """
    Iterative SMPL parameter regressor (inspired by HMR).

    Architecture:
        ResNet-50 encoder → global average pool →
        iterative regression head (n_iter rounds)
        → shape params (10-D) + pose params (72-D)

    Args:
        n_iter: Number of iterative regression steps.
    """

    SHAPE_DIM = 10
    POSE_DIM = 72   # 24 joints × 3 (axis-angle)
    NPOSE = POSE_DIM

    def __init__(self, n_iter: int = 3):
        super().__init__()
        self.n_iter = n_iter

        # --- Encoder ---
        from torchvision.models import resnet50, ResNet50_Weights
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # (B, 2048, 1, 1)
        enc_dim = 2048

        # --- Regression head ---
        param_dim = self.SHAPE_DIM + self.POSE_DIM  # 82
        self.regressor = nn.Sequential(
            nn.Linear(enc_dim + param_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, param_dim),
        )

        # Initial parameter estimate (zero shape, rest pose)
        self.register_buffer(
            "init_params",
            torch.zeros(1, param_dim),
        )

        logger.info(f"BodyReconstructionModel initialised (n_iter={n_iter})")

    def forward(
        self, image: torch.Tensor
    ):
        """
        Args:
            image: (B, 3, 224, 224) float tensor.

        Returns:
            shape_params: (B, 10)
            pose_params:  (B, 72)
        """
        B = image.shape[0]

        # Encode image
        feat = self.encoder(image)          # (B, 2048, 1, 1)
        feat = feat.view(B, -1)             # (B, 2048)

        # Iterative regression
        params = self.init_params.expand(B, -1)  # (B, 82)
        for _ in range(self.n_iter):
            x = torch.cat([feat, params], dim=1)  # (B, 2048+82)
            delta = self.regressor(x)              # (B, 82)
            params = params + delta

        shape_params = params[:, : self.SHAPE_DIM]   # (B, 10)
        pose_params = params[:, self.SHAPE_DIM :]    # (B, 72)
        return shape_params, pose_params


# ===========================================================================
# FiLM conditioning helpers
# ===========================================================================

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation: applies scale γ and shift β to node features,
    both predicted from a conditioning vector.
    """

    def __init__(self, feature_dim: int, cond_dim: int):
        super().__init__()
        self.gamma_net = nn.Linear(cond_dim, feature_dim)
        self.beta_net = nn.Linear(cond_dim, feature_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (N, feature_dim) node features.
            cond: (1, cond_dim)    conditioning vector (broadcast over nodes).
        Returns:
            (N, feature_dim) modulated features.
        """
        gamma = self.gamma_net(cond)  # (1, feature_dim)
        beta = self.beta_net(cond)    # (1, feature_dim)
        return gamma * x + beta


# ===========================================================================
# ConditionalGarmentDrapingModel
# ===========================================================================

class ConditionalGarmentDrapingModel(nn.Module):
    """
    FiLM-conditioned Graph Neural Network for garment draping.

    Each layer applies:
        1. Graph convolution (message passing over edges).
        2. FiLM conditioning with SMPL params + template embedding.
        3. ReLU activation.

    Output is per-node 3-D offset vectors added to template vertices.

    Args:
        node_input_dim: Dimension of input node features (3 for xyz).
        hidden_dim:     Hidden feature dimension.
        cond_dim:       Conditioning vector size (smpl_params + template_emb).
        emb_dim:        Template embedding dimension (part of cond_dim).
        num_layers:     Number of GNN layers.
    """

    def __init__(
        self,
        node_input_dim: int = 3,
        hidden_dim: int = 128,
        cond_dim: int = 98,     # 82 (smpl) + 16 (template emb)
        emb_dim: int = 16,
        num_layers: int = 4,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.emb_dim = emb_dim

        # Input projection
        self.input_proj = nn.Linear(node_input_dim, hidden_dim)

        # GNN layers + FiLM layers
        self.conv_layers = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for _ in range(num_layers):
            # Simple graph conv: aggregate neighbour mean + self transform
            self.conv_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.film_layers.append(FiLMLayer(hidden_dim, cond_dim))
            self.norm_layers.append(nn.LayerNorm(hidden_dim))

        # Output head: predict 3-D offset
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )

        logger.info(
            f"ConditionalGarmentDrapingModel initialised: "
            f"{num_layers} layers, hidden={hidden_dim}, cond={cond_dim}"
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        smpl_params: torch.Tensor,
        template_emb: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_features: (N, 3)   xyz coordinates of template vertices.
            edge_index:    (2, E)   COO edge index (undirected).
            smpl_params:   (1, 82)  concatenated shape + pose parameters.
            template_emb:  (1, 16)  template embedding.
            batch_index:   (N,)     batch assignment for each node.

        Returns:
            offsets: (N, 3) per-vertex displacement vectors.
        """
        # Build conditioning vector (broadcast over all nodes)
        cond = torch.cat([smpl_params, template_emb], dim=1)  # (1, cond_dim)

        # Input projection
        h = F.relu(self.input_proj(node_features))  # (N, hidden_dim)

        N = h.shape[0]
        src, dst = edge_index[0], edge_index[1]

        for conv, film, norm in zip(
            self.conv_layers, self.film_layers, self.norm_layers
        ):
            # --- Message passing: mean aggregation ---
            if src.numel() > 0:
                msg = h[src]                                     # (E, hidden_dim)
                agg = torch.zeros_like(h)
                agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
                # Count neighbours for mean
                count = torch.zeros(N, 1, device=h.device)
                count.scatter_add_(0, dst.unsqueeze(1), torch.ones(src.shape[0], 1, device=h.device))
                count = count.clamp(min=1.0)
                agg = agg / count
            else:
                agg = torch.zeros_like(h)

            # Transform
            h_new = conv(h + agg)              # (N, hidden_dim)

            # FiLM conditioning
            h_new = film(h_new, cond)          # (N, hidden_dim)

            # Residual + LayerNorm
            h = norm(h + F.relu(h_new))        # (N, hidden_dim)

        offsets = self.output_head(h)          # (N, 3)
        return offsets
