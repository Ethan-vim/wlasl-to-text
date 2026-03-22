"""Tests for src.models — ST-GCN encoder, prototypical network, classifier, and unified build_model."""

import pytest
import torch
import torch.nn as nn

from src.models.stgcn import (
    STGCNEncoder,
    build_stgcn_encoder,
    DropPath,
    AttentionPool,
    CrossBranchAttention,
    STGCNBlock,
    STGCNBranch,
    build_spatial_graph,
    HAND_EDGES,
    HAND_NUM_JOINTS,
)
from src.models.prototypical import PrototypicalNetwork
from src.models.classifier import STGCNClassifier, build_classifier
from src.models import build_model
from src.training.config import Config


NUM_KP = 543
B, T = 2, 16


# ---------------------------------------------------------------------------
# STGCNClassifier
# ---------------------------------------------------------------------------


class TestSTGCNClassifier:
    def _make_cfg(self, **overrides):
        """Create a minimal Config for ST-GCN classifier tests."""
        defaults = dict(
            approach="stgcn_ce",
            num_keypoints=NUM_KP,
            num_classes=10,
            wlasl_variant=10,
            d_model=32,
            dropout=0.1,
            head_dropout=0.1,
            T=T,
            use_motion=False,
        )
        defaults.update(overrides)
        return Config(**defaults)

    def test_forward_shape(self):
        """STGCNClassifier output has shape (B, num_classes)."""
        cfg = self._make_cfg()
        encoder = build_stgcn_encoder(cfg)
        model = STGCNClassifier(encoder, num_classes=10, dropout=0.1)
        x = torch.randn(B, T, NUM_KP * 3)
        out = model(x)
        assert out.shape == (B, 10)

    def test_classify_matches_forward(self):
        """classify() returns same as forward()."""
        cfg = self._make_cfg()
        encoder = build_stgcn_encoder(cfg)
        model = STGCNClassifier(encoder, num_classes=10, dropout=0.1)
        model.eval()
        x = torch.randn(B, T, NUM_KP * 3)
        with torch.no_grad():
            out_fwd = model(x)
            out_cls = model.classify(x)
        torch.testing.assert_close(out_fwd, out_cls)

    def test_build_classifier_from_config(self):
        """build_classifier creates model from config."""
        cfg = self._make_cfg()
        model = build_classifier(cfg)
        assert isinstance(model, STGCNClassifier)
        assert model.num_classes == 10
        model.eval()
        x = torch.randn(1, T, NUM_KP * 3)
        out = model(x)
        assert out.shape == (1, 10)

    def test_unified_build_model_ce(self):
        """build_model dispatches to classifier for stgcn_ce."""
        cfg = self._make_cfg(approach="stgcn_ce")
        model = build_model(cfg)
        assert isinstance(model, STGCNClassifier)

    def test_unified_build_model_proto(self):
        """build_model dispatches to prototypical for stgcn_proto."""
        cfg = self._make_cfg(approach="stgcn_proto")
        model = build_model(cfg)
        assert isinstance(model, PrototypicalNetwork)

    def test_model_is_differentiable(self):
        """Classifier model supports backprop."""
        cfg = self._make_cfg()
        model = build_classifier(cfg)
        x = torch.randn(B, T, NUM_KP * 3)
        out = model(x)
        loss = out.sum()
        loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad

    def test_forward_with_motion(self):
        """STGCNClassifier works with motion features (6 channels per keypoint)."""
        cfg = self._make_cfg(use_motion=True)
        model = build_classifier(cfg)
        x = torch.randn(B, T, NUM_KP * 6)
        out = model(x)
        assert out.shape == (B, 10)

    def test_head_has_two_linear_layers(self):
        """Enhanced head should have two Linear layers."""
        cfg = self._make_cfg()
        model = build_classifier(cfg)
        linear_layers = [m for m in model.head if isinstance(m, nn.Linear)]
        assert len(linear_layers) == 2

    def test_head_has_layernorm(self):
        """Classification head should use LayerNorm (not BatchNorm) to avoid mode collapse."""
        cfg = self._make_cfg()
        model = build_classifier(cfg)
        ln_layers = [m for m in model.head if isinstance(m, nn.LayerNorm)]
        assert len(ln_layers) == 1
        bn_layers = [m for m in model.head if isinstance(m, nn.BatchNorm1d)]
        assert len(bn_layers) == 0


# ---------------------------------------------------------------------------
# STGCNEncoder normalize_embeddings
# ---------------------------------------------------------------------------


class TestNormalizeEmbeddings:
    def test_normalized_by_default(self):
        """Default encoder produces L2-normalized outputs."""
        encoder = STGCNEncoder(num_keypoints=NUM_KP, embedding_dim=32)
        encoder.eval()
        x = torch.randn(B, T, NUM_KP * 3)
        with torch.no_grad():
            emb = encoder(x)
        norms = emb.norm(dim=1)
        torch.testing.assert_close(norms, torch.ones(B), atol=1e-5, rtol=1e-5)

    def test_unnormalized_when_disabled(self):
        """Encoder with normalize_embeddings=False does NOT produce unit-norm."""
        encoder = STGCNEncoder(num_keypoints=NUM_KP, embedding_dim=32, normalize_embeddings=False)
        encoder.eval()
        x = torch.randn(B, T, NUM_KP * 3)
        with torch.no_grad():
            emb = encoder(x)
        norms = emb.norm(dim=1)
        # Should NOT all be 1.0
        assert not torch.allclose(norms, torch.ones(B), atol=1e-3)

    def test_build_reads_config_flag(self):
        """build_stgcn_encoder reads normalize_embeddings from config."""
        cfg = Config(approach="stgcn_ce", normalize_embeddings=False, wlasl_variant=10, num_classes=10)
        encoder = build_stgcn_encoder(cfg)
        assert encoder.normalize_embeddings is False

    def test_build_defaults_to_true(self):
        """build_stgcn_encoder defaults normalize_embeddings to True."""
        cfg = Config(approach="stgcn_proto", wlasl_variant=10, num_classes=10)
        encoder = build_stgcn_encoder(cfg)
        assert encoder.normalize_embeddings is True


# ---------------------------------------------------------------------------
# DropPath
# ---------------------------------------------------------------------------


class TestDropPath:
    def test_identity_at_eval(self):
        """DropPath is identity during eval mode."""
        dp = DropPath(drop_prob=0.5)
        dp.eval()
        x = torch.randn(4, 16)
        out = dp(x)
        torch.testing.assert_close(out, x)

    def test_zero_prob_identity(self):
        """DropPath with p=0 is always identity."""
        dp = DropPath(drop_prob=0.0)
        dp.train()
        x = torch.randn(4, 16)
        out = dp(x)
        torch.testing.assert_close(out, x)

    def test_drops_during_train(self):
        """DropPath with high prob should zero some samples during training."""
        dp = DropPath(drop_prob=0.99)
        dp.train()
        torch.manual_seed(42)
        x = torch.ones(100, 16)
        out = dp(x)
        # With p=0.99, most samples should be zeroed
        zero_rows = (out.abs().sum(dim=1) == 0).sum().item()
        assert zero_rows > 50  # at least half should be dropped


# ---------------------------------------------------------------------------
# AttentionPool
# ---------------------------------------------------------------------------


class TestAttentionPool:
    def test_output_shape(self):
        """AttentionPool produces correct (B, C) output."""
        pool = AttentionPool(channels=64)
        x = torch.randn(2, 64, 16, 21)  # (B, C, T, V)
        out = pool(x)
        assert out.shape == (2, 64)

    def test_weights_sum_to_one(self):
        """Attention weights should sum to 1 along temporal dimension."""
        pool = AttentionPool(channels=64)
        x = torch.randn(2, 64, 16, 21)
        # Access internal weights
        x_v = x.mean(dim=3)  # (B, C, T)
        x_t = x_v.permute(0, 2, 1)  # (B, T, C)
        import torch.nn.functional as F
        weights = F.softmax(pool.attn(x_t).squeeze(-1), dim=1)
        sums = weights.sum(dim=1)
        torch.testing.assert_close(sums, torch.ones(2), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# CrossBranchAttention
# ---------------------------------------------------------------------------


class TestCrossBranchAttention:
    def test_output_shape(self):
        """CrossBranchAttention preserves embedding dim."""
        cba = CrossBranchAttention(embed_dim=64, num_heads=4)
        features = [torch.randn(2, 64) for _ in range(3)]
        out = cba(features)
        assert len(out) == 3
        for feat in out:
            assert feat.shape == (2, 64)

    def test_is_differentiable(self):
        """CrossBranchAttention supports backprop."""
        cba = CrossBranchAttention(embed_dim=64, num_heads=4)
        features = [torch.randn(2, 64, requires_grad=True) for _ in range(3)]
        out = cba(features)
        loss = sum(f.sum() for f in out)
        loss.backward()
        for f in features:
            assert f.grad is not None


# ---------------------------------------------------------------------------
# DilatedTCN
# ---------------------------------------------------------------------------


class TestDilatedTCN:
    def test_preserves_temporal_dim(self):
        """Dilated conv doesn't change T dimension."""
        adj = build_spatial_graph(HAND_EDGES, HAND_NUM_JOINTS)
        block = STGCNBlock(64, 64, adj, dilation=4)
        x = torch.randn(2, 64, 16, 21)
        out = block(x)
        assert out.shape[2] == 16  # T preserved


# ---------------------------------------------------------------------------
# JointImportance
# ---------------------------------------------------------------------------


class TestJointImportance:
    def test_learnable(self):
        """Joint importance weights are learnable parameters."""
        adj = build_spatial_graph(HAND_EDGES, HAND_NUM_JOINTS)
        branch = STGCNBranch(3, [32, 64], adj, HAND_NUM_JOINTS)
        assert isinstance(branch.joint_importance, nn.Parameter)
        assert branch.joint_importance.requires_grad


# ---------------------------------------------------------------------------
# Model with 75 keypoints
# ---------------------------------------------------------------------------


class TestModelWith75Keypoints:
    def test_model_75_keypoints(self):
        """Model works with num_keypoints=75."""
        cfg = Config(
            approach="stgcn_ce", num_keypoints=75, wlasl_variant=10,
            num_classes=10, d_model=32, dropout=0.1, head_dropout=0.1,
            use_motion=False,
        )
        model = build_classifier(cfg)
        x = torch.randn(B, T, 75 * 3)
        out = model(x)
        assert out.shape == (B, 10)

    def test_model_75_keypoints_with_motion(self):
        """Model works with num_keypoints=75 and motion features."""
        cfg = Config(
            approach="stgcn_ce", num_keypoints=75, wlasl_variant=10,
            num_classes=10, d_model=32, dropout=0.1, head_dropout=0.1,
            use_motion=True,
        )
        model = build_classifier(cfg)
        x = torch.randn(B, T, 75 * 6)
        out = model(x)
        assert out.shape == (B, 10)


# ---------------------------------------------------------------------------
# New features combined
# ---------------------------------------------------------------------------


class TestNewFeaturesCombined:
    def _make_cfg(self, **overrides):
        defaults = dict(
            approach="stgcn_ce", num_keypoints=75, wlasl_variant=10,
            num_classes=10, d_model=32, dropout=0.1, head_dropout=0.1,
            use_motion=False,
        )
        defaults.update(overrides)
        return Config(**defaults)

    def test_attention_pool_in_model(self):
        """Model builds with attention pooling enabled."""
        cfg = self._make_cfg(use_attention_pool=True)
        model = build_classifier(cfg)
        x = torch.randn(B, T, 75 * 3)
        out = model(x)
        assert out.shape == (B, 10)

    def test_drop_path_in_model(self):
        """Model builds with stochastic depth enabled."""
        cfg = self._make_cfg(drop_path_rate=0.1)
        model = build_classifier(cfg)
        x = torch.randn(B, T, 75 * 3)
        out = model(x)
        assert out.shape == (B, 10)

    def test_cross_attention_in_model(self):
        """Model builds with cross-branch attention enabled."""
        cfg = self._make_cfg(use_cross_attention=True)
        model = build_classifier(cfg)
        x = torch.randn(B, T, 75 * 3)
        out = model(x)
        assert out.shape == (B, 10)

    def test_auxiliary_heads(self):
        """Model with aux_loss_weight > 0 creates auxiliary heads."""
        cfg = self._make_cfg(aux_loss_weight=0.3)
        model = build_classifier(cfg)
        assert hasattr(model, "aux_heads")
        assert len(model.aux_heads) == 3

    def test_forward_with_aux(self):
        """forward_with_aux returns logits and aux logits."""
        cfg = self._make_cfg(aux_loss_weight=0.3)
        model = build_classifier(cfg)
        x = torch.randn(B, T, 75 * 3)
        logits, aux_list = model.forward_with_aux(x)
        assert logits.shape == (B, 10)
        assert len(aux_list) == 3
        for aux in aux_list:
            assert aux.shape == (B, 10)

    def test_forward_with_aux_no_aux_heads(self):
        """forward_with_aux with aux_loss_weight=0 returns empty list."""
        cfg = self._make_cfg(aux_loss_weight=0.0)
        model = build_classifier(cfg)
        x = torch.randn(B, T, 75 * 3)
        logits, aux_list = model.forward_with_aux(x)
        assert logits.shape == (B, 10)
        assert aux_list == []

    def test_all_features_combined(self):
        """Model builds with ALL new features enabled simultaneously."""
        cfg = self._make_cfg(
            use_attention_pool=True,
            drop_path_rate=0.1,
            use_cross_attention=True,
            aux_loss_weight=0.3,
        )
        model = build_classifier(cfg)
        x = torch.randn(B, T, 75 * 3)
        out = model(x)
        assert out.shape == (B, 10)

    def test_forward_with_branches(self):
        """Encoder forward_with_branches returns embedding + branch features."""
        cfg = self._make_cfg()
        encoder = build_stgcn_encoder(cfg)
        x = torch.randn(B, T, 75 * 3)
        emb, branch_feats = encoder.forward_with_branches(x)
        assert emb.shape == (B, 128)  # default embedding_dim
        assert len(branch_feats) == 3
