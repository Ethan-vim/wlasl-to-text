"""Tests for src.training.config — Config dataclass, load/save."""

from pathlib import Path

import pytest
import yaml

from src.training.config import Config, load_config, save_config


class TestConfigDefaults:
    def test_default_values(self):
        cfg = Config()
        assert cfg.approach == "stgcn_ce"
        assert cfg.num_keypoints == 543
        assert cfg.T == 64
        assert cfg.num_classes == 100

    def test_wlasl_variant_derives_num_classes(self):
        cfg = Config(wlasl_variant=300)
        assert cfg.num_classes == 300

    def test_custom_num_classes_overridden_by_variant(self):
        # __post_init__ overrides num_classes based on wlasl_variant
        cfg = Config(wlasl_variant=100, num_classes=999)
        assert cfg.num_classes == 100

    def test_variant_scales_model_architecture(self):
        cfg100 = Config(wlasl_variant=100)
        assert cfg100.d_model == 128
        assert cfg100.nhead == 4
        assert cfg100.num_layers == 2
        assert cfg100.dropout == 0.1

        cfg300 = Config(wlasl_variant=300)
        assert cfg300.d_model == 192
        assert cfg300.nhead == 6
        assert cfg300.num_layers == 4
        assert cfg300.dropout == 0.3

        cfg1000 = Config(wlasl_variant=1000)
        assert cfg1000.d_model == 256
        assert cfg1000.nhead == 8
        assert cfg1000.num_layers == 5
        assert cfg1000.dropout == 0.4

        cfg2000 = Config(wlasl_variant=2000)
        assert cfg2000.d_model == 384
        assert cfg2000.nhead == 8
        assert cfg2000.num_layers == 6
        assert cfg2000.dropout == 0.5

    def test_new_fields_exist(self):
        cfg = Config()
        assert hasattr(cfg, "use_motion")
        assert hasattr(cfg, "mixup_alpha")
        assert hasattr(cfg, "use_tta")

    def test_use_motion_default(self):
        cfg = Config()
        assert cfg.use_motion is True

    def test_mixup_alpha_default(self):
        cfg = Config()
        assert cfg.mixup_alpha == 0.2

    def test_use_tta_default(self):
        cfg = Config()
        assert cfg.use_tta is False

    def test_normalize_embeddings_default(self):
        cfg = Config()
        assert cfg.normalize_embeddings is True

    def test_new_accuracy_config_fields(self):
        """Config has new accuracy improvement fields with correct defaults."""
        cfg = Config()
        assert cfg.use_attention_pool is False
        assert cfg.drop_path_rate == 0.0
        assert cfg.use_cross_attention is False
        assert cfg.aux_loss_weight == 0.0


class TestLoadConfig:
    def test_load_minimal(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text("T: 32\n")
        cfg = load_config(p)
        assert cfg.T == 32

    def test_unknown_key_ignored(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text("T: 32\nnonsense_key: 42\n")
        cfg = load_config(p)
        assert cfg.T == 32
        assert not hasattr(cfg, "nonsense_key")

    def test_load_full_yaml(self, tmp_config_yaml):
        cfg = load_config(tmp_config_yaml)
        assert cfg.approach == "stgcn_ce"
        assert cfg.T == 16
        assert cfg.d_model == 64

    def test_load_new_fields(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text("use_motion: true\nmixup_alpha: 0.3\nuse_tta: true\n")
        cfg = load_config(p)
        assert cfg.use_motion is True
        assert cfg.mixup_alpha == 0.3
        assert cfg.use_tta is True


class TestSaveConfig:
    def test_roundtrip(self, tmp_path):
        # wlasl_variant=300 → __post_init__ sets d_model=192; verify roundtrip preserves it
        cfg = Config(T=16, wlasl_variant=300, use_motion=True, mixup_alpha=0.5)
        out = tmp_path / "saved.yaml"
        save_config(cfg, out)
        loaded = load_config(out)
        assert loaded.T == 16
        assert loaded.d_model == 192
        assert loaded.use_motion is True
        assert loaded.mixup_alpha == 0.5

    def test_file_created(self, tmp_path):
        cfg = Config()
        out = tmp_path / "subdir" / "config.yaml"
        save_config(cfg, out)
        assert out.exists()

    def test_yaml_readable(self, tmp_path):
        cfg = Config()
        out = tmp_path / "config.yaml"
        save_config(cfg, out)
        with open(out) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert "approach" in data
