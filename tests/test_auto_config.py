"""Tests for scripts/auto_config.py — hardware detection and config generation."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# Import the module under test
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import auto_config  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cuda_hw():
    """Simulated CUDA hardware (mid-tier GPU)."""
    return auto_config.HardwareInfo(
        device="cuda",
        device_name="NVIDIA RTX 3070",
        vram_gb=8.0,
        cuda_version="12.1",
        cpu_cores=8,
        platform_name="Linux-6.5.0-x86_64",
        torch_version="2.2.0",
        gpu_count=1,
    )


@pytest.fixture
def high_cuda_hw():
    """Simulated high-tier CUDA hardware."""
    return auto_config.HardwareInfo(
        device="cuda",
        device_name="NVIDIA A100",
        vram_gb=40.0,
        cuda_version="12.1",
        cpu_cores=32,
        platform_name="Linux-6.5.0-x86_64",
        torch_version="2.2.0",
        gpu_count=4,
    )


@pytest.fixture
def low_cuda_hw():
    """Simulated low-tier CUDA hardware."""
    return auto_config.HardwareInfo(
        device="cuda",
        device_name="NVIDIA GTX 1650",
        vram_gb=4.0,
        cuda_version="11.8",
        cpu_cores=4,
        platform_name="Windows-10-x86_64",
        torch_version="2.1.0",
        gpu_count=1,
    )


@pytest.fixture
def mps_hw():
    """Simulated Apple Silicon hardware."""
    return auto_config.HardwareInfo(
        device="mps",
        device_name="Apple arm64",
        vram_gb=0.0,
        cuda_version="",
        cpu_cores=10,
        platform_name="macOS-14.0-arm64",
        torch_version="2.2.0",
        gpu_count=0,
    )


@pytest.fixture
def cpu_hw():
    """Simulated CPU-only hardware."""
    return auto_config.HardwareInfo(
        device="cpu",
        device_name="Intel i7",
        vram_gb=0.0,
        cuda_version="",
        cpu_cores=8,
        platform_name="Linux-6.5.0-x86_64",
        torch_version="2.2.0",
        gpu_count=0,
    )


# ---------------------------------------------------------------------------
# TestDetermineTier
# ---------------------------------------------------------------------------


class TestDetermineTier:
    def test_high_tier(self, high_cuda_hw):
        assert auto_config.determine_tier(high_cuda_hw) == "high"

    def test_mid_tier(self, cuda_hw):
        assert auto_config.determine_tier(cuda_hw) == "mid"

    def test_low_tier(self, low_cuda_hw):
        assert auto_config.determine_tier(low_cuda_hw) == "low"

    def test_mps_is_cpu_tier(self, mps_hw):
        assert auto_config.determine_tier(mps_hw) == "cpu"

    def test_cpu_is_cpu_tier(self, cpu_hw):
        assert auto_config.determine_tier(cpu_hw) == "cpu"

    def test_boundary_16gb(self):
        hw = auto_config.HardwareInfo(
            device="cuda", device_name="GPU", vram_gb=16.0,
            cuda_version="12.1", cpu_cores=8, platform_name="Linux",
            torch_version="2.2.0", gpu_count=1,
        )
        assert auto_config.determine_tier(hw) == "high"

    def test_boundary_8gb(self):
        hw = auto_config.HardwareInfo(
            device="cuda", device_name="GPU", vram_gb=8.0,
            cuda_version="12.1", cpu_cores=8, platform_name="Linux",
            torch_version="2.2.0", gpu_count=1,
        )
        assert auto_config.determine_tier(hw) == "mid"

    def test_boundary_below_4gb(self):
        hw = auto_config.HardwareInfo(
            device="cuda", device_name="GPU", vram_gb=3.5,
            cuda_version="11.8", cpu_cores=4, platform_name="Linux",
            torch_version="2.1.0", gpu_count=1,
        )
        assert auto_config.determine_tier(hw) == "low"


# ---------------------------------------------------------------------------
# TestBuildConfigValues
# ---------------------------------------------------------------------------


class TestBuildConfigValues:
    def test_stgcn_ce_mid(self, cuda_hw):
        cfg = auto_config.build_config_values("stgcn_ce", 100, "mid", cuda_hw)
        assert cfg["approach"] == "stgcn_ce"
        assert cfg["batch_size"] == 32
        assert cfg["T"] == 64
        assert cfg["fp16"] is True
        assert cfg["wlasl_variant"] == 100
        assert cfg["d_model"] == 128
        assert cfg["label_smoothing"] == 0.1
        assert cfg["mixup_alpha"] == 0.2
        assert cfg["head_dropout"] == 0.2

    def test_stgcn_ce_cpu(self, cpu_hw):
        cfg = auto_config.build_config_values("stgcn_ce", 100, "cpu", cpu_hw)
        assert cfg["batch_size"] == 8
        assert cfg["fp16"] is False
        assert cfg["num_workers"] == 2

    def test_stgcn_ce_high(self, high_cuda_hw):
        cfg = auto_config.build_config_values("stgcn_ce", 300, "high", high_cuda_hw)
        assert cfg["batch_size"] == 64
        assert cfg["approach"] == "stgcn_ce"
        assert cfg["wlasl_variant"] == 300

    def test_stgcn_proto_mid(self, cuda_hw):
        cfg = auto_config.build_config_values("stgcn_proto", 100, "mid", cuda_hw)
        assert cfg["approach"] == "stgcn_proto"
        assert cfg["batch_size"] == 32
        assert cfg["fp16"] is True
        assert cfg["n_way"] == 10
        assert cfg["k_shot"] == 3
        assert cfg["q_query"] == 2
        assert cfg["num_episodes"] == 200

    def test_stgcn_proto_cpu(self, cpu_hw):
        cfg = auto_config.build_config_values("stgcn_proto", 100, "cpu", cpu_hw)
        assert cfg["batch_size"] == 8
        assert cfg["fp16"] is False
        assert cfg["num_workers"] == 2

    def test_stgcn_proto_mps(self, mps_hw):
        cfg = auto_config.build_config_values("stgcn_proto", 100, "cpu", mps_hw)
        assert cfg["num_workers"] == 0

    def test_num_workers_cuda(self, cuda_hw):
        cfg = auto_config.build_config_values("stgcn_ce", 100, "mid", cuda_hw)
        assert cfg["num_workers"] == min(8, cuda_hw.cpu_cores)

    def test_num_workers_cpu(self, cpu_hw):
        cfg = auto_config.build_config_values("stgcn_ce", 100, "cpu", cpu_hw)
        assert cfg["num_workers"] == min(2, cpu_hw.cpu_cores)

    def test_buffer_size_matches_t(self, cuda_hw):
        cfg = auto_config.build_config_values("stgcn_ce", 100, "mid", cuda_hw)
        assert cfg["buffer_size"] == cfg["T"]


# ---------------------------------------------------------------------------
# TestRenderYaml
# ---------------------------------------------------------------------------


class TestRenderYaml:
    def test_stgcn_ce_valid_yaml(self, cuda_hw):
        values = auto_config.build_config_values("stgcn_ce", 100, "mid", cuda_hw)
        content = auto_config.render_yaml("stgcn_ce", values, cuda_hw, "mid")
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict)
        assert parsed["approach"] == "stgcn_ce"
        assert parsed["label_smoothing"] == 0.1

    def test_stgcn_proto_valid_yaml(self, cuda_hw):
        values = auto_config.build_config_values("stgcn_proto", 100, "mid", cuda_hw)
        content = auto_config.render_yaml("stgcn_proto", values, cuda_hw, "mid")
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict)
        assert parsed["approach"] == "stgcn_proto"
        assert parsed["n_way"] == 10

    def test_header_present(self, cuda_hw):
        values = auto_config.build_config_values("stgcn_ce", 100, "mid", cuda_hw)
        content = auto_config.render_yaml("stgcn_ce", values, cuda_hw, "mid")
        assert "Auto-generated by scripts/auto_config.py" in content
        assert "NVIDIA RTX 3070" in content
        assert "Tier: mid" in content

    def test_fp16_false_in_cpu_yaml(self, cpu_hw):
        values = auto_config.build_config_values("stgcn_ce", 100, "cpu", cpu_hw)
        content = auto_config.render_yaml("stgcn_ce", values, cpu_hw, "cpu")
        parsed = yaml.safe_load(content)
        assert parsed["fp16"] is False


# ---------------------------------------------------------------------------
# TestDetectHardware
# ---------------------------------------------------------------------------


class TestDetectHardware:
    def test_device_override_cpu(self):
        hw = auto_config.detect_hardware(device_override="cpu")
        assert hw.device == "cpu"
        assert hw.cpu_cores > 0
        assert hw.torch_version != ""

    def test_returns_hardware_info(self):
        hw = auto_config.detect_hardware()
        assert isinstance(hw, auto_config.HardwareInfo)
        assert hw.device in ("cuda", "mps", "cpu")
        assert hw.cpu_cores > 0


# ---------------------------------------------------------------------------
# TestMainDryRun (integration)
# ---------------------------------------------------------------------------


class TestMainDryRun:
    def test_dry_run_stgcn_ce(self):
        result = subprocess.run(
            [sys.executable, "scripts/auto_config.py", "--approach", "stgcn_ce", "--dry-run"],
            capture_output=True,
            text=True,
            cwd=str(auto_config.PROJECT_ROOT),
        )
        assert result.returncode == 0
        assert "WLASL Auto-Config" in result.stdout
        assert "approach: stgcn_ce" in result.stdout

    def test_dry_run_stgcn_proto(self):
        result = subprocess.run(
            [sys.executable, "scripts/auto_config.py",
             "--approach", "stgcn_proto", "--variant", "300", "--device", "cpu", "--dry-run"],
            capture_output=True,
            text=True,
            cwd=str(auto_config.PROJECT_ROOT),
        )
        assert result.returncode == 0
        assert "approach: stgcn_proto" in result.stdout
        assert "wlasl_variant: 300" in result.stdout

    def test_write_and_load(self, tmp_path):
        out = tmp_path / "test.yaml"
        result = subprocess.run(
            [sys.executable, "scripts/auto_config.py",
             "--approach", "stgcn_ce", "--output", str(out)],
            capture_output=True,
            text=True,
            cwd=str(auto_config.PROJECT_ROOT),
        )
        assert result.returncode == 0
        assert out.exists()
        parsed = yaml.safe_load(out.read_text())
        assert parsed["approach"] == "stgcn_ce"


# ---------------------------------------------------------------------------
# TestCEConfigDefaults
# ---------------------------------------------------------------------------


class TestCEConfigDefaults:
    """Verify that the Config dataclass exposes CE-specific fields."""

    def test_default_approach_is_stgcn_ce(self):
        from src.training.config import Config
        cfg = Config()
        assert cfg.approach == "stgcn_ce"

    def test_ce_fields_exist(self):
        from src.training.config import Config
        cfg = Config()
        assert hasattr(cfg, "label_smoothing")
        assert hasattr(cfg, "mixup_alpha")
        assert hasattr(cfg, "head_dropout")

    def test_ce_field_defaults(self):
        from src.training.config import Config
        cfg = Config(approach="stgcn_ce")
        assert isinstance(cfg.label_smoothing, float)
        assert isinstance(cfg.mixup_alpha, float)
        assert isinstance(cfg.head_dropout, float)

    def test_ce_config_from_yaml(self, tmp_path):
        """A CE YAML config round-trips through load_config."""
        from src.training.config import load_config
        cfg_path = tmp_path / "ce.yaml"
        cfg_path.write_text(
            "approach: stgcn_ce\n"
            "wlasl_variant: 100\n"
            "label_smoothing: 0.1\n"
            "mixup_alpha: 0.2\n"
            "head_dropout: 0.3\n"
        )
        cfg = load_config(cfg_path)
        assert cfg.approach == "stgcn_ce"
        assert cfg.label_smoothing == 0.1
        assert cfg.mixup_alpha == 0.2
        assert cfg.head_dropout == 0.3


# ---------------------------------------------------------------------------
# TestValidApproaches
# ---------------------------------------------------------------------------


class TestValidApproaches:
    """Verify only stgcn_ce and stgcn_proto are valid approaches."""

    def test_valid_approaches_tuple(self):
        assert auto_config.VALID_APPROACHES == ("stgcn_ce", "stgcn_proto")

    def test_no_pose_video_fusion(self):
        for invalid in ("pose", "video", "fusion"):
            assert invalid not in auto_config.VALID_APPROACHES

    def test_invalid_approach_rejected(self):
        result = subprocess.run(
            [sys.executable, "scripts/auto_config.py", "--approach", "pose", "--dry-run"],
            capture_output=True,
            text=True,
            cwd=str(auto_config.PROJECT_ROOT),
        )
        assert result.returncode != 0
