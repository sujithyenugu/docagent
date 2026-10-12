"""Unit tests for LocalModelClient (embedded Apple MLX backend).

All MLX/huggingface_hub calls are mocked via sys.modules injection — no real
`mlx_vlm` package or model download is required to run these tests, and they
run correctly on any platform (not just Apple Silicon).
"""

import os
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import local_model
from local_model import (
    LocalModelClient,
    LocalModelResponse,
    LocalModelMetadata,
    DownloadProgress,
    download_model,
    get_model_download_size,
    _strip_code_fence,
    is_platform_supported,
    is_model_ready,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_engine_cache():
    """The MLX engine is cached at module scope (loaded once per process) —
    clear it between tests so each test controls its own load()/generate()
    mocks instead of reusing a previous test's cached engine."""
    local_model._engine_cache.clear()
    yield
    local_model._engine_cache.clear()


def _install_mock_mlx_vlm(
    response_text: str = "## Extracted\nSome content",
    load_side_effect=None,
):
    """Inject fake `mlx_vlm`, `mlx_vlm.utils`, `mlx_vlm.prompt_utils` modules
    into sys.modules so `local_model`'s lazy `from mlx_vlm import ...`
    imports resolve to mocks instead of requiring the real package.

    Returns the mock `load` and `generate` callables so tests can assert on
    call counts/args.
    """
    mock_model = MagicMock(name="mlx_model")
    mock_processor = MagicMock(name="mlx_processor")

    mock_load = MagicMock(return_value=(mock_model, mock_processor))
    if load_side_effect is not None:
        mock_load.side_effect = load_side_effect

    mock_generate = MagicMock(return_value=MagicMock(text=response_text))
    mock_load_config = MagicMock(return_value={"model_type": "gemma3"})
    mock_apply_chat_template = MagicMock(side_effect=lambda processor, config, prompt, **kw: prompt)

    mlx_vlm_module = MagicMock(name="mlx_vlm")
    mlx_vlm_module.load = mock_load
    mlx_vlm_module.generate = mock_generate

    mlx_vlm_utils_module = MagicMock(name="mlx_vlm.utils")
    mlx_vlm_utils_module.load_config = mock_load_config

    mlx_vlm_prompt_utils_module = MagicMock(name="mlx_vlm.prompt_utils")
    mlx_vlm_prompt_utils_module.apply_chat_template = mock_apply_chat_template

    modules = {
        "mlx_vlm": mlx_vlm_module,
        "mlx_vlm.utils": mlx_vlm_utils_module,
        "mlx_vlm.prompt_utils": mlx_vlm_prompt_utils_module,
    }
    return modules, mock_load, mock_generate


def _install_mock_hf_hub(available: bool = True):
    """Inject a fake `huggingface_hub` module for is_model_ready() tests."""
    mock_snapshot_download = MagicMock()
    mock_try_to_load_from_cache = MagicMock()
    if available:
        # Any real, nonzero-size file stands in for a cached weights path.
        mock_try_to_load_from_cache.return_value = local_model.__file__
    else:
        mock_snapshot_download.side_effect = Exception("not cached")
        mock_try_to_load_from_cache.return_value = None

    mock_hf_hub = MagicMock(name="huggingface_hub")
    mock_hf_hub.snapshot_download = mock_snapshot_download
    mock_hf_hub.try_to_load_from_cache = mock_try_to_load_from_cache
    return {"huggingface_hub": mock_hf_hub}, mock_snapshot_download


# ---------------------------------------------------------------------------
# Test: basic markdown return from the local model
# ---------------------------------------------------------------------------

class TestReturnsMarkdown:
    def test_returns_markdown_from_local_model(self, tmp_path):
        """parse() on an image file returns a LocalModelResponse with .markdown."""
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)  # minimal PNG header

        modules, _, _ = _install_mock_mlx_vlm("# Invoice\n**Total**: 100")

        with patch.dict("sys.modules", modules):
            client = LocalModelClient(repo_id="test/repo")
            result = client.parse(img)

        assert isinstance(result, LocalModelResponse)
        assert "Invoice" in result.markdown
        assert "Total" in result.markdown

    def test_returns_local_model_response_type(self, tmp_path):
        img = tmp_path / "doc.jpg"
        img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 50)  # minimal JPEG header

        modules, _, _ = _install_mock_mlx_vlm("some markdown")

        with patch.dict("sys.modules", modules):
            client = LocalModelClient()
            result = client.parse(img)

        assert isinstance(result, LocalModelResponse)
        assert isinstance(result.metadata, LocalModelMetadata)


# ---------------------------------------------------------------------------
# Test: PDF converted page-by-page
# ---------------------------------------------------------------------------

class TestPdfPageByPage:
    def test_pdf_converted_page_by_page(self, tmp_path):
        """Each PDF page generates exactly one generate() call."""
        pdf_path = tmp_path / "doc.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        modules, _, mock_generate = _install_mock_mlx_vlm("page content")

        mock_page = MagicMock()
        mock_pixmap = MagicMock()
        mock_pixmap.tobytes.return_value = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        mock_page.get_pixmap.return_value = mock_pixmap

        mock_fitz_doc = MagicMock()
        mock_fitz_doc.__len__ = lambda self: 3
        mock_fitz_doc.load_page.return_value = mock_page

        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_fitz_doc
        mock_fitz.Matrix.return_value = MagicMock()

        with patch.dict("sys.modules", {**modules, "fitz": mock_fitz}):
            client = LocalModelClient()
            result = client.parse(pdf_path)

        assert mock_generate.call_count == 3
        assert result.metadata.page_count == 3

    def test_pdf_page_count_matches_pages(self, tmp_path):
        pdf_path = tmp_path / "single.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        modules, _, _ = _install_mock_mlx_vlm("content")

        mock_page = MagicMock()
        mock_pixmap = MagicMock()
        mock_pixmap.tobytes.return_value = b"\x89PNG\r\n\x1a\n"
        mock_page.get_pixmap.return_value = mock_pixmap

        mock_fitz_doc = MagicMock()
        mock_fitz_doc.__len__ = lambda self: 1
        mock_fitz_doc.load_page.return_value = mock_page

        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_fitz_doc
        mock_fitz.Matrix.return_value = MagicMock()

        with patch.dict("sys.modules", {**modules, "fitz": mock_fitz}):
            client = LocalModelClient()
            result = client.parse(pdf_path)

        assert result.metadata.page_count == 1


# ---------------------------------------------------------------------------
# Test: model weights unavailable (no cache, no internet) → RuntimeError
# ---------------------------------------------------------------------------

class TestRaisesWhenWeightsUnavailable:
    def test_raises_runtime_error_when_load_fails(self, tmp_path):
        """A load() failure (e.g. first run with no internet) is surfaced as
        a clear RuntimeError mentioning the repo id and the download."""
        img = tmp_path / "doc.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        modules, _, _ = _install_mock_mlx_vlm(
            load_side_effect=Exception("no internet, no cached weights")
        )

        with patch.dict("sys.modules", modules):
            client = LocalModelClient(repo_id="mlx-community/gemma-3-4b-it-4bit")
            with pytest.raises(RuntimeError) as exc_info:
                client.parse(img)

        assert "mlx-community/gemma-3-4b-it-4bit" in str(exc_info.value)
        assert "internet connection" in str(exc_info.value)

    def test_raises_runtime_error_when_mlx_not_installed(self, tmp_path, monkeypatch):
        """On a platform without mlx-vlm installed, parse() fails clearly
        instead of an ImportError leaking out."""
        img = tmp_path / "doc.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        monkeypatch.setattr(local_model, "is_platform_supported", lambda: True)
        # Ensure `import mlx_vlm` genuinely fails (not previously mocked in).
        for mod in ("mlx_vlm", "mlx_vlm.utils", "mlx_vlm.prompt_utils"):
            sys.modules.pop(mod, None)
        with patch.dict("sys.modules", {"mlx_vlm": None}):
            client = LocalModelClient()
            with pytest.raises(RuntimeError, match="MLX is not installed"):
                client.parse(img)

    def test_raises_runtime_error_on_unsupported_platform(self, tmp_path, monkeypatch):
        img = tmp_path / "doc.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        monkeypatch.setattr(local_model, "is_platform_supported", lambda: False)

        client = LocalModelClient()
        with pytest.raises(RuntimeError, match="Apple Silicon macOS"):
            client.parse(img)


# ---------------------------------------------------------------------------
# Test: stripping a whole-response code fence
# ---------------------------------------------------------------------------

class TestStripCodeFence:
    def test_strips_markdown_language_fence(self):
        wrapped = "```markdown\n# Title\nSome content\n```"
        assert _strip_code_fence(wrapped) == "# Title\nSome content"

    def test_strips_bare_fence(self):
        wrapped = "```\n# Title\nSome content\n```"
        assert _strip_code_fence(wrapped) == "# Title\nSome content"

    def test_leaves_unfenced_text_untouched(self):
        text = "# Title\nSome content"
        assert _strip_code_fence(text) == text

    def test_leaves_text_with_internal_fence_untouched(self):
        text = "Some notes\n```python\nprint(1)\n```\nmore text"
        assert _strip_code_fence(text) == text

    def test_parse_unwraps_fenced_response(self, tmp_path):
        img = tmp_path / "doc.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        modules, _, _ = _install_mock_mlx_vlm("```markdown\n# Invoice\n**Total**: 100\n```")

        with patch.dict("sys.modules", modules):
            client = LocalModelClient()
            result = client.parse(img)

        assert not result.markdown.startswith("```")
        assert "# Invoice" in result.markdown


# ---------------------------------------------------------------------------
# Test: metadata shape
# ---------------------------------------------------------------------------

class TestMetadataShape:
    def test_metadata_has_page_count_and_duration(self, tmp_path):
        img = tmp_path / "doc.jpeg"
        img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 50)

        modules, _, _ = _install_mock_mlx_vlm("content")

        with patch.dict("sys.modules", modules):
            client = LocalModelClient()
            result = client.parse(img)

        assert hasattr(result.metadata, "page_count")
        assert hasattr(result.metadata, "duration_ms")
        assert result.metadata.page_count == 1
        assert isinstance(result.metadata.duration_ms, int)
        assert result.metadata.duration_ms >= 0

    def test_duration_ms_is_non_negative(self, tmp_path):
        img = tmp_path / "doc.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        modules, _, _ = _install_mock_mlx_vlm("ok")

        with patch.dict("sys.modules", modules):
            client = LocalModelClient()
            result = client.parse(img)

        assert result.metadata.duration_ms >= 0


# ---------------------------------------------------------------------------
# Test: env var model repo
# ---------------------------------------------------------------------------

class TestEnvVarModelRepo:
    def test_uses_env_var_model_repo(self, monkeypatch):
        """LocalModelClient picks up LOCAL_MODEL_REPO from environment."""
        monkeypatch.setattr(local_model, "_DEFAULT_REPO", "mlx-community/custom-model")

        client = LocalModelClient()

        assert client._model == "mlx-community/custom-model"
        assert client.model_id == "mlx-community/custom-model"

    def test_constructor_arg_overrides_default(self, monkeypatch):
        """Explicit constructor argument takes precedence over the default repo."""
        monkeypatch.setattr(local_model, "_DEFAULT_REPO", "mlx-community/env-model")

        client = LocalModelClient(repo_id="mlx-community/explicit-model")

        assert client._model == "mlx-community/explicit-model"

    def test_default_repo_when_no_override(self):
        client = LocalModelClient()
        assert client._model == local_model._DEFAULT_REPO


# ---------------------------------------------------------------------------
# Test: platform support / model readiness
# ---------------------------------------------------------------------------

class TestPlatformSupport:
    def test_supported_on_apple_silicon_macos(self, monkeypatch):
        monkeypatch.setattr(local_model.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(local_model.platform, "machine", lambda: "arm64")
        assert is_platform_supported() is True

    def test_unsupported_on_intel_macos(self, monkeypatch):
        monkeypatch.setattr(local_model.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(local_model.platform, "machine", lambda: "x86_64")
        assert is_platform_supported() is False

    def test_unsupported_on_linux(self, monkeypatch):
        monkeypatch.setattr(local_model.platform, "system", lambda: "Linux")
        monkeypatch.setattr(local_model.platform, "machine", lambda: "arm64")
        assert is_platform_supported() is False


class TestModelReady:
    def test_ready_when_weights_cached(self, monkeypatch):
        monkeypatch.setattr(local_model, "is_platform_supported", lambda: True)
        modules, _ = _install_mock_hf_hub(available=True)
        with patch.dict("sys.modules", modules):
            assert is_model_ready("mlx-community/gemma-3-4b-it-4bit") is True

    def test_not_ready_when_weights_missing(self, monkeypatch):
        monkeypatch.setattr(local_model, "is_platform_supported", lambda: True)
        modules, _ = _install_mock_hf_hub(available=False)
        with patch.dict("sys.modules", modules):
            assert is_model_ready("mlx-community/gemma-3-4b-it-4bit") is False

    def test_not_ready_on_unsupported_platform(self, monkeypatch):
        monkeypatch.setattr(local_model, "is_platform_supported", lambda: False)
        assert is_model_ready("mlx-community/gemma-3-4b-it-4bit") is False

    def test_checks_only_files_mlx_actually_needs(self, monkeypatch):
        """Regression test: must pass allow_patterns matching what's actually
        downloaded, or this always reports "not ready" even when the model
        is fully cached — snapshot_download(local_files_only=True) with no
        allow_patterns requires *every* file in the repo (README, LICENSE,
        alternate checkpoints, ...) to be present locally, not just the ones
        mlx_vlm.load() needs and that download_model() actually fetches."""
        monkeypatch.setattr(local_model, "is_platform_supported", lambda: True)
        mock_snapshot_download = MagicMock()
        mock_hf_hub = MagicMock()
        mock_hf_hub.snapshot_download = mock_snapshot_download
        with patch.dict("sys.modules", {"huggingface_hub": mock_hf_hub}):
            is_model_ready("some/repo")
        _, kwargs = mock_snapshot_download.call_args
        assert kwargs.get("allow_patterns") == local_model._MODEL_ALLOW_PATTERNS

    def test_not_ready_when_download_only_partially_completed(self, monkeypatch):
        """Regression test: snapshot_download(local_files_only=True) can
        report success even when the cache is genuinely incomplete (e.g. a
        cancelled or still-in-progress download, with no cached file-tree
        manifest to check completeness against) — reproduced live against a
        real partial cache. Must also verify the weights file itself."""
        monkeypatch.setattr(local_model, "is_platform_supported", lambda: True)
        mock_hf_hub = MagicMock()
        mock_hf_hub.snapshot_download.return_value = "/fake/snapshot/dir"
        mock_hf_hub.try_to_load_from_cache.return_value = None  # weights not actually there
        with patch.dict("sys.modules", {"huggingface_hub": mock_hf_hub}):
            assert is_model_ready("some/repo") is False


# ---------------------------------------------------------------------------
# Test: unsupported file type
# ---------------------------------------------------------------------------

class TestUnsupportedFileType:
    def test_raises_value_error_for_unsupported_extension(self, tmp_path):
        doc = tmp_path / "doc.tiff"
        doc.write_bytes(b"\x00" * 10)

        client = LocalModelClient()
        with pytest.raises(ValueError, match="Unsupported file type"):
            client.parse(doc)


# ---------------------------------------------------------------------------
# Test: tqdm monitor-thread deadlock prevention
# ---------------------------------------------------------------------------

class TestQuietTqdm:
    def test_disables_hf_hub_progress_bars_env_var(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
        local_model._quiet_tqdm()
        assert os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") == "1"

    def test_disables_tqdm_monitor_thread(self):
        import tqdm as tqdm_module
        tqdm_module.tqdm.monitor_interval = 999  # simulate the risky default
        local_model._quiet_tqdm()
        assert tqdm_module.tqdm.monitor_interval == 0


# ---------------------------------------------------------------------------
# Test: download progress state
# ---------------------------------------------------------------------------

class TestDownloadProgress:
    def test_fraction_zero_when_total_unknown(self):
        progress = DownloadProgress()
        assert progress.fraction == 0.0

    def test_fraction_computed_from_downloaded_and_total(self):
        progress = DownloadProgress()
        progress.total_bytes = 100
        progress.add(25)
        assert progress.fraction == 0.25

    def test_fraction_capped_at_one(self):
        progress = DownloadProgress()
        progress.total_bytes = 100
        progress.add(150)
        assert progress.fraction == 1.0

    def test_add_accumulates_across_calls(self):
        progress = DownloadProgress()
        progress.add(10)
        progress.add(15)
        assert progress.downloaded_bytes == 25


# ---------------------------------------------------------------------------
# Test: download size lookup
# ---------------------------------------------------------------------------

def _make_sibling(name: str, size: int) -> MagicMock:
    sibling = MagicMock()
    sibling.rfilename = name
    sibling.size = size
    return sibling


class TestDiskBytesUsed:
    """_disk_bytes_used() is the transport-agnostic progress source — tested
    against a real filesystem, not mocks, since it's pure file I/O."""

    def test_returns_zero_when_repo_dir_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
        assert local_model._disk_bytes_used("some/repo") == 0

    def test_sums_file_sizes_including_incomplete(self, tmp_path, monkeypatch):
        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
        blobs_dir = tmp_path / "models--some--repo" / "blobs"
        blobs_dir.mkdir(parents=True)
        (blobs_dir / "abc").write_bytes(b"x" * 100)
        (blobs_dir / "def.incomplete").write_bytes(b"y" * 50)
        assert local_model._disk_bytes_used("some/repo") == 150

    def test_repo_id_slashes_map_to_double_dash(self, tmp_path, monkeypatch):
        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
        blobs_dir = tmp_path / "models--mlx-community--gemma-3-4b-it-4bit" / "blobs"
        blobs_dir.mkdir(parents=True)
        (blobs_dir / "weights").write_bytes(b"z" * 10)
        assert local_model._disk_bytes_used("mlx-community/gemma-3-4b-it-4bit") == 10

    def test_does_not_double_count_snapshot_symlinks(self, tmp_path, monkeypatch):
        """Regression test: huggingface_hub's cache stores each file twice —
        once as the real blob, once as a symlink under snapshots/<rev>/ that
        points to it. Path.is_file() follows symlinks, so counting both
        doubles every completed file's size — reproduced live as the
        progress bar showing e.g. "4054 MB / 3281 MB" (over 100%)."""
        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
        repo_dir = tmp_path / "models--some--repo"
        blobs_dir = repo_dir / "blobs"
        snapshot_dir = repo_dir / "snapshots" / "abc123"
        blobs_dir.mkdir(parents=True)
        snapshot_dir.mkdir(parents=True)

        blob_path = blobs_dir / "deadbeef"
        blob_path.write_bytes(b"x" * 1000)
        (snapshot_dir / "model.safetensors").symlink_to(blob_path)

        assert local_model._disk_bytes_used("some/repo") == 1000


class TestCleanupStaleIncompleteFiles:
    """Regression tests for the leaked-disk-space bug: repeated cancel/retry
    cycles left multiple stale `.incomplete` files for the same blob,
    inflating _disk_bytes_used() past 100% and wasting real disk space."""

    def test_removes_incomplete_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
        blobs_dir = tmp_path / "models--some--repo" / "blobs"
        blobs_dir.mkdir(parents=True)
        stale_1 = blobs_dir / "abc.11111111.incomplete"
        stale_2 = blobs_dir / "abc.22222222.incomplete"
        stale_1.write_bytes(b"x" * 1000)
        stale_2.write_bytes(b"y" * 2000)

        local_model._cleanup_stale_incomplete_files("some/repo")

        assert not stale_1.exists()
        assert not stale_2.exists()

    def test_leaves_completed_blobs_untouched(self, tmp_path, monkeypatch):
        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
        blobs_dir = tmp_path / "models--some--repo" / "blobs"
        blobs_dir.mkdir(parents=True)
        completed = blobs_dir / "deadbeef"
        completed.write_bytes(b"z" * 500)
        (blobs_dir / "stale.incomplete").write_bytes(b"w" * 500)

        local_model._cleanup_stale_incomplete_files("some/repo")

        assert completed.exists()
        assert completed.stat().st_size == 500

    def test_noop_when_repo_dir_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
        # Should not raise even though nothing exists yet.
        local_model._cleanup_stale_incomplete_files("some/repo")

    def test_download_model_starts_with_accurate_progress_after_stale_cleanup(
        self, tmp_path, monkeypatch
    ):
        """End-to-end regression check: stale .incomplete files left before a
        download starts must not inflate this attempt's progress reading."""
        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
        blobs_dir = tmp_path / "models--some--repo" / "blobs"
        blobs_dir.mkdir(parents=True)
        # Simulate leftover garbage from earlier abandoned attempts.
        (blobs_dir / "model.11111111.incomplete").write_bytes(b"x" * 5_000_000)
        (blobs_dir / "model.22222222.incomplete").write_bytes(b"y" * 5_000_000)

        siblings = [_make_sibling("model.safetensors", 1000)]
        mock_info = MagicMock(siblings=siblings)

        def fake_hf_hub_download(repo_id, filename, tqdm_class=None):
            # Simulate this attempt writing a fresh, small .incomplete file.
            (blobs_dir / "model.33333333.incomplete").write_bytes(b"z" * 100)
            return "/fake/path"

        progress = DownloadProgress()
        with patch("huggingface_hub.HfApi") as MockApi, \
             patch("huggingface_hub.hf_hub_download", side_effect=fake_hf_hub_download):
            MockApi.return_value.model_info.return_value = mock_info
            download_model("some/repo", progress, threading.Event())

        # Stale multi-megabyte leftovers must be gone before we measure.
        assert progress.downloaded_bytes == 100


class TestGetModelDownloadSize:
    def test_sums_only_files_matching_allow_patterns(self):
        siblings = [
            _make_sibling("model.safetensors", 1000),
            _make_sibling("config.json", 10),
            _make_sibling("README.md", 500),  # not in allow patterns — excluded
        ]
        mock_info = MagicMock(siblings=siblings)
        with patch("huggingface_hub.HfApi") as MockApi:
            MockApi.return_value.model_info.return_value = mock_info
            total = get_model_download_size("some/repo")
        assert total == 1010

    def test_returns_zero_when_api_unreachable(self):
        with patch("huggingface_hub.HfApi", side_effect=Exception("offline")):
            assert get_model_download_size("some/repo") == 0


# ---------------------------------------------------------------------------
# Test: cancellable, progress-reporting download
# ---------------------------------------------------------------------------

class TestDownloadModel:
    """Progress now comes from polling bytes-on-disk (_disk_bytes_used), not
    from huggingface_hub's tqdm_class hooks — those are only used to check
    for cancellation, since hf_xet (the default transfer backend for most
    mlx-community repos) doesn't reliably call update()/update_transfer()
    with real byte counts. See download_model()'s docstring."""

    def test_downloads_each_matching_file_and_marks_done(self):
        siblings = [_make_sibling("model.safetensors", 1000), _make_sibling("config.json", 10)]
        mock_info = MagicMock(siblings=siblings)
        downloaded_files = []

        def fake_hf_hub_download(repo_id, filename, tqdm_class=None):
            downloaded_files.append(filename)
            bar = tqdm_class(total=1)
            bar.update(1)
            bar.update_transfer(1)
            return "/fake/path"

        progress = DownloadProgress()
        cancel_event = threading.Event()

        with patch("huggingface_hub.HfApi") as MockApi, \
             patch("huggingface_hub.hf_hub_download", side_effect=fake_hf_hub_download), \
             patch("local_model._disk_bytes_used", return_value=42):
            MockApi.return_value.model_info.return_value = mock_info
            download_model("some/repo", progress, cancel_event)

        assert downloaded_files == ["model.safetensors", "config.json"]
        assert progress.done is True
        assert progress.error is None
        assert progress.cancelled is False
        assert progress.downloaded_bytes == 42  # from the disk poller, not update()

    def test_progress_reflects_disk_usage_poller(self):
        """The disk poller is the source of truth for downloaded_bytes —
        confirm the final value comes from _disk_bytes_used(repo_id)."""
        siblings = [_make_sibling("model.safetensors", 1000)]
        mock_info = MagicMock(siblings=siblings)

        def fake_hf_hub_download(repo_id, filename, tqdm_class=None):
            return "/fake/path"

        progress = DownloadProgress()
        with patch("huggingface_hub.HfApi") as MockApi, \
             patch("huggingface_hub.hf_hub_download", side_effect=fake_hf_hub_download), \
             patch("local_model._disk_bytes_used", return_value=777) as mock_disk_usage:
            MockApi.return_value.model_info.return_value = mock_info
            download_model("some/repo", progress, threading.Event())

        mock_disk_usage.assert_called_with("some/repo")
        assert progress.downloaded_bytes == 777

    def test_cancellation_via_update_stops_before_remaining_files(self):
        siblings = [_make_sibling("a.safetensors", 1000), _make_sibling("b.safetensors", 1000)]
        mock_info = MagicMock(siblings=siblings)
        cancel_event = threading.Event()
        progress = DownloadProgress()
        call_count = {"n": 0}

        def fake_hf_hub_download(repo_id, filename, tqdm_class=None):
            call_count["n"] += 1
            cancel_event.set()  # simulate the user clicking Cancel mid-file
            bar = tqdm_class(total=1)
            bar.update(1)  # should raise DownloadCancelled since cancel_event is set
            return "/fake/path"

        with patch("huggingface_hub.HfApi") as MockApi, \
             patch("huggingface_hub.hf_hub_download", side_effect=fake_hf_hub_download):
            MockApi.return_value.model_info.return_value = mock_info
            download_model("some/repo", progress, cancel_event)

        assert progress.cancelled is True
        assert progress.done is False
        assert call_count["n"] == 1  # second file never attempted

    def test_cancellation_via_update_transfer_stops_before_remaining_files(self):
        """update_transfer() (hf_xet's own progress-report method) must also
        be checked for cancellation, not just update()."""
        siblings = [_make_sibling("a.safetensors", 1000), _make_sibling("b.safetensors", 1000)]
        mock_info = MagicMock(siblings=siblings)
        cancel_event = threading.Event()
        progress = DownloadProgress()
        call_count = {"n": 0}

        def fake_hf_hub_download(repo_id, filename, tqdm_class=None):
            call_count["n"] += 1
            cancel_event.set()
            bar = tqdm_class(total=1)
            bar.update_transfer(1)
            return "/fake/path"

        with patch("huggingface_hub.HfApi") as MockApi, \
             patch("huggingface_hub.hf_hub_download", side_effect=fake_hf_hub_download):
            MockApi.return_value.model_info.return_value = mock_info
            download_model("some/repo", progress, cancel_event)

        assert progress.cancelled is True
        assert call_count["n"] == 1

    def test_error_is_captured_not_raised(self):
        progress = DownloadProgress()
        with patch("huggingface_hub.HfApi", side_effect=RuntimeError("network down")):
            download_model("some/repo", progress, threading.Event())
        assert progress.error is not None
        assert "network down" in progress.error
        assert progress.done is False
        assert progress.cancelled is False
