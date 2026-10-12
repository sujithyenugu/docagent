"""LocalModelClient — embedded Apple MLX backend for local document extraction.

Runs a vision-language model directly in-process via `mlx-vlm`, entirely
on-device (no external server, no network calls after the one-time model
download). Apple Silicon macOS only.

The model weights are cached via the standard huggingface_hub cache
(~/.cache/huggingface, or $HF_HOME) and downloaded automatically on first
use (~2.5 GB for the default model).
"""

import os
import platform
import tempfile
import threading
import time
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

_DEFAULT_REPO = os.environ.get("LOCAL_MODEL_REPO", "mlx-community/Qwen2.5-VL-3B-Instruct-4bit")

_EXTRACT_PROMPT = (
    "Extract all text, tables, and key-value pairs from this document image. "
    "Return clean Markdown. Preserve table structure. Do not add commentary."
)


def _quiet_tqdm() -> None:
    """Disable tqdm's background monitor thread and huggingface_hub's own
    console progress bars.

    Observed in practice: on a fresh multi-file model download (parallel
    workers via hf_xet), tqdm's monitor thread and the downloading thread can
    both deadlock trying to acquire tqdm's global class lock during progress
    bar teardown — the whole process hangs forever with ~0% CPU. Neither of
    these console bars is ever visible in this app (headless Streamlit/FastAPI
    server), so disabling them removes the risk with no loss of functionality.
    Our own download_model() progress bar (a plain tqdm.tqdm subclass, used
    one file at a time) is unaffected by HF_HUB_DISABLE_PROGRESS_BARS, which
    only gates huggingface_hub's own internal wrapper class.
    """
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        import tqdm as tqdm_module
        tqdm_module.tqdm.monitor_interval = 0
    except ImportError:
        pass


_quiet_tqdm()

# Mirrors mlx_vlm.utils.get_model_path()'s default allow_patterns — kept in
# sync so our own pre-download fetches exactly what mlx_vlm.load() would.
_MODEL_ALLOW_PATTERNS = [
    "*.json", "*.jsonl", "*.safetensors", "*.py", "*.model", "*.tiktoken",
    "*.txt", "*.jinja",
]

# Loaded (model, processor, config) tuples, keyed by repo id — loaded once per
# process and reused across requests. `_engine_lock` guards the one-time load;
# `_generation_lock` serializes generate() calls, since MLX's Metal command
# streams are thread-local and concurrent generation on one model is a known
# crash source (https://github.com/Blaizzy/mlx-vlm/issues/1591, #1614).
_engine_cache: dict[str, tuple] = {}
_engine_lock = threading.Lock()
_generation_lock = threading.Lock()


def _strip_code_fence(text: str) -> str:
    """Unwrap a response that's entirely wrapped in a single ``` code fence.

    Local vision models frequently wrap their whole Markdown answer in a
    fenced code block (e.g. ```markdown ... ```) despite being asked not to.
    Left as-is, Streamlit renders that as a literal, non-wrapping code block
    instead of flowing text — this restores it to plain Markdown.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    lines = stripped.split("\n")
    if len(lines) < 2 or lines[-1].strip() != "```":
        return text
    return "\n".join(lines[1:-1])


def is_platform_supported() -> bool:
    """MLX requires Apple Silicon macOS."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def is_model_ready(repo_id: str | None = None) -> bool:
    """Cheap, network-free check: are this model's weights already cached?

    Must pass the same allow_patterns mlx_vlm.load() actually needs — without
    it, snapshot_download(local_files_only=True) checks for *every* file in
    the repo (README, LICENSE, alternate checkpoint formats, ...), which we
    never download, so this would always report "not ready" even when the
    model is fully cached and usable.

    snapshot_download(local_files_only=True) alone isn't a reliable enough
    signal on its own, though: with no cached file-tree manifest to check
    completeness against (e.g. a download that was cancelled partway, or is
    still in progress), it has been observed to report success even though
    the actual weights file is still missing. So this also directly verifies
    the model weights file itself is present and non-empty.
    """
    if not is_platform_supported():
        return False
    try:
        from huggingface_hub import snapshot_download, try_to_load_from_cache
    except ImportError:
        return False

    resolved_repo = repo_id or _DEFAULT_REPO
    try:
        snapshot_download(
            resolved_repo,
            local_files_only=True,
            allow_patterns=_MODEL_ALLOW_PATTERNS,
        )
    except Exception:
        return False

    weights_path = try_to_load_from_cache(resolved_repo, "model.safetensors")
    if not isinstance(weights_path, str):
        return False
    try:
        return Path(weights_path).stat().st_size > 0
    except OSError:
        return False


class DownloadCancelled(Exception):
    """Raised internally to abort an in-progress model download."""


class DownloadProgress:
    """Thread-safe download state a UI can poll for a live progress bar."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.downloaded_bytes = 0
        self.total_bytes = 0
        self.done = False
        self.cancelled = False
        self.error: str | None = None

    def add(self, n: int) -> None:
        with self._lock:
            self.downloaded_bytes += n

    @property
    def fraction(self) -> float:
        with self._lock:
            if self.total_bytes <= 0:
                return 0.0
            return min(1.0, self.downloaded_bytes / self.total_bytes)


def get_model_download_size(repo_id: str | None = None) -> int:
    """Total bytes to download for repo_id (best-effort; 0 if unknown/offline)."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo_id or _DEFAULT_REPO, files_metadata=True)
    except Exception:
        return 0
    return sum(
        (sibling.size or 0)
        for sibling in (info.siblings or [])
        if any(fnmatch(sibling.rfilename, pat) for pat in _MODEL_ALLOW_PATTERNS)
    )


def _repo_cache_dir(repo_id: str) -> Path | None:
    """Local cache directory for repo_id (models--org--repo), or None if the
    huggingface_hub cache location can't be resolved."""
    try:
        from huggingface_hub import constants
        cache_root = Path(constants.HF_HUB_CACHE)
    except Exception:
        return None
    return cache_root / ("models--" + repo_id.replace("/", "--"))


def _disk_bytes_used(repo_id: str) -> int:
    """Best-effort total bytes currently on disk for this repo, including
    in-progress `.incomplete` files.

    Used as a transport-agnostic progress signal: hf_xet (the default
    transfer backend for most mlx-community repos) has been observed to not
    reliably call the tqdm_class byte-progress hooks huggingface_hub exposes
    for its own HTTP fallback path, leaving the progress bar stuck at 0%
    despite real data flowing to disk. Polling actual bytes-on-disk works
    regardless of which transport backend is doing the writing.

    Must skip symlinks: huggingface_hub's cache layout stores each file
    under snapshots/<revision>/<filename> as a symlink into blobs/<hash> —
    Path.is_file() follows symlinks, so counting both would double-count
    every completed file (observed live: >100%, e.g. "4054 MB / 3281 MB").
    """
    root = _repo_cache_dir(repo_id)
    if root is None or not root.exists():
        return 0
    total = 0
    try:
        for path in root.rglob("*"):
            if path.is_file() and not path.is_symlink():
                try:
                    total += path.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _cleanup_stale_incomplete_files(repo_id: str) -> None:
    """Remove leftover `.incomplete` temp files from previous, abandoned
    download attempts before starting a new one.

    huggingface_hub never reuses or cleans up another attempt's `.incomplete`
    file — each hf_hub_download() call picks a fresh random suffix. Combined
    with the cancel grace-period timeout (which deliberately leaves a
    non-cooperating native transfer thread to finish or die on its own
    rather than block the UI), repeated cancel/retry cycles pile up multiple
    stale partial files for the same blob. Reproduced live: five leftover
    `.incomplete` files (~4.6 GB total) from one testing session alone,
    inflating _disk_bytes_used() well past the real total (e.g. showing
    "4054 MB / 3281 MB" — over 100%) and wasting real disk space forever.

    Safe to unlink even if an old orphaned thread still has one of these
    files open for writing (standard POSIX unlink-while-open semantics): the
    write continues into the now-unlinked inode harmlessly, and that old
    attempt simply fails silently when it tries to finalize — nothing reads
    its result anymore since the UI already gave up waiting on it.
    """
    root = _repo_cache_dir(repo_id)
    if root is None or not root.exists():
        return
    for path in root.rglob("*.incomplete"):
        try:
            path.unlink()
        except OSError:
            pass


def download_model(
    repo_id: str, progress: DownloadProgress, cancel_event: threading.Event
) -> None:
    """Download a model's weights with live, cancellable, byte-level progress.

    Safe to call from a background thread — this is pure network I/O with no
    MLX calls. (MLX array creation must stay on one consistent thread — see
    _run_generate's docstring on Metal stream thread-affinity — so the actual
    mlx_vlm.load()/generate() calls happen afterwards, on the caller's thread,
    once this download has already populated the local cache; mlx_vlm.load()
    will then find everything cached and do no further downloading.)

    Byte-level progress comes from directly polling bytes-on-disk in a small
    background poller thread (see _disk_bytes_used) — not from
    huggingface_hub's own tqdm_class hooks, which have been observed to
    never fire for repos transferred via hf_xet (the default backend for
    most mlx-community repos). Cancellation hooks into huggingface_hub's
    per-file `tqdm_class` extension point instead, checked between files and
    from within the plain-HTTP fallback path's own per-chunk callback.

    The object passed as `tqdm_class` deliberately does NOT subclass the real
    `tqdm.tqdm` (and this module never imports `tqdm` at all): tqdm's global
    monitor thread and class-level lock were observed in practice to
    deadlock this background download thread against Streamlit's own
    threads — a real hang reproduced twice, with everything at ~0% CPU
    forever. huggingface_hub's own internal progress aggregator
    (`_AggregatedTqdm` in its `_snapshot_download.py`) is likewise a plain
    duck-typed object for the same reason, not a real tqdm subclass — this
    mirrors that pattern.

    Known limitation, observed in practice: once a transfer is underway via
    hf_xet, cancel_event isn't always noticed promptly (hf_xet's native
    thread pool doesn't check it) — the caller is responsible for giving up
    on waiting after a grace period rather than blocking forever; the
    orphaned daemon thread finishes or dies with the process, harmlessly.
    """
    from huggingface_hub import HfApi, hf_hub_download

    _cleanup_stale_incomplete_files(repo_id)
    progress.total_bytes = get_model_download_size(repo_id)

    class _ProgressReporter:
        """Minimal tqdm-compatible duck type — no dependency on tqdm itself.
        Only used for cancellation checks; byte progress comes from the disk
        poller below, which works regardless of transfer backend."""

        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self) -> "_ProgressReporter":
            return self

        def __exit__(self, *exc_info) -> bool:
            return False

        def _check_cancelled(self) -> None:
            if cancel_event.is_set():
                raise DownloadCancelled("Download cancelled by user.")

        def update(self, n: int = 1) -> None:
            self._check_cancelled()

        def update_transfer(self, n: int = 1) -> None:
            self._check_cancelled()

        def close(self) -> None:
            pass

        def __getattr__(self, _name):
            return lambda *a, **kw: None

    stop_poll = threading.Event()

    def _poll_disk_usage() -> None:
        while not stop_poll.is_set():
            progress.downloaded_bytes = _disk_bytes_used(repo_id)
            stop_poll.wait(0.5)

    poller = threading.Thread(target=_poll_disk_usage, daemon=True)
    poller.start()

    try:
        info = HfApi().model_info(repo_id, files_metadata=True)
        filenames = [
            sibling.rfilename
            for sibling in (info.siblings or [])
            if any(fnmatch(sibling.rfilename, pat) for pat in _MODEL_ALLOW_PATTERNS)
        ]
        for filename in filenames:
            if cancel_event.is_set():
                raise DownloadCancelled("Download cancelled by user.")
            hf_hub_download(repo_id, filename, tqdm_class=_ProgressReporter)
        progress.done = True
    except DownloadCancelled:
        progress.cancelled = True
    except Exception as exc:
        progress.error = str(exc)
    finally:
        stop_poll.set()
        progress.downloaded_bytes = _disk_bytes_used(repo_id)


def _load_engine(repo_id: str) -> tuple:
    """Lazily load (and cache) the MLX model + processor + config for repo_id."""
    if repo_id in _engine_cache:
        return _engine_cache[repo_id]

    with _engine_lock:
        if repo_id in _engine_cache:  # double-checked locking
            return _engine_cache[repo_id]

        if not is_platform_supported():
            raise RuntimeError(
                "Local extraction requires Apple Silicon macOS (MLX). "
                "This environment doesn't support it — use Cloud mode instead."
            )

        try:
            from mlx_vlm import load
            from mlx_vlm.utils import load_config
        except ImportError as exc:
            raise RuntimeError(
                "MLX is not installed. Run `pip install -r requirements.txt` "
                "on a Mac to enable local extraction."
            ) from exc

        try:
            model, processor = load(repo_id)
            config = load_config(repo_id)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load local model '{repo_id}'. If this is the "
                "first run, an internet connection is required once to "
                "download the model (~2.5 GB for the default model). "
                f"Original error: {exc}"
            ) from exc

        _engine_cache[repo_id] = (model, processor, config)
        return _engine_cache[repo_id]


def _run_generate(
    prompt: str,
    *,
    image_path: str | None,
    max_tokens: int,
    temperature: float,
    repo_id: str | None = None,
) -> str:
    """Run one generation call against the shared MLX engine."""
    active_repo = repo_id or _DEFAULT_REPO
    model, processor, config = _load_engine(active_repo)

    # Only imported after _load_engine succeeds, so platform/install errors
    # surface as _load_engine's friendly RuntimeError, not a raw ImportError.
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    formatted_prompt = apply_chat_template(
        processor, config, prompt, num_images=1 if image_path else 0
    )

    with _generation_lock:
        result = generate(
            model,
            processor,
            formatted_prompt,
            image=[image_path] if image_path else None,
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=False,
        )

    return _strip_code_fence(result.text or "")


def generate_text(
    prompt: str,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    repo_id: str | None = None,
) -> str:
    """Run a text-only completion against the shared local engine.

    Used by form_filler.py's JSON-extraction stage, which needs no image.
    """
    return _run_generate(
        prompt, image_path=None, max_tokens=max_tokens, temperature=temperature,
        repo_id=repo_id,
    )


@dataclass
class LocalModelMetadata:
    page_count: int
    duration_ms: int


@dataclass
class LocalModelResponse:
    markdown: str
    metadata: LocalModelMetadata


class LocalModelClient:
    """Runs document extraction through the embedded MLX vision-language model.

    Args:
        repo_id: Hugging Face repo id of the MLX vision model to use.
                 Override with LOCAL_MODEL_REPO env var or pass directly.
    """

    def __init__(self, repo_id: str | None = None) -> None:
        self._model = repo_id or _DEFAULT_REPO
        self.model_id = self._model

    def parse(self, document: Path, model: str | None = None) -> LocalModelResponse:
        """Extract text, tables, and key-value pairs from a document.

        Supports PDF (converted page-by-page via pymupdf) and image files
        (JPEG, PNG). Returns a LocalModelResponse with .markdown and
        .metadata(.page_count, .duration_ms).

        Args:
            document: Path to the document file.
            model: Override the model repo id for this call only.

        Raises:
            RuntimeError: If the platform is unsupported or the model can't
                          be loaded (e.g. first-run download failed).
            ValueError: If the file extension is not supported.
        """
        active_repo = model or self._model
        suffix = document.suffix.lower()

        if suffix not in {".pdf", ".jpg", ".jpeg", ".png"}:
            raise ValueError(
                f"Unsupported file type '{suffix}'. "
                "Supported: .pdf, .jpg, .jpeg, .png"
            )

        start = time.time()
        if suffix == ".pdf":
            markdown, page_count = self._parse_pdf(document, active_repo)
        else:
            markdown, page_count = self._parse_image(document, active_repo)
        duration_ms = int((time.time() - start) * 1000)

        return LocalModelResponse(
            markdown=markdown,
            metadata=LocalModelMetadata(page_count=page_count, duration_ms=duration_ms),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_pdf(self, path: Path, repo_id: str) -> tuple[str, int]:
        """Convert each PDF page to an image and extract via the vision model."""
        import fitz  # pymupdf

        doc = fitz.open(str(path))
        pages: list[str] = []
        try:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # Render at 96 DPI — keeps image size (and generation cost) modest.
                mat = fitz.Matrix(96 / 72, 96 / 72)
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(img_bytes)
                    tmp_path = tmp.name
                try:
                    page_md = _run_generate(
                        _EXTRACT_PROMPT, image_path=tmp_path,
                        max_tokens=2048, temperature=0.1, repo_id=repo_id,
                    )
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
                pages.append(page_md)
        finally:
            doc.close()

        combined = "\n\n---\n\n".join(pages) if len(pages) > 1 else (pages[0] if pages else "")
        return combined, len(pages)

    def _parse_image(self, path: Path, repo_id: str) -> tuple[str, int]:
        """Extract text from a single image file."""
        markdown = _run_generate(
            _EXTRACT_PROMPT, image_path=str(path),
            max_tokens=2048, temperature=0.1, repo_id=repo_id,
        )
        return markdown, 1
