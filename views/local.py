"""Local extraction page — Qwen2.5-VL-3B via embedded Apple MLX."""

import logging
import tempfile
import threading
import time
from pathlib import Path

import streamlit as st

from common import document_source_picker, page_header, render_pdf_first_page, render_result
from local_model import (
    DownloadProgress,
    LocalModelClient,
    download_model,
    is_model_ready,
    is_platform_supported,
)

_MB = 1024 * 1024
# hf_xet's native transfer threads don't always notice cancel_event promptly
# — give up waiting after this long rather than polling forever, so the UI
# never gets stuck. The orphaned daemon thread finishes or dies with the
# process, harmlessly.
_CANCEL_GRACE_SECONDS = 6


def _run_extraction(doc_bytes: bytes, suffix: str, doc_filename: str) -> None:
    """Run the actual (fast, by now-cached-model) extraction and render results."""
    tmp_path: str | None = None
    pdf_preview: bytes | None = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, prefix="tmp_upload"
        ) as tmp:
            tmp.write(doc_bytes)
            tmp_path = tmp.name

        client_local = LocalModelClient()
        with st.spinner("Extracting via local model — this may take a moment…"):
            response = client_local.parse(document=Path(tmp_path))

        logging.info(
            "Local parse complete | pages=%s duration_ms=%s",
            response.metadata.page_count,
            response.metadata.duration_ms,
        )
        meta = response.metadata

        if suffix == ".pdf":
            pdf_preview = render_pdf_first_page(tmp_path)

    except Exception as e:
        st.error(str(e))
        st.stop()
    finally:
        if tmp_path and Path(tmp_path).exists():
            Path(tmp_path).unlink()

    chips_html = (
        f'<span class="da-chip">📄 {meta.page_count} page(s)</span>'
        f'<span class="da-chip">⏱ {meta.duration_ms} ms</span>'
        f'<span class="da-chip">🔒 Local — no data sent</span>'
    )

    render_result(
        success_msg=(
            f"Done — {meta.page_count} page(s), {meta.duration_ms} ms (local model, 0 credits)"
        ),
        chips_html=chips_html,
        markdown=response.markdown,
        doc_bytes=doc_bytes,
        suffix=suffix,
        doc_filename=doc_filename,
        pdf_preview=pdf_preview,
        download_filename="extracted.md",
    )


def render() -> None:
    page_header(
        "Local Extraction",
        "Runs Qwen2.5-VL-3B on your own machine via Apple MLX — no document ever leaves this device.",
    )

    download_error = st.session_state.pop("local_download_error", None)
    if download_error:
        st.error(f"Model download failed: {download_error}")

    with st.sidebar:
        st.markdown("**Local Model**")
        if not is_platform_supported():
            st.caption("⚠️ Requires Apple Silicon macOS. Use Cloud mode on this machine.")
        elif is_model_ready():
            st.caption("✅ Model downloaded — ready to extract.")
        else:
            st.caption("First extraction downloads the model (~2.5 GB), once.")
        st.caption("Set LOCAL_MODEL_REPO in .env to use a different model.")

    st.markdown(
        """
        <div class="da-banner da-banner-local">
            🖥️ <strong>Local Model Active</strong> — your documents never leave this machine.
            Runs entirely on-device via Apple MLX — no setup, no external server.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Powered by Apple MLX · Qwen2.5-VL-3B (local, no external API calls)")

    doc_bytes, suffix, doc_filename = document_source_picker("local")

    downloading = st.session_state.get("local_downloading", False)

    if st.button("Extract →", type="primary", disabled=downloading):
        if doc_bytes is None:
            st.error("Please upload or select a document first.")
            st.stop()

        st.session_state.local_pending_doc = (doc_bytes, suffix, doc_filename)

        if is_model_ready():
            st.session_state.local_downloading = False
        else:
            progress = DownloadProgress()
            cancel_event = threading.Event()
            repo_id = LocalModelClient()._model
            thread = threading.Thread(
                target=download_model,
                args=(repo_id, progress, cancel_event),
                daemon=True,
            )
            thread.start()
            st.session_state.local_download_progress = progress
            st.session_state.local_download_cancel = cancel_event
            st.session_state.local_download_thread = thread
            st.session_state.local_downloading = True

    # ── Download in progress: live progress bar + cancel ────────────────────
    if st.session_state.get("local_downloading"):
        progress: DownloadProgress = st.session_state.local_download_progress
        cancel_event: threading.Event = st.session_state.local_download_cancel
        thread: threading.Thread = st.session_state.local_download_thread

        col_bar, col_cancel = st.columns([5, 1])
        with col_bar:
            if progress.total_bytes > 0:
                pct = progress.fraction
                # Clamp the displayed MB alongside the (already-clamped)
                # percentage — the disk-usage estimate can drift slightly
                # above the metadata-based total (e.g. extra cache bookkeeping
                # files), which would otherwise show as "over 100%".
                shown_downloaded = min(progress.downloaded_bytes, progress.total_bytes)
                st.progress(
                    pct,
                    text=(
                        f"Downloading local model (first run only)… {pct * 100:.0f}%  "
                        f"({shown_downloaded / _MB:.0f} MB / "
                        f"{progress.total_bytes / _MB:.0f} MB)"
                    ),
                )
            else:
                st.progress(0.0, text="Downloading local model (first run only)… preparing")
        with col_cancel:
            if st.button("Cancel", key="local_cancel_download"):
                cancel_event.set()
                st.session_state.local_cancel_requested_at = time.time()
                st.info("Cancelling…")

        if thread.is_alive():
            requested_at = st.session_state.get("local_cancel_requested_at")
            if requested_at and (time.time() - requested_at) > _CANCEL_GRACE_SECONDS:
                # Widgets drawn earlier in this run (e.g. the Extract button)
                # reflect session_state as it was at the *start* of this run,
                # not the updates we just made — st.stop() would leave them
                # looking stuck disabled until the next interaction. Rerun
                # immediately instead; st.toast (unlike st.warning/st.error)
                # survives across it, so the message still reaches the user.
                st.session_state.local_downloading = False
                st.session_state.pop("local_pending_doc", None)
                st.session_state.pop("local_cancel_requested_at", None)
                st.toast(
                    "Cancel requested, but the download is taking a moment to stop "
                    "in the background. It's safe to click Extract again.",
                    icon="⚠️",
                )
                st.rerun()
            time.sleep(0.4)
            st.rerun()

        # Thread finished this rerun — resolve the outcome.
        st.session_state.local_downloading = False
        st.session_state.pop("local_cancel_requested_at", None)
        if progress.cancelled:
            st.session_state.pop("local_pending_doc", None)
            st.toast("Download cancelled. Click Extract to try again.", icon="🛑")
            st.rerun()
        if progress.error:
            error_message = progress.error
            st.session_state.pop("local_pending_doc", None)
            st.session_state.local_download_error = error_message
            st.rerun()
        # progress.done — fall through to run the extraction below.

    # ── Run extraction once we have a pending doc and aren't downloading ────
    pending = st.session_state.get("local_pending_doc")
    if pending and not st.session_state.get("local_downloading"):
        st.session_state.pop("local_pending_doc", None)
        _run_extraction(*pending)
