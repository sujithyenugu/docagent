"""Cloud extraction page — Landing AI AIDE."""

import logging
import os
import tempfile
from pathlib import Path

import streamlit as st
from landingai_ade import LandingAIADE

import storage
from common import document_source_picker, page_header, render_pdf_first_page, render_result

_SAMPLE_MARKDOWN = """# Invoice #INV-2023-001
**Date**: October 24, 2023
**Company**: Acme Corp Ltd.
**Address**: 123 Industrial Estate, Mumbai, MH 400001

## Items

| Description | Quantity | Unit Price (₹) | Total (₹) |
|---|---|---|---|
| Industrial Steel Pipes (Grade A) | 50 | 500.00 | 25,000.00 |
| Heavy Duty Valves | 10 | 1,500.00 | 15,000.00 |
| Installation Services | 1 | 10,000.00 | 10,000.00 |

---
**Subtotal**: ₹50,000.00
**Tax (18% GST)**: ₹9,000.00
**Grand Total**: ₹59,000.00

*Note: This is a static sample for demo mode.*
"""


def render() -> None:
    page_header(
        "Cloud Extraction",
        "Sends documents to Landing AI's AIDE API for high-accuracy structured extraction.",
    )

    with st.sidebar:
        st.markdown("**Landing AI API Key**")
        env_key = os.getenv("VISION_AGENT_API_KEY", "")
        api_key = st.text_input(
            "Landing AI API Key",
            value=env_key,
            type="password",
            label_visibility="collapsed",
            placeholder="Paste key here…",
            help="Set VISION_AGENT_API_KEY in your .env file, or paste it here.",
        )

        if "last_api_key" not in st.session_state:
            st.session_state.last_api_key = api_key
            st.session_state.is_preview_mode = not bool(api_key.strip())
        elif st.session_state.last_api_key != api_key:
            st.session_state.is_preview_mode = not bool(api_key.strip())
            st.session_state.last_api_key = api_key

        is_preview_mode = st.session_state.is_preview_mode

        st.divider()
        st.markdown("**Mode**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Demo", use_container_width=True):
                st.session_state.is_preview_mode = True
        with col2:
            if st.button("Full", use_container_width=True):
                st.session_state.is_preview_mode = False
        is_preview_mode = st.session_state.is_preview_mode

        st.divider()
        st.markdown("**Model**")
        model = st.selectbox(
            "Model",
            options=["dpt-2-latest", "dpt-2-mini"],
            label_visibility="collapsed",
            format_func=lambda m: (
                "dpt-2-latest — high accuracy (3 cr/page)"
                if m == "dpt-2-latest"
                else "dpt-2-mini — faster, if available on your key (1.5 cr/page)"
            ),
        )

        st.divider()
        st.markdown("**Storage**")
        s3_env_enabled = storage.s3_enabled()
        s3_consent = st.toggle(
            "Save to S3",
            value=s3_env_enabled,
            help=(
                "Save uploaded documents and extraction outputs to your S3 bucket. "
                "Requires S3_BUCKET_NAME and AWS_REGION in .env."
            ),
        )
        st.caption("Documents are sent to Landing AI servers.")

    if is_preview_mode:
        banner_cls, banner_msg = "da-banner-preview", (
            "🟡 <strong>Demo Mode</strong> — showing a sample extraction. "
            "Add a Landing AI key in the sidebar to extract real documents."
        )
    else:
        banner_cls, banner_msg = "da-banner-cloud", (
            "☁️ <strong>Cloud Mode Active</strong> — documents are sent to Landing AI for extraction."
        )
    st.markdown(f'<div class="da-banner {banner_cls}">{banner_msg}</div>', unsafe_allow_html=True)
    st.caption("Powered by Landing AI ADE")

    doc_bytes, suffix, doc_filename = document_source_picker("cloud")

    if st.button("Extract →", type="primary"):
        if is_preview_mode:
            with st.spinner("Extracting document (simulated)…"):
                import time
                time.sleep(1.5)
            st.success("Done (Demo) — 1 page, 0 credits used, 1500ms")
            st.markdown('<div class="da-label">Extracted Text</div>', unsafe_allow_html=True)
            st.download_button(
                label="Download .md",
                data=_SAMPLE_MARKDOWN,
                file_name="extracted_sample.md",
                mime="text/markdown",
            )
            with st.container(key="da_result_scroll"):
                st.markdown(_SAMPLE_MARKDOWN)
            st.stop()

        if not api_key.strip():
            st.error("Please enter your Landing AI API Key in the sidebar to run a full extraction.")
            st.stop()

        if doc_bytes is None:
            st.error("Please upload or select a document first.")
            st.stop()

        tmp_path: str | None = None
        pdf_preview: bytes | None = None

        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix, prefix="tmp_upload"
            ) as tmp:
                tmp.write(doc_bytes)
                tmp_path = tmp.name

            client_cloud = LandingAIADE(apikey=api_key)
            with st.spinner("Extracting document…"):
                response = client_cloud.parse(document=Path(tmp_path), model=model)

            logging.info(
                "Parse complete | file=%s model=%s pages=%s credits=%s duration_ms=%s chunks=%d",
                response.metadata.filename,
                model,
                response.metadata.page_count,
                response.metadata.credit_usage,
                response.metadata.duration_ms,
                len(response.chunks),
            )
            if response.metadata.failed_pages:
                logging.warning("Failed pages: %s", response.metadata.failed_pages)

            meta = response.metadata

            if suffix == ".pdf":
                pdf_preview = render_pdf_first_page(tmp_path)

        except Exception as e:
            st.error(str(e))
            st.stop()
        finally:
            if tmp_path and Path(tmp_path).exists():
                Path(tmp_path).unlink()

        if s3_consent:
            try:
                s3 = storage.build_backend(st.session_state.session_id)
                s3.upload_document(doc_bytes, doc_filename)
                s3.upload_output(response.markdown, "extracted.md")
                st.sidebar.success(f"Saved to S3 (session: {st.session_state.session_id[:8]}…)")
            except RuntimeError as exc:
                st.sidebar.warning(f"S3 storage skipped: {exc}")
                logging.warning("S3 upload failed: %s", exc)

        chips_html = (
            f'<span class="da-chip">📄 {meta.page_count} page(s)</span>'
            f'<span class="da-chip">💳 {meta.credit_usage} credits</span>'
            f'<span class="da-chip">⏱ {meta.duration_ms} ms</span>'
        )

        render_result(
            success_msg=(
                f"Done — {meta.page_count} page(s), {meta.credit_usage} credits, {meta.duration_ms} ms"
            ),
            chips_html=chips_html,
            markdown=response.markdown,
            doc_bytes=doc_bytes,
            suffix=suffix,
            doc_filename=doc_filename,
            pdf_preview=pdf_preview,
            download_filename="extracted.md",
        )
