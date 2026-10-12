"""Shared UI helpers and styling for DocumentAgent's multipage Streamlit app."""

import uuid
from pathlib import Path

import streamlit as st

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}

# ── Design tokens ────────────────────────────────────────────────────────────
INK = "#0f172a"
MUTED = "#64748b"
BORDER = "#e2e8f0"
SURFACE = "#f8fafc"
ACCENT = "#4f46e5"
ACCENT_DARK = "#4338ca"
CLOUD_COLOR = "#0ea5e9"
LOCAL_COLOR = "#16a34a"


def init_session() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())


def inject_theme() -> None:
    """Global styling shared by every page — turns default Streamlit chrome
    into a clean, modern website look."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"], [data-testid="stAppViewContainer"] * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        #MainMenu, footer { visibility: hidden; }
        header[data-testid="stHeader"] { background: transparent; }
        [data-testid="stAppViewContainer"] { background: #ffffff; }
        [data-testid="stMainBlockContainer"] { padding-top: 1.5rem; max-width: 1180px; }

        /* Streamlit's own top navigation (st.navigation position="top") */
        [data-testid="stNavigation"] {
            background: #ffffff; border-bottom: 1px solid var(--da-border, #e2e8f0);
            padding: 0.4rem 0;
        }
        [data-testid="stNavigation"] a { font-weight: 600 !important; }

        div.stButton > button, div.stDownloadButton > button {
            border-radius: 10px; font-weight: 600; border: 1px solid #e2e8f0;
        }
        div.stButton > button[kind="primary"] {
            background: #4f46e5; border-color: #4f46e5;
        }
        div.stButton > button[kind="primary"]:hover {
            background: #4338ca; border-color: #4338ca;
        }

        /* ── Reusable component classes ─────────────────────────────── */
        .da-topline { font-size: 0.78rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.1em; color: #4f46e5; margin-bottom: 0.5rem; }

        .da-hero { margin-top: 0.75rem; }
        .da-hero h1 { font-size: 2.4rem; font-weight: 800; margin: 0 0 0.4rem 0;
            letter-spacing: -0.03em; color: #0f172a; }
        .da-hero p { font-size: 1.08rem; color: #64748b; max-width: 640px; margin: 0 0 1rem 0; }

        .da-badges { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 1.25rem; }
        .da-badge { display: inline-flex; align-items: center; gap: 5px; font-size: 0.78rem;
            font-weight: 600; padding: 4px 12px; border-radius: 999px; }
        .da-badge-privacy { background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }
        .da-badge-local { background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }

        .da-banner { padding: 12px 18px; border-radius: 12px; margin-bottom: 1.5rem;
            font-size: 0.92rem; font-weight: 500; border: 1px solid transparent; }
        .da-banner-local { background: #eff6ff; color: #1e40af; border-color: #bfdbfe; }
        .da-banner-preview { background: #fffbeb; color: #92400e; border-color: #fde68a; }
        .da-banner-cloud { background: #ecfdf5; color: #065f46; border-color: #a7f3d0; }

        .da-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px;
            padding: 1.5rem 1.7rem; box-shadow: 0 1px 2px rgba(15,23,42,0.04); }

        .da-howto h3 { font-size: 0.95rem; font-weight: 700; margin: 0 0 0.9rem 0; color: #1e293b; }
        .da-steps { display: flex; gap: 2rem; flex-wrap: wrap; }
        .da-step { display: flex; align-items: flex-start; gap: 10px; min-width: 220px; }
        .da-step-num { width: 26px; height: 26px; border-radius: 50%; background: #4f46e5;
            color: white; font-size: 0.78rem; font-weight: 700; display: flex;
            align-items: center; justify-content: center; flex-shrink: 0; margin-top: 1px; }
        .da-step-text { font-size: 0.87rem; color: #475569; line-height: 1.45; }
        .da-step-sub { color: #94a3b8; }

        .da-label { font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.08em; color: #64748b; margin-bottom: 0.5rem; }

        .da-chips { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 0.85rem; }
        .da-chip { background: #f1f5f9; color: #334155; font-size: 0.78rem; font-weight: 600;
            padding: 4px 11px; border-radius: 999px; border: 1px solid #e2e8f0; }

        div.st-key-da_result_scroll { border: 1px solid #e2e8f0; border-radius: 12px;
            background: #ffffff; max-height: 640px; overflow-y: auto; overflow-x: hidden;
            padding: 1rem 1.2rem; line-height: 1.65; }
        div.st-key-da_result_scroll p, div.st-key-da_result_scroll li,
        div.st-key-da_result_scroll h1, div.st-key-da_result_scroll h2,
        div.st-key-da_result_scroll h3, div.st-key-da_result_scroll h4 {
            overflow-wrap: anywhere; word-break: break-word; white-space: normal;
        }
        div.st-key-da_result_scroll pre, div.st-key-da_result_scroll code {
            white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word;
        }
        div.st-key-da_result_scroll table { display: block; overflow-x: auto; max-width: 100%; }
        div.st-key-da_result_scroll img { max-width: 100%; height: auto; }

        /* ── Mode selector cards on the home page ───────────────────── */
        .da-mode-card { display: block; border: 1px solid #e2e8f0; border-radius: 18px;
            padding: 1.8rem 1.8rem 1.6rem; height: 100%; text-decoration: none !important;
            transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease; }
        .da-mode-card:hover { transform: translateY(-3px);
            box-shadow: 0 12px 24px rgba(15,23,42,0.08); }
        .da-mode-cloud { border-top: 4px solid #0ea5e9; }
        .da-mode-local { border-top: 4px solid #16a34a; }
        .da-mode-icon { font-size: 1.8rem; margin-bottom: 0.6rem; display: block; }
        .da-mode-card h3 { font-size: 1.15rem; font-weight: 700; margin: 0 0 0.4rem 0; color: #0f172a; }
        .da-mode-card p { font-size: 0.9rem; color: #64748b; line-height: 1.5; margin: 0 0 0.9rem 0; }
        .da-mode-tags { display: flex; gap: 6px; flex-wrap: wrap; }
        .da-mode-tag { font-size: 0.72rem; font-weight: 600; color: #475569; background: #f1f5f9;
            border-radius: 999px; padding: 2px 9px; }
        /* The real st.page_link button, visually docked under each mode card */
        div[data-testid="stPageLink"] { margin-top: -0.7rem; margin-bottom: 1.5rem; }
        div[data-testid="stPageLink"] a {
            display: flex; align-items: center; justify-content: center;
            background: #0f172a; padding: 11px 16px; border-radius: 0 0 14px 14px;
            text-decoration: none !important;
        }
        div[data-testid="stPageLink"] a:hover { background: #1e293b; }
        div[data-testid="stPageLink"] a p {
            color: #ffffff !important; margin: 0; font-weight: 700; font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="da-hero">
            <div class="da-topline">DocumentAgent</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_sample_files() -> list[str]:
    samples_dir = Path("samples")
    if not samples_dir.is_dir():
        return []
    return [
        f.name
        for f in sorted(samples_dir.iterdir())
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def render_pdf_first_page(path: str) -> bytes | None:
    """Return PNG bytes for the first page of a PDF (for preview only)."""
    try:
        import fitz
        doc = fitz.open(path)
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img = pix.tobytes("png")
        doc.close()
        return img
    except Exception:
        return None


def document_source_picker(key_prefix: str):
    """Renders the upload/sample picker shared by both extraction pages.

    Returns (doc_bytes, suffix, filename) or (None, None, None) if nothing
    is selected yet.
    """
    sample_files = get_sample_files()

    if sample_files:
        input_mode = st.radio(
            "Document source",
            options=["Upload", "Use sample"],
            horizontal=True,
            key=f"{key_prefix}_input_mode",
        )
    else:
        input_mode = "Upload"

    if input_mode == "Upload":
        uploaded_file = st.file_uploader(
            "Drop your document here",
            type=["pdf", "png", "jpg", "jpeg"],
            label_visibility="collapsed",
            key=f"{key_prefix}_uploader",
        )
        if uploaded_file is None:
            return None, None, None
        return uploaded_file.read(), Path(uploaded_file.name).suffix.lower(), uploaded_file.name

    chosen_sample = st.selectbox(
        "Select a sample document", options=sample_files, key=f"{key_prefix}_sample"
    )
    if not chosen_sample:
        return None, None, None
    sample_path = Path("samples") / chosen_sample
    if not sample_path.exists():
        return None, None, None
    return sample_path.read_bytes(), sample_path.suffix.lower(), sample_path.name


def render_result(
    *,
    success_msg: str,
    chips_html: str,
    markdown: str,
    doc_bytes: bytes,
    suffix: str,
    doc_filename: str,
    pdf_preview: bytes | None,
    download_filename: str,
) -> None:
    st.success(success_msg)

    col_doc, col_result = st.columns([1, 1], gap="large")

    with col_doc:
        st.markdown('<div class="da-label">Document</div>', unsafe_allow_html=True)
        if suffix in {".jpg", ".jpeg", ".png"}:
            st.image(doc_bytes, caption=doc_filename, use_container_width=True)
        elif suffix == ".pdf" and pdf_preview:
            st.image(pdf_preview, caption=f"{doc_filename} — page 1", use_container_width=True)
        else:
            st.info("Preview not available for this file type.")

    with col_result:
        st.markdown('<div class="da-label">Extracted Text</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="da-chips">{chips_html}</div>', unsafe_allow_html=True)
        st.download_button(
            label="Download .md",
            data=markdown,
            file_name=download_filename,
            mime="text/markdown",
        )
        with st.container(key="da_result_scroll"):
            st.markdown(markdown)
