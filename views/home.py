"""Landing page — choose Cloud or Local extraction."""

import streamlit as st

from common import page_header


def render() -> None:
    page_header(
        "Extract structured text from any document",
        "Pick how you want your documents processed. Both modes produce the same "
        "clean Markdown output — the difference is where the extraction happens.",
    )

    st.markdown(
        """
        <div class="da-badges">
            <span class="da-badge da-badge-privacy">🔒 Privacy conscious</span>
            <span class="da-badge da-badge-local">📄 PDF · PNG · JPG</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_cloud, col_local = st.columns(2, gap="large")

    with col_cloud:
        st.markdown(
            """
            <div class="da-mode-card da-mode-cloud">
                <span class="da-mode-icon">☁️</span>
                <h3>Cloud Extraction</h3>
                <p>High-accuracy extraction via Landing AI's AIDE API. Best for
                invoices, ledgers, ID documents, and government forms where
                accuracy matters most.</p>
                <div class="da-mode-tags">
                    <span class="da-mode-tag">Highest accuracy</span>
                    <span class="da-mode-tag">Needs API key</span>
                    <span class="da-mode-tag">Optional S3 storage</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        from nav import CLOUD

        st.page_link(CLOUD, label="Open Cloud mode →", use_container_width=True)

    with col_local:
        st.markdown(
            """
            <div class="da-mode-card da-mode-local">
                <span class="da-mode-icon">🖥️</span>
                <h3>Local Extraction</h3>
                <p>Runs Gemma 3 4B on your own machine via Apple MLX. No document
                or image ever leaves your device — ideal for sensitive or
                offline work.</p>
                <div class="da-mode-tags">
                    <span class="da-mode-tag">100% offline</span>
                    <span class="da-mode-tag">No API key</span>
                    <span class="da-mode-tag">No setup</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        from nav import LOCAL

        st.page_link(LOCAL, label="Open Local mode →", use_container_width=True)

    st.markdown("<div style='height: 1.75rem'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="da-card da-howto">
            <h3>How it works</h3>
            <div class="da-steps">
                <div class="da-step">
                    <div class="da-step-num">1</div>
                    <div class="da-step-text">Choose Cloud or Local above<br>
                        <span class="da-step-sub">Each opens its own dedicated page</span>
                    </div>
                </div>
                <div class="da-step">
                    <div class="da-step-num">2</div>
                    <div class="da-step-text">Upload a document<br>
                        <span class="da-step-sub">PDF, PNG, or JPG</span>
                    </div>
                </div>
                <div class="da-step">
                    <div class="da-step-num">3</div>
                    <div class="da-step-text">Copy or download<br>
                        <span class="da-step-sub">Clean structured Markdown output</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
