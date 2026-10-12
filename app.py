import logging
import logging.handlers
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from common import init_session, inject_theme
from nav import PAGES

_log_dir = Path.home() / "Library" / "Logs" / "DocumentAgent"
_log_dir.mkdir(parents=True, exist_ok=True)
_log_handler = logging.handlers.RotatingFileHandler(
    _log_dir / "documentagent.log", maxBytes=10 * 1024 * 1024, backupCount=3
)
_log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logging.basicConfig(level=logging.DEBUG, handlers=[_log_handler])

load_dotenv()

st.set_page_config(page_title="DocumentAgent", page_icon="📄", layout="wide")

init_session()
inject_theme()

nav = st.navigation(PAGES, position="top")
nav.run()
