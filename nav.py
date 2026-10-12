"""Central registry of st.Page objects — imported by app.py to build the
navigation, and by views/home.py to link to Cloud/Local from the mode cards."""

import streamlit as st

from views import cloud, home, local

HOME = st.Page(home.render, title="Home", icon="🏠", url_path="home", default=True)
CLOUD = st.Page(cloud.render, title="Cloud", icon="☁️", url_path="cloud")
LOCAL = st.Page(local.render, title="Local", icon="🖥️", url_path="local")

PAGES = [HOME, CLOUD, LOCAL]
