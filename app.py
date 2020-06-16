import streamlit as st
import numpy as np

progress_bar = st.sidebar.progress(0)
frame_text = st.sidebar.empty()

for frame_num, a in enumerate(np.linspace(0.0, 4 * np.pi, 100)):
    # Here were setting value for these two elements.
    progress_bar.progress(frame_num)
    frame_text.text("Frame %i/100" % (frame_num + 1))