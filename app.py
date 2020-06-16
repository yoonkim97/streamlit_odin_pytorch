import streamlit as st
import numpy as np
import time

model_bar = st.progress(0)
for p in range(100):
    model_bar.progress(p + 1)