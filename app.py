import streamlit as st
import geopandas as gpd
import pandas as pd
from src.arp.io import load_candidates_geojson

st.set_page_config(page_title="Archaeo Resonance Explorer", layout="wide")
st.title("Archaeo-Resonance Explorer")

with st.sidebar:
    st.header("Controls")
    w_g = st.slider("w_g (geometry)", 0.0, 1.0, 0.18, 0.01)
    w_h = st.slider("w_h (harmonics)", 0.0, 1.0, 0.26, 0.01)
    w_m = st.slider("w_m (mag)", 0.0, 1.0, 0.18, 0.01)
    w_l = st.slider("w_l (lidar)", 0.0, 1.0, 0.22, 0.01)
    w_s = st.slider("w_s (symbolic)", 0.0, 1.0, 0.16, 0.01)
    theta = st.slider("theta (bias)", 0.0, 1.0, 0.5, 0.01)
    lam = st.slider("lambda (sigmoid)", 0.1, 10.0, 6.0, 0.1)

candidates = load_candidates_geojson("examples/known_sites.geojson")
st.write("## Candidate map & list")
st.dataframe(pd.DataFrame(candidates))
