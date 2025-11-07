import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import json
import plotly.express as px
import rasterio

# ---------------------------
# Utility: Load candidates
# ---------------------------
def load_candidates(path_or_file):
    if path_or_file is None:
        return None
    try:
        if hasattr(path_or_file, "read"):
            name = getattr(path_or_file, "name", "uploaded")
            if name.endswith((".geojson", ".json")):
                data = json.load(path_or_file)
                gdf = gpd.GeoDataFrame.from_features(data["features"])
                return gdf
            elif name.endswith(".csv"):
                return pd.read_csv(path_or_file)
        else:
            if str(path_or_file).endswith(".geojson"):
                return gpd.read_file(path_or_file)
            elif str(path_or_file).endswith(".csv"):
                return pd.read_csv(path_or_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


# ---------------------------
# Fusion computation
# ---------------------------
def compute_S(G, H, M, L, Ssym, w_g, w_h, w_m, w_l, w_s, theta, lam):
    Lscore = w_g * G + w_h * H + w_m * M + w_l * L + w_s * Ssym
    S = 1 / (1 + np.exp(-lam * (Lscore - theta)))
    return S


# ---------------------------
# DEM extraction
# ---------------------------
def extract_elevation_from_raster(raster_path, lat, lon):
    elevations = []
    try:
        with rasterio.open(raster_path) as src:
            for la, lo in zip(lat, lon):
                try:
                    row, col = src.index(lo, la)
                    val = src.read(1)[row, col]
                    elevations.append(val)
                except Exception:
                    elevations.append(np.nan)
    except Exception:
        elevations = [np.nan] * len(lat)
    return elevations


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Archaeo-Resonance Explorer", layout="wide")
st.title("🌍 Archaeo-Resonance Explorer")

st.markdown("""
Upload or select example candidate sites to compute and visualize resonance likelihoods.
Use the **Fusion Controls** in the sidebar to adjust weighting.
""")

# Sidebar controls
st.sidebar.header("Fusion Controls")
w_g = st.sidebar.slider("w_g (geometry)", 0.0, 1.0, 0.18)
w_h = st.sidebar.slider("w_h (harmonics)", 0.0, 1.0, 0.26)
w_m = st.sidebar.slider("w_m (magnetic)", 0.0, 1.0, 0.18)
w_l = st.sidebar.slider("w_l (LIDAR)", 0.0, 1.0, 0.22)
w_s = st.sidebar.slider("w_s (symbolic)", 0.0, 1.0, 0.16)
theta = st.sidebar.slider("θ (bias)", 0.0, 1.0, 0.5)
lam = st.sidebar.slider("λ (sigmoid)", 0.1, 10.0, 6.0)

# Upload or example selection
uploaded = st.file_uploader("Upload candidate sites (.geojson or .csv)")
example_options = {
    "Example A (GeoJSON)": "examples/known_sites_A.geojson",
    "Example B (GeoJSON)": "examples/known_sites_B.geojson",
}
selected_example = st.selectbox("Or choose an example", ["None"] + list(example_options.keys()))

# Optional DEM
dem_file = st.file_uploader("Optional DEM (.tif or .asc)", type=["tif", "asc"])

if uploaded:
    candidates = load_candidates(uploaded)
elif selected_example != "None":
    candidates = load_candidates(example_options[selected_example])
else:
    candidates = None


# ---------------------------
# Main logic
# ---------------------------
if candidates is not None:
    required_cols = ["G", "H", "M", "L", "Ssym"]
    missing = [c for c in required_cols if c not in candidates.columns]

    if missing:
        st.warning(f"Missing required columns: {', '.join(missing)}")
        if st.button("🧩 Auto-fix missing columns"):
            for col in missing:
                candidates[col] = np.random.uniform(0.3, 0.8, len(candidates))
            st.success("Added missing columns with default random values.")
            missing = []

    if not missing:
        candidates["S"] = compute_S(
            candidates["G"], candidates["H"], candidates["M"],
            candidates["L"], candidates["Ssym"],
            w_g, w_h, w_m, w_l, w_s, theta, lam
        )

        # Ensure coordinates
        if "geometry" in candidates:
            try:
                gdf = gpd.GeoDataFrame(candidates, geometry="geometry", crs="EPSG:4326")
                candidates["lat"] = gdf.geometry.centroid.y
                candidates["lon"] = gdf.geometry.centroid.x
            except Exception:
                st.warning("Geometry column found but could not extract coordinates.")
        else:
            if "lat" not in candidates or "lon" not in candidates:
                st.warning("No spatial data found. Generating synthetic coordinates.")
                candidates["lat"] = np.random.uniform(35, 40, len(candidates))
                candidates["lon"] = np.random.uniform(-5, 5, len(candidates))

        # DEM or synthetic elevation
        if dem_file is not None:
            st.info("Extracting elevation from uploaded DEM...")
            candidates["elevation"] = extract_elevation_from_raster(
                dem_file, candidates["lat"], candidates["lon"]
            )
        else:
            st.info("No DEM uploaded — generating synthetic terrain.")
            base = np.sin(candidates["lat"] / 2) * 100 + np.cos(candidates["lon"] / 2) * 100
            candidates["elevation"] = base + np.random.uniform(-20, 20, len(candidates))

        candidates["z"] = candidates["elevation"].fillna(0) + candidates["S"] * 100

        # ---------------------------
        # Tabs
        # ---------------------------
        tabs = st.tabs([
            "📊 Data",
            "📈 Scatter",
            "🔥 Heatmap",
            "🏔️ 3D Terrain",
            "📉 Feature Importance",
        ])

        # 📊 Data Tab
        with tabs[0]:
            st.subheader("Candidate Site Data")
            st.dataframe(
                candidates[["lat", "lon", "G", "H", "M", "L", "Ssym", "S", "elevation"]],
                use_container_width=True
            )
            csv = candidates.to_csv(index=False).encode("utf-8")
            st.download_button(
                "💾 Download Results as CSV",
                csv,
                "archaeo_resonance_results.csv",
                "text/csv",
                key="download-csv",
            )

        # 📈 Scatter Tab
        with tabs[1]:
            st.subheader("Scatter Map")
            fig = px.scatter_mapbox(
                candidates, lat="lat", lon="lon", color="S",
                color_continuous_scale="Viridis", zoom=5,
                hover_data=["S"], size="S", mapbox_style="open-street-map"
            )
            st.plotly_chart(fig, use_container_width=True)

        # 🔥 Heatmap Tab
        with tabs[2]:
            st.subheader("Heatmap View")
            try:
                fig = px.density_mapbox(
                    candidates, lat="lat", lon="lon", z="S",
                    radius=25, zoom=5, mapbox_style="open-street-map",
                    color_continuous_scale="Inferno"
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create heatmap: {e}")

        # 🏔️ 3D Terrain Tab
        with tabs[3]:
            st.subheader("3D Terrain View")
            try:
                fig = px.scatter_3d(
                    candidates, x="lon", y="lat", z="z",
                    color="S", color_continuous_scale="Viridis",
                    size="S", title="3D Terrain + Resonance Score"
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render 3D terrain: {e}")

        # 📉 Feature Importance Tab
        with tabs[4]:
            st.subheader("Feature Importance (Weight Contribution)")
            weights = {
                "Geometry (w_g)": w_g,
                "Harmonics (w_h)": w_h,
                "Magnetic (w_m)": w_m,
                "LIDAR (w_l)": w_l,
                "Symbolic (w_s)": w_s,
            }
            df_weights = pd.DataFrame(list(weights.items()), columns=["Feature", "Weight"])
            fig = px.bar(
                df_weights, x="Feature", y="Weight",
                color="Weight", color_continuous_scale="Viridis",
                title="Relative Contribution of Each Feature"
            )
            fig.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

            # Compute actual average contribution
            candidates["geometry_contrib"] = w_g * candidates["G"]
            candidates["harmonics_contrib"] = w_h * candidates["H"]
            candidates["magnetic_contrib"] = w_m * candidates["M"]
            candidates["lidar_contrib"] = w_l * candidates["L"]
            candidates["symbolic_contrib"] = w_s * candidates["Ssym"]

            contrib_means = candidates[
                ["geometry_contrib", "harmonics_contrib", "magnetic_contrib", "lidar_contrib", "symbolic_contrib"]
            ].mean().reset_index()

            contrib_means.columns = ["Feature", "Avg Contribution"]

            st.markdown("#### Average Feature Contribution to Site Likelihood (Across Dataset)")
            st.dataframe(contrib_means, use_container_width=True)

else:
    st.info("👋 Upload a candidate sites file or select an example to get started.")
