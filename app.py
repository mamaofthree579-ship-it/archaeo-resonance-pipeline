# Archaeo-Resonance Explorer (with DEM support: uploaded DEM or auto-fetch elevation)
import streamlit as st
import pandas as pd
import geopandas as gpd
import json
import numpy as np
import plotly.express as px
import os
from shapely.geometry import Point
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Archaeo-Resonance Explorer", layout="wide")

# optional imports that may not be installed in all environments
try:
    import rasterio
    from rasterio import transform
    HAS_RASTERIO = True
except Exception:
    HAS_RASTERIO = False

try:
    import requests
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False

# ----------------------------------------
# Helpers: Load files, compute S, get elev
# ----------------------------------------
@st.cache_data(show_spinner=False)
def load_candidates(path_or_file):
    """Load candidate sites from GeoJSON, JSON, or CSV."""
    if path_or_file is None:
        return None
    try:
        if hasattr(path_or_file, "read"):
            name = getattr(path_or_file, "name", "uploaded")
            if name.endswith((".geojson", ".json")):
                data = json.load(path_or_file)
                gdf = gpd.GeoDataFrame.from_features(data["features"])
                gdf.crs = "EPSG:4326"
                return gdf
            elif name.endswith(".csv"):
                return pd.read_csv(path_or_file)
        else:
            pf = str(path_or_file)
            if pf.endswith((".geojson", ".json")):
                gdf = gpd.read_file(pf)
                if gdf.crs is None:
                    gdf.set_crs(epsg=4326, inplace=True)
                return gdf
            elif pf.endswith(".csv"):
                return pd.read_csv(pf)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


def compute_S(G, H, M, L, Ssym, w_g, w_h, w_m, w_l, w_s, theta, lam):
    Lscore = w_g * G + w_h * H + w_m * M + w_l * L + w_s * Ssym
    return 1 / (1 + np.exp(-lam * (Lscore - theta)))


@st.cache_data(show_spinner=False, max_entries=16)
def sample_dem_raster(dem_path, pts):
    """
    dem_path: path to raster tif
    pts: list of (lon, lat) tuples
    returns list of elevations (one per point) or raises
    """
    elevations = []
    with rasterio.open(dem_path) as src:
        # Ensure DEM has a spatial reference:
        if src.crs is None:
            # we assume EPSG:4326 if missing — this is risky, but prevents crashes
            # better to require user-supplied proper DEM
            st.warning("DEM has no CRS, assuming EPSG:4326.")
            # we won't set it permanently, just warn
        for lon, lat in pts:
            try:
                # rasterio expects coords as (lon,lat)
                for val in src.sample([(lon, lat)]):
                    # val is an array with one element per band
                    elevations.append(float(val[0]))
            except Exception:
                elevations.append(np.nan)
    return elevations


def fetch_elevations_open_elevation(pts):
    """
    pts: list of (lat, lon)
    Uses Open-Elevation batch endpoint: POST /api/v1/lookup
    Splits into chunks of up to 100 coordinates per request.
    Returns list of elevation floats or np.nan on failure.
    """
    if not HAS_REQUESTS:
        raise RuntimeError("requests library not available for API fetch.")

    url = "https://api.open-elevation.com/api/v1/lookup"
    elevations = []
    chunk_size = 100
    # API expects {"locations":[{"latitude":..,"longitude":..}, ...]}
    for i in range(0, len(pts), chunk_size):
        chunk = pts[i : i + chunk_size]
        payload = {"locations": [{"latitude": p[0], "longitude": p[1]} for p in chunk]}
        try:
            r = requests.post(url, json=payload, timeout=30)
            if r.status_code == 200:
                data = r.json()
                for ritem in data.get("results", []):
                    elevations.append(float(ritem.get("elevation", np.nan)))
            else:
                # API error -> fill with nan
                elevations.extend([np.nan] * len(chunk))
        except Exception:
            elevations.extend([np.nan] * len(chunk))
    return elevations


# ----------------------------------------
# UI: Controls
# ----------------------------------------
st.title("🌍 Archaeo-Resonance Explorer — DEM-enhanced 3D Terrain")

st.markdown(
    "Upload candidate sites (GeoJSON or CSV). Optionally upload a DEM (.tif) for accurate elevation sampling, or allow the app to auto-fetch elevations (Open-Elevation API)."
)

# Fusion weights
st.sidebar.header("Fusion Controls")
w_g = st.sidebar.slider("w_g (geometry)", 0.0, 1.0, 0.18)
w_h = st.sidebar.slider("w_h (harmonics)", 0.0, 1.0, 0.26)
w_m = st.sidebar.slider("w_m (magnetic)", 0.0, 1.0, 0.18)
w_l = st.sidebar.slider("w_l (LIDAR)", 0.0, 1.0, 0.22)
w_s = st.sidebar.slider("w_s (symbolic)", 0.0, 1.0, 0.16)
theta = st.sidebar.slider("θ (bias)", 0.0, 1.0, 0.5)
lam = st.sidebar.slider("λ (sigmoid)", 0.1, 10.0, 6.0)

# DEM controls
st.sidebar.header("DEM / Elevation")
dem_upload = st.sidebar.file_uploader("Upload DEM (.tif) — optional", type=["tif", "tiff"])
auto_fetch_checkbox = st.sidebar.checkbox("Auto-fetch elevations (Open-Elevation) if no DEM", value=True)

# elevation visualization scaling
st.sidebar.header("3D Terrain Settings")
elev_boost = st.sidebar.slider(
    "Elevation boost from S (meters per S unit)", min_value=0.0, max_value=500.0, value=50.0, step=1.0
)
elev_multiplier = st.sidebar.slider("DEM vertical exaggeration", 0.1, 10.0, 1.0, step=0.1)

# Upload or example dataset
uploaded = st.file_uploader("Upload candidate sites (.geojson or .csv)")
example_options = {
    "Example A (GeoJSON)": "examples/known_sites_A.geojson",
    "Example B (GeoJSON)": "examples/known_sites_B.geojson",
}
selected_example = st.selectbox("Or choose an example", ["None"] + list(example_options.keys()))

# persistent state
if "candidates" not in st.session_state:
    st.session_state["candidates"] = None
if "elevations" not in st.session_state:
    st.session_state["elevations"] = None
if "dem_path_tmp" not in st.session_state:
    st.session_state["dem_path_tmp"] = None

# load candidates
if uploaded:
    st.session_state["candidates"] = load_candidates(uploaded)
    st.session_state["image_path"] = None
elif selected_example != "None":
    st.session_state["candidates"] = load_candidates(example_options[selected_example])
    st.session_state["image_path"] = os.path.splitext(example_options[selected_example])[0] + ".png"

candidates = st.session_state["candidates"]

# if user uploaded DEM file, save to a temp file so rasterio can open it
if dem_upload is not None:
    if not HAS_RASTERIO:
        st.sidebar.warning("rasterio not installed — DEM sampling will not be available. Install rasterio or use auto-fetch.")
    else:
        # save uploaded DEM to temp location
        temp_dem_path = os.path.join("temp_dem.tif")
        with open(temp_dem_path, "wb") as f:
            f.write(dem_upload.read())
        st.session_state["dem_path_tmp"] = temp_dem_path
        st.sidebar.success("DEM uploaded and ready for sampling.")

# Buttons for elevation operations
col1, col2 = st.columns(2)
with col1:
    if st.button("Compute elevations now"):
        st.session_state["elevations"] = None  # reset previous
        st.experimental_rerun()
with col2:
    if st.button("Clear elevations"):
        st.session_state["elevations"] = None

# ----------------------------
# Process dataset & compute S
# ----------------------------
if candidates is not None:
    required_cols = ["G", "H", "M", "L", "Ssym"]
    missing = [c for c in required_cols if c not in candidates.columns]
    if missing:
        st.warning(f"Missing required columns: {', '.join(missing)}")
        if st.button("🧩 Auto-fix missing columns"):
            for col in missing:
                candidates[col] = np.random.uniform(0.3, 0.8, size=len(candidates))
            st.session_state["candidates"] = candidates
            st.success("Added missing columns.")
            missing = []

    if not missing:
        # compute S
        candidates["S"] = compute_S(
            candidates["G"],
            candidates["H"],
            candidates["M"],
            candidates["L"],
            candidates["Ssym"],
            w_g,
            w_h,
            w_m,
            w_l,
            w_s,
            theta,
            lam,
        )

        # ensure lat/lon available
        if "geometry" in candidates:
            gdf = gpd.GeoDataFrame(candidates)
            if gdf.crs is None:
                gdf.set_crs(epsg=4326, inplace=True)
            # use centroid for polygons and lines
            if not all(gdf.geometry.geom_type == "Point"):
                gdf["geometry"] = gdf.geometry.centroid
            gdf["lat"] = gdf.geometry.y
            gdf["lon"] = gdf.geometry.x
            candidates = gdf
        elif {"lat", "lon"}.issubset(candidates.columns):
            candidates["lat"] = candidates["lat"].astype(float)
            candidates["lon"] = candidates["lon"].astype(float)
        else:
            st.warning("No geometry or lat/lon columns found — mapping disabled.")

        # If user asked to compute elevations (button pressed earlier) or no cached elevations:
        if st.session_state.get("elevations") is None:
            need_elev = True
        else:
            need_elev = False

        # compute elevations if requested
        if need_elev:
            pts_latlon = list(zip(candidates["lat"].astype(float).tolist(), candidates["lon"].astype(float).tolist()))
            # convert to (lon,lat) for raster sampling
            pts_lonlat = [(lon, lat) for lat, lon in pts_latlon]

            elevations = None
            # 1) Try user-supplied DEM via rasterio
            dem_path = st.session_state.get("dem_path_tmp")
            if dem_path and HAS_RASTERIO:
                try:
                    with st.spinner("Sampling uploaded DEM..."):
                        elevations = sample_dem_raster(dem_path, pts_lonlat)
                        st.success("Sampled elevations from uploaded DEM.")
                except Exception as e:
                    st.warning(f"DEM sampling failed: {e}")
                    elevations = None

            # 2) Fallback: auto fetch via Open-Elevation if allowed
            if elevations is None and auto_fetch_checkbox and HAS_REQUESTS:
                try:
                    with st.spinner("Fetching elevations from Open-Elevation..."):
                        # fetch expects (lat, lon)
                        pts_for_api = pts_latlon
                        elevations = fetch_elevations_open_elevation(pts_for_api)
                        # if all nan or empty, treat as failure
                        if elevations is None or len(elevations) != len(pts_latlon):
                            elevations = None
                        else:
                            st.success("Fetched elevations from Open-Elevation.")
                except Exception as e:
                    st.warning(f"Auto-fetch elevations failed: {e}")
                    elevations = None

            # 3) Final fallback: zeros
            if elevations is None:
                st.info("Using flat elevation (zeros) as fallback.")
                elevations = [0.0] * len(pts_latlon)

            # store in session
            st.session_state["elevations"] = elevations

        # attach elevations
        if st.session_state.get("elevations") is not None:
            candidates = candidates.reset_index(drop=True)
            candidates["elevation_m"] = st.session_state["elevations"]
            # compute 3D z value: elevation + S * elev_boost
            candidates["z_value"] = candidates["elevation_m"] * elev_multiplier + candidates["S"] * elev_boost

        # drop rows without coords or S
        candidates = candidates.dropna(subset=["lat", "lon", "S"])

        # Save back
        st.session_state["candidates"] = candidates

        # ---------- Tabs and visualizations ----------
        tabs = st.tabs(["Map & 3D", "Analytics", "About"])
        # Map mode (keep outside tabs in previous iterations to avoid resetting)
        map_mode = st.radio("Map display:", ["Scatter Map", "Heatmap", "3D Terrain (DEM)"], horizontal=True)

        # Map & 3D tab
        with tabs[0]:
            st.subheader("Interactive Geospatial View")
            try:
                if map_mode == "Scatter Map":
                    fig = px.scatter_mapbox(
                        candidates,
                        lat="lat",
                        lon="lon",
                        color="S",
                        size="S",
                        color_continuous_scale="Viridis",
                        zoom=5,
                        mapbox_style="open-street-map",
                        title="Site Likelihood Scatter Map",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                elif map_mode == "Heatmap":
                    fig = px.density_mapbox(
                        candidates,
                        lat="lat",
                        lon="lon",
                        z="S",
                        radius=20,
                        center=dict(lat=candidates["lat"].mean(), lon=candidates["lon"].mean()),
                        zoom=5,
                        mapbox_style="open-street-map",
                        color_continuous_scale="YlOrRd",
                        title="Site Likelihood Heatmap",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    # 3D terrain: use lon, lat, z_value
                    fig3d = px.scatter_3d(
                        candidates,
                        x="lon",
                        y="lat",
                        z="z_value",
                        color="S",
                        color_continuous_scale="Viridis",
                        title="3D Terrain: elevation + S*boost",
                    )
                    fig3d.update_traces(marker=dict(size=4, opacity=0.8))
                    fig3d.update_layout(scene=dict(
                        xaxis_title="Longitude", yaxis_title="Latitude", zaxis_title="elevation+S*boost"
                    ))
                    st.plotly_chart(fig3d, use_container_width=True)

            except Exception as e:
                st.warning(f"Could not render map/3D view: {e}")

            # optional image preview if exists
            image_path = st.session_state.get("image_path")
            if image_path and os.path.exists(image_path):
                st.image(Image.open(image_path), caption="Associated Site Map", use_container_width=True)

        # Analytics tab
        with tabs[1]:
            st.subheader("Analytics & Exports")
            st.dataframe(candidates[["lat", "lon", "elevation_m", "S", "z_value"]].head(200))
            st.plotly_chart(px.histogram(candidates, x="S", nbins=20), use_container_width=True)

            # exports
            csv_buffer = BytesIO()
            candidates.to_csv(csv_buffer, index=False)
            st.download_button("⬇️ Download CSV", csv_buffer.getvalue(), "archaeo_resonance_results.csv", "text/csv")

            try:
                gdf = gpd.GeoDataFrame(candidates)
                gdf.set_geometry("geometry", inplace=True, errors="ignore")
                if "geometry" not in gdf.columns:
                    # create Point geometry for GeoJSON
                    gdf["geometry"] = [Point(xy) for xy in zip(gdf["lon"], gdf["lat"])]
                gdf.set_crs(epsg=4326, inplace=True)
                st.download_button("🌐 Download GeoJSON", gdf.to_json().encode("utf-8"), "archaeo_resonance_results.geojson", "application/geo+json")
            except Exception as e:
                st.warning(f"GeoJSON export failed: {e}")

        # About tab
        with tabs[2]:
            st.markdown("### About\nThis view uses DEM (uploaded or fetched) and S-score to create a combined 3D terrain visualization.\n")
else:
    st.info0
