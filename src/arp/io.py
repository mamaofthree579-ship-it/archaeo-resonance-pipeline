import geopandas as gpd
import pandas as pd

def load_candidates_geojson(path: str):
    """
    Load candidate sites from a GeoJSON or CSV file.
    Returns a GeoDataFrame or DataFrame with columns:
    ['lat', 'lon', 'G', 'H', 'M', 'L', 'Ssym']
    """
    try:
        gdf = gpd.read_file(path)
        # Ensure lat/lon columns exist
        if 'geometry' in gdf.columns:
            gdf['lat'] = gdf.geometry.y
            gdf['lon'] = gdf.geometry.x
        return gdf
    except Exception:
        # fallback: load as CSV
        return pd.read_csv(path)
