import geopandas as gpd
import rasterio
import rasterio.mask
from rasterio.features import geometry_mask

# Define file paths
shapefile_path = r"G:\Shiva_WS\Pit_Detection_WS\Output_files\final_pit_boundaries.shp"
dem_path = r"G:\Shiva_WS\Pit_Detection_WS\Input_files\TEST_7\slopemap.tif"
output_path = r"G:\Shiva_WS\Pit_Detection_WS\Input_files\TEST_7\clipped_slopemap.tif"

# Load the shapefile and extract the boundary
gdf = gpd.read_file(shapefile_path)
boundary = gdf.geometry.unary_union

# Ensure the shapefile and DEM have the same CRS
with rasterio.open(dem_path) as dem_src:
    if gdf.crs != dem_src.crs:
        gdf = gdf.to_crs(dem_src.crs)
    # Mask DEM using the shapefile boundary
    out_image, out_transform = rasterio.mask.mask(dem_src, [boundary], crop=True)
    out_meta = dem_src.meta.copy()

# Update metadata to reflect new dimensions and transform
out_meta.update({
    "driver": "GTiff",
    "height": out_image.shape[1],
    "width": out_image.shape[2],
    "transform": out_transform,
})

# Write the clipped DEM to a new file
with rasterio.open(output_path, "w", **out_meta) as dest:
    dest.write(out_image)

print(f"Clipped DEM saved to {output_path}")
