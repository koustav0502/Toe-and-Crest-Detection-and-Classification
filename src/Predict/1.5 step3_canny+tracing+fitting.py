#code b
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, osr, ogr
from skimage import feature
import cv2
import geopandas as gpd
from shapely.geometry import LineString, box
import os
from math import atan2, degrees
from numba import jit
from time import time
from datetime import datetime

gdal.UseExceptions()

def create_mask_boundary_buffer(mask_path, buffer_distance=2):
    """
    Create a buffer around the mask boundary to exclude edge lines.
    """
    print(f"\nCreating boundary buffer with distance {buffer_distance}...")
    mask_gdf = gpd.read_file(mask_path)
    boundaries = mask_gdf.boundary
    boundary_buffer = boundaries.buffer(buffer_distance)
    return gpd.GeoDataFrame(geometry=boundary_buffer, crs=mask_gdf.crs)

def create_geodataframe_from_lines(line_strings, crs):
    print(f"\nCreating GeoDataFrame with {len(line_strings)} line features...")
    return gpd.GeoDataFrame({'geometry': line_strings}, geometry='geometry', crs=crs)

def edge_tracing(binary_image):
    print("\nStarting edge tracing process...")
    visited = np.zeros_like(binary_image, dtype=bool)
    all_lines = []
    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    for i in range(binary_image.shape[0]):
        for j in range(binary_image.shape[1]):
            if binary_image[i, j] == 1 and not visited[i, j]:
                line_coords = []
                p = (j, i)
                line_coords.append(p)
                visited[i, j] = True

                while True:
                    neighbor_found = False
                    for offset in neighbor_offsets:
                        x, y = p[0] + offset[0], p[1] + offset[1]
                        if (0 <= x < binary_image.shape[1] and 
                            0 <= y < binary_image.shape[0] and
                            binary_image[y, x] == 1 and 
                            not visited[y, x]):
                            neighbor_found = True
                            p = (x, y)
                            line_coords.append(p)
                            visited[y, x] = True
                            break

                    if not neighbor_found:
                        break

                if len(line_coords) > 1:
                    all_lines.append(LineString(line_coords))
    
    print(f"Found {len(all_lines)} line features")
    return all_lines

@jit(nopython=True)
def pixel_to_geo(x, y, geotransform):
    geo_x = geotransform[0] + x * geotransform[1] + y * geotransform[2]
    geo_y = geotransform[3] + x * geotransform[4] + y * geotransform[5]
    return geo_x, geo_y

def is_on_or_outside_boundary(line, dem_boundary_box, boundary_buffer_gdf, tolerance=1e-6):
    """
    Check if a line lies on or extends beyond the DEM boundary or intersects with the mask boundary buffer.
    """
    if not dem_boundary_box.contains(line.buffer(tolerance)):
        return True
    return any(buffer.intersects(line) for buffer in boundary_buffer_gdf.geometry)

def create_inner_mask(mask_array, buffer_pixels=3):
    """
    Create an inner mask by eroding the original mask to avoid edge effects.
    """
    kernel = np.ones((buffer_pixels, buffer_pixels), np.uint8)
    return cv2.erode(mask_array, kernel, iterations=1)

def main():
    print(f"\n{'='*50}")
    print(f"Starting processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")
    
    total_start_time = time()
    
    input_path = r"G:\Shiva_WS\Pit_Detection_WS\Input_files\TEST_7\slopemap.tif"
    mask_path = r"G:\Shiva_WS\Pit_Detection_WS\Output_files\final_pit_boundaries.shp"
    
    if not os.path.exists(input_path):
        print(f"Error: The file {input_path} does not exist.")
        return
    
    try:
        print(f"\nLoading input file: {os.path.basename(input_path)}")
        dataset = gdal.Open(input_path)
        if dataset is None:
            print(f"Error: Unable to open the file {input_path}.")
            return
        
        input_image = dataset.ReadAsArray()
        geotransform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()
        srs = osr.SpatialReference(wkt=projection)
        
        dem_x_min, dem_y_min = pixel_to_geo(0, input_image.shape[0], geotransform)
        dem_x_max, dem_y_max = pixel_to_geo(input_image.shape[1], 0, geotransform)
        dem_boundary_box = box(dem_x_min, dem_y_min, dem_x_max, dem_y_max)
        
        boundary_buffer_gdf = create_mask_boundary_buffer(mask_path, buffer_distance=2)
        
        mask_ds = gdal.GetDriverByName('MEM').Create('', input_image.shape[1], input_image.shape[0], 1, gdal.GDT_Byte)
        mask_ds.SetGeoTransform(geotransform)
        mask_ds.SetProjection(projection)
        
        shapefile = ogr.Open(mask_path)
        layer = shapefile.GetLayer()
        gdal.RasterizeLayer(mask_ds, [1], layer, burn_values=[1])
        
        mask_array = mask_ds.ReadAsArray()
        inner_mask = create_inner_mask(mask_array, buffer_pixels=3)
        masked_input_image = np.ma.masked_array(input_image, mask=(inner_mask == 0))
        
        print("\nApplying bilateral filter...")
        blurred = cv2.bilateralFilter(
            masked_input_image.filled(0).astype(np.float32),
            d=27,
            sigmaColor=95,
            sigmaSpace=95
        )
        
        print("\nPerforming Canny edge detection...")
        edges = feature.canny(
            blurred,
            sigma=50,
            low_threshold=0.05,
            high_threshold=0.2
        )
        
        binary_image = edges.astype(np.uint8)
        line_strings = edge_tracing(binary_image)
        
        print("\nConverting to georeferenced coordinates and filtering lines...")
        geo_fitted_lines = []
        for line in line_strings:
            geo_coords = [
                pixel_to_geo(x, y, geotransform) 
                for x, y in line.coords
            ]
            geo_line = LineString(geo_coords)
            if not is_on_or_outside_boundary(geo_line, dem_boundary_box, boundary_buffer_gdf):
                geo_fitted_lines.append(geo_line)
        
        print(f"Retained {len(geo_fitted_lines)} lines after filtering")
        
        crs = srs.ExportToProj4()
        gdf = create_geodataframe_from_lines(geo_fitted_lines, crs)
        
        output_shapefile = r'C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\result\new\after feeding edges\updline_fitting_output_test_shiva_test7_clipped_1.5.shp'
        output_geojson = r'C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\result\new\after feeding edges\updline_fitting_output_shiva_test7_clipped_1.5.geojson'
        
        print(f"\nSaving Shapefile: {os.path.basename(output_shapefile)}")
        gdf.to_file(output_shapefile)
        
        print(f"Saving GeoJSON: {os.path.basename(output_geojson)}")
        gdf.to_file(output_geojson, driver='GeoJSON')
        
        total_time = time() - total_start_time
        print(f"\n{'='*50}")
        print(f"Processing completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"\nError processing the dataset: {str(e)}")
        return

if __name__ == "__main__":
    main()
