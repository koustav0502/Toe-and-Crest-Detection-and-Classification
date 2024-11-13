import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, osr
from skimage import feature
import cv2
import geopandas as gpd
from shapely.geometry import LineString, box
import os
import pandas as pd
from math import atan2, degrees
from numba import jit
import numpy.typing as npt
from time import time
from datetime import datetime
from rasterio.mask import mask
import rasterio

gdal.UseExceptions()

def create_geodataframe_from_lines(line_strings, crs):
    print(f"\nCreating GeoDataFrame with {len(line_strings)} line features...")
    return gpd.GeoDataFrame(
        {'geometry': line_strings}, 
        geometry='geometry', 
        crs=crs
    )

@jit(nopython=True)
def find_neighbors(i: int, j: int, binary_image: npt.NDArray) -> list:
    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    neighbors = []
    for di, dj in neighbor_offsets:
        ni, nj = i + di, j + dj
        if (0 <= ni < binary_image.shape[0] and 
            0 <= nj < binary_image.shape[1] and 
            binary_image[ni, nj] == 1):
            neighbors.append((ni, nj))
    return neighbors

def edge_tracing(binary_image: npt.NDArray) -> list:
    print("\nStarting edge tracing process...")
    start_time = time()
    
    visited = np.zeros_like(binary_image, dtype=bool)
    all_lines = []
    
    print("Finding edge pixels...")
    edge_pixels = np.column_stack(np.where(binary_image == 1))
    total_pixels = len(edge_pixels)
    print(f"Found {total_pixels} edge pixels to process")
    
    processed_count = 0
    last_percentage = 0
    
    for i, j in edge_pixels:
        if not visited[i, j]:
            line_coords = [(j, i)]  # (x, y)
            visited[i, j] = True
            
            stack = [(i, j)]
            while stack:
                current_i, current_j = stack.pop()
                neighbors = find_neighbors(current_i, current_j, binary_image)
                
                for ni, nj in neighbors:
                    if not visited[ni, nj]:
                        visited[ni, nj] = True
                        line_coords.append((nj, ni))
                        stack.append((ni, nj))
            
            if len(line_coords) > 1:
                all_lines.append(LineString(line_coords))
        
        processed_count += 1
        percentage = (processed_count * 100) // total_pixels
        if percentage > last_percentage and percentage % 10 == 0:
            print(f"Edge tracing progress: {percentage}% complete")
            last_percentage = percentage
    
    elapsed_time = time() - start_time
    print(f"\nEdge tracing completed in {elapsed_time:.2f} seconds")
    print(f"Found {len(all_lines)} line features")
    
    return all_lines

@jit(nopython=True)
def calculate_angle_numba(vec1: npt.NDArray, vec2: npt.NDArray) -> float:
    det = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    dot = np.dot(vec1, vec2)
    angle = atan2(det, dot)
    return abs(degrees(angle))

def fit_lines_to_pixel_lists(pixel_lists: list, delta_l: int = 2, alpha_d: float = 70) -> list:
    print("\nStarting line fitting process...")
    start_time = time()
    
    line_lists = []
    total_lines = len(pixel_lists)
    print(f"Processing {total_lines} line features")
    
    for idx, line in enumerate(pixel_lists, 1):
        if idx % 100 == 0:
            print(f"Fitting progress: {(idx * 100) // total_lines}% ({idx}/{total_lines})")
        
        pixel_list = np.array(list(line.coords))
        if len(pixel_list) < delta_l:
            continue
        
        current_lines = []
        segments = []
        
        for i in range(0, len(pixel_list) - delta_l, delta_l):
            segment = pixel_list[i:i + delta_l + 1]
            segments.append(LineString([segment[0], segment[-1]]))
        
        if segments:
            first_line = segments[0]
            current_lines.append(first_line)
            
            first_vec = np.array(list(first_line.coords)[-1]) - np.array(list(first_line.coords)[0])
            
            for segment in segments[1:]:
                seg_vec = np.array(list(segment.coords)[-1]) - np.array(list(segment.coords)[0])
                angle = calculate_angle_numba(first_vec, seg_vec)
                
                if angle > alpha_d:
                    line_lists.append(current_lines)
                    current_lines = [segment]
                    first_line = segment
                    first_vec = seg_vec
                else:
                    current_lines.append(segment)
            
            if current_lines:
                line_lists.append(current_lines)
    
    elapsed_time = time() - start_time
    print(f"\nLine fitting completed in {elapsed_time:.2f} seconds")
    print(f"Generated {len(line_lists)} fitted line groups")
    
    return line_lists

@jit(nopython=True)
def pixel_to_geo(x: float, y: float, geotransform: tuple) -> tuple:
    geo_x = geotransform[0] + x * geotransform[1] + y * geotransform[2]
    geo_y = geotransform[3] + x * geotransform[4] + y * geotransform[5]
    return geo_x, geo_y

def main():
    print(f"\n{'='*50}")
    print(f"Starting processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")
    
    total_start_time = time()
    
    # File path for the input .tif file
    input_path = r"G:\Shiva_WS\Pit_Detection_WS\Input_files\TEST_7\slopemap.tif"
    polygon_path = r"G:\Shiva_WS\Pit_Detection_WS\Output_files\final_pit_boundaries.shp"
    
    if not os.path.exists(input_path) or not os.path.exists(polygon_path):
        print("Error: Input file or polygon file does not exist.")
        return
    
    # Load polygon and reproject if needed
    polygon = gpd.read_file(polygon_path)

    # Open DEM raster and reproject polygon to match DEM CRS
    with rasterio.open(input_path) as src:
        raster_crs = src.crs
        polygon = polygon.to_crs(raster_crs)
        raster_bounds = src.bounds
        print("Raster Bounds:", raster_bounds)
        print("Polygon Bounds:", polygon.total_bounds)

    # Clip polygon to raster bounds to avoid out-of-bounds areas
    polygon = polygon.clip(box(*raster_bounds))

    # Check if polygon is empty after clipping
    if polygon.empty:
        print("Error: Polygon does not overlap with the raster extent after clipping.")
        return

    # Apply the mask to restrict to the area within the polygon
    with rasterio.open(input_path) as src:
        masked_image, masked_transform = mask(src, polygon.geometry, crop=True)
        geotransform = masked_transform  # Use transform directly from mask output
        projection = src.crs.to_string()
    
    # Continue processing with masked image
    # Edge detection
    print("\nPerforming Canny edge detection...")
    edge_start = time()
    edges = feature.canny(
        masked_image[0].astype(np.float32),
        sigma=50,
        low_threshold=0.05,
        high_threshold=0.2,
        use_quantiles=True
    )
    print(f"Edge detection completed in {time() - edge_start:.2f} seconds")
    
    # Perform edge tracing and line fitting
    binary_image = edges.astype(np.uint8)
    line_strings = edge_tracing(binary_image)
    fitted_lines = fit_lines_to_pixel_lists(line_strings, delta_l=2, alpha_d=70)
    
    # Convert to georeferenced coordinates
    print("\nConverting to georeferenced coordinates...")
    geo_start = time()
    geo_fitted_lines = []
    for line_list in fitted_lines:
        for line in line_list:
            geo_coords = [
                pixel_to_geo(x, y, geotransform) 
                for x, y in line.coords
            ]
            geo_fitted_lines.append(LineString(geo_coords))
    print(f"Coordinate conversion completed in {time() - geo_start:.2f} seconds")
    
    # Create and save GeoDataFrame
    print("\nPreparing to save outputs...")
    crs = projection
    gdf = create_geodataframe_from_lines(geo_fitted_lines, crs)
    
    # Save outputs
    output_shapefile = r'C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\result\new\after feeding edges\line_fitting_output_test_shiva_test7.shp'
    output_geojson = r'C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\result\new\after feeding edges\line_fitting_output_shiva_test7.geojson'
    
    print(f"\nSaving Shapefile: {os.path.basename(output_shapefile)}")
    gdf.to_file(output_shapefile)
    
    print(f"Saving GeoJSON: {os.path.basename(output_geojson)}")
    gdf.to_file(output_geojson, driver='GeoJSON')
    
    total_time = time() - total_start_time
    print(f"\n{'='*50}")
    print(f"Processing completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"{'='*50}")
    
if __name__ == "__main__":
    main()
