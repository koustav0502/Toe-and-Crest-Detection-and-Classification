import os
from skimage import feature
from osgeo import gdal, osr
import joblib
import cv2
import numpy as np
from shapely.geometry import LineString
from math import atan2, degrees
from sklearn.cluster import DBSCAN
import geopandas as gpd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing
import pandas as pd
import time
from datetime import datetime
from numba import jit, prange
import numpy as np
from numba.typed import List

def print_timestamp(message):
    """Print a message with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")

@jit(nopython=True)
def get_elevation_numba(x, y, dem_data, geotransform):
    pixel_x = int((x - geotransform[0]) / geotransform[1])
    pixel_y = int((y - geotransform[3]) / geotransform[5])
    if 0 <= pixel_x < dem_data.shape[1] and 0 <= pixel_y < dem_data.shape[0]:
        return dem_data[pixel_y, pixel_x]
    return np.nan

@jit(nopython=True)
def calculate_features_numba(sample_points, dem_data, geotransform, delta):
    elevations = np.zeros(len(sample_points))
    for i in range(len(sample_points)):
        elevations[i] = get_elevation_numba(sample_points[i, 0], sample_points[i, 1], 
                                          dem_data, geotransform)
    
    if np.any(np.isnan(elevations)):
        return None
    
    derivatives = np.diff(elevations) / delta
    uncertainty = np.std(elevations)
    
    return np.concatenate([derivatives, np.array([uncertainty])])

def extract_features_batch(lines_batch, dem_data, geotransform):
    features_list = []
    centroids_list = []
    valid_lines = []
    
    for line in lines_batch:
        coords = list(line.coords)
        if len(coords) < 2:
            continue
            
        coords = np.array(coords)
        mid_point = line.interpolate(0.5, normalized=True)
        mid_x, mid_y = mid_point.x, mid_point.y
        
        dx = coords[-1][0] - coords[0][0]
        dy = coords[-1][1] - coords[0][1]
        length = np.sqrt(dx**2 + dy**2)
        
        if length == 0:
            continue
            
        ortho_x, ortho_y = -dy / length, dx / length
        delta = 5
        
        multipliers = np.array([-2, -1, 0, 1, 2]) * delta
        sample_points = np.array([(mid_x + m * ortho_x, mid_y + m * ortho_y) 
                                 for m in multipliers])
        
        features = calculate_features_numba(sample_points, dem_data, geotransform, delta)
        if features is None:
            continue
            
        features = np.append(features, length)
        features_list.append(features)
        centroids_list.append((mid_x, mid_y))
        valid_lines.append(line)
    
    return features_list, centroids_list, valid_lines

def classify_lines(lines, pipeline, label_encoder, dem_data, geotransform):
    print_timestamp(f"Starting classification of {len(lines)} lines")
    start_time = time.time()
    
    # Split lines into batches for parallel processing
    batch_size = 1000  # Adjust based on your system's memory
    num_batches = (len(lines) + batch_size - 1) // batch_size
    batches = [lines[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
    
    # Process batches in parallel
    all_features = []
    all_centroids = []
    all_valid_lines = []
    
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    print_timestamp(f"Processing {num_batches} batches using {num_workers} workers")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for batch in batches:
            future = executor.submit(extract_features_batch, batch, dem_data, geotransform)
            futures.append(future)
        
        for future in as_completed(futures):
            features_batch, centroids_batch, valid_lines_batch = future.result()
            if features_batch:
                all_features.extend(features_batch)
                all_centroids.extend(centroids_batch)
                all_valid_lines.extend(valid_lines_batch)
    
    if not all_features:
        print_timestamp("No valid features found! Returning empty list")
        return []
    
    # Convert to numpy arrays for faster processing
    features_array = np.array(all_features)
    centroids_array = np.array(all_centroids)
    
    # Optimized DBSCAN clustering
    print_timestamp("Starting DBSCAN clustering...")
    clustering = DBSCAN(eps=10, min_samples=2, n_jobs=-1).fit(centroids_array)
    
    # Prepare features for classification
    features_with_clusters = np.column_stack((features_array, clustering.labels_))
    
    # Batch prediction
    print_timestamp("Making predictions...")
    batch_size = 10000  # Adjust based on your system's memory
    predictions = []
    
    for i in range(0, len(features_with_clusters), batch_size):
        batch = features_with_clusters[i:i + batch_size]
        batch_predictions = pipeline.predict(batch)
        predictions.extend(batch_predictions)
    
    predicted_labels = label_encoder.inverse_transform(predictions)
    
    print_timestamp(f"Classification completed in {time.time() - start_time:.2f} seconds")
    return list(zip(all_valid_lines, predicted_labels))

def pixel_to_geo(x, y, geotransform):
    geo_x = geotransform[0] + x * geotransform[1] + y * geotransform[2]
    geo_y = geotransform[3] + x * geotransform[4] + y * geotransform[5]
    return geo_x, geo_y

def load_model(model_path):
    print_timestamp(f"Loading model from: {model_path}")
    start_time = time.time()
    with open(model_path, 'rb') as f:
        data = joblib.load(f)
    print_timestamp(f"Model loading completed in {time.time() - start_time:.2f} seconds")
    return data['pipeline'], data['label_encoder']

def main():
    print_timestamp("Starting line classification script")
    total_start_time = time.time()

    # Paths
    dem_file = r"G:\Mine Data\sanaindupur\DSM_7.62cm-pix_Sanaindupur_36052023_Y2022.tif"
    model_path = r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\models\1. 3mines_94%\updatedsvm_model_with_passes_and_preprocessing_pass3.joblib"
    edges_shp_file = r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\result\new\after feeding edges\line_fitting_output_test_sanaindupur.shp"
    
    print_timestamp(f"Loading DEM file: {dem_file}")
    dem_start_time = time.time()
    # Load DEM data using GDAL's memory mapping
    dem_dataset = gdal.Open(dem_file)
    dem_data = dem_dataset.ReadAsArray()
    geotransform = dem_dataset.GetGeoTransform()
    projection = dem_dataset.GetProjection()
    srs = osr.SpatialReference(wkt=projection)
    print_timestamp(f"DEM data loaded in {time.time() - dem_start_time:.2f} seconds")
    print_timestamp(f"DEM shape: {dem_data.shape}")

    # Load model
    pipeline, label_encoder = load_model(model_path)

    # Read shapefile
    print_timestamp(f"Loading shapefile: {edges_shp_file}")
    shapefile_start_time = time.time()
    edges_gdf = gpd.read_file(edges_shp_file)
    print_timestamp(f"Shapefile loaded in {time.time() - shapefile_start_time:.2f} seconds")
    print_timestamp(f"Number of edges loaded: {len(edges_gdf)}")

    # Ensure CRS matches
    print_timestamp("Checking coordinate reference systems...")
    dem_crs = srs.ExportToWkt()
    if edges_gdf.crs != dem_crs:
        print_timestamp("Reprojecting edges to match DEM CRS...")
        reproject_start_time = time.time()
        edges_gdf = edges_gdf.to_crs(dem_crs)
        print_timestamp(f"Reprojection completed in {time.time() - reproject_start_time:.2f} seconds")

    # Extract geometries
    print_timestamp("Extracting line geometries...")
    geometry_start_time = time.time()
    line_strings = edges_gdf.geometry.values
    print_timestamp(f"Geometries extracted in {time.time() - geometry_start_time:.2f} seconds")

    # Classify lines
    classified_lines = classify_lines(line_strings, pipeline, label_encoder, dem_data, geotransform)
    
    # Create GeoDataFrame
    print_timestamp("Creating GeoDataFrame with classified lines...")
    gdf_start_time = time.time()
    classified_gdf = gpd.GeoDataFrame({
        'geometry': [line for line, _ in classified_lines],
        'classification': [cls for _, cls in classified_lines]
    }, crs=dem_crs)
    print_timestamp(f"GeoDataFrame created in {time.time() - gdf_start_time:.2f} seconds")
    
    # Define output paths
    output_shapefile = r'C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\result\new\after feeding edges\classified_lines_sanaindupur.shp'
    output_geojson = r'C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\result\new\after feeding edges\classified_lines_sanaindupur.geojson'

    # Save files
    print_timestamp("Saving results...")
    save_start_time = time.time()
    classified_gdf.to_file(output_shapefile)
    classified_gdf.to_file(output_geojson, driver='GeoJSON')
    print_timestamp(f"Results saved in {time.time() - save_start_time:.2f} seconds")
    
    total_time = time.time() - total_start_time
    print_timestamp(f"Script completed successfully!")
    print_timestamp(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print_timestamp(f"Results saved to:")
    print_timestamp(f"- Shapefile: {output_shapefile}")
    print_timestamp(f"- GeoJSON: {output_geojson}")

    # Print summary statistics
    print_timestamp("\nSummary Statistics:")
    print_timestamp(f"Total lines processed: {len(line_strings)}")
    print_timestamp(f"Valid lines classified: {len(classified_lines)}")
    print_timestamp(f"Invalid/skipped lines: {len(line_strings) - len(classified_lines)}")
    
    # Print unique classifications and their counts
    classifications = classified_gdf['classification'].value_counts()
    print_timestamp("\nClassification Results:")
    for class_name, count in classifications.items():
        print_timestamp(f"- {class_name}: {count} lines")

if __name__ == "__main__":
    main()