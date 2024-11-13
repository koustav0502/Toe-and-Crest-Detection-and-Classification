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
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing
import pandas as pd
import time
from datetime import datetime

def print_timestamp(message):
    """Print a message with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")

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

def extract_features(args):
    line, dem_data, geotransform = args
    
    def get_elevation(x, y):
        pixel_x = int((x - geotransform[0]) / geotransform[1])
        pixel_y = int((y - geotransform[3]) / geotransform[5])
        if 0 <= pixel_x < dem_data.shape[1] and 0 <= pixel_y < dem_data.shape[0]:
            return dem_data[pixel_y, pixel_x]
        return None

    coords = list(line.coords)
    if len(coords) < 2:
        return None, None

    mid_point = line.interpolate(0.5, normalized=True)
    mid_x, mid_y = mid_point.x, mid_point.y

    dx = coords[-1][0] - coords[0][0]
    dy = coords[-1][1] - coords[0][1]
    length = np.sqrt(dx**2 + dy**2)
    if length == 0:
        return None, None

    # Vectorized calculations
    ortho_x, ortho_y = -dy / length, dx / length
    delta = 5

    # Calculate all sample points at once
    multipliers = np.array([-2, -1, 0, 1, 2]) * delta
    sample_points = np.array([(mid_x + m * ortho_x, mid_y + m * ortho_y) for m in multipliers])
    
    # Vectorized elevation calculation
    elevations = np.array([get_elevation(x, y) for x, y in sample_points])
    if None in elevations:
        return None, None

    # Vectorized derivative calculation
    derivatives = np.diff(elevations) / delta
    
    uncertainty = np.std(elevations)
    centroid = (mid_x, mid_y)

    features = list(derivatives) + [uncertainty, length]
    return features, centroid

def classify_lines(lines, pipeline, label_encoder, dem_data, geotransform):
    print_timestamp(f"Starting classification of {len(lines)} lines")
    start_time = time.time()

    # Prepare arguments for parallel processing
    args = [(line, dem_data, geotransform) for line in lines]
    
    # Use number of CPU cores minus 1 to avoid overloading
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    print_timestamp(f"Using {num_workers} worker threads for parallel processing")
    
    all_centroids = []
    features = []
    valid_lines = []
    
    # Parallel feature extraction
    feature_start_time = time.time()
    print_timestamp("Starting parallel feature extraction...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(extract_features, args))
    print_timestamp(f"Feature extraction completed in {time.time() - feature_start_time:.2f} seconds")
    
    # Process results
    process_start_time = time.time()
    print_timestamp("Processing extracted features...")
    for line, (feature, centroid) in zip(lines, results):
        if feature is not None:
            features.append(feature)
            all_centroids.append(centroid)
            valid_lines.append(line)

    print_timestamp(f"Found {len(valid_lines)} valid lines out of {len(lines)} total lines")
    
    if not features:
        print_timestamp("No valid features found! Returning empty list")
        return []

    # Convert to numpy arrays for faster processing
    features = np.array(features)
    all_centroids = np.array(all_centroids)

    # Optimized DBSCAN clustering
    cluster_start_time = time.time()
    print_timestamp("Starting DBSCAN clustering...")
    clustering = DBSCAN(eps=10, min_samples=2, n_jobs=-1).fit(all_centroids)
    print_timestamp(f"Clustering completed in {time.time() - cluster_start_time:.2f} seconds")
    
    # Vectorized feature stacking
    print_timestamp("Preparing features for classification...")
    features_with_clusters = np.column_stack((features, clustering.labels_))
    
    # Batch prediction
    prediction_start_time = time.time()
    print_timestamp("Making predictions...")
    predictions = pipeline.predict(features_with_clusters)
    predicted_labels = label_encoder.inverse_transform(predictions)
    print_timestamp(f"Predictions completed in {time.time() - prediction_start_time:.2f} seconds")

    print_timestamp(f"Total classification time: {time.time() - start_time:.2f} seconds")
    return list(zip(valid_lines, predicted_labels))

def main():
    print_timestamp("Starting line classification script")
    total_start_time = time.time()

    # Paths
    dem_file = r"C:\Users\Copy\Desktop\files_kp\DATA\toe and crest classification 2\Dem and ortho\DEM.tif"
    model_path = r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\models\updated2svm_model_with_passes_and_preprocessing_pass3.joblib"
    edges_shp_file = r'C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\result\new\after feeding edges\line_fitting_output_test_dem1.shp'
    
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
    output_shapefile = r'C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\result\new\after feeding edges\classified_lines_dem1.shp'
    output_geojson = r'C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\result\new\after feeding edges\classified_lines_dem1.geojson'

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

#sanaindupur classification output:
# [2024-11-05 10:25:43] Starting line classification script
# [2024-11-05 10:25:43] Loading DEM file: G:\Mine Data\sanaindupur\DSM_7.62cm-pix_Sanaindupur_36052023_Y2022.tif
# t.
#   warnings.warn(
# [2024-11-05 10:25:47] DEM data loaded in 4.30 seconds
# [2024-11-05 10:25:47] DEM shape: (18242, 29785)
# [2024-11-05 10:25:47] Loading model from: C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\models\updated2svm_model_with_passes_and_preprocessing_pass3.joblib
# [2024-11-05 10:25:47] Model loading completed in 0.00 seconds
# [2024-11-05 10:25:47] Loading shapefile: C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\result\new\after feeding edges\line_fitting_output_test_sanaindupur.shp
# [2024-11-05 10:26:00] Shapefile loaded in 12.72 seconds
# [2024-11-05 10:26:00] Number of edges loaded: 3283464
# [2024-11-05 10:26:00] Checking coordinate reference systems...
# [2024-11-05 10:26:00] Reprojecting edges to match DEM CRS...
# [2024-11-05 10:26:01] Reprojection completed in 1.55 seconds
# [2024-11-05 10:26:01] Extracting line geometries...
# [2024-11-05 10:26:01] Geometries extracted in 0.00 seconds
# [2024-11-05 10:26:01] Starting classification of 3283464 lines
# [2024-11-05 10:26:03] Using 23 worker threads for parallel processing
# [2024-11-05 10:26:03] Starting parallel feature extraction...
# [2024-11-05 10:30:33] Feature extraction completed in 270.03 seconds
# [2024-11-05 10:30:33] Processing extracted features...
# [2024-11-05 10:30:35] Found 2211128 valid lines out of 3283464 total lines
# [2024-11-05 10:30:37] Starting DBSCAN clustering...
# [2024-11-05 11:15:27] Clustering completed in 2690.47 seconds
# [2024-11-05 11:15:27] Preparing features for classification...
# [2024-11-05 11:15:27] Making predictions...
# [2024-11-05 13:05:01] Predictions completed in 6573.68 seconds
# [2024-11-05 13:05:01] Total classification time: 9539.69 seconds
# [2024-11-05 13:05:03] Creating GeoDataFrame with classified lines...
# [2024-11-05 13:05:05] GeoDataFrame created in 2.18 seconds
# [2024-11-05 13:05:05] Saving results...
# c:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\src\Predict\2. test_new_edges.py:206: UserWarning: Column names longer than 10 characters will be truncated when saved to ESRI Shapefile.
#   classified_gdf.to_file(output_shapefile)
# C:\Users\Copy\AppData\Roaming\Python\Python312\site-packages\pyogrio\raw.py:709: RuntimeWarning: Normalized/laundered field name: 'classification' to 'classifica'
#   ogr_write(
# [2024-11-05 13:05:47] Results saved in 41.34 seconds
# [2024-11-05 13:05:47] Script completed successfully!
# [2024-11-05 13:05:47] Total processing time: 9604.04 seconds (160.07 minutes)
# [2024-11-05 13:05:47] Results saved to:
# [2024-11-05 13:05:47] - Shapefile: C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\result\new\after feeding edges\classified_lines_sanaindupur.shp
# [2024-11-05 13:05:47] - GeoJSON: C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\result\new\after feeding edges\classified_lines_sanaindupur.geojson
# [2024-11-05 13:05:47]
# Summary Statistics:
# [2024-11-05 13:05:47] Total lines processed: 3283464
# [2024-11-05 13:05:47] Valid lines classified: 2211128
# [2024-11-05 13:05:47] Invalid/skipped lines: 1072336
# [2024-11-05 13:05:47] 
# Classification Results:
# [2024-11-05 13:05:47] - neither: 1562822 lines
# [2024-11-05 13:05:47] - crest: 324301 lines
# [2024-11-05 13:05:47] - toe: 324005 lines