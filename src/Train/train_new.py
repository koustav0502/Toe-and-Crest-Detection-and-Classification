import numpy as np
import os
import pandas as pd
import geopandas as gpd
from osgeo import gdal
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import DBSCAN
import joblib
import time

gdal.UseExceptions()

# Function to extract features and centroids
def extract_features(line, dem_data, geotransform):
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
    ortho_x, ortho_y = -dy / length, dx / length

    delta = 5
    sample_points = [
        (mid_x - 2 * delta * ortho_x, mid_y - 2 * delta * ortho_y),
        (mid_x - delta * ortho_x, mid_y - delta * ortho_y),
        (mid_x, mid_y),
        (mid_x + delta * ortho_x, mid_y + delta * ortho_y),
        (mid_x + 2 * delta * ortho_x, mid_y + 2 * delta * ortho_y)
    ]

    elevations = [get_elevation(x, y) for x, y in sample_points]
    if any(e is None for e in elevations):
        return None, None

    derivatives = [
        (elevations[1] - elevations[0]) / delta,
        (elevations[2] - elevations[1]) / delta,
        (elevations[3] - elevations[2]) / delta,
        (elevations[4] - elevations[3]) / delta
    ]

    uncertainty = np.std(elevations)
    centroid = (mid_x, mid_y)  # Used only for clustering

    # Exclude mid_x and mid_y from features
    features = derivatives + [uncertainty, length]

    return features, centroid

# Function to process data and extract features
def process_combined_data(mines_data):
    features = []
    labels = []
    groups = []
    centroids = []

    for mine_data in mines_data:
        crest_gdf = gpd.read_file(mine_data['crest_file'])
        toe_gdf = gpd.read_file(mine_data['toe_file'])
        neither_gdf = gpd.read_file(mine_data['neither_file'])
        print('DATA LOADED>>>')
        crest_gdf['class'] = 'crest'
        toe_gdf['class'] = 'toe'
        neither_gdf['class'] = 'neither'

        gdf = pd.concat([crest_gdf, toe_gdf, neither_gdf])

        dem_dataset = gdal.Open(mine_data['dem_file'])
        dem_data = dem_dataset.ReadAsArray()
        geotransform = dem_dataset.GetGeoTransform()

        for idx, row in gdf.iterrows():
            line = row['geometry']
            feature, centroid = extract_features(line, dem_data, geotransform)

            if feature is not None:
                features.append(feature)
                labels.append(row['class'])
                groups.append(mine_data['mine_name'])
                centroids.append(centroid)

    return np.array(features), np.array(labels), np.array(groups), np.array(centroids)

# Function to perform DBSCAN clustering
def cluster_and_assign_lines(features, centroids, eps=10, min_samples=2):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(centroids)
    features_with_clusters = np.column_stack((features, cluster_labels))
    return features_with_clusters

# Function to perform training for a single pass
def perform_training_pass(mines_data, pass_num, model_output_path):
    features, labels, groups, centroids = process_combined_data(mines_data)
    print('COMBINED DATA PROCESSED>>>')
    # Cluster lines by centroids
    features_with_clusters = cluster_and_assign_lines(features, centroids)
    # Now features_with_clusters has shape (n_samples, 7)

    # Encode labels
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    print('pipeline defined...')
    # Define pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC())
    ])
    print('grid search started...')
    # Define parameter grid for grid search
    param_grid = {
        'svm__C': [0.1, 1, 10],
        'svm__gamma': ['scale', 'auto'],
        'svm__kernel': ['rbf','linear'],
        'svm__class_weight': ['balanced']
    }

    # Use GroupKFold
    gkf = GroupKFold(n_splits=2)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=gkf.split(features_with_clusters, encoded_labels, groups),
        verbose=10,
        n_jobs=-1,
        scoring='f1_weighted'
    )

    # Train the model
    grid_search.fit(features_with_clusters, encoded_labels)

    best_model = grid_search.best_estimator_

    # Save both the model and label encoder
    data_to_save = {'pipeline': best_model, 'label_encoder': le}
    pass_model_output_path = model_output_path.replace('.joblib', f'_pass{pass_num}.joblib')
    joblib.dump(data_to_save, pass_model_output_path)

    return best_model, grid_search, le

# Function to perform grid search in 3 passes
def perform_grid_search(mines_data_passes, model_output_path):
    txt_log_path = model_output_path.replace('.joblib', '_training_log.txt')

    # Prepare to log results
    with open(txt_log_path, 'w') as log_file:
        start_time_total = time.time()

        for pass_num, mines_data in enumerate(mines_data_passes, start=1):
            start_time_pass = time.time()
            print(f'started pass {pass_num}...')
            # Train and update model
            best_model, grid_search, le = perform_training_pass(mines_data, pass_num, model_output_path)

            end_time_pass = time.time()
            duration_pass = end_time_pass - start_time_pass
            log_file.write(f"Pass {pass_num} completed in {duration_pass:.2f} seconds\n")
            log_file.write(f"Best Parameters for Pass {pass_num}: {grid_search.best_params_}\n")
            log_file.write(f"Best Score for Pass {pass_num}: {grid_search.best_score_:.4f}\n")

            # Classification report and confusion matrix
            features, labels, groups, centroids = process_combined_data(mines_data)
            features_with_clusters = cluster_and_assign_lines(features, centroids)
            encoded_labels = le.transform(labels)  # Encode labels using the same LabelEncoder

            y_pred = best_model.predict(features_with_clusters)
            y_true = labels  # These are the original string labels

            # Convert integer predictions back to original labels
            y_pred_labels = le.inverse_transform(y_pred)  # Use le.inverse_transform

            log_file.write(f"\nClassification Report for Pass {pass_num}:\n")
            log_file.write(classification_report(y_true, y_pred_labels))
            log_file.write("\nConfusion Matrix:\n")
            log_file.write(np.array2string(confusion_matrix(y_true, y_pred_labels)))
            log_file.write("\n")

        end_time_total = time.time()
        duration_total = end_time_total - start_time_total
        log_file.write(f"Total training time: {duration_total:.2f} seconds\n")


# Main script
mines_data_passes = [
    # Pass 1 (Part 1 of each mine)
    [
        {
            'crest_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\crest\partitioned features\crest_mine1_part1.shp",
            'toe_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\toe\partitioned features\toe_mine1_part1.shp",
            'neither_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\neither\partitioned features\neither_mine1_part1.shp",
            'dem_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\dem\Mine 1\DEM.tif",
            'mine_name': 'Mine 1'
        },
        {
            'crest_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\crest\partitioned features\crest_mine2_part1.shp",
            'toe_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\toe\partitioned features\toe_mine2_part1.shp",
            'neither_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\neither\partitioned features\neither_mine2_part1.shp",
            'dem_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\dem\Mine 2\DSM_9.02cm_Pix_KCM_JSL_Y2022_JIndal chromite.tif",
            'mine_name': 'Mine 2'
        },
        {
            'crest_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\crest\partitioned features\crest_mine3_part1.shp",
            'toe_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\toe\partitioned features\toe_mine3_part1.shp",
            'neither_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\neither\partitioned features\neither_mine3_part1.shp",
            'dem_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\dem\Mine 3\DSM_8cm_Pix_Patabeda14Hect_Y2022.tif",
            'mine_name': 'Mine 3'
        },
        {
            'crest_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\crest\partitioned features\crest_mine4_part1.shp",
            'toe_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\toe\partitioned features\toe_mine4_part1.shp",
            'neither_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\neither\partitioned features\neither_mine4_part1.shp",
            'dem_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\dem\MIne 4\Mine4_part1_DSM_9.8cm_pix_NIM_ESL_Y2022.tif",
            'mine_name': 'Mine 4'
        }
    ],
    # Pass 2 (Part 2 of each mine)
    [
        {
            'crest_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\crest\partitioned features\crest_mine1_part2.shp",
            'toe_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\toe\partitioned features\toe_mine1_part2.shp",
            'neither_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\neither\partitioned features\neither_mine1_part2.shp",
            'dem_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\dem\Mine 1\DEM.tif",
            'mine_name': 'Mine 1'
        },
        {
            'crest_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\crest\partitioned features\crest_mine2_part2.shp",
            'toe_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\toe\partitioned features\toe_mine2_part2.shp",
            'neither_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\neither\partitioned features\neither_mine2_part2.shp",
            'dem_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\dem\Mine 2\DSM_9.02cm_Pix_KCM_JSL_Y2022_JIndal chromite.tif",
            'mine_name': 'Mine 2'
        },
        {
            'crest_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\crest\partitioned features\crest_mine3_part2.shp",
            'toe_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\toe\partitioned features\toe_mine3_part2.shp",
            'neither_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\neither\partitioned features\neither_mine3_part2.shp",
            'dem_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\dem\Mine 3\DSM_8cm_Pix_Patabeda14Hect_Y2022.tif",
            'mine_name': 'Mine 3'
        },
        {
            'crest_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\crest\partitioned features\crest_mine4_part2.shp",
            'toe_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\toe\partitioned features\toe_mine4_part2.shp",
            'neither_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\neither\partitioned features\neither_mine4_part2.shp",
            'dem_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\dem\MIne 4\Mine4_part2_part3_DSM_9.8cm_pix_NIMM_ESL_Y2022.tif",
            'mine_name': 'Mine 4'
        }
    ],
    # Pass 3 (Part 3 of each mine)
    [
        {
            'crest_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\crest\partitioned features\crest_mine1_part3.shp",
            'toe_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\toe\partitioned features\toe_mine1_part3.shp",
            'neither_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\neither\partitioned features\neither_mine1_part3.shp",
            'dem_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\dem\Mine 1\DEM.tif",
            'mine_name': 'Mine 1'
        },
        {
            'crest_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\crest\partitioned features\crest_mine2_part3.shp",
            'toe_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\toe\partitioned features\toe_mine2_part3.shp",
            'neither_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\neither\partitioned features\neither_mine2_part3.shp",
            'dem_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\dem\Mine 2\DSM_9.02cm_Pix_KCM_JSL_Y2022_JIndal chromite.tif",
            'mine_name': 'Mine 2'
        },
        {
            'crest_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\crest\partitioned features\crest_mine3_part3.shp",
            'toe_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\toe\partitioned features\toe_mine3_part3.shp",
            'neither_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\neither\partitioned features\neither_mine3_part3.shp",
            'dem_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\dem\Mine 3\DSM_8cm_Pix_Patabeda14Hect_Y2022.tif",
            'mine_name': 'Mine 3'
        },
        {
            'crest_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\crest\partitioned features\crest_mine4_part3.shp",
            'toe_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\toe\partitioned features\toe_mine4_part3.shp",
            'neither_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\neither\partitioned features\neither_mine4_part3.shp",
            'dem_file': r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\data\training\dem\MIne 4\Mine4_part2_part3_DSM_9.8cm_pix_NIMM_ESL_Y2022.tif",
            'mine_name': 'Mine 4'
        }
    ]
]

perform_grid_search(mines_data_passes, r"C:\Users\Copy\Desktop\files_kp\Toe_crest_classifier_model\models\updated2svm_model_with_passes_and_preprocessing.joblib")
