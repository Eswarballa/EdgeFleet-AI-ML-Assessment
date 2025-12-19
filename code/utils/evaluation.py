import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

def parse_xml_ground_truth(xml_path: str) -> pd.DataFrame:
    """
    Parses a CVAT-style XML file to extract ground truth annotations.
    Returns a DataFrame with columns: [frame_index, x_gt, y_gt, visible_gt]
    """
    try:
        tree = ET.parse(xml_path)
    except FileNotFoundError:
        return pd.DataFrame()
        
    root = tree.getroot()
    
    data = []
    
    # Iterate over 'track' elements. Assuming label="Ball" is the one we want.
    for track in root.findall('track'):
        if track.get('label') != 'Ball':
            continue
            
        for point in track.findall('points'):
            frame_idx = int(point.get('frame'))
            # 'points' attribute is "x,y"
            points_str = point.get('points')
            try:
                x_str, y_str = points_str.split(',')
                x, y = float(x_str), float(y_str)
            except ValueError:
                continue # Skip malformed points
            
            # 'occluded' 1 means hidden, 0 means visible
            occluded = int(point.get('occluded', 0))
            # We define 'visible' as NOT occluded
            visible = 1 if occluded == 0 else 0
            
            data.append({
                'frame_index': frame_idx,
                'x_gt': x,
                'y_gt': y,
                'visible_gt': visible
            })
            
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values('frame_index').reset_index(drop=True)
    else:
        # Return empty DF with expected columns to avoid errors later
        df = pd.DataFrame(columns=['frame_index', 'x_gt', 'y_gt', 'visible_gt'])
        
    return df

def load_system_predictions(csv_path: str) -> pd.DataFrame:
    """
    Loads system predictions from CSV.
    Expected columns: frame_index, x_centroid, y_centroid, visibility_flag
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame(columns=['frame_index', 'x_centroid', 'y_centroid', 'visibility_flag'])

    # Ensure columns match expected types
    df['frame_index'] = df['frame_index'].astype(int)
    
    # Handle potentially missing or empty values if any
    df['x_centroid'] = pd.to_numeric(df['x_centroid'], errors='coerce')
    df['y_centroid'] = pd.to_numeric(df['y_centroid'], errors='coerce')
    df['visibility_flag'] = pd.to_numeric(df['visibility_flag'], errors='coerce').fillna(0).astype(int)
    
    return df

def compute_metrics(gt_df: pd.DataFrame, pred_df: pd.DataFrame, distance_threshold: int = 20) -> Dict:
    """
    Computes evaluation metrics for a single video.
    """
    if gt_df.empty:
        return {}
        
    if pred_df.empty:
         # If no predictions, recall is 0, precision undefined (or 0), drift undefined
        total_visible_gt = len(gt_df[gt_df['visible_gt'] == 1])
        return {
            'total_visible_gt': total_visible_gt,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': total_visible_gt,
            'recall': 0.0,
            'precision': 0.0,
            'avg_drift': np.nan
        }

    # Merge on frame_index
    merged = pd.merge(gt_df, pred_df, on='frame_index', how='left', suffixes=('', '_pred'))
    
    visible_gt_frames = gt_df[gt_df['visible_gt'] == 1]
    total_visible_gt = len(visible_gt_frames)
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    accumulated_drift = 0.0
    matched_visible_count = 0
    
    for _, row in merged.iterrows():
        is_gt_visible = row['visible_gt'] == 1
        
        # Prediction is present if x_centroid is not NaN and visibility_flag is 1
        has_prediction = pd.notna(row['x_centroid']) and row['visibility_flag'] == 1
        
        if is_gt_visible:
            if has_prediction:
                # Calculate distance
                dist = np.sqrt((row['x_gt'] - row['x_centroid'])**2 + (row['y_gt'] - row['y_centroid'])**2)
                
                if dist <= distance_threshold:
                    true_positives += 1
                    accumulated_drift += dist
                    matched_visible_count += 1
                else:
                    # Predicted but too far -> False Negative for the correct object
                    false_negatives += 1
            else:
                false_negatives += 1
        else:
            # GT is not visible
            if has_prediction:
                false_positives += 1
                
    recall = true_positives / total_visible_gt if total_visible_gt > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    
    avg_drift = accumulated_drift / matched_visible_count if matched_visible_count > 0 else np.nan
    
    return {
        'total_visible_gt': total_visible_gt,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'recall': recall,
        'precision': precision,
        'avg_drift': avg_drift
    }
