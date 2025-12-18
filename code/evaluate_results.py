import os
import glob
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import argparse
from typing import Dict, List, Tuple

def parse_xml_ground_truth(xml_path: str) -> pd.DataFrame:
    """
    Parses a CVAT-style XML file to extract ground truth annotations.
    Returns a DataFrame with columns: [frame_index, x_gt, y_gt, visible_gt]
    """
    tree = ET.parse(xml_path)
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
            # We define 'visible' as NOT occluded AND NOT outside (though current XMLs don't show 'outside')
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
    return df

def load_system_predictions(csv_path: str) -> pd.DataFrame:
    """
    Loads system predictions from CSV.
    Expected columns: frame_index, x_centroid, y_centroid, visibility_flag
    """
    df = pd.read_csv(csv_path)
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

    # Merge on frame_index
    merged = pd.merge(gt_df, pred_df, on='frame_index', how='left', suffixes=('', '_pred'))
    
    total_frames = len(gt_df)
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

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Define paths
    annotations_root_dir = os.path.join(project_root, 'annotations')
    gt_dir = os.path.join(project_root, 'data', 'TEST', 'Annotations')
    results_dir = os.path.join(project_root, 'results')
    
    detectors = ['yolo', 'classical']
    
    summary_results = []
    detailed_results = []
    
    print(f"{'Detector':<12} | {'Video':<20} | {'Recall':<8} | {'Precision':<10} | {'Drift (px)':<10} | {'TP':<5} | {'FP':<5} | {'FN':<5}")
    print("-" * 100)
    
    for detector in detectors:
        det_dir = os.path.join(annotations_root_dir, detector)
        if not os.path.exists(det_dir):
            print(f"Directory not found: {det_dir}")
            continue
            
        csv_files = glob.glob(os.path.join(det_dir, '*.csv'))
        
        det_metrics = {
            'tp': 0, 'fp': 0, 'fn': 0, 'total_drift': 0, 'drift_count': 0, 'total_gt': 0
        }
        
        for csv_file in sorted(csv_files):
            basename = os.path.basename(csv_file)
            parts = basename.split('_')
            video_name = parts[0] 
            
            # Construct GT path
            gt_filename = f"{video_name}_annotations.xml"
            gt_path = os.path.join(gt_dir, gt_filename)
            
            if not os.path.exists(gt_path):
                continue
                
            gt_df = parse_xml_ground_truth(gt_path)
            pred_df = load_system_predictions(csv_file)
            
            metrics = compute_metrics(gt_df, pred_df)
            
            if metrics:
                drift_str = f"{metrics['avg_drift']:.2f}" if pd.notna(metrics['avg_drift']) else "N/A"
                print(f"{detector:<12} | {video_name:<20} | {metrics['recall']:.2f}     | {metrics['precision']:.2f}       | {drift_str:<10} | {metrics['true_positives']:<5} | {metrics['false_positives']:<5} | {metrics['false_negatives']:<5}")
                
                # Append to detail list
                detailed_results.append({
                    'Detector': detector,
                    'Video': video_name,
                    'Recall': metrics['recall'],
                    'Precision': metrics['precision'],
                    'Avg Drift': metrics['avg_drift'],
                    'TP': metrics['true_positives'],
                    'FP': metrics['false_positives'],
                    'FN': metrics['false_negatives'],
                    'Total GT Visible': metrics['total_visible_gt']
                })
                
                det_metrics['tp'] += metrics['true_positives']
                det_metrics['fp'] += metrics['false_positives']
                det_metrics['fn'] += metrics['false_negatives']
                det_metrics['total_gt'] += metrics['total_visible_gt']
                
                if pd.notna(metrics['avg_drift']):
                    det_metrics['total_drift'] += metrics['avg_drift'] * metrics['true_positives'] # Weighted sum
                    det_metrics['drift_count'] += metrics['true_positives']
        
        # Aggregate for detector
        overall_recall = det_metrics['tp'] / det_metrics['total_gt'] if det_metrics['total_gt'] > 0 else 0
        overall_precision = det_metrics['tp'] / (det_metrics['tp'] + det_metrics['fp']) if (det_metrics['tp'] + det_metrics['fp']) > 0 else 0
        overall_drift = det_metrics['total_drift'] / det_metrics['drift_count'] if det_metrics['drift_count'] > 0 else 0
        
        summary_results.append({
            'Detector': detector,
            'Recall': overall_recall,
            'Precision': overall_precision,
            'Avg Drift': overall_drift,
            'Total TP': det_metrics['tp'],
            'Total FP': det_metrics['fp'],
            'Total FN': det_metrics['fn']
        })
        
    print("-" * 100)
    print("\nOVERALL SUMMARY:")
    summary_df = pd.DataFrame(summary_results)
    print(summary_df.to_string(index=False))
    
    # Save results to CSV
    os.makedirs(results_dir, exist_ok=True)
    
    summary_csv_path = os.path.join(results_dir, "evaluation_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSummary results saved to: {summary_csv_path}")
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_csv_path = os.path.join(results_dir, "evaluation_details.csv")
    detailed_df.to_csv(detailed_csv_path, index=False)
    print(f"Detailed results saved to: {detailed_csv_path}")

if __name__ == "__main__":
    main()
