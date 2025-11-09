#!/usr/bin/env python3
"""
Enrich COUGHVID tile manifest with clip-level JSON attributes as weak labels.
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from tqdm import tqdm

# Try to import orjson, fallback to json
try:
    import orjson
    JSON_LOAD = orjson.loads
    JSON_AVAILABLE = 'orjson'
except ImportError:
    import json
    JSON_LOAD = json.load
    JSON_AVAILABLE = 'json'


def to_bool(x: Any) -> Optional[bool]:
    """
    Coerce value to boolean.
    Accepts: True/False, 1/0, "yes"/"no", "true"/"false", "y"/"n" (case-insensitive).
    Returns None if cannot be coerced.
    """
    if x is None:
        return None
    
    # Already boolean
    if isinstance(x, bool):
        return x
    
    # Numeric
    if isinstance(x, (int, float)):
        return bool(x)
    
    # String
    if isinstance(x, str):
        x_lower = x.lower().strip()
        if x_lower in ('true', 'yes', 'y', '1'):
            return True
        elif x_lower in ('false', 'no', 'n', '0'):
            return False
    
    return None


def parse_json_file(json_path: Path) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Parse a JSON file and return (stem, parsed_dict).
    Returns (stem, None) if parsing fails.
    """
    try:
        with open(json_path, 'rb') as f:
            content = f.read()
        
        if JSON_AVAILABLE == 'orjson':
            data = orjson.loads(content)
        else:
            data = json.loads(content.decode('utf-8'))
        
        stem = json_path.stem
        return (stem, data)
    except Exception as e:
        warnings.warn(f"Failed to parse {json_path}: {e}")
        return (json_path.stem, None)


def index_json_files(coughvid_root: Path, num_workers: int = 8) -> Dict[str, Dict[str, Any]]:
    """
    Recursively scan for *.json files and parse them in parallel.
    Returns dict {stem -> parsed_json}.
    """
    json_files = list(coughvid_root.rglob('*.json'))
    print(f"Found {len(json_files)} JSON files")
    
    json_dict = {}
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(parse_json_file, json_path): json_path 
                   for json_path in json_files}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Parsing JSONs"):
            stem, data = future.result()
            if data is not None:
                json_dict[stem] = data
    
    print(f"Successfully parsed {len(json_dict)} JSON files")
    return json_dict


def find_expert_labels_dict(json_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Find the first key k where k.lower().startswith("expert_labels") 
    and isinstance(json[k], dict).
    Returns the dict or None.
    """
    if json_data is None:
        return None
    
    for key, value in json_data.items():
        if isinstance(key, str) and key.lower().startswith("expert_labels"):
            if isinstance(value, dict):
                return value
    return None


def process_attributes(row: pd.Series, json_data: Optional[Dict[str, Any]], 
                      attributes: list) -> Dict[str, Any]:
    """
    Process attributes for a single row.
    Looks for expert_labels dict first, then falls back to top-level.
    Returns dict of new column values.
    """
    result = {}
    
    if json_data is None:
        # No JSON found
        for attr in attributes:
            result[f'attr_{attr}'] = np.nan
            result[f'attr_{attr}_mask'] = 0
        result['has_json'] = 0
        return result
    
    result['has_json'] = 1
    
    # Find expert_labels dict
    expert_labels = find_expert_labels_dict(json_data)
    
    # Process boolean attributes: wheezing, stridor, choking, congestion
    bool_attributes = ["wheezing", "stridor", "choking", "congestion"]
    for attr in bool_attributes:
        if attr in attributes:
            # Check expert_labels first, then fallback to top-level
            attr_value = None
            if expert_labels is not None and attr in expert_labels:
                attr_value = expert_labels[attr]
            elif attr in json_data:
                attr_value = json_data[attr]
            
            if attr_value is not None:
                bool_val = to_bool(attr_value)
                if bool_val is not None:
                    result[f'attr_{attr}'] = 1 if bool_val else 0
                    result[f'attr_{attr}_mask'] = 1
                else:
                    # Present but not coercible to bool
                    result[f'attr_{attr}'] = np.nan
                    result[f'attr_{attr}_mask'] = 0
            else:
                # Missing
                result[f'attr_{attr}'] = np.nan
                result[f'attr_{attr}_mask'] = 0
    
    # Handle wet/dry via cough_type
    if "wet" in attributes:
        if expert_labels is not None and "cough_type" in expert_labels:
            cough_type = str(expert_labels["cough_type"]).lower().strip()
            if cough_type in ("wet", "productive"):
                result['attr_wet'] = 1
                result['attr_wet_mask'] = 1
            elif cough_type == "dry":
                result['attr_wet'] = 0
                result['attr_wet_mask'] = 1
            else:
                # Unknown or other
                result['attr_wet'] = np.nan
                result['attr_wet_mask'] = 0
        else:
            # Missing cough_type
            result['attr_wet'] = np.nan
            result['attr_wet_mask'] = 0
    
    return result


def map_quality_to_score(quality_str: str) -> int:
    """
    Map quality string to integer score.
    good=3, acceptable/moderate=2, poor/bad=1, unknown/other=0
    """
    quality_lower = str(quality_str).lower().strip()
    if quality_lower == "good":
        return 3
    elif quality_lower in ("acceptable", "moderate"):
        return 2
    elif quality_lower in ("poor", "bad"):
        return 1
    else:
        return 0


def process_extras(row: pd.Series, json_data: Optional[Dict[str, Any]], 
                   min_quality: int) -> Tuple[Dict[str, Any], bool]:
    """
    Process extra attributes (quality, severity, diagnosis, cough_detected).
    Returns (dict of new column values, should_drop_row).
    """
    result = {}
    should_drop = False
    
    if json_data is None:
        result['quality_raw'] = np.nan
        result['quality_score'] = np.nan
        result['severity_raw'] = np.nan
        result['severity_is_pseudocough'] = 0
        result['diagnosis_raw'] = np.nan
        result['diagnosis_covid_like'] = 0
        result['cough_detected_score'] = np.nan
        return result, should_drop
    
    # Quality (string)
    if 'quality' in json_data:
        quality_raw = str(json_data['quality'])
        result['quality_raw'] = quality_raw
        quality_score = map_quality_to_score(quality_raw)
        result['quality_score'] = quality_score
        # Apply min_quality filter to quality_score if quality_raw exists
        if quality_score < min_quality:
            should_drop = True
    else:
        result['quality_raw'] = np.nan
        result['quality_score'] = np.nan
    
    # Severity (string)
    if 'severity' in json_data:
        severity_raw = str(json_data['severity'])
        result['severity_raw'] = severity_raw
        severity_lower = severity_raw.lower().strip()
        result['severity_is_pseudocough'] = 1 if severity_lower == "pseudocough" else 0
    else:
        result['severity_raw'] = np.nan
        result['severity_is_pseudocough'] = 0
    
    # Diagnosis (string)
    if 'diagnosis' in json_data:
        diagnosis_raw = str(json_data['diagnosis']).lower()
        result['diagnosis_raw'] = diagnosis_raw
        result['diagnosis_covid_like'] = 1 if 'covid' in diagnosis_raw else 0
    else:
        result['diagnosis_raw'] = np.nan
        result['diagnosis_covid_like'] = 0
    
    # Top-level cough_detected (float)
    if 'cough_detected' in json_data:
        try:
            cough_detected = float(json_data['cough_detected'])
            result['cough_detected_score'] = cough_detected
        except (ValueError, TypeError):
            result['cough_detected_score'] = np.nan
    else:
        result['cough_detected_score'] = np.nan
    
    return result, should_drop


def main():
    parser = argparse.ArgumentParser(
        description='Enrich COUGHVID tile manifest with JSON attributes'
    )
    parser.add_argument('--coughvid-root', type=str, required=True,
                        help='Directory containing paired *.wav and *.json files')
    parser.add_argument('--manifest-in', type=str, required=True,
                        help='Path to input tiles CSV')
    parser.add_argument('--manifest-out', type=str, 
                        default='data/feats_cv/manifests/coughvid_tiles_attrs.csv',
                        help='Path to output enriched CSV')
    parser.add_argument('--min-quality', type=int, default=1,
                        help='Minimum quality threshold (default: 1)')
    parser.add_argument('--jobs', type=int, default=8,
                        help='Number of parallel JSON parsing workers (default: 8)')
    
    args = parser.parse_args()
    
    # Index JSON files
    coughvid_root = Path(args.coughvid_root)
    if not coughvid_root.exists():
        print(f"Error: --coughvid-root does not exist: {coughvid_root}", file=sys.stderr)
        sys.exit(1)
    
    print("Indexing JSON files...")
    json_dict = index_json_files(coughvid_root, args.jobs)
    print()
    
    # Load manifest
    print(f"Loading manifest from {args.manifest_in}...")
    # Use dtype hints for memory efficiency
    dtype_hints = {
        'start_sample': 'int32',
        'end_sample': 'int32',
        'sr': 'int32',
    }
    try:
        manifest_df = pd.read_csv(args.manifest_in, dtype=dtype_hints, low_memory=False)
    except Exception as e:
        print(f"Error loading manifest: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded {len(manifest_df)} rows")
    
    # Find source column
    if 'src' in manifest_df.columns:
        src_col = 'src'
    elif 'source' in manifest_df.columns:
        src_col = 'source'
    else:
        print("Error: Manifest must contain either 'src' or 'source' column", file=sys.stderr)
        sys.exit(1)
    
    # Derive stems
    print("Deriving stems from source paths...")
    manifest_df['_stem'] = manifest_df[src_col].apply(
        lambda x: Path(x).stem if pd.notna(x) else None
    )
    
    # Attributes to process
    attributes = ["wet", "wheezing", "stridor", "choking", "congestion"]
    
    # Process rows
    print("Processing attributes...")
    
    # Initialize new columns with efficient dtypes
    for attr in attributes:
        manifest_df[f'attr_{attr}'] = np.nan  # float32 will be used where possible
        manifest_df[f'attr_{attr}_mask'] = 0  # int8 for memory efficiency
    
    manifest_df['has_json'] = 0  # int8
    manifest_df['quality_raw'] = np.nan  # object (string)
    manifest_df['quality_score'] = np.nan  # float32
    manifest_df['severity_raw'] = np.nan  # object (string)
    manifest_df['severity_is_pseudocough'] = 0  # int8
    manifest_df['diagnosis_raw'] = np.nan  # object (string)
    manifest_df['diagnosis_covid_like'] = 0  # int8
    manifest_df['cough_detected_score'] = np.nan  # float32
    
    # Track rows to drop
    rows_to_drop = []
    
    # Process each row
    for idx, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Enriching rows"):
        # Only process coughvid rows
        if row.get('dataset') != 'coughvid':
            continue
        
        stem = row['_stem']
        if stem is None:
            continue
        
        json_data = json_dict.get(stem)
        
        # Process attributes
        attr_results = process_attributes(row, json_data, attributes)
        for key, value in attr_results.items():
            manifest_df.at[idx, key] = value
        
        # Process extras
        extra_results, should_drop = process_extras(row, json_data, args.min_quality)
        for key, value in extra_results.items():
            manifest_df.at[idx, key] = value
        
        if should_drop:
            rows_to_drop.append(idx)
    
    # No need to compute severity_z anymore (severity is now string-based)
    
    # Drop rows with quality < min_quality
    rows_before_drop = len(manifest_df)
    if rows_to_drop:
        manifest_df = manifest_df.drop(index=rows_to_drop)
    rows_after_drop = len(manifest_df)
    
    # Remove temporary column
    manifest_df = manifest_df.drop(columns=['_stem'])
    
    # Optimize dtypes for memory efficiency
    print("Optimizing column dtypes for memory efficiency...")
    for attr in attributes:
        # Convert mask columns to int8
        mask_col = f'attr_{attr}_mask'
        if mask_col in manifest_df.columns:
            manifest_df[mask_col] = manifest_df[mask_col].astype('int8')
        # Convert attr columns to float32 where possible
        attr_col = f'attr_{attr}'
        if attr_col in manifest_df.columns:
            manifest_df[attr_col] = manifest_df[attr_col].astype('float32')
    
    # Convert other integer columns
    if 'has_json' in manifest_df.columns:
        manifest_df['has_json'] = manifest_df['has_json'].astype('int8')
    if 'quality_score' in manifest_df.columns:
        manifest_df['quality_score'] = manifest_df['quality_score'].astype('float32')
    if 'severity_is_pseudocough' in manifest_df.columns:
        manifest_df['severity_is_pseudocough'] = manifest_df['severity_is_pseudocough'].astype('int8')
    if 'diagnosis_covid_like' in manifest_df.columns:
        manifest_df['diagnosis_covid_like'] = manifest_df['diagnosis_covid_like'].astype('int8')
    if 'cough_detected_score' in manifest_df.columns:
        manifest_df['cough_detected_score'] = manifest_df['cough_detected_score'].astype('float32')
    
    # Save output
    out_path = Path(args.manifest_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving enriched manifest to {out_path}...")
    manifest_df.to_csv(out_path, index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total rows in: {rows_before_drop}")
    print(f"Rows kept: {rows_after_drop}")
    print(f"Rows dropped (quality_score < {args.min_quality}): {rows_before_drop - rows_after_drop}")
    
    # JSON coverage
    coughvid_rows = manifest_df[manifest_df['dataset'] == 'coughvid']
    if len(coughvid_rows) > 0:
        json_coverage = (coughvid_rows['has_json'] == 1).sum() / len(coughvid_rows) * 100
        print(f"\nJSON coverage (coughvid rows): {json_coverage:.1f}% ({coughvid_rows['has_json'].sum()}/{len(coughvid_rows)})")
        
        # Attribute statistics
        print("\nAttribute statistics (coughvid rows):")
        for attr in attributes:
            mask_col = f'attr_{attr}_mask'
            if mask_col in coughvid_rows.columns:
                labeled_count = (coughvid_rows[mask_col] == 1).sum()
                if labeled_count > 0:
                    attr_col = f'attr_{attr}'
                    positive_count = (coughvid_rows[attr_col] == 1).sum()
                    positive_rate = positive_count / labeled_count * 100
                    print(f"  {attr}:")
                    print(f"    Labeled: {labeled_count} ({labeled_count/len(coughvid_rows)*100:.1f}%)")
                    print(f"    Positive rate: {positive_rate:.1f}% ({positive_count}/{labeled_count})")
                else:
                    print(f"  {attr}: No labels")
        
        # Quality histogram (quality_raw)
        if 'quality_raw' in coughvid_rows.columns:
            quality_present = coughvid_rows['quality_raw'].notna()
            if quality_present.sum() > 0:
                print("\nQuality distribution (quality_raw):")
                quality_values = coughvid_rows.loc[quality_present, 'quality_raw']
                print(f"  Count: {len(quality_values)}")
                # Histogram
                hist_counts = quality_values.value_counts().sort_index()
                print("  Histogram:")
                if len(hist_counts) > 0:
                    max_count = hist_counts.max()
                    for val, count in hist_counts.items():
                        bar = '#' * int(count / max_count * 40) if max_count > 0 else ''
                        print(f"    {val}: {count:4d} {bar}")
        
        # Severity stats
        if 'severity_raw' in coughvid_rows.columns:
            severity_present = coughvid_rows['severity_raw'].notna()
            if severity_present.sum() > 0:
                print(f"\nSeverity: {severity_present.sum()} values present")
                pseudocough_count = (coughvid_rows['severity_is_pseudocough'] == 1).sum()
                print(f"  Pseudocough: {pseudocough_count} rows")
        
        # Diagnosis stats
        if 'diagnosis_covid_like' in coughvid_rows.columns:
            covid_like_count = (coughvid_rows['diagnosis_covid_like'] == 1).sum()
            print(f"\nDiagnosis COVID-like: {covid_like_count} rows")
        
        # Cough detected stats
        if 'cough_detected_score' in coughvid_rows.columns:
            cough_detected_present = coughvid_rows['cough_detected_score'].notna()
            if cough_detected_present.sum() > 0:
                cough_detected_values = coughvid_rows.loc[cough_detected_present, 'cough_detected_score']
                print(f"\nCough detected score: {cough_detected_present.sum()} values present")
                print(f"  Mean: {cough_detected_values.mean():.2f}")
                print(f"  Min: {cough_detected_values.min():.2f}")
                print(f"  Max: {cough_detected_values.max():.2f}")
    
    print(f"\n✓ Enriched manifest saved to: {out_path.absolute()}")


if __name__ == '__main__':
    main()

