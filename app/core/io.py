import pandas as pd
import base64
import io
import json
from typing import Tuple, Optional, Dict, List
try:
    import anndata as ad
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False
    print("Warning: anndata not installed. H5AD support disabled.")


def parse_upload_contents(contents: str, filename: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Parse uploaded file contents and return DataFrame or error message
    
    Args:
        contents: Base64 encoded file contents from dcc.Upload
        filename: Name of the uploaded file
        
    Returns:
        Tuple of (DataFrame, error_message)
    """
    try:
        # Decode the base64 string
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Try to read the file based on extension
        if filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif filename.endswith('.txt'):
            # Try different separators for text files
            try:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep='\t')
            except:
                try:
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep=' ')
                except:
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep=',')
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(decoded))
        elif filename.endswith('.h5ad'):
            if not ANNDATA_AVAILABLE:
                return None, "H5AD files require anndata package. Please install: pip install anndata"
            # Parse h5ad file (similar to VT3D approach)
            df = parse_h5ad(io.BytesIO(decoded))
            if df is None:
                return None, "Failed to parse H5AD file. Ensure it contains spatial coordinates."
        else:
            return None, f"Unsupported file format: {filename}"
        
        print(f"Loaded file: {filename}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"First few rows:\n{df.head()}")
        
        return df, None
        
    except Exception as e:
        return None, f"Error parsing file: {str(e)}"


def validate_dataframe(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Validate DataFrame has required columns and clean data
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (validated_DataFrame, error_message)
    """
    if df is None or df.empty:
        return None, "DataFrame is empty"
    
    # Normalize column names to lowercase
    df.columns = df.columns.str.lower().str.strip()
    print(f"After normalization, columns: {list(df.columns)}")
    
    # Check for required columns
    required_cols = ['x', 'y', 'z']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        return None, f"Missing required columns: {', '.join(missing_cols)}. Found columns: {', '.join(df.columns)}"
    
    # Check if x, y, z are numeric
    for col in required_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                return None, f"Column '{col}' contains non-numeric values that cannot be converted"
    
    # Count NaN values
    nan_counts = df[required_cols].isna().sum()
    total_nans = nan_counts.sum()
    
    if total_nans > 0:
        # Drop rows with NaN in x, y, z
        original_len = len(df)
        df = df.dropna(subset=required_cols)
        dropped = original_len - len(df)
        print(f"Dropped {dropped} rows with NaN values in x/y/z coordinates")
    
    if len(df) == 0:
        return None, "No valid data rows after removing NaN values"
    
    return df, None


def parse_h5ad(file_buffer: io.BytesIO) -> Optional[pd.DataFrame]:
    """
    Parse h5ad file following VT3D approach
    
    Args:
        file_buffer: BytesIO buffer containing h5ad data
        
    Returns:
        DataFrame with spatial coordinates and metadata
    """
    if not ANNDATA_AVAILABLE:
        return None
    
    try:
        # Load anndata object
        adata = ad.read_h5ad(file_buffer)
        
        # Extract spatial coordinates (VT3D uses 'spatial3D' by default)
        spatial_keys = ['spatial3D', 'spatial', 'X_spatial', 'X_umap', 'X_pca']
        coords = None
        spatial_key_used = None
        
        for key in spatial_keys:
            if key in adata.obsm.keys():
                coords = adata.obsm[key]
                spatial_key_used = key
                print(f"Found spatial coordinates in obsm['{key}']")
                break
        
        if coords is None:
            print("No spatial coordinates found. Available keys:", list(adata.obsm.keys()))
            return None
        
        # Create DataFrame with coordinates
        if coords.shape[1] == 2:
            # 2D coordinates, add z=0
            df = pd.DataFrame({
                'x': coords[:, 0],
                'y': coords[:, 1],
                'z': 0
            })
        elif coords.shape[1] >= 3:
            # 3D coordinates
            df = pd.DataFrame({
                'x': coords[:, 0],
                'y': coords[:, 1],
                'z': coords[:, 2]
            })
        else:
            print(f"Invalid coordinate dimension: {coords.shape[1]}")
            return None
        
        # Add metadata from obs (annotations like cell types, clusters)
        for col in adata.obs.columns:
            df[col] = adata.obs[col].values
        
        print(f"Loaded H5AD: {adata.n_obs} cells, {adata.n_vars} genes")
        print(f"Coordinate key: {spatial_key_used}, shape: {coords.shape}")
        print(f"Available annotations: {list(adata.obs.columns)}")
        
        return df
        
    except Exception as e:
        print(f"Error parsing H5AD file: {str(e)}")
        return None


def export_to_json(df: pd.DataFrame, filename: str, annotations: List[str] = None) -> str:
    """
    Export data to JSON format following VT3D structure
    
    Args:
        df: DataFrame with x, y, z coordinates
        filename: Output filename
        annotations: List of annotation columns to export
        
    Returns:
        Path to exported JSON file
    """
    try:
        # Point cloud data
        point_data = df[['x', 'y', 'z']].values.tolist()
        
        # Annotation data
        anno_data = {}
        if annotations:
            for anno in annotations:
                if anno in df.columns:
                    anno_data[anno] = df[anno].tolist()
        
        # Summary
        summary = {
            'n_points': len(df),
            'box': {
                'xmin': float(df['x'].min()),
                'xmax': float(df['x'].max()),
                'ymin': float(df['y'].min()),
                'ymax': float(df['y'].max()),
                'zmin': float(df['z'].min()),
                'zmax': float(df['z'].max())
            },
            'annotations': list(anno_data.keys())
        }
        
        # Save JSON
        output = {
            'points': point_data,
            'annotations': anno_data,
            'summary': summary
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f)
        
        print(f"Exported {len(df)} points to {filename}")
        return filename
        
    except Exception as e:
        print(f"Error exporting to JSON: {str(e)}")
        return None


def get_column_info(df: pd.DataFrame) -> dict:
    """
    Get information about DataFrame columns for UI display
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with column information
    """
    info = {
        'total_rows': len(df),
        'columns': list(df.columns),
        'numeric_cols': list(df.select_dtypes(include=['number']).columns),
        'categorical_cols': []
    }
    
    # Identify categorical columns (non-numeric or numeric with few unique values)
    for col in df.columns:
        if col not in ['new_x', 'new_y', 'new_z']:
            if not pd.api.types.is_numeric_dtype(df[col]) or df[col].nunique() < 20:
                info['categorical_cols'].append(col)
    
    return info
