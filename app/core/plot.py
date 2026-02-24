import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Optional


def create_pointcloud_figure(
    df: pd.DataFrame,
    point_size: float = 3,
    opacity: float = 0.8,
    color_by: Optional[str] = None,
    filter_col: Optional[str] = None,
    filter_values: Optional[list] = None,
    sampling_ratio: float = 1.0
):
    """
    Create a Plotly 3D scatter plot for point cloud visualization
    
    Args:
        df: DataFrame with x, y, z columns
        point_size: Size of points
        opacity: Opacity of points (0-1)
        color_by: Column name to color by
        filter_col: Column to filter by
        filter_values: Values to keep if filtering
        
    Returns:
        Plotly Figure object
    """
    print(f"create_pointcloud_figure called: df.shape={df.shape}, color_by={color_by}, sampling={sampling_ratio}")
    
    # Apply filtering if specified
    plot_df = df.copy()
    if filter_col and filter_values and filter_col in df.columns:
        plot_df = df[df[filter_col].isin(filter_values)]
    
    # Apply sampling for performance
    if sampling_ratio < 1.0 and len(plot_df) > 1000:
        sample_size = max(100, int(len(plot_df) * sampling_ratio))
        plot_df = plot_df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} points from {len(df)} (ratio={sampling_ratio})")
    
    print(f"After filtering and sampling: {len(plot_df)} points")
    
    if len(plot_df) == 0:
        return create_empty_figure("No data points after filtering")
    
    # Prepare hover data
    hover_cols = ['x', 'y', 'z']
    if color_by and color_by in plot_df.columns:
        hover_cols.append(color_by)
    
    hover_text = plot_df[hover_cols].apply(
        lambda row: '<br>'.join([f'{col}: {row[col]}' for col in hover_cols]),
        axis=1
    )
    
    # Create scatter plot
    fig = go.Figure()
    
    if color_by and color_by in plot_df.columns:
        # Color by specified column
        if pd.api.types.is_numeric_dtype(plot_df[color_by]):
            # Continuous coloring
            fig.add_trace(go.Scatter3d(
                x=plot_df['x'],
                y=plot_df['y'],
                z=plot_df['z'],
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=plot_df[color_by],
                    colorscale='Viridis',
                    opacity=opacity,
                    colorbar=dict(title=color_by)
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                name='Points'
            ))
        else:
            # Categorical coloring
            categories = plot_df[color_by].unique()
            for cat in categories:
                cat_df = plot_df[plot_df[color_by] == cat]
                fig.add_trace(go.Scatter3d(
                    x=cat_df['x'],
                    y=cat_df['y'],
                    z=cat_df['z'],
                    mode='markers',
                    marker=dict(
                        size=point_size,
                        opacity=opacity
                    ),
                    name=str(cat),
                    text=cat_df[hover_cols].apply(
                        lambda row: '<br>'.join([f'{col}: {row[col]}' for col in hover_cols]),
                        axis=1
                    ),
                    hovertemplate='%{text}<extra></extra>'
                ))
    else:
        # Single color
        fig.add_trace(go.Scatter3d(
            x=plot_df['x'],
            y=plot_df['y'],
            z=plot_df['z'],
            mode='markers',
            marker=dict(
                size=point_size,
                color='#1f77b4',
                opacity=opacity
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            name='Points'
        ))
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, showgrid=False, zeroline=False, showticklabels=False, title=''),
            yaxis=dict(visible=False, showgrid=False, zeroline=False, showticklabels=False, title=''),
            zaxis=dict(visible=False, showgrid=False, zeroline=False, showticklabels=False, title=''),
            aspectmode='data'
        ),
        legend=dict(
            yanchor="top",
            y=0.95,  # 降低图例位置，避免与工具栏重叠
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 1)",
            bordercolor="#d2d2d7",
            borderwidth=1
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='white',
        plot_bgcolor='white',
        height=700
    )
    
    return fig


def create_mesh_figure(
    vertices: np.ndarray,
    faces: np.ndarray,
    mesh_opacity: float = 1.0,
    show_points: bool = False,
    point_cloud: Optional[np.ndarray] = None,
    point_size: float = 2,
    point_df: Optional[pd.DataFrame] = None,
    color_by: Optional[str] = None
):
    """
    Create a Plotly 3D mesh plot
    
    Args:
        vertices: Nx3 array of vertex coordinates
        faces: Mx3 array of face indices
        mesh_opacity: Opacity of mesh surface (0-1)
        show_points: Whether to overlay point cloud
        point_cloud: Optional Nx3 array of point coordinates
        point_size: Size of overlay points
        point_df: Optional DataFrame with point data for coloring
        color_by: Optional column name to color points by
        point_cloud: Optional Nx3 array of point coordinates
        point_size: Size of overlay points
        
    Returns:
        Plotly Figure object
    """
    if len(vertices) == 0 or len(faces) == 0:
        return create_empty_figure("No mesh data to display")
    
    fig = go.Figure()
    
    # Add mesh with optimized lighting and consistent normals
    # Normals are now computed correctly in mesh generation, enabling smooth shading
    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=mesh_opacity,
        color='lightgray',
        flatshading=False,   # 使用平滑着色，配合一致的法线
        lighting=dict(
            ambient=0.35,
            diffuse=0.55,
            specular=0.8,
            roughness=0.2,
            fresnel=0.1
        ),
        lightposition=dict(
            x=120,
            y=120,
            z=1500
        ),
        name='Mesh'
    ))
    
    # Optionally add point cloud overlay with color support
    if show_points and point_cloud is not None and len(point_cloud) > 0:
        # Apply color by if specified
        if color_by and point_df is not None and color_by in point_df.columns:
            if pd.api.types.is_numeric_dtype(point_df[color_by]):
                # Continuous coloring
                fig.add_trace(go.Scatter3d(
                    x=point_cloud[:, 0],
                    y=point_cloud[:, 1],
                    z=point_cloud[:, 2],
                    mode='markers',
                    marker=dict(
                        size=point_size,
                        color=point_df[color_by],
                        colorscale='Viridis',
                        opacity=0.6,
                        colorbar=dict(title=color_by, x=1.02)
                    ),
                    name='Points',
                    hovertemplate=f'x: %{{x}}<br>y: %{{y}}<br>z: %{{z}}<br>{color_by}: %{{marker.color}}<extra></extra>'
                ))
            else:
                # Categorical coloring
                categories = point_df[color_by].unique()
                for cat in categories:
                    mask = point_df[color_by] == cat
                    cat_points = point_cloud[mask]
                    fig.add_trace(go.Scatter3d(
                        x=cat_points[:, 0],
                        y=cat_points[:, 1],
                        z=cat_points[:, 2],
                        mode='markers',
                        marker=dict(
                            size=point_size,
                            opacity=0.6
                        ),
                        name=str(cat),
                        hovertemplate=f'x: %{{x}}<br>y: %{{y}}<br>z: %{{z}}<br>{color_by}: {cat}<extra></extra>'
                    ))
        else:
            # No coloring, use default
            fig.add_trace(go.Scatter3d(
                x=point_cloud[:, 0],
                y=point_cloud[:, 1],
                z=point_cloud[:, 2],
                mode='markers',
                marker=dict(
                    size=point_size,
                    color='darkblue',
                    opacity=0.3
                ),
                name='Points'
            ))
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, showgrid=False, zeroline=False, showticklabels=False, title=''),
            yaxis=dict(visible=False, showgrid=False, zeroline=False, showticklabels=False, title=''),
            zaxis=dict(visible=False, showgrid=False, zeroline=False, showticklabels=False, title=''),
            aspectmode='data'
        ),
        legend=dict(
            yanchor="top",
            y=0.95,  # 降低图例位置，避免与工具栏重叠
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#d2d2d7",
            borderwidth=1
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='white',
        plot_bgcolor='white',
        height=700
    )
    
    return fig


def create_empty_figure(message: str = "No data to display"):
    """Create an empty figure with a message"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color='gray')
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor='white',
        plot_bgcolor='white',
        height=700
    )
    return fig
