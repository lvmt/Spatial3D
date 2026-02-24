from dash import Input, Output, State, html, ALL
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
import pyvista as pv
import time
from core.plot import create_pointcloud_figure, create_mesh_figure, create_empty_figure
from core.params import MESH_METHODS, get_default_params, parse_radii
from core.state import format_status_message
from layout.components import create_slider, create_input, create_checkbox, create_control_group
import sys
import os

# Add parent directory to path to import algo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from algorithms.algo import (
    construct_pc, mesh4pyvista, mesh4alphashape, mesh4ballpivoting,
    mesh4poisson, mesh4marchingcube, mesh4voxel
)


def register_vis_callbacks(app):
    """Register callbacks for visualization updates"""
    
    @app.callback(
        [Output('mode-specific-controls', 'children'),
         Output('point-controls', 'style'),
         Output('mesh-controls', 'style')],
        Input('vis-mode', 'value')
    )
    def update_mode_controls(mode):
        """Show/hide controls based on visualization mode"""
        if mode == 'mesh':
            from layout.layout import create_mesh_method_controls
            return (
                create_mesh_method_controls(),
                {'display': 'none'},
                {'display': 'block'}
            )
        else:
            return (
                html.Div(),
                {'display': 'block'},
                {'display': 'none'}
            )
    
    
    @app.callback(
        Output('mesh-params', 'children'),
        Input('mesh-method', 'value')
    )
    def update_mesh_params(method):
        """Update parameter inputs based on selected mesh method"""
        if not method:
            raise PreventUpdate
        
        method_config = MESH_METHODS.get(method, {})
        params = method_config.get('params', {})
        
        param_inputs = []
        for param_name, param_config in params.items():
            if param_config['type'] == 'number':
                param_inputs.append(
                    html.Div([
                        create_slider(
                            {'type': 'mesh-param', 'param': param_name},
                            min_val=param_config.get('min', 0),
                            max_val=param_config.get('max', 10),
                            step=param_config.get('step', 0.1),
                            value=param_config['default'],
                            label=param_name.replace('_', ' ').title()
                        ),
                        html.Div(
                            param_config.get('description', ''),
                            className='help-text',
                            style={'marginTop': '5px', 'fontSize': '12px', 'color': '#86868b', 'fontStyle': 'italic'}
                        )
                    ])
                )
            elif param_config['type'] == 'boolean':
                param_inputs.append(
                    html.Div([
                        create_checkbox(
                            {'type': 'mesh-param', 'param': param_name},
                            param_name.replace('_', ' ').title(),
                            value=param_config['default']
                        ),
                        html.Div(
                            param_config.get('description', ''),
                            className='help-text',
                            style={'marginTop': '5px', 'fontSize': '12px', 'color': '#86868b', 'fontStyle': 'italic'}
                        )
                    ])
                )
            elif param_config['type'] == 'text':
                param_inputs.append(
                    html.Div([
                        create_input(
                            {'type': 'mesh-param', 'param': param_name},
                            input_type='text',
                            value=param_config['default'],
                            label=param_name.replace('_', ' ').title(),
                            placeholder=param_config.get('description', '')
                        ),
                        html.Div(
                            param_config.get('description', ''),
                            className='help-text',
                            style={'marginTop': '5px', 'fontSize': '12px', 'color': '#86868b', 'fontStyle': 'italic'}
                        )
                    ])
                )
        
        return param_inputs
    
    
    @app.callback(
        [Output('3d-plot', 'figure'),
         Output('mesh-data-store', 'data'),
         Output('status-text', 'children', allow_duplicate=True)],
        [Input('data-store', 'data'),
         Input('vis-mode', 'value'),
         Input('point-size', 'value'),
         Input('point-opacity', 'value'),
         Input('expand-z', 'value'),
         Input('mesh-opacity', 'value'),
         Input('show-points-overlay', 'value'),
         Input('color-by', 'value'),
         Input('filter-column', 'value'),
         Input('filter-values', 'value'),
         Input('sampling-ratio', 'value')],
        prevent_initial_call='initial_duplicate'
    )
    def update_visualization(data_json, mode, point_size, point_opacity, expand_z,
                           mesh_opacity, show_points, color_by, filter_col, filter_values,
                           sampling_ratio):
        """Update the 3D visualization based on current settings"""
        print(f"\n=== VISUALIZATION CALLBACK TRIGGERED ===")
        print(f"Mode: {mode}")
        print(f"Data available: {data_json is not None}")
        print(f"Point size: {point_size}, Opacity: {point_opacity}")
        print(f"Expand Z: {expand_z}")
        print(f"Color by: {color_by}, Filter col: {filter_col}")
        
        if not data_json:
            print("No data - returning empty figure")
            return create_empty_figure("Upload data to begin"), None, "No data loaded"
        
        # Load dataframe
        print("Loading dataframe...")
        df = pd.read_json(data_json, orient='split')
        print(f"DataFrame loaded: shape={df.shape}, columns={list(df.columns)}")
        
        try:
            if mode == 'pointcloud':
                print("Creating point cloud figure...")
                plot_df = df.copy()
                z_scale = float(expand_z) if expand_z else 1.0
                plot_df['z'] = plot_df['z'].astype(float) * z_scale

                # Create point cloud visualization
                fig = create_pointcloud_figure(
                    df=plot_df,
                    point_size=point_size,
                    opacity=point_opacity,
                    color_by=color_by,
                    filter_col=filter_col,
                    filter_values=filter_values,
                    sampling_ratio=sampling_ratio if sampling_ratio else 1.0
                )
                status = format_status_message(df=df, mode='pointcloud')
                print(f"Point cloud figure created successfully")
                return fig, None, status
            
            elif mode == 'mesh':
                print("Mesh mode selected - returning message to select method")
                return create_empty_figure("Select a mesh method and it will render"), None, "Select mesh method"
                
        except Exception as e:
            error_msg = f"Visualization error: {str(e)}"
            print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            status = format_status_message(error=error_msg)
            return create_empty_figure(error_msg), None, status
    
    
    @app.callback(
        [Output('3d-plot', 'figure', allow_duplicate=True),
         Output('mesh-data-store', 'data', allow_duplicate=True),
         Output('status-text', 'children', allow_duplicate=True)],
        [Input('render-mesh-btn', 'n_clicks')],
        [State('mesh-method', 'value'),
         State('data-store', 'data'),
         State('vis-mode', 'value'),
         State('mesh-opacity', 'value'),
         State('show-points-overlay', 'value'),
         State('color-by', 'value'),
         State('filter-column', 'value'),
         State('filter-values', 'value'),
         State('sampling-ratio', 'value'),
         State({'type': 'mesh-param', 'param': ALL}, 'value'),
         State({'type': 'mesh-param', 'param': ALL}, 'id')],
        prevent_initial_call=True
    )
    def render_mesh_on_button_click(n_clicks, mesh_method, data_json, mode, mesh_opacity, show_points, 
                                    color_by, filter_col, filter_values, sampling_ratio,
                                    param_values, param_ids):
        """Generate mesh when Render button is clicked, with global filtering and color support"""
        print(f"\n=== RENDER MESH BUTTON CLICKED (n_clicks={n_clicks}) ===")
        print(f"Method: {mesh_method}, Mode: {mode}, Color by: {color_by}")
        print(f"Filter: column={filter_col}, values={filter_values}, sampling={sampling_ratio}")
        
        if not n_clicks or not data_json or mode != 'mesh' or not mesh_method:
            print("Conditions not met for mesh generation")
            raise PreventUpdate
        
        try:
            # Load dataframe
            df = pd.read_json(data_json, orient='split')
            print(f"Original dataframe: {len(df)} points")
            
            # Apply global filtering (same as point cloud visualization)
            if filter_col and filter_values and filter_col in df.columns:
                df = df[df[filter_col].isin(filter_values)]
                print(f"After filtering by {filter_col}: {len(df)} points")
            
            # Apply sampling
            if sampling_ratio < 1.0 and len(df) > 1000:
                sample_size = max(100, int(len(df) * sampling_ratio))
                df = df.sample(n=sample_size, random_state=42)
                print(f"After sampling at {sampling_ratio*100:.0f}%: {len(df)} points")
            
            if len(df) == 0:
                return create_empty_figure("No data points after filtering"), None, "❌ No data after filtering"
            
            print(f"Generating mesh for {len(df)} points using {mesh_method}")
            
            start_time = time.time()
            
            # Construct point cloud directly using pyvista
            points_array = df[['x', 'y', 'z']].to_numpy().astype(np.float64)
            pc = pv.PolyData(points_array)
            print(f"Point cloud created with {pc.n_points} points")
            
            # Extract parameters from State using ALL pattern
            params = {}
            for param_id, param_value in zip(param_ids, param_values):
                param_name = param_id['param']
                params[param_name] = param_value
                print(f"  {param_name} = {param_value}")
            
            # Fill missing params with defaults
            default_params = get_default_params(mesh_method)
            for key, value in default_params.items():
                if key not in params:
                    params[key] = value
            
            print(f"Final parameters: {params}")
            
            # Call appropriate mesh function from algo.py
            if mesh_method == 'pyvista':
                mesh = mesh4pyvista(
                    pc, 
                    alpha=params.get('alpha', 2.0),
                    smooth_iter=int(params.get('smooth_iter', 1000))
                )
            elif mesh_method == 'alphashape':
                mesh = mesh4alphashape(
                    pc, 
                    alpha=params.get('alpha', 2.0),
                    smooth_iter=int(params.get('smooth_iter', 500))
                )
            elif mesh_method == 'ballpivoting':
                radii_str = params.get('radii', '1.0,2.0,3.0')
                radii = parse_radii(radii_str)
                mesh = mesh4ballpivoting(
                    pc, 
                    radii=radii,
                    smooth_iter=int(params.get('smooth_iter', 500))
                )
            elif mesh_method == 'poisson':
                # Fix linear_fit: convert empty list or any value to boolean
                linear_fit_value = params.get('linear_fit', False)
                if isinstance(linear_fit_value, list):
                    linear_fit_value = bool(linear_fit_value)  # Empty list -> False
                mesh = mesh4poisson(
                    pc,
                    depth=int(params.get('depth', 8)),
                    width=int(params.get('width', 0)),
                    scale=params.get('scale', 1.1),
                    linear_fit=bool(linear_fit_value),
                    density_threshold=params.get('density_threshold', 0.01),
                    smooth_iter=int(params.get('smooth_iter', 500))
                )
            elif mesh_method == 'marchingcube':
                mesh = mesh4marchingcube(
                    pc,
                    levelset=params.get('levelset', 0),
                    mc_scale_factor=params.get('mc_scale_factor', 1.0),
                    dist_sample_num=int(params.get('dist_sample_num', 100)),
                    smooth_iter=int(params.get('smooth_iter', 500))
                )
            elif mesh_method == 'voxel':
                mesh = mesh4voxel(
                    pc,
                    grid_size=int(params.get('grid_size', 60)),
                    sigma=float(params.get('sigma', 2.5)),
                    iso_percentile=float(params.get('iso_percentile', 70)),
                    hole_size_factor=float(params.get('hole_size_factor', 0.6)),
                    smooth_iter=int(params.get('smooth_iter', 30)),
                    closure_smooth_factor=float(params.get('closure_smooth_factor', 1.0)),
                )
            else:
                raise ValueError(f"Unknown mesh method: {mesh_method}")
            
            runtime = time.time() - start_time
            print(f"Mesh generated in {runtime:.2f}s")
            
            # Convert PyVista mesh to numpy arrays for Plotly
            mesh_for_plot = mesh.extract_surface().triangulate().clean()
            vertices = np.array(mesh_for_plot.points)
            faces = mesh_for_plot.faces.reshape(-1, 4)[:, 1:4]  # Triangulated faces
            print(f"Mesh has {len(vertices)} vertices and {len(faces)} faces")
            
            # Store mesh data
            mesh_data = {
                'vertices': vertices.tolist(),
                'faces': faces.tolist()
            }
            
            # Create mesh figure
            show_overlay = 'checked' in (show_points or [])
            point_cloud_array = None
            overlay_df = None
            overlay_note = ""

            if show_overlay:
                overlay_df = df.copy()
                overlay_points = overlay_df[['x', 'y', 'z']].to_numpy(dtype=np.float64)

                try:
                    overlay_pc = pv.PolyData(overlay_points)
                    enclosed = overlay_pc.select_enclosed_points(
                        mesh_for_plot,
                        tolerance=0.0,
                        check_surface=False,
                    )
                    inside_mask = np.asarray(enclosed.point_data['SelectedPoints']).astype(bool)
                    overlay_df = overlay_df.loc[inside_mask].reset_index(drop=True)
                    point_cloud_array = overlay_df[['x', 'y', 'z']].to_numpy(dtype=np.float64)
                    overlay_note = f"\n📍 Overlay points inside mesh: {len(point_cloud_array):,}/{len(df):,}"
                    print(f"Overlay filter: kept {len(point_cloud_array)} inside points from {len(df)}")
                except Exception as overlay_err:
                    print(f"Warning: overlay inside-mesh filtering failed: {overlay_err}")
                    point_cloud_array = np.empty((0, 3), dtype=np.float64)
                    overlay_df = df.iloc[0:0].copy()
                    overlay_note = "\n⚠ Overlay inside-mesh filtering failed"
            
            fig = create_mesh_figure(
                vertices=vertices,
                faces=faces,
                mesh_opacity=mesh_opacity if mesh_opacity else 0.3,
                show_points=show_overlay,
                point_cloud=point_cloud_array,
                point_size=2,
                point_df=overlay_df if show_overlay else None,
                color_by=color_by if show_overlay else None
            )
            
            status = format_status_message(
                df=df,
                mode='mesh',
                method=mesh_method,
                runtime=runtime
            ) + overlay_note
            
            print("Mesh figure created successfully")
            return fig, mesh_data, status
            
        except ValueError as e:
            error_msg = str(e)
            print(f"Mesh generation failed: {error_msg}")
            status = format_status_message(error=f"Mesh generation failed: {error_msg}")
            return create_empty_figure(error_msg), None, status
        except Exception as e:
            error_msg = f"Error generating mesh: {str(e)}"
            print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            status = format_status_message(error=error_msg)
            return create_empty_figure(error_msg), None, status


def extract_params_from_children(method, children_list):
    """Extract parameter values from UI components (sliders/inputs/checkboxes)"""
    if not children_list:
        return get_default_params(method)
    
    params = {}
    method_config = MESH_METHODS.get(method, {})
    default_params = method_config.get('params', {})
    
    # Extract values from children components
    for child in children_list:
        if not child:
            continue
            
        # Get component props
        props = child.get('props', {}) if isinstance(child, dict) else {}
        children_in_child = props.get('children', [])
        
        # Look for input components with id starting with 'param-'
        for item in (children_in_child if isinstance(children_in_child, list) else [children_in_child]):
            if not isinstance(item, dict):
                continue
                
            item_props = item.get('props', {})
            item_id = item_props.get('id', '')
            
            if isinstance(item_id, str) and item_id.startswith('param-'):
                param_name = item_id.replace('param-', '')
                
                # Get value from slider/input/checkbox
                if 'value' in item_props:
                    params[param_name] = item_props['value']
    
    # Fill in missing params with defaults
    for param_name, param_config in default_params.items():
        if param_name not in params:
            params[param_name] = param_config['default']
    
    return params
