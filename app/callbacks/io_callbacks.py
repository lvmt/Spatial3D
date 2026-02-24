from dash import Input, Output, State
from dash.exceptions import PreventUpdate
import json
from core.io import parse_upload_contents, validate_dataframe, get_column_info
from core.state import format_status_message


def register_io_callbacks(app):
    """Register callbacks for file I/O operations"""
    
    @app.callback(
        [Output('data-store', 'data'),
         Output('upload-status', 'children'),
         Output('data-info', 'children'),
         Output('column-info', 'children'),
         Output('color-by', 'options'),
         Output('filter-column', 'options'),
         Output('status-text', 'children')],
        Input('upload-data', 'contents'),
        State('upload-data', 'filename')
    )
    def handle_upload(contents, filename):
        """Handle file upload and data validation"""
        if contents is None:
            raise PreventUpdate
        
        # Parse file
        df, parse_error = parse_upload_contents(contents, filename)
        if parse_error:
            return (
                None,
                f"❌ {parse_error}",
                "",
                "",
                [],
                [],
                f"Error: {parse_error}"
            )
        
        # Validate data
        df, validate_error = validate_dataframe(df)
        if validate_error:
            return (
                None,
                f"❌ {validate_error}",
                "",
                "",
                [],
                [],
                f"Error: {validate_error}"
            )
        
        # Get column information
        col_info = get_column_info(df)
        
        # Create data info display
        data_info = f"✓ Loaded: {filename}"
        column_info = f"Rows: {col_info['total_rows']:,} | Columns: {len(col_info['columns'])}"
        
        # Create dropdown options for color and filter
        annotation_cols = [col for col in col_info['columns'] if col not in ['x', 'y', 'z']]
        dropdown_options = [{'label': col, 'value': col} for col in annotation_cols]
        
        # Store data as JSON
        data_json = df.to_json(orient='split')
        
        # Status message
        status = format_status_message(df=df, mode='pointcloud')
        
        return (
            data_json,
            "✓ Upload successful",
            data_info,
            column_info,
            dropdown_options,
            dropdown_options,
            status
        )
    
    
    @app.callback(
        [Output('filter-values-container', 'style'),
         Output('filter-values', 'options'),
         Output('filter-values', 'value')],
        [Input('filter-column', 'value'),
         Input('data-store', 'data')]
    )
    def update_filter_values(filter_col, data_json):
        """Update filter value options when filter column changes"""
        if not filter_col or not data_json:
            return {'display': 'none'}, [], None
        
        import pandas as pd
        df = pd.read_json(data_json, orient='split')
        
        if filter_col not in df.columns:
            return {'display': 'none'}, [], None
        
        # Get unique values
        unique_vals = sorted(df[filter_col].unique())
        options = [{'label': str(val), 'value': val} for val in unique_vals]
        
        return {'display': 'block'}, options, unique_vals
