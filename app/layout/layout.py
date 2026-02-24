from dash import html, dcc
from layout.components import (
    create_card, create_control_group, create_dropdown, 
    create_slider, create_radio_items, create_checkbox,
    create_input, create_loading, create_section_header
)


def create_upload_section():
    """Create file upload card"""
    return create_card([
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                html.I(className='upload-icon'),
                html.Div('Drag and Drop or Click to Upload', className='upload-text'),
                html.Div('Supported: .csv, .xlsx, .xls, .txt, .h5ad', className='upload-subtext')
            ]),
            className='upload-area',
            multiple=False
        ),
        html.Div(id='upload-status', className='upload-status')
    ], title='📁 Upload Data')


def create_data_preview_section():
    """Create data preview card"""
    return create_card([
        html.Div(id='data-info', className='data-info'),
        html.Div(id='column-info', className='column-info')
    ], title='📊 Data Preview', card_id='data-preview-card')


def create_visualization_controls():
    """Create visualization control card"""
    return create_card([
        # Mode selection
        create_radio_items(
            'vis-mode',
            options=[
                {'label': '📍 Point Cloud', 'value': 'pointcloud'},
                {'label': '🔺 Mesh', 'value': 'mesh'}
            ],
            value='pointcloud',
            label='Visualization Mode'
        ),
        
        html.Div(id='mode-specific-controls', children=[]),
        
        html.Hr(),
        
        # Point cloud controls
        html.Div([
            create_slider(
                'point-size',
                min_val=1,
                max_val=10,
                step=0.5,
                value=3,
                label='Point Size',
                marks={1: '1', 5: '5', 10: '10'}
            ),
            
            create_slider(
                'point-opacity',
                min_val=0,
                max_val=1,
                step=0.05,
                value=0.8,
                label='Opacity',
                marks={0: '0', 0.5: '0.5', 1: '1'}
            ),

            create_slider(
                'expand-z',
                min_val=1,
                max_val=30,
                step=5,
                value=1,
                label='Expand Z (Z scale factor)',
                marks={1: '1x', 5: '5x', 10: '10x', 15: '15x', 20: '20x', 25: '25x', 30: '30x'}
            ),
            
            html.Hr(style={'margin': '10px 0'}),
            
            html.Div([
                html.Label('Performance Settings', className='control-label', style={'fontWeight': 'bold'}),
                html.Div('For datasets >10,000 points', className='help-text'),
            ]),
            
            create_slider(
                'sampling-ratio',
                min_val=0.01,
                max_val=1.0,
                step=0.01,
                value=1.0,
                label='Sampling Ratio (1.0 = all points)',
                marks={0.01: '1%', 0.25: '25%', 0.5: '50%', 0.75: '75%', 1.0: '100%'}
            ),
        ], id='point-controls'),
        
        # Mesh controls (hidden by default)
        html.Div([
            create_slider(
                'mesh-opacity',
                min_val=0,
                max_val=1,
                step=0.05,
                value=0.3,
                label='Mesh Opacity',
                marks={0: '0', 0.5: '0.5', 1: '1'}
            ),
            
            create_checkbox('show-points-overlay', 'Show Points Overlay', value=False),
            
            html.Div(
                '💡 如果有points在mesh外面，这是正常的几何重建现象，mesh是对点云的"逼近"而非精确包裹。',
                className='help-text',
                style={
                    'marginTop': '8px',
                    'marginLeft': '24px',
                    'fontSize': '12px',
                    'color': '#86868b',
                    'fontStyle': 'italic',
                    'lineHeight': '1.4'
                }
            ),
        ], id='mesh-controls', style={'display': 'none'}),
        
        html.Hr(),
        
        # Color controls
        create_dropdown(
            'color-by',
            options=[],
            value=None,
            placeholder='None (single color)',
            label='Color By'
        ),
        
        # Filter controls
        create_dropdown(
            'filter-column',
            options=[],
            value=None,
            placeholder='None',
            label='Filter By Column'
        ),
        
        html.Div([
            create_dropdown(
                'filter-values',
                options=[],
                value=None,
                placeholder='Select values to show',
                multi=True
            )
        ], id='filter-values-container', style={'display': 'none'}),
        
    ], title='🎨 Visualization Controls')


def create_mesh_method_controls():
    """Create mesh method selection and parameter controls"""
    return html.Div([
        html.Hr(),
        
        create_dropdown(
            'mesh-method',
            options=[
                {'label': 'Marching Cubes', 'value': 'marchingcube'},
                {'label': 'Voxel (Density + Contour)', 'value': 'voxel'}
            ],
            value='voxel',
            placeholder='Select mesh method',
            label='Mesh Method'
        ),
        
        html.Div(id='mesh-params', children=[]),
        
        html.Div([
            html.Button(
                '🔄 Render Mesh',
                id='render-mesh-btn',
                className='btn btn-success',
                n_clicks=0,
                style={
                    'width': '100%',
                    'marginTop': '15px',
                    'padding': '10px',
                    'fontSize': '16px',
                    'fontWeight': 'bold',
                    'backgroundColor': '#28a745',
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '5px',
                    'cursor': 'pointer'
                }
            )
        ])
    ])


def create_export_section():
    """Create export controls card"""
    return create_card([
        html.Div([
            html.Button('📥 Export Point Cloud (JSON)', id='export-points-btn', className='btn btn-primary', style={'width': '100%', 'marginBottom': '10px'}),
            html.Button('📦 Export Mesh (OBJ)', id='export-mesh-obj-btn', className='btn btn-primary', style={'width': '100%', 'marginBottom': '10px'}),
            html.Button('📊 Export Mesh (JSON)', id='export-mesh-json-btn', className='btn btn-primary', style={'width': '100%'}),
        ]),
        html.Div(id='export-status', style={'marginTop': '10px', 'fontSize': '12px', 'color': '#666'})
    ], title='💾 Export Data')


def create_status_section():
    """Create status display card"""
    return create_card([
        html.Pre(id='status-text', children='Ready', className='status-text')
    ], title='📝 Status')


def create_main_layout():
    """Create the main application layout"""
    return html.Div([
        # Main container
        html.Div([
            # Left sidebar with controls
            html.Div([
                create_upload_section(),
                create_data_preview_section(),
                create_visualization_controls(),
                # create_export_section(),
                create_status_section()
            ], className='sidebar'),
            
            # Right main visualization area
            html.Div([
                create_card([
                    create_loading(
                        dcc.Graph(
                            id='3d-plot',
                            config={
                                'displayModeBar': True,
                                'displaylogo': False,
                                'modeBarButtonsToRemove': ['select2d', 'lasso2d']
                            }
                        )
                    )
                ], title='🎯 3D Visualization')
            ], className='main-content')
        ], className='container'),
        
        # Hidden stores for data
        dcc.Store(id='data-store'),
        dcc.Store(id='mesh-data-store')
        
    ], className='app-wrapper')
