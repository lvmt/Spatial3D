"""
State management helpers for the Dash app
"""

def create_initial_state():
    """Create initial app state"""
    return {
        'data': None,
        'mode': 'pointcloud',
        'mesh_method': None,
        'color_by': None,
        'filter_col': None,
        'filter_values': []
    }


def format_status_message(df=None, mode='pointcloud', method=None, error=None, runtime=None):
    """
    Format status message for display
    
    Args:
        df: DataFrame with data
        mode: Current visualization mode
        method: Current mesh method (if applicable)
        error: Error message (if any)
        runtime: Algorithm runtime in seconds (if applicable)
        
    Returns:
        Formatted status string
    """
    messages = []
    
    if error:
        messages.append(f"❌ Error: {error}")
        return '\n'.join(messages)
    
    if df is not None:
        messages.append(f"✓ Loaded {len(df):,} points")
    
    if mode == 'pointcloud':
        messages.append(f"📍 Mode: Point Cloud")
    elif mode == 'mesh' and method:
        messages.append(f"🔺 Mode: Mesh ({method})")
    
    if runtime is not None:
        messages.append(f"⏱️ Computation time: {runtime:.2f}s")
    
    return '\n'.join(messages) if messages else 'Ready'
