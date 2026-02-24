import sys
from pathlib import Path

# Add algorithms directory to Python path
algorithms_path = Path(__file__).parent / 'algorithms'
sys.path.insert(0, str(algorithms_path))

from dash import Dash
from layout.layout import create_main_layout
from callbacks.callbacks import register_all_callbacks


# Initialize the Dash app
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    title='3D Point Cloud & Mesh Viewer'
)

# Set up the layout
app.layout = create_main_layout()

# Register all callbacks
register_all_callbacks(app)

# Run the server
if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting 3D Point Cloud & Mesh Viewer")
    print("=" * 60)
    print("📍 Open your browser and navigate to: http://127.0.0.1:8050")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=8050)
