from callbacks.io_callbacks import register_io_callbacks
from callbacks.vis_callbacks import register_vis_callbacks


def register_all_callbacks(app):
    """Register all application callbacks"""
    register_io_callbacks(app)
    register_vis_callbacks(app)
