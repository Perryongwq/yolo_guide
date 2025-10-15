"""
Shared utilities to avoid circular imports
"""
import threading
import logging
from configreader.configreader import return_config
from icecream import ic, install
import sys
from contextlib import asynccontextmanager
import logging, signal, asyncio
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")

# Configure icecream
ic.configureOutput(
    includeContext=True,  # Shows file and line number
    prefix="🍦 ",  # Optional: adds emoji prefix for easy spotting
    outputFunction=lambda *args: print(*args, file=sys.stderr),  # Print to stderr
)

# Install ic globally
install()

# Suppress noisy third-party loggers globally
logging.getLogger("pika").setLevel(logging.WARNING)
logging.getLogger("pika.connection").setLevel(logging.WARNING)
logging.getLogger("pika.callback").setLevel(logging.WARNING)
logging.getLogger("pika.adapters").setLevel(logging.WARNING)

# Thread-safe config access
_config = return_config(0)
_config_lock = threading.RLock()


def get_config():
    """Thread-safe config getter"""
    with _config_lock:
        return _config


def get_ic():
    """Get icecream instance"""
    return ic


def update_config(new_config):
    """Thread-safe config setter"""
    global _config
    with _config_lock:
        _config = new_config

def render_error_template(template_name, request, config):
    """Render an error template with common parameters."""
    return templates.TemplateResponse(
        template_name,
        {
            "request": request,
            "fileserverurl": config['CDN_SERVER']
        }
    )
