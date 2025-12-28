"""Start the Streamlit UI using configuration from .env."""

import subprocess
import sys

from src.utils.config import settings

if __name__ == "__main__":
    # Run streamlit with port from config
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "ui/app.py",
        "--server.port",
        str(settings.streamlit_port),
        "--server.address",
        settings.streamlit_host,
    ]
    subprocess.run(cmd)

