#!/usr/bin/env python
"""
Run the PolyBot Streamlit dashboard.

Usage:
    python scripts/run_dashboard.py

Or directly with Streamlit:
    streamlit run dashboard/app.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the Streamlit dashboard."""
    dashboard_path = Path(__file__).parent.parent / "dashboard" / "app.py"

    if not dashboard_path.exists():
        print(f"Error: Dashboard not found at {dashboard_path}")
        sys.exit(1)

    print("Starting PolyBot Dashboard...")
    print("Open http://localhost:8501 in your browser")
    print("Press Ctrl+C to stop")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard_path),
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--server.headless=true",
        ]
    )


if __name__ == "__main__":
    main()
