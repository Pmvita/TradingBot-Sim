#!/usr/bin/env python3
"""Standalone GUI Launcher for Trading Bot Simulator."""

import sys
import os
import subprocess
from pathlib import Path

def main():
    """Launch the appropriate GUI."""
    if len(sys.argv) < 2:
        print("🤖 Trading Bot Simulator GUI Launcher")
        print("=" * 40)
        print("Usage:")
        print("  python gui_launcher.py streamlit [port]")
        print("  python gui_launcher.py dash [port]")
        print("\nExamples:")
        print("  python gui_launcher.py streamlit 8501")
        print("  python gui_launcher.py dash 8050")
        print("\nDefault ports:")
        print("  Streamlit: 8501")
        print("  Dash: 8050")
        sys.exit(1)
    
    interface = sys.argv[1].lower()
    port = int(sys.argv[2]) if len(sys.argv) > 2 else (8501 if interface == "streamlit" else 8050)
    
    # Get the project root
    project_root = Path(__file__).parent
    
    if interface == "streamlit":
        print(f"🚀 Launching Streamlit GUI on port {port}...")
        print(f"📱 Open your browser to: http://localhost:{port}")
        
        # Run streamlit using poetry with full path
        cmd = [
            "/Users/petermvita/.local/bin/poetry", "run", "streamlit", "run",
            str(project_root / "src" / "tbs" / "gui" / "streamlit_app.py"),
            "--server.port", str(port),
            "--server.address", "localhost",
            "--server.headless", "true"
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to launch Streamlit: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n👋 Streamlit GUI stopped")
            
    elif interface == "dash":
        print(f"🚀 Launching Dash GUI on port {port}...")
        print(f"📱 Open your browser to: http://localhost:{port}")
        
        # Add src to path and run dash
        sys.path.insert(0, str(project_root / "src"))
        
        try:
            from tbs.gui.dash_app import app
            app.run_server(debug=False, host="0.0.0.0", port=port)
        except ImportError as e:
            print(f"❌ Failed to import Dash app: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n👋 Dash GUI stopped")
            
    else:
        print(f"❌ Unknown interface: {interface}")
        print("Available interfaces: streamlit, dash")
        sys.exit(1)

if __name__ == "__main__":
    main()
