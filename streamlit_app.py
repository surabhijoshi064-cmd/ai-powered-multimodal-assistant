"""
MindfulAI — Entrypoint redirecting to app.py.
Preserves backwards compatibility for 'streamlit run streamlit_app.py'.
Canonical launch command: 'streamlit run app.py'
"""
import sys
import runpy
from pathlib import Path

root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

app_path = root_dir / "app.py"
runpy.run_path(str(app_path), run_name="__main__")

