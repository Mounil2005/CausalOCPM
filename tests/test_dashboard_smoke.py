"""
Dashboard smoke test.

Runs the full Streamlit app script (app/dashboard.py) end-to-end via
streamlit.testing.v1.AppTest and asserts no uncaught exception escapes.

Unlike `curl`-ing a running `streamlit run` server, AppTest actually executes
the per-session script body — including every `with tabN:` block — the same
way a real browser session would. A plain HTTP check of the server only
confirms the process started; it does not execute tab code at all.

Run with: pytest -v tests/test_dashboard_smoke.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit.testing.v1 import AppTest

DASHBOARD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "dashboard.py"
)


def test_dashboard_runs_without_exception():
    at = AppTest.from_file(DASHBOARD_PATH, default_timeout=180)
    at.run()
    assert not at.exception, (
        "Dashboard raised an uncaught exception on default load: "
        f"{[str(e) for e in at.exception]}"
    )


def test_dashboard_renders_all_tabs():
    at = AppTest.from_file(DASHBOARD_PATH, default_timeout=180)
    at.run()
    assert not at.exception
    assert len(at.tabs) == 5, f"Expected 5 tabs, found {len(at.tabs)}"
