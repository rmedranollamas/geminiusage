"""Dummy quota provider for open-source version."""
from typing import Any, Dict

def get_status() -> Dict[str, Any]:
    """Returns a dummy status indicating no quota provider is active.

    To implement a custom quota provider (e.g. for internal tools),
    override this file with your own implementation that returns
    the active plan and remaining quota percentage for each model.
    """
    return {
        "running": False,
        "connected": False,
        "models": [],
        "email": "",
        "plan": "",
    }
