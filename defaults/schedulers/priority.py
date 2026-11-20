"""Strict priority. A larger priority value runs first.

A missing priority is treated as 0. Equal priorities fall back to arrival order.
"""


def select(ready, now_ms):
    """Return the highest-priority waiting job."""
    return min(
        ready,
        key=lambda job: (
            -job.get("priority", 0),
            job["arrival_ms"],
            job["id"],
        ),
    )
