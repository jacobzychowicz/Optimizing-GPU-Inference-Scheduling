"""Shortest predicted job first.

Uses predicted_duration_ms from the workload, which can disagree with the
model's real runtime if you edit one and not the other.
"""


def select(ready, now_ms):
    """Return the job with the smallest predicted duration."""
    return min(
        ready,
        key=lambda job: (
            job["predicted_duration_ms"],
            job["arrival_ms"],
            job["id"],
        ),
    )
