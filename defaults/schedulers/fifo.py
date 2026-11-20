"""First come, first served. The earliest arrival runs next."""


def select(ready, now_ms):
    """Return the job with the smallest arrival time. Ties break on id."""
    return min(ready, key=lambda job: (job["arrival_ms"], job["id"]))
