"""Priority that rises while a job waits, so low-priority work is not stuck forever.

A job gains 1 priority point for every AGING_MS of wait. At the moment a job
arrives, this matches strict priority.
"""

AGING_MS = 1000


def select(ready, now_ms):
    """Return the job with the highest aged priority."""

    def score(job):
        waited = max(0, now_ms - job["arrival_ms"])
        aged = job.get("priority", 0) + waited / AGING_MS
        return (-aged, job["arrival_ms"], job["id"])

    return min(ready, key=score)
