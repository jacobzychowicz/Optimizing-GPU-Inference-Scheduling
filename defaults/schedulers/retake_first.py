"""Interactive work first, without starving long jobs: earliest deadline first.

Each kind of request gets a target finish time. A retake (someone is waiting on
it) should start within RETAKE_SLO_MS of arriving; a script segment within
SEGMENT_SLO_MS. The job with the earliest deadline (arrival + target) runs
next. A new retake beats segments that are not yet close to their own deadline,
but a segment that has waited its full target beats new retakes, so neither side
can be starved.
"""

RETAKE_SLO_MS = 2000
SEGMENT_SLO_MS = 600000
OTHER_SLO_MS = 60000

_SLO = {"retake": RETAKE_SLO_MS, "script_segment": SEGMENT_SLO_MS}


def select(ready, now_ms):
    """Return the job with the earliest deadline."""
    return min(
        ready,
        key=lambda job: (job["arrival_ms"] + _SLO.get(job.get("kind"), OTHER_SLO_MS), job["arrival_ms"], job["id"]),
    )
