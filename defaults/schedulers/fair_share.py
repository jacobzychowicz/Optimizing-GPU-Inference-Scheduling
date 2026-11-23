"""Fair share across users: whoever has the least work waiting goes next.

One user's 300-line script can't hold the GPU while another user waits on a
single retake: the retake's user has 1 request queued, so it runs next. Between
two big scripts it alternates, because serving one shrinks its backlog. Within
a user, requests run in arrival order. Jobs without a "user" share one queue,
which makes this plain first come, first served.
"""


def select(ready, now_ms):
    """Return the oldest job of the user with the smallest backlog."""
    backlog = {}
    for job in ready:
        user = job.get("user", "")
        backlog[user] = backlog.get(user, 0) + 1
    return min(ready, key=lambda job: (backlog[job.get("user", "")], job["arrival_ms"], job["id"]))
