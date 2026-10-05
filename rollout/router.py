import asyncio


class Router:
    def __init__(self, actors):
        self.actors = actors
        self.n = len(actors)
        self._sessions = {}
        self._load = [0] * self.n

    async def start(self):


        await asyncio.gather(*(a.start.remote() for a in self.actors))

    async def stop(self):

        return

    def _pick(self):
        return min(range(self.n), key=lambda i: self._load[i])

    async def generate(self, session_id, payload):


        idx = self._sessions.get(session_id)
        if idx is None:
            idx = self._pick()
            self._sessions[session_id] = idx
            self._load[idx] += 1
        return await self.actors[idx].submit.remote(payload)

    def release(self, session_id):
        idx = self._sessions.pop(session_id, None)
        if idx is not None:
            self._load[idx] -= 1
            self.actors[idx].release.remote(session_id)

    async def stats(self):
        results = await asyncio.gather(*(a.stats.remote() for a in self.actors))
        return {r["id"]: r for r in results}
