import os


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("FLA_DISABLE_TENSOR_CACHE", "1")

import argparse

import ray
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, model_validator

from rollout.router import Router
from rollout.dashboard import DASHBOARD_HTML


_ENV_DENY = {"CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "RAY_ADDRESS"}

app = FastAPI()
router: Router = None


class GenRequest(BaseModel):


    model_config = ConfigDict(extra="forbid")

    session_id: str
    prompt_ids: list[int]
    stop_ids: list[list[int]] = []
    max_new_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    split_ids: list[int] = []
    subs: list[tuple[int, list[int]]] = []
    swap: list[int] = []

    ephem_at: int | None = None
    swap_at: int | None = None

    @model_validator(mode="after")
    def require_swap_at(self):
        assert not self.swap or self.swap_at is not None, \
            "swap  requires an explicit boundary when nonempty:  swap_at"
        return self


@app.post("/generate")
async def generate(req: GenRequest):
    return await router.generate(req.session_id, req.model_dump())


class ReleaseRequest(BaseModel):
    session_id: str


@app.post("/release")
async def release(req: ReleaseRequest):
    router.release(req.session_id)
    return {"released": req.session_id}


@app.get("/stats")
async def stats():
    return await router.stats()


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


@app.on_event("startup")
async def _startup():
    await router.start()


@app.on_event("shutdown")
async def _shutdown():
    if router is not None:
        await router.stop()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--max-reside", type=int, default=128)
    p.add_argument("--max-token", type=int, default=1048576)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--offload", action=argparse.BooleanOptionalAction, default=True,
                   help="Offload least-recently-used inactive KV only when free pages are insufficient")
    p.add_argument("--offload-cpu-gb", type=float, default=16.)
    p.add_argument("--offload-transfer-mb", type=float, default=64.)
    args = p.parse_args()

    env_vars = {k: v for k, v in os.environ.items() if k not in _ENV_DENY}


    os.environ.pop("RAY_ADDRESS", None)
    ray.init(address="local", runtime_env={"env_vars": env_vars})

    n = int(ray.cluster_resources().get("GPU", 0))
    if n == 0:
        raise RuntimeError("Ray  found no available  GPU")


    from rollout.engine_actor import EngineActor

    actors = [
        EngineActor.remote(
            engine_id=i,
            model_path=args.model,
            max_token=args.max_token,
            max_reside=args.max_reside,
            offload=args.offload,
            offload_cpu_gb=args.offload_cpu_gb, offload_transfer_mb=args.offload_transfer_mb)
        for i in range(n)
    ]

    global router
    router = Router(actors)


    uvicorn.run(app, host=args.host, port=args.port, timeout_keep_alive=10**9)


if __name__ == "__main__":
    main()
