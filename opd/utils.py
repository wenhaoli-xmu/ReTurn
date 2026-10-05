import os
import random
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist


def _latest(path, marker):
    path = Path(path)
    if (path / marker).exists():
        return path
    checkpoints = [
        item for item in path.glob("step-*")
        if item.is_dir() and (item / marker).exists()
    ]
    return max(checkpoints, key=lambda item: int(item.name.rsplit("-", 1)[-1]), default=None)


def latest_lora(path):
    return _latest(path, "adapter_config.json")


def latest_full(path):
    return _latest(path, "config.json")


def setup_distributed():
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(device)
    return rank, world, device


@dataclass
class Server:
    process: subprocess.Popen
    url: str

    def stop(self):
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            self.process.wait(30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop()


def _free_port():
    lo, hi = 20000, 32000
    for _ in range(64):
        port = random.randint(lo, hi)
        with socket.socket() as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"{lo}-{hi}  contains no available port ")


def launch_server(
        lora_path,
        model,
        port=None,
        max_reside=32,
        max_token=1048576,
        timeout=1800,
        attempts=3):

    checkpoint = latest_lora(lora_path) if lora_path else None
    if checkpoint:
        raise ValueError("The current rollout server requires merged full model weights")
    last = None
    for attempt in range(attempts):
        this_port = port or _free_port()
        cmd = [
            sys.executable, "-m", "rollout.server", "--model", model,
            "--port", str(this_port), "--max-reside", str(max_reside),
            "--max-token", str(max_token),
        ]
        process = subprocess.Popen(cmd, start_new_session=True)
        server = Server(process, f"http://127.0.0.1:{this_port}")
        deadline = time.time() + timeout
        while time.time() < deadline:
            if process.poll() is not None:
                last = RuntimeError(
                    f"rollout server exited with {process.returncode} (port {this_port})")
                break
            try:
                with urllib.request.urlopen(server.url + "/stats", timeout=2):
                    return server
            except Exception:
                time.sleep(1)
        else:
            last = TimeoutError(f"rollout server startup timed out (port {this_port})")
        server.stop()
        print(f"[launch_server]  attempt  {attempt + 1}/{attempts}  launch attempt failed ：{last}", flush=True)
        if port:
            break
    raise last


def _fla_available():
    try:
        import fla.modules
        import fla.ops.gated_delta_rule
    except ImportError:
        return False
    return True


def training_linear_fallback():

    toolkit = True
    try:
        from tilelang.contrib.nvcc import find_cuda_path
        find_cuda_path()
    except Exception:
        toolkit = False
        os.environ["FLA_TILELANG"] = "0"
    hopper = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
    if (hopper and not toolkit) or not _fla_available():
        import transformers.utils.import_utils as imports
        imports.is_flash_linear_attention_available = lambda: False


def validate_tokenizers(student, teacher):
    from transformers import AutoTokenizer

    left = AutoTokenizer.from_pretrained(student)
    right = AutoTokenizer.from_pretrained(teacher)
    if left.get_vocab() != right.get_vocab():
        raise ValueError("OPD requires identical student and teacher token ID mappings")
