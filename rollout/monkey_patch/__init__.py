import threading

from triton.runtime.autotuner import Autotuner

_autotune_lock = threading.Lock()


def _install_autotune_lock():
    orig_run = Autotuner.run
    if getattr(orig_run, "_fla_serialized", False):
        return

    def run(self, *args, **kwargs):

        if len(self.configs) <= 1:
            return orig_run(self, *args, **kwargs)


        with _autotune_lock:
            return orig_run(self, *args, **kwargs)

    run._fla_serialized = True
    Autotuner.run = run


_install_autotune_lock()
