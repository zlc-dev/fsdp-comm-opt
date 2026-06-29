import threading
import torch
import torch.distributed as dist


class AggregatedWork(dist.Work):

    def __init__(self, works, postprocess_fn=None):
        super().__init__()

        self.works = works
        self.postprocess_fn = postprocess_fn

        self._done = False
        self._exception = None
        self._lock = threading.Lock()

        # 用于 CUDA stream 同步
        self._event = None

    def _run_postprocess_once(self):
        with self._lock:
            if self._done:
                return

            if self.postprocess_fn is not None:
                self.postprocess_fn()

            # 记录当前 stream 完成事件
            if torch.cuda.is_available():
                self._event = torch.cuda.Event()
                self._event.record(torch.cuda.current_stream())

            self._done = True

    def is_completed(self):
        if not all(w.is_completed() for w in self.works):
            return False

        self._run_postprocess_once()
        return True

    def is_success(self):
        try:
            self.wait()
            return True
        except Exception:
            return False

    def exception(self):
        return self._exception

    def wait(self, timeout=None):
        try:
            for w in self.works:
                w.wait(timeout)

            self._run_postprocess_once()

            if self._event is not None:
                self._event.synchronize()

            return True

        except Exception as e:
            self._exception = e
            raise

    def block_current_stream(self):
        for w in self.works:
            w.block_current_stream()

        if self._event is not None:
            torch.cuda.current_stream().wait_event(self._event)

    def get_future(self):
        futs = [w.get_future() for w in self.works]

        fut = torch.futures.collect_all(futs)

        def _callback(_):
            self._run_postprocess_once()
            return self.result()

        return fut.then(_callback)

    def result(self):
        self.wait()
        return [w.result() for w in self.works]