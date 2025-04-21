# Author: Peter Jin

from typing import Any, Optional, Union
from dataclasses import dataclass
import concurrent.futures
import json
import multiprocessing as mp
import os
import subprocess
import time
import traceback
import urllib.request

try:
    import sglang
    import sglang.srt.entrypoints.http_server
    import sglang.srt.server_args
    import sglang.srt.utils
except ImportError:
    sglang = None


def _sglang_server_init(tx, server_args):
    def _post_init():
        tx.send(None)
    sglang.srt.entrypoints.http_server.launch_server(
        server_args,
        launch_callback=_post_init,
    )


def _sglang_server_heartbeat(
    _host: str = None,
    _port: int = None,
):
    req_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    req = urllib.request.Request(
        f"http://{_host}:{_port}/health_generate",
        headers=req_headers,
    )
    with urllib.request.urlopen(req, timeout=2) as out:
        out_data = out.read()
    output = out_data.decode("utf-8")
    return {
        "output": output,
    }


def _sglang_server_submit(
    input_ids: list = None,
    sampling_params: dict = None,
    _ctr: int = None,
    _host: int = None,
    _port: int = None,
):
    req_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    req_body = {
        "input_ids": input_ids,
        "sampling_params": sampling_params,
    }
    req_data = json.dumps(req_body).encode("utf-8")
    req = urllib.request.Request(
        f"http://{_host}:{_port}/generate",
        headers=req_headers,
        data=req_data,
    )
    with urllib.request.urlopen(req) as out:
        out_data = out.read()
    output = json.loads(out_data.decode("utf-8"))
    return {
        "output": output,
        "_ctr": _ctr,
    }


@dataclass
class SGLangRequest:
    _ctr: int
    _key: Any
    _output: Any

    def _counter(self) -> int:
        return self._ctr

    def key(self) -> Any:
        return self._key

    def result(self) -> Any:
        return self._output


class SGLangServerExecutor:
    def __init__(
        self,
        max_workers: int = 256,
        server_host: int = "127.0.0.1",
        server_port: int = 30000,
        backend: str = "spawn",
        # NB(peter): upgrade sglang here.
        subprocess_venv_path: str = None,
        **kwargs
    ):
        if server_host is not None:
            assert kwargs.get("host", None) is None
            kwargs["host"] = server_host
        if server_port is not None:
            assert kwargs.get("port", None) is None
            kwargs["port"] = server_port
        kwargs.setdefault("skip_tokenizer_init", True)
        kwargs.setdefault("log_level", "warning")
        # NB(peter): fa3 backend requires sglang >= 0.4.5.
        kwargs.setdefault("attention_backend", "fa3")
        if backend == "spawn":
            server_args = sglang.srt.server_args.ServerArgs(
                **kwargs
            )
            mpctx = mp.get_context("spawn")
            rx, tx = mpctx.Pipe(False)
            proc = mpctx.Process(
                target=_sglang_server_init,
                args=(tx, server_args,),
            )
            proc.start()
            self._server_host = server_host
            self._server_port = server_port
            self._server_proc = proc
            self._server_pid = proc.pid
        elif backend == "subprocess":
            # TODO(peter): alternative subprocess-based implementation
            # to avoid sglang pip requirement (but need existing venv).
            if subprocess_venv_path is not None:
                python_path = os.path.join(subprocess_venv_path, "bin/python")
            else:
                python_path = "python"
            cmd = [
                python_path,
                "-m",
                "sglang.launch_server",
            ]
            for key, arg in kwargs.items():
                if isinstance(arg, bool):
                    if arg:
                        cmd.append(
                            f"--{key.replace('_', '-')}"
                        )
                    else:
                        cmd.append(
                            f"--no-{key.replace('_', '-')}"
                        )
                else:
                    cmd.append(
                        f"--{key.replace('_', '-')}"
                    )
                    if isinstance(arg, float):
                        cmd.append(str(arg))
                    elif isinstance(arg, int):
                        cmd.append(str(arg))
                    elif isinstance(arg, str):
                        cmd.append(arg)
                    else:
                        raise NotImplementedError
            print(f"DEBUG: SGLangServerExecutor: subprocess command = {cmd}")
            proc = subprocess.Popen(cmd, shell=False, text=True)
            self._server_host = server_host
            self._server_port = server_port
            self._server_proc = proc
            self._server_pid = proc.pid
        else:
            raise NotImplementedError
        assert self._server_pid is not None
        self._server_backend = backend
        self._pool_ctr = 0
        self._pool_dict = dict()
        self._pool_work = set()
        self._pool_exec = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        )
        print(f"DEBUG: SGLangServerExecutor: post init...")
        if backend == "spawn":
            _ = rx.recv()
        elif backend == "subprocess":
            heartbeat_args = {
                "_host": self._server_host,
                "_port": self._server_port,
            }
            while True:
                try:
                    w = self._pool_exec.submit(
                        _sglang_server_heartbeat,
                        **heartbeat_args
                    )
                    work = [w]
                    for w in concurrent.futures.as_completed(work):
                        output = w.result()
                        #print(f"DEBUG: SGLangServerExecutor: post init: heartbeat output = {output}")
                        print(f"DEBUG: SGLangServerExecutor: post init: heartbeat: ok")
                except Exception as e:
                    print(f"DEBUG: SGLangServerExecutor: post init: retry heartbeat: exception = {e}")
                    print(f"DEBUG: SGLangServerExecutor: post init: retry heartbeat: sleep...")
                    time.sleep(10.0)
                    continue
                break
        print(f"DEBUG: SGLangServerExecutor: post init: done")

    def join(self):
        print(f"DEBUG: SGLangServerExecutor: join...")
        if self._server_backend == "spawn":
            sglang.srt.utils.kill_process_tree(self._server_pid)
            self._server_proc = None
        elif self._server_backend == "subprocess":
            self._server_proc.kill()
            self._server_proc = None
            self._server_pid = None
        else:
            raise NotImplementedError

    def submit(
        self,
        input_ids: Union[None, list[int], list[list[int]]] = None,
        prompt_token_ids: Union[None, list[int], list[list[int]]] = None,
        sampling_params: Union[None, dict[str, Any], list[dict[str, Any]]] = None,
        keys: Optional[list[Any]] = None,
    ):
        if (
            input_ids is not None and
            prompt_token_ids is not None
        ):
            assert False, (
                "SGLangServerExecutor.submit supports either `input_ids` or `prompt_token_ids` but not both"
            )
        elif (
            input_ids is None and
            prompt_token_ids is not None
        ):
            input_ids = prompt_token_ids
        assert input_ids is not None

        if (
            isinstance(input_ids, list) and
            len(input_ids) > 0
        ):
            if isinstance(input_ids[0], int):
                input_ids = [input_ids]

        batch_size = len(input_ids)
        work = []

        for batch_idx in range(batch_size):
            params = None
            if isinstance(sampling_params, dict):
                params = sampling_params
            elif isinstance(sampling_params, list):
                if len(sampling_params) <= 0:
                    raise NotImplementedError
                if isinstance(sampling_params[batch_idx], dict):
                    params = sampling_params[batch_idx]
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
            assert params is not None

            submit_args = {
                "input_ids": input_ids[batch_idx],
                "sampling_params": params,
                "_ctr": self._pool_ctr,
                "_host": self._server_host,
                "_port": self._server_port,
            }
            w = self._pool_exec.submit(
                _sglang_server_submit,
                **submit_args
            )
            self._pool_dict[w] = (self._pool_ctr, keys[batch_idx])
            self._pool_work.add(w)
            self._pool_ctr += 1
            work.append(w)

        return work

    def as_completed(self):
        # NB(peter): the `wait` version _should be_ re-entrant-safe.
        # for w in concurrent.futures.as_completed(self._pool_work):
        while self._pool_work:
            done, work = concurrent.futures.wait(
                self._pool_work,
                return_when=concurrent.futures.FIRST_COMPLETED
            )
            self._pool_work = work
            for w in done:
                ctr, key = self._pool_dict.pop(w)
                result = w.result()
                assert ctr == result["_ctr"]
                output = result["output"]
                yield SGLangRequest(
                    _ctr=ctr,
                    _key=key,
                    _output=output,
                )
