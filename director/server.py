"""Antigravity Director Visual Web Server.
Zero-dependency asynchronous HTTP server providing real-time telemetry,
interactive prompt routing, and Pareto frontier benchmarking visualization.
"""
import os
import re
import sys
import json
import asyncio
import threading
import traceback
import subprocess
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, Optional

from director.config import CONFIG
from director.profiler import HardwareSnapshot, compare_metrics
from director.ollama_worker import OllamaWorker
from director.frontier_client import FrontierClient
from director.orchestrator import HybridDirector
from director.benchmark_tasks import BENCHMARK_TASKS, TASK_VALIDATORS
from director.runner import BenchmarkRunner

# Global event loop in dedicated background thread
_loop: Optional[asyncio.AbstractEventLoop] = None
_thread: Optional[threading.Thread] = None
_director: Optional[HybridDirector] = None
_runner: Optional[BenchmarkRunner] = None
_active_benchmark_status: Dict[str, Any] = {"running": False, "progress": 0, "total": 0, "current_task": ""}

def _get_event_loop() -> asyncio.AbstractEventLoop:
    global _loop, _thread, _director, _runner
    if _loop is None or not _loop.is_running():
        _loop = asyncio.new_event_loop()
        def run_loop(l):
            asyncio.set_event_loop(l)
            l.run_forever()
        _thread = threading.Thread(target=run_loop, args=(_loop,), daemon=True)
        _thread.start()
        _director = HybridDirector()
        _runner = BenchmarkRunner(_director)
    return _loop

def run_async(coro):
    loop = _get_event_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=180.0)

class DirectorHTTPRequestHandler(BaseHTTPRequestHandler):
    server_version = "AntigravityDirector/2.0"

    def log_message(self, format, *args):
        # Concise logging
        pass

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, default=lambda o: getattr(o, "dict", lambda: str(o))()).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, file_path: str, content_type: str = "text/html"):
        if not os.path.exists(file_path):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return
        with open(file_path, "rb") as f:
            content = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            web_dir = os.path.join(os.path.dirname(__file__), "web")
            index_path = os.path.join(web_dir, "index.html")
            return self._send_file(index_path, "text/html")

        elif path == "/api/status":
            try:
                snap = HardwareSnapshot.get_snapshot()
                _get_event_loop()
                running_models = run_async(_director.ollama.get_running_models())
                vram_total_gb = sum(m.get("size_vram_gb", 0.0) for m in running_models)
                
                resp = {
                    "hardware": snap,
                    "vram_total_gb": round(vram_total_gb, 2),
                    "running_models": running_models,
                    "frontier_model": CONFIG.frontier_model,
                    "default_local_model": CONFIG.default_local_model,
                    "benchmark_status": _active_benchmark_status,
                }
                return self._send_json(resp)
            except Exception as e:
                return self._send_json({"error": str(e)}, status=500)

        elif path == "/api/models":
            try:
                _get_event_loop()
                all_models = run_async(_director.ollama.list_models())
                running_models = run_async(_director.ollama.get_running_models())
                return self._send_json({
                    "installed_models": all_models,
                    "running_models": running_models,
                    "frontier_model": CONFIG.frontier_model,
                    "specialists": _director.specialists,
                })
            except Exception as e:
                return self._send_json({"error": str(e)}, status=500)

        elif path == "/api/tasks":
            tasks = [
                {
                    "id": t.id,
                    "category": t.category,
                    "title": t.title,
                    "prompt": t.prompt,
                }
                for t in BENCHMARK_TASKS
            ]
            return self._send_json({"tasks": tasks})

        elif path == "/api/benchmark/latest":
            # Search for latest json report
            target = "specialist_benchmark.json"
            if os.path.exists(target):
                with open(target, "r") as f:
                    data = json.load(f)
                return self._send_json(data)
            return self._send_json({"error": "No benchmark data found"}, status=404)

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Endpoint not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)
        payload = {}
        if post_data:
            try:
                payload = json.loads(post_data.decode("utf-8"))
            except Exception:
                pass

        if path == "/api/run":
            prompt = payload.get("prompt", "").strip()
            mode = payload.get("mode", "speculative_auto")
            category = payload.get("category", None)
            override_model = payload.get("model", None)
            task_id = payload.get("task_id", "interactive_task")

            if not prompt:
                return self._send_json({"error": "Prompt cannot be empty"}, status=400)

            try:
                _get_event_loop()
                # Create custom director instance if model is overridden
                director = _director
                if override_model and override_model != "auto":
                    director = HybridDirector(local_model=override_model)

                # Check if this task has a registered validator
                validator = TASK_VALIDATORS.get(task_id)

                result = run_async(
                    director.execute_task(
                        mode=mode,
                        task_id=task_id,
                        prompt=prompt,
                        category=category,
                        validator_func=validator,
                    )
                )

                # If comparing against baseline, run a lightweight estimate
                baseline_frontier = run_async(
                    director.execute_frontier_only(task_id=f"{task_id}_baseline", prompt=prompt)
                )
                result.metrics = compare_metrics(baseline_frontier.metrics, result.metrics)

                return self._send_json({
                    "task_id": result.task_id,
                    "mode": result.mode,
                    "output": result.output,
                    "verified": result.verified,
                    "fallback_triggered": result.fallback_triggered,
                    "details": result.details,
                    "metrics": result.metrics.dict(),
                    "baseline_tokens": baseline_frontier.metrics.total_tokens,
                    "baseline_cost_usd": baseline_frontier.metrics.frontier_cost_usd,
                })
            except Exception as e:
                traceback.print_exc()
                return self._send_json({"error": str(e), "trace": traceback.format_exc()}, status=500)

        elif path == "/api/benchmark":
            tasks_subset = payload.get("tasks", None)
            run_speculative = payload.get("run_speculative", True)

            try:
                _get_event_loop()
                selected_tasks = None
                if tasks_subset:
                    selected_tasks = [t for t in BENCHMARK_TASKS if t.id in tasks_subset]

                summary = run_async(_runner.run_all(tasks=selected_tasks, run_speculative=run_speculative))
                
                # Save to disk as benchmark_latest.json
                with open("specialist_benchmark.json", "w") as f:
                    json.dump(summary.dict(), f, indent=2)

                return self._send_json(summary.dict())
            except Exception as e:
                traceback.print_exc()
                return self._send_json({"error": str(e)}, status=500)

        elif path == "/api/execute":
            code = payload.get("code", "")
            clean = re.sub(r"<think>[\s\S]*?</think>", "", code).strip()
            match = re.search(r"```(?:python)?\s*([\s\S]*?)\s*```", clean)
            target_code = match.group(1).strip() if match else clean

            try:
                proc = subprocess.run(
                    [sys.executable, "-c", target_code],
                    capture_output=True,
                    text=True,
                    timeout=8.0,
                )
                return self._send_json({
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "exit_code": proc.returncode,
                })
            except subprocess.TimeoutExpired:
                return self._send_json({"stdout": "", "stderr": "Execution timed out (8s limit).", "exit_code": -1})
            except Exception as e:
                return self._send_json({"stdout": "", "stderr": str(e), "exit_code": 1})

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Endpoint not found")

def start_server(port: int = 8501, host: str = "127.0.0.1"):
    _get_event_loop()
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, DirectorHTTPRequestHandler)
    print(f"\n🚀 Antigravity Director Web Dashboard running at http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server...")
        httpd.server_close()

if __name__ == "__main__":
    start_server()
