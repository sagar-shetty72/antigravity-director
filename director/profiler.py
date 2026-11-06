"""Resource & Telemetry Profiler.
Measures latency, token usage, cost, CPU, and RAM consumption.
"""
import time
import psutil
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from director.config import CONFIG

class ExecutionMetrics(BaseModel):
    mode: str  # "frontier_only", "local_only", "hybrid", "speculative_auto"
    task_id: str
    success: bool = True
    error_message: Optional[str] = None
    route: str = "direct"  # "direct", "director_plan", "speculative_direct", "speculative_escalated"
    
    # Latency
    wall_time_ms: float = 0.0
    ttft_ms: Optional[float] = None
    
    # Token usage
    frontier_input_tokens: int = 0
    frontier_output_tokens: int = 0
    local_input_tokens: int = 0
    local_output_tokens: int = 0
    
    # Derived Token & Cost
    total_tokens: int = 0
    frontier_tokens_saved: int = 0
    frontier_cost_usd: float = 0.0
    frontier_cost_saved_usd: float = 0.0
    frontier_cost_saved_pct: float = 0.0
    local_tokens_per_sec: float = 0.0
    
    # System Compute Metrics
    cpu_percent_start: float = 0.0
    cpu_percent_end: float = 0.0
    ram_used_gb_start: float = 0.0
    ram_used_gb_end: float = 0.0
    ram_wired_gb_start: float = 0.0
    ram_wired_gb_end: float = 0.0
    ram_delta_mb: float = 0.0
    
    metadata: Dict[str, Any] = Field(default_factory=dict)

class HardwareSnapshot:
    @staticmethod
    def get_snapshot() -> Dict[str, float]:
        vm = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=None)
        wired = getattr(vm, "wired", 0.0)
        return {
            "ram_total_gb": round(vm.total / (1024 ** 3), 2),
            "ram_used_gb": round(vm.used / (1024 ** 3), 2),
            "ram_available_gb": round(vm.available / (1024 ** 3), 2),
            "ram_wired_gb": round(wired / (1024 ** 3), 2),
            "ram_percent": vm.percent,
            "cpu_percent": cpu,
        }

class ProfilerContext:
    def __init__(self, mode: str, task_id: str):
        self.mode = mode
        self.task_id = task_id
        self.start_time = 0.0
        self.end_time = 0.0
        self.snap_start: Dict[str, float] = {}
        self.snap_end: Dict[str, float] = {}
        self.metrics = ExecutionMetrics(mode=mode, task_id=task_id)

    def __enter__(self):
        # Primer sample for cpu_percent
        psutil.cpu_percent(interval=None)
        self.snap_start = HardwareSnapshot.get_snapshot()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.snap_end = HardwareSnapshot.get_snapshot()
        
        self.metrics.wall_time_ms = round((self.end_time - self.start_time) * 1000, 2)
        self.metrics.cpu_percent_start = self.snap_start["cpu_percent"]
        self.metrics.cpu_percent_end = self.snap_end["cpu_percent"]
        self.metrics.ram_used_gb_start = self.snap_start["ram_used_gb"]
        self.metrics.ram_used_gb_end = self.snap_end["ram_used_gb"]
        self.metrics.ram_wired_gb_start = self.snap_start.get("ram_wired_gb", 0.0)
        self.metrics.ram_wired_gb_end = self.snap_end.get("ram_wired_gb", 0.0)
        self.metrics.ram_delta_mb = round((self.snap_end["ram_used_gb"] - self.snap_start["ram_used_gb"]) * 1024, 2)
        
        # Calculate cost
        f_in_cost = (self.metrics.frontier_input_tokens / 1_000_000) * CONFIG.cost.frontier_input_cost_per_m
        f_out_cost = (self.metrics.frontier_output_tokens / 1_000_000) * CONFIG.cost.frontier_output_cost_per_m
        self.metrics.frontier_cost_usd = round(f_in_cost + f_out_cost, 6)
        
        self.metrics.total_tokens = (
            self.metrics.frontier_input_tokens + self.metrics.frontier_output_tokens +
            self.metrics.local_input_tokens + self.metrics.local_output_tokens
        )
        
        if self.metrics.local_output_tokens > 0 and self.metrics.wall_time_ms > 0:
            self.metrics.local_tokens_per_sec = round(
                self.metrics.local_output_tokens / (self.metrics.wall_time_ms / 1000), 2
            )
            
        if exc_val is not None:
            self.metrics.success = False
            self.metrics.error_message = str(exc_val)

def compare_metrics(baseline: ExecutionMetrics, evaluated: ExecutionMetrics) -> ExecutionMetrics:
    """Computes delta metrics of an evaluated run against the baseline."""
    baseline_f_tokens = baseline.frontier_input_tokens + baseline.frontier_output_tokens
    eval_f_tokens = evaluated.frontier_input_tokens + evaluated.frontier_output_tokens
    
    tokens_saved = baseline_f_tokens - eval_f_tokens
    evaluated.frontier_tokens_saved = max(0, tokens_saved)
    
    cost_saved = baseline.frontier_cost_usd - evaluated.frontier_cost_usd
    evaluated.frontier_cost_saved_usd = round(max(0.0, cost_saved), 6)
    
    if baseline.frontier_cost_usd > 0:
        evaluated.frontier_cost_saved_pct = round(
            (evaluated.frontier_cost_saved_usd / baseline.frontier_cost_usd) * 100, 2
        )
    return evaluated
