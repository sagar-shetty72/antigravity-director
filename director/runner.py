"""Benchmark Runner: Executes comparative trials across Frontier-Only, Local-Only, and Hybrid modes."""
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from director.orchestrator import HybridDirector, TaskResult
from director.benchmark_tasks import BENCHMARK_TASKS, TASK_VALIDATORS, BenchmarkTask
from director.profiler import compare_metrics, ExecutionMetrics

class TaskComparison(BaseModel):
    task: BenchmarkTask
    frontier_only: TaskResult
    local_only: TaskResult
    hybrid: TaskResult
    speculative: Optional[TaskResult] = None
    tokens_saved: int = 0
    cost_saved_usd: float = 0.0
    cost_saved_pct: float = 0.0
    speculative_tokens_saved: int = 0
    speculative_cost_saved_usd: float = 0.0

class BenchmarkSummary(BaseModel):
    total_tasks: int = 0
    frontier_accuracy_pct: float = 0.0
    local_accuracy_pct: float = 0.0
    hybrid_accuracy_pct: float = 0.0
    speculative_accuracy_pct: float = 0.0
    total_frontier_tokens_baseline: int = 0
    total_frontier_tokens_hybrid: int = 0
    total_frontier_tokens_speculative: int = 0
    total_tokens_saved: int = 0
    overall_tokens_saved_pct: float = 0.0
    total_tokens_saved_speculative: int = 0
    overall_speculative_saved_pct: float = 0.0
    total_cost_baseline_usd: float = 0.0
    total_cost_hybrid_usd: float = 0.0
    total_cost_speculative_usd: float = 0.0
    total_cost_saved_usd: float = 0.0
    total_cost_saved_speculative_usd: float = 0.0
    comparisons: List[TaskComparison] = Field(default_factory=list)

class BenchmarkRunner:
    def __init__(self, director: Optional[HybridDirector] = None):
        self.director = director or HybridDirector()

    async def run_task(self, task: BenchmarkTask, run_speculative: bool = True) -> TaskComparison:
        validator = TASK_VALIDATORS.get(task.id)

        # 1. Frontier-Only
        res_frontier = await self.director.execute_frontier_only(
            task_id=task.id,
            prompt=task.prompt,
        )
        if validator:
            res_frontier.verified = validator(res_frontier.output)

        # 2. Local-Only
        res_local = await self.director.execute_local_only(
            task_id=task.id,
            prompt=task.prompt,
            category=task.category,
        )
        if validator:
            res_local.verified = validator(res_local.output)

        # 3. Hybrid Orchestration
        res_hybrid = await self.director.execute_hybrid(
            task_id=task.id,
            prompt=task.prompt,
            category=task.category,
            validator_func=validator,
        )
        if validator and not res_hybrid.fallback_triggered:
            res_hybrid.verified = validator(res_hybrid.output)

        # Calculate comparative metrics for Hybrid
        res_hybrid.metrics = compare_metrics(res_frontier.metrics, res_hybrid.metrics)

        # 4. Speculative Auto (Zero-Cost First)
        res_spec = None
        spec_tokens_saved = 0
        spec_cost_saved = 0.0
        if run_speculative:
            res_spec = await self.director.execute_speculative_auto(
                task_id=task.id,
                prompt=task.prompt,
                category=task.category,
                validator_func=validator,
            )
            res_spec.metrics = compare_metrics(res_frontier.metrics, res_spec.metrics)
            spec_tokens_saved = res_spec.metrics.frontier_tokens_saved
            spec_cost_saved = res_spec.metrics.frontier_cost_saved_usd

        return TaskComparison(
            task=task,
            frontier_only=res_frontier,
            local_only=res_local,
            hybrid=res_hybrid,
            speculative=res_spec,
            tokens_saved=res_hybrid.metrics.frontier_tokens_saved,
            cost_saved_usd=res_hybrid.metrics.frontier_cost_saved_usd,
            cost_saved_pct=res_hybrid.metrics.frontier_cost_saved_pct,
            speculative_tokens_saved=spec_tokens_saved,
            speculative_cost_saved_usd=spec_cost_saved,
        )

    async def run_all(
        self,
        tasks: Optional[List[BenchmarkTask]] = None,
        run_speculative: bool = True,
    ) -> BenchmarkSummary:
        selected_tasks = tasks or BENCHMARK_TASKS
        comparisons: List[TaskComparison] = []

        for task in selected_tasks:
            comp = await self.run_task(task, run_speculative=run_speculative)
            comparisons.append(comp)

        summary = BenchmarkSummary(total_tasks=len(comparisons), comparisons=comparisons)
        if not comparisons:
            return summary

        f_correct = sum(1 for c in comparisons if c.frontier_only.verified)
        l_correct = sum(1 for c in comparisons if c.local_only.verified)
        h_correct = sum(1 for c in comparisons if c.hybrid.verified)

        summary.frontier_accuracy_pct = round((f_correct / len(comparisons)) * 100, 1)
        summary.local_accuracy_pct = round((l_correct / len(comparisons)) * 100, 1)
        summary.hybrid_accuracy_pct = round((h_correct / len(comparisons)) * 100, 1)

        summary.total_frontier_tokens_baseline = sum(
            c.frontier_only.metrics.frontier_input_tokens + c.frontier_only.metrics.frontier_output_tokens
            for c in comparisons
        )
        summary.total_frontier_tokens_hybrid = sum(
            c.hybrid.metrics.frontier_input_tokens + c.hybrid.metrics.frontier_output_tokens
            for c in comparisons
        )
        summary.total_tokens_saved = sum(c.tokens_saved for c in comparisons)

        if summary.total_frontier_tokens_baseline > 0:
            summary.overall_tokens_saved_pct = round(
                (summary.total_tokens_saved / summary.total_frontier_tokens_baseline) * 100, 2
            )

        summary.total_cost_baseline_usd = round(
            sum(c.frontier_only.metrics.frontier_cost_usd for c in comparisons), 6
        )
        summary.total_cost_hybrid_usd = round(
            sum(c.hybrid.metrics.frontier_cost_usd for c in comparisons), 6
        )
        summary.total_cost_saved_usd = round(
            sum(c.cost_saved_usd for c in comparisons), 6
        )

        if run_speculative:
            s_correct = sum(1 for c in comparisons if c.speculative and c.speculative.verified)
            summary.speculative_accuracy_pct = round((s_correct / len(comparisons)) * 100, 1)
            summary.total_frontier_tokens_speculative = sum(
                (c.speculative.metrics.frontier_input_tokens + c.speculative.metrics.frontier_output_tokens)
                if c.speculative else 0
                for c in comparisons
            )
            summary.total_tokens_saved_speculative = sum(c.speculative_tokens_saved for c in comparisons)
            if summary.total_frontier_tokens_baseline > 0:
                summary.overall_speculative_saved_pct = round(
                    (summary.total_tokens_saved_speculative / summary.total_frontier_tokens_baseline) * 100, 2
                )
            summary.total_cost_speculative_usd = round(
                sum((c.speculative.metrics.frontier_cost_usd if c.speculative else 0.0) for c in comparisons), 6
            )
            summary.total_cost_saved_speculative_usd = round(
                sum(c.speculative_cost_saved_usd for c in comparisons), 6
            )

        return summary
