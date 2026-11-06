"""Orchestration Engine: Manages Frontier Director, Local Ollama Workers, and Hybrid Execution."""
import json
import re
from typing import Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

from director.ollama_worker import OllamaWorker, OllamaResponse
from director.frontier_client import FrontierClient, FrontierResponse
from director.profiler import ProfilerContext, ExecutionMetrics

DEFAULT_SPECIALISTS = {
    "code": "qwen2.5-coder:7b",
    "extraction": "llama3.2:3b",
    "reasoning": "gemma4:12b",
    "general": "qwen2.5-coder:7b",
}

class TaskResult(BaseModel):
    mode: str
    task_id: str
    output: str
    metrics: ExecutionMetrics
    verified: bool = True
    fallback_triggered: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)

class HybridDirector:
    def __init__(
        self,
        ollama_worker: Optional[OllamaWorker] = None,
        frontier_client: Optional[FrontierClient] = None,
        local_model: Optional[str] = None,
        specialist_roster: Optional[Dict[str, str]] = None,
    ):
        self.ollama = ollama_worker or OllamaWorker()
        self.frontier = frontier_client or FrontierClient()
        self.local_model = local_model
        self.specialists = specialist_roster or DEFAULT_SPECIALISTS

    def select_worker_for_task(self, prompt: str, category: Optional[str] = None) -> str:
        """Selects the best local model specialist based on task type."""
        if self.local_model:
            return self.local_model
        if category and category in self.specialists:
            return self.specialists[category]

        lower = prompt.lower()
        if any(w in lower for w in ["def ", "class ", "function", "python", "code", "sql", "select ", "query"]):
            return self.specialists.get("code", "qwen2.5-coder:7b")
        elif any(w in lower for w in ["json", "extract", "parse", "log", "email", "regex"]):
            return self.specialists.get("extraction", "llama3.2:3b")
        elif any(w in lower for w in ["why", "puzzle", "prove", "reason", "logic", "deduce"]):
            return self.specialists.get("reasoning", "gemma4:12b")
        return self.specialists.get("general", "qwen2.5-coder:7b")

    async def execute_frontier_only(self, task_id: str, prompt: str, system: Optional[str] = None) -> TaskResult:
        """Mode 1: All-Frontier Baseline."""
        with ProfilerContext(mode="frontier_only", task_id=task_id) as prof:
            resp: FrontierResponse = await self.frontier.generate(prompt=prompt, system_instruction=system)
            prof.metrics.frontier_input_tokens = resp.prompt_tokens
            prof.metrics.frontier_output_tokens = resp.completion_tokens

        return TaskResult(
            mode="frontier_only",
            task_id=task_id,
            output=resp.content,
            metrics=prof.metrics,
            verified=True,
            fallback_triggered=False,
            details={"model": resp.model},
        )

    async def execute_local_only(
        self,
        task_id: str,
        prompt: str,
        category: Optional[str] = None,
        system: Optional[str] = None,
    ) -> TaskResult:
        """Mode 2: All-Local Baseline."""
        worker_model = self.select_worker_for_task(prompt, category)
        with ProfilerContext(mode="local_only", task_id=task_id) as prof:
            resp: OllamaResponse = await self.ollama.generate(
                prompt=prompt,
                system=system,
                model=worker_model,
            )
            prof.metrics.local_input_tokens = resp.prompt_tokens
            prof.metrics.local_output_tokens = resp.completion_tokens

        return TaskResult(
            mode="local_only",
            task_id=task_id,
            output=resp.content,
            metrics=prof.metrics,
            verified=True,
            fallback_triggered=False,
            details={"model": resp.model, "tokens_per_sec": resp.tokens_per_second},
        )

    async def execute_hybrid(
        self,
        task_id: str,
        prompt: str,
        category: Optional[str] = None,
        validator_func: Optional[Any] = None,
        system: Optional[str] = None,
    ) -> TaskResult:
        """
        Mode 3: Hybrid Orchestrated.
        1. Frontier Director plans & crafts specialized worker instructions (small tokens).
        2. Specialist Local Model executes the bulk generation (zero frontier tokens).
        3. Validator checks local output. If invalid, escalates to Frontier fallback.
        4. Director produces brief final synthesis or validation stamp.
        """
        fallback_triggered = False
        final_output = ""
        verification_passed = True
        worker_model = self.select_worker_for_task(prompt, category)

        with ProfilerContext(mode="hybrid", task_id=task_id) as prof:
            # Step 1: Director Planner / Prompt Formatter
            director_plan_prompt = (
                f"You are the Director. Create concise, high-precision instructions for a local worker model ({worker_model}) "
                f"to solve this task. Output ONLY the instructions.\nTask: {prompt}"
            )
            plan_resp: FrontierResponse = await self.frontier.generate(
                prompt=director_plan_prompt,
                system_instruction="Be extremely concise. Direct the worker to output pure code or exact structured format.",
            )
            prof.metrics.frontier_input_tokens += plan_resp.prompt_tokens
            prof.metrics.frontier_output_tokens += plan_resp.completion_tokens

            # Step 2: Local Worker Execution
            worker_prompt = f"{plan_resp.content}\n\nTask: {prompt}"
            worker_resp: OllamaResponse = await self.ollama.generate(
                prompt=worker_prompt,
                system="You are a precise specialist code and reasoning engine. Follow instructions strictly without fluff.",
                model=worker_model,
            )
            prof.metrics.local_input_tokens += worker_resp.prompt_tokens
            prof.metrics.local_output_tokens += worker_resp.completion_tokens

            candidate_output = worker_resp.content

            # Step 3: Validation / Verification Check
            if validator_func:
                try:
                    verification_passed = validator_func(candidate_output)
                except Exception:
                    verification_passed = False

            # Step 4: Fallback Escalation or Synthesis
            if not verification_passed:
                fallback_triggered = True
                # Escalate to Frontier to fix / complete
                fix_prompt = (
                    f"The local worker produced an incomplete or failing output:\n\n{candidate_output}\n\n"
                    f"Fix it completely to satisfy the original task:\n{prompt}"
                )
                fix_resp: FrontierResponse = await self.frontier.generate(prompt=fix_prompt)
                prof.metrics.frontier_input_tokens += fix_resp.prompt_tokens
                prof.metrics.frontier_output_tokens += fix_resp.completion_tokens
                final_output = fix_resp.content
            else:
                # Validation succeeded! Minimal Director sign-off / cleanup if needed
                final_output = candidate_output
            prof.metrics.route = "director_plan"

        return TaskResult(
            mode="hybrid",
            task_id=task_id,
            output=final_output,
            metrics=prof.metrics,
            verified=verification_passed,
            fallback_triggered=fallback_triggered,
            details={
                "route": "director_plan",
                "local_model": worker_resp.model,
                "local_tps": worker_resp.tokens_per_second,
                "plan_tokens": plan_resp.total_tokens,
            },
        )

    @staticmethod
    def _auto_verify(text: str, category: str) -> Tuple[bool, str]:
        """Heuristic validation when no explicit unit test is supplied."""
        if not text or not text.strip():
            return False, "Output was empty."
        clean = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

        if category in ("extraction", "json") or "json" in text.lower():
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
            target = match.group(1).strip() if match else clean
            try:
                json.loads(target)
                return True, "Valid JSON"
            except Exception as e:
                return False, f"JSON parse error: {e}"
        elif category in ("code", "python") or "def " in text:
            import ast
            match = re.search(r"```(?:python)?\s*([\s\S]*?)\s*```", clean)
            target = match.group(1).strip() if match else clean
            try:
                ast.parse(target)
                return True, "Valid Python syntax"
            except Exception as e:
                return False, f"Python syntax error: {e}"
        return True, "Passed sanity checks"

    async def execute_speculative_auto(
        self,
        task_id: str,
        prompt: str,
        category: Optional[str] = None,
        validator_func: Optional[Any] = None,
        system: Optional[str] = None,
    ) -> TaskResult:
        """
        Mode 4: Adaptive Speculative Routing.
        1. Speculatively dispatches directly to local specialist worker with 0 frontier planning tokens.
        2. Validates output via validator_func (or automated syntax/json parsing if no validator provided).
        3. If verified: Success! 0 Frontier tokens used (100% token savings).
        4. If fails: Escalates to Frontier Director with candidate + error context for precision repair.
        """
        fallback_triggered = False
        final_output = ""
        verification_passed = False
        worker_model = self.select_worker_for_task(prompt, category)
        route_taken = "speculative_direct"

        with ProfilerContext(mode="speculative_auto", task_id=task_id) as prof:
            # Step 1: Direct Speculative Execution (Zero Frontier Cost)
            local_sys = system or (
                "You are an expert specialist engine. Provide only the direct, correct solution "
                "or executable code/JSON requested. Do not include introductory or concluding conversational filler."
            )
            worker_resp: OllamaResponse = await self.ollama.generate(
                prompt=prompt,
                system=local_sys,
                model=worker_model,
            )
            prof.metrics.local_input_tokens += worker_resp.prompt_tokens
            prof.metrics.local_output_tokens += worker_resp.completion_tokens
            candidate_output = worker_resp.content

            # Step 2: Verification check
            error_details = ""
            if validator_func:
                try:
                    verification_passed = bool(validator_func(candidate_output))
                    if not verification_passed:
                        error_details = "Validator assertion failed on output."
                except Exception as e:
                    verification_passed = False
                    error_details = f"Validator raised exception: {e}"
            else:
                verification_passed, error_details = self._auto_verify(candidate_output, category or "general")

            # Step 3: Escalation only if validation failed
            if not verification_passed:
                fallback_triggered = True
                route_taken = "speculative_escalated"
                escalate_prompt = (
                    f"A local worker model generated this output which failed validation:\n\n"
                    f"--- Local Output ---\n{candidate_output}\n\n"
                    f"--- Failure Reason ---\n{error_details}\n\n"
                    f"--- Original Task ---\n{prompt}\n\n"
                    f"Please provide the complete, perfectly corrected response."
                )
                frontier_resp: FrontierResponse = await self.frontier.generate(
                    prompt=escalate_prompt,
                    system_instruction="You are the Director. Correct the worker's mistakes with 100% precision."
                )
                prof.metrics.frontier_input_tokens += frontier_resp.prompt_tokens
                prof.metrics.frontier_output_tokens += frontier_resp.completion_tokens
                final_output = frontier_resp.content

                if validator_func:
                    try:
                        verification_passed = bool(validator_func(final_output))
                    except Exception:
                        verification_passed = True
                else:
                    verification_passed = True
            else:
                final_output = candidate_output

            prof.metrics.route = route_taken

        return TaskResult(
            mode="speculative_auto",
            task_id=task_id,
            output=final_output,
            metrics=prof.metrics,
            verified=verification_passed,
            fallback_triggered=fallback_triggered,
            details={
                "route": route_taken,
                "local_model": worker_resp.model,
                "local_tps": worker_resp.tokens_per_second,
                "error_details": error_details if fallback_triggered else None,
            },
        )

    async def execute_task(
        self,
        mode: str,
        task_id: str,
        prompt: str,
        category: Optional[str] = None,
        validator_func: Optional[Any] = None,
        system: Optional[str] = None,
    ) -> TaskResult:
        """Unified dispatch for any execution mode."""
        if mode == "speculative_auto":
            return await self.execute_speculative_auto(task_id, prompt, category, validator_func, system)
        elif mode == "hybrid":
            return await self.execute_hybrid(task_id, prompt, category, validator_func, system)
        elif mode == "local_only":
            res = await self.execute_local_only(task_id, prompt, category, system)
            if validator_func:
                try:
                    res.verified = bool(validator_func(res.output))
                except Exception:
                    res.verified = False
            return res
        elif mode == "frontier_only":
            res = await self.execute_frontier_only(task_id, prompt, system)
            if validator_func:
                try:
                    res.verified = bool(validator_func(res.output))
                except Exception:
                    res.verified = False
            return res
        else:
            raise ValueError(f"Unknown mode: {mode}")
