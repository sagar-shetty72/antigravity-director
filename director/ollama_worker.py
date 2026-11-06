"""Ollama Worker Client.
Handles async interaction with local models, extracts native Ollama token & latency metrics.
"""
import httpx
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from director.config import CONFIG

class OllamaResponse(BaseModel):
    model: str
    content: str
    thinking: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_duration_ms: float = 0.0
    load_duration_ms: float = 0.0
    eval_duration_ms: float = 0.0
    tokens_per_second: float = 0.0
    raw: Dict[str, Any] = Field(default_factory=dict)

class OllamaWorker:
    def __init__(self, base_url: str = CONFIG.ollama_base_url):
        self.base_url = base_url.rstrip("/")

    async def list_models(self) -> List[str]:
        """Fetch list of models available in Ollama."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]

    async def get_running_models(self) -> List[Dict[str, Any]]:
        """Fetch models currently loaded into memory/VRAM from Ollama /api/ps."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/ps")
                if resp.status_code == 200:
                    data = resp.json()
                    models = []
                    for m in data.get("models", []):
                        vram_bytes = m.get("size_vram", 0)
                        models.append({
                            "name": m.get("name", ""),
                            "model": m.get("model", ""),
                            "size_vram_gb": round(vram_bytes / (1024 ** 3), 2),
                            "parameter_size": m.get("details", {}).get("parameter_size", "unknown"),
                            "quantization": m.get("details", {}).get("quantization_level", "unknown"),
                            "expires_at": m.get("expires_at", ""),
                        })
                    return models
        except Exception:
            pass
        return []

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        json_format: bool = False,
        temperature: float = 0.2,
        timeout: float = CONFIG.timeout_seconds,
    ) -> OllamaResponse:
        """Call Ollama /api/generate and parse content and token stats."""
        model_name = model or CONFIG.default_local_model
        payload: Dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        if system:
            payload["system"] = system
        if json_format:
            payload["format"] = "json"

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()

        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        eval_dur_ns = data.get("eval_duration", 0)
        eval_dur_ms = eval_dur_ns / 1_000_000.0
        load_dur_ms = data.get("load_duration", 0) / 1_000_000.0
        total_dur_ms = data.get("total_duration", 0) / 1_000_000.0
        
        tps = 0.0
        if eval_dur_ms > 0 and completion_tokens > 0:
            tps = round(completion_tokens / (eval_dur_ms / 1000.0), 2)

        return OllamaResponse(
            model=model_name,
            content=data.get("response", "").strip(),
            thinking=data.get("thinking", None),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_duration_ms=round(total_dur_ms, 2),
            load_duration_ms=round(load_dur_ms, 2),
            eval_duration_ms=round(eval_dur_ms, 2),
            tokens_per_second=tps,
            raw=data,
        )
