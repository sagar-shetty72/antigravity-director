import os
from pydantic import BaseModel

def _load_env_file():
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k not in os.environ:
                        os.environ[k] = v.strip("'\"")

_load_env_file()

class CostConfig(BaseModel):
    # Rates per 1,000,000 tokens (Frontier model baseline)
    frontier_input_cost_per_m: float = 1.25   # $1.25 / 1M input tokens
    frontier_output_cost_per_m: float = 5.00  # $5.00 / 1M output tokens
    local_cost_per_m: float = 0.00            # Local inference cost ($0)

class DirectorConfig(BaseModel):
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    default_local_model: str = os.getenv("DEFAULT_LOCAL_MODEL", "gemma4:12b")
    frontier_model: str = os.getenv("FRONTIER_MODEL", "gemini-3.5-flash-lite")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    cost: CostConfig = CostConfig()
    timeout_seconds: float = 60.0
    max_retries: int = 2

CONFIG = DirectorConfig()
