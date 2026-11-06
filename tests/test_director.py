"""Unit tests for the Antigravity Director framework."""
import pytest
import asyncio
from director.profiler import ProfilerContext, compare_metrics, ExecutionMetrics
from director.benchmark_tasks import (
    validate_palindrome,
    validate_lru_cache,
    validate_flatten_dict,
    validate_contact_extraction,
    validate_sql_query,
    validate_log_parsing,
    validate_retry_decorator,
    validate_schema_checker,
    validate_sql_window,
)
from director.orchestrator import HybridDirector
from director.frontier_client import FrontierClient

def test_palindrome_validator():
    good_code = (
        "def is_palindrome(s: str) -> bool:\n"
        "    clean = [c.lower() for c in s if c.isalnum()]\n"
        "    return clean == clean[::-1]\n"
    )
    bad_code = "def is_palindrome(s: str) -> bool:\n    return True\n"
    assert validate_palindrome(good_code) is True
    assert validate_palindrome(bad_code) is False

def test_lru_cache_validator():
    good_code = (
        "class LRUCache:\n"
        "    def __init__(self, capacity: int):\n"
        "        self.capacity = capacity\n"
        "        self.cache = {}\n"
        "    def get(self, key: int) -> int:\n"
        "        if key not in self.cache:\n"
        "            return -1\n"
        "        val = self.cache.pop(key)\n"
        "        self.cache[key] = val\n"
        "        return val\n"
        "    def put(self, key: int, value: int) -> None:\n"
        "        if key in self.cache:\n"
        "            self.cache.pop(key)\n"
        "        elif len(self.cache) >= self.capacity:\n"
        "            first = next(iter(self.cache))\n"
        "            del self.cache[first]\n"
        "        self.cache[key] = value\n"
    )
    bad_code = "class LRUCache:\n    pass\n"
    assert validate_lru_cache(good_code) is True
    assert validate_lru_cache(bad_code) is False

def test_flatten_dict_validator():
    good_code = (
        "def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:\n"
        "    items = []\n"
        "    for k, v in d.items():\n"
        "        new_key = f'{parent_key}{sep}{k}' if parent_key else k\n"
        "        if isinstance(v, dict):\n"
        "            items.extend(flatten_dict(v, new_key, sep=sep).items())\n"
        "        else:\n"
        "            items.append((new_key, v))\n"
        "    return dict(items)\n"
    )
    assert validate_flatten_dict(good_code) is True

def test_sql_query_validator():
    good_sql = "SELECT country, COUNT(id) FROM users WHERE is_active = 1 GROUP BY country ORDER BY COUNT(id) DESC;"
    bad_sql = "SELECT * FROM users;"
    assert validate_sql_query(good_sql) is True
    assert validate_sql_query(bad_sql) is False

def test_log_parsing_validator():
    good_json = '{"ip": "192.168.1.105", "method": "POST", "path": "/api/v1/checkout", "status_code": 500}'
    bad_json = '{"ip": "192.168.1.105", "status_code": 200}'
    assert validate_log_parsing(good_json) is True
    assert validate_log_parsing(bad_json) is False

def test_profiler_and_compare_metrics():
    baseline = ExecutionMetrics(
        mode="frontier_only",
        task_id="test_1",
        frontier_input_tokens=1000,
        frontier_output_tokens=500,
        frontier_cost_usd=0.00375,
    )
    hybrid = ExecutionMetrics(
        mode="hybrid",
        task_id="test_1",
        frontier_input_tokens=100,
        frontier_output_tokens=50,
        frontier_cost_usd=0.000375,
    )
    res = compare_metrics(baseline, hybrid)
    assert res.frontier_tokens_saved == 1350
    assert res.frontier_cost_saved_usd > 0
    assert res.frontier_cost_saved_pct == 90.0

def test_retry_decorator_validator():
    good_code = (
        "import time\n"
        "def retry(max_attempts=3, delay=0.01):\n"
        "    def decorator(fn):\n"
        "        def wrapper(*args, **kwargs):\n"
        "            last_err = None\n"
        "            for attempt in range(max_attempts):\n"
        "                try:\n"
        "                    return fn(*args, **kwargs)\n"
        "                except Exception as e:\n"
        "                    last_err = e\n"
        "                    time.sleep(delay)\n"
        "            raise last_err\n"
        "        return wrapper\n"
        "    return decorator\n"
    )
    bad_code = "def retry(max_attempts=3, delay=0.01):\n    def dec(fn): return fn\n    return dec\n"
    assert validate_retry_decorator(good_code) is True
    assert validate_retry_decorator(bad_code) is False

def test_schema_checker_validator():
    good_code = (
        "def validate_user(user: dict) -> bool:\n"
        "    if not isinstance(user, dict): return False\n"
        "    if not user.get('username') or not isinstance(user['username'], str): return False\n"
        "    if not isinstance(user.get('age'), int) or user['age'] < 18: return False\n"
        "    if not isinstance(user.get('email'), str) or '@' not in user['email']: return False\n"
        "    if not isinstance(user.get('roles'), list) or len(user['roles']) == 0: return False\n"
        "    return True\n"
    )
    bad_code = "def validate_user(user: dict) -> bool:\n    return True\n"
    assert validate_schema_checker(good_code) is True
    assert validate_schema_checker(bad_code) is False

def test_sql_window_validator():
    good_sql = "SELECT name, department_id, salary, ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary DESC) as dept_rank FROM employees;"
    bad_sql = "SELECT * FROM employees ORDER BY salary DESC;"
    assert validate_sql_window(good_sql) is True
    assert validate_sql_window(bad_sql) is False

def test_auto_verify():
    valid_py, _ = HybridDirector._auto_verify("```python\ndef hello():\n    return 42\n```", "code")
    assert valid_py is True
    invalid_py, _ = HybridDirector._auto_verify("```python\ndef hello(:\n```", "code")
    assert invalid_py is False

    valid_json, _ = HybridDirector._auto_verify('{"status": "ok", "code": 200}', "extraction")
    assert valid_json is True
    invalid_json, _ = HybridDirector._auto_verify('{"status": "ok", ', "extraction")
    assert invalid_json is False

@pytest.mark.asyncio
async def test_simulated_frontier_client():
    client = FrontierClient(api_key="")
    resp = await client.generate("Write a palindrome function in python")
    assert resp.total_tokens > 0
    assert "is_palindrome" in resp.content
