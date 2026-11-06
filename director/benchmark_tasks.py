"""Benchmark Task Suite for Hybrid Model Orchestration.
Includes real validation logic (unit tests, JSON validation, regex).
"""
import json
import re
from typing import Callable, Dict, Any, List
from pydantic import BaseModel

class BenchmarkTask(BaseModel):
    id: str
    category: str
    title: str
    prompt: str

def extract_code_block(text: str) -> str:
    """Extracts python code from markdown block or returns raw text, stripping think tags."""
    clean = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    match = re.search(r"```(?:python)?\s*([\s\S]*?)\s*```", clean)
    if match:
        return match.group(1).strip()
    return clean.strip()

def validate_palindrome(text: str) -> bool:
    """Validates python code for is_palindrome."""
    code = extract_code_block(text)
    scope: Dict[str, Any] = {}
    try:
        exec(code, scope, scope)
        fn = scope.get("is_palindrome")
        if not callable(fn):
            return False
        assert fn("A man, a plan, a canal: Panama") is True
        assert fn("race a car") is False
        assert fn("Was it a car or a cat I saw?") is True
        assert fn("") is True
        return True
    except Exception:
        return False

def validate_lru_cache(text: str) -> bool:
    """Validates LRUCache implementation."""
    code = extract_code_block(text)
    scope: Dict[str, Any] = {}
    try:
        exec(code, scope, scope)
        cls = scope.get("LRUCache")
        if not cls:
            return False
        cache = cls(2)
        cache.put(1, 1)
        cache.put(2, 2)
        assert cache.get(1) == 1
        cache.put(3, 3)  # evicts key 2
        assert cache.get(2) == -1
        cache.put(4, 4)  # evicts key 1
        assert cache.get(1) == -1
        assert cache.get(3) == 3
        assert cache.get(4) == 4
        return True
    except Exception:
        return False

def validate_flatten_dict(text: str) -> bool:
    """Validates flatten_dict implementation."""
    code = extract_code_block(text)
    scope: Dict[str, Any] = {}
    try:
        exec(code, scope, scope)
        fn = scope.get("flatten_dict")
        if not callable(fn):
            return False
        nested = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
        expected = {"a": 1, "b.c": 2, "b.d.e": 3}
        assert fn(nested) == expected
        return True
    except Exception:
        return False

def validate_contact_extraction(text: str) -> bool:
    """Validates structured JSON output."""
    clean = text.strip()
    # Strip markdown if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
    if match:
        clean = match.group(1).strip()
    try:
        data = json.loads(clean)
        return (
            data.get("name") == "Sarah Connor" and
            data.get("email") == "sarah@skynet-defense.org" and
            data.get("role") == "Lead Architect"
        )
    except Exception:
        return False

def validate_sql_query(text: str) -> bool:
    """Validates SQL query structure and clauses."""
    clean = extract_code_block(text).upper()
    return (
        "SELECT" in clean and
        "FROM USERS" in clean and
        "GROUP BY" in clean and
        "COUNT(" in clean and
        ("ORDER BY" in clean or "HAVING" in clean)
    )

def validate_log_parsing(text: str) -> bool:
    """Validates log parsing extraction output."""
    clean = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
    if match:
        clean = match.group(1).strip()
    try:
        data = json.loads(clean)
        return (
            data.get("status_code") == 500 and
            data.get("ip") == "192.168.1.105" and
            data.get("method") == "POST"
        )
    except Exception:
        return False

def validate_logic_puzzle(text: str) -> bool:
    """Validates logic deduction puzzle answer."""
    lower = text.lower()
    return ("tuesday" in lower or "wednesday" in lower) and "alice" in lower

# Registry of benchmark tasks
BENCHMARK_TASKS: List[BenchmarkTask] = [
    BenchmarkTask(
        id="task_palindrome",
        category="code",
        title="Palindrome Function",
        prompt=(
            "Write a Python function `def is_palindrome(s: str) -> bool:` that returns True if the string is a palindrome, "
            "ignoring case, spaces, and punctuation. Return ONLY executable python code."
        ),
    ),
    BenchmarkTask(
        id="task_lru",
        category="code",
        title="LRU Cache Implementation",
        prompt=(
            "Implement a class `LRUCache(capacity: int)` with methods `get(key: int) -> int` and `put(key: int, value: int) -> None`. "
            "Return -1 if key does not exist. Evict least recently used when full. Return ONLY executable python code."
        ),
    ),
    BenchmarkTask(
        id="task_flatten",
        category="code",
        title="Flatten Nested Dict",
        prompt=(
            "Write a Python function `def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:` that flattens "
            "a nested dictionary using dot notation for nested keys. Return ONLY executable python code."
        ),
    ),
    BenchmarkTask(
        id="task_extract_json",
        category="extraction",
        title="Structured JSON Extraction",
        prompt=(
            "Extract the contact details from this text and return a valid JSON object with keys 'name', 'email', and 'role'. "
            "Text: 'Hello team, please welcome Sarah Connor as our new Lead Architect. You can reach her at sarah@skynet-defense.org.'\n"
            "Output ONLY valid JSON."
        ),
    ),
    BenchmarkTask(
        id="task_sql_query",
        category="code",
        title="Analytical SQL Query",
        prompt=(
            "Write an SQL query on table 'users' with columns (id, country, signup_date, is_active). "
            "Find the total count of active users per country who signed up in 2026, ordering from highest to lowest. "
            "Return ONLY the SQL query."
        ),
    ),
    BenchmarkTask(
        id="task_parse_log",
        category="extraction",
        title="Server Log Parsing",
        prompt=(
            "Parse this server log line into a JSON object with keys 'ip', 'method', 'path', and 'status_code' (as integer):\n"
            "'2026-11-06 10:15:22 192.168.1.105 POST /api/v1/checkout HTTP/1.1 500 Internal Server Error'\n"
            "Output ONLY valid JSON."
        ),
    ),
    BenchmarkTask(
        id="task_retry_decorator",
        category="code",
        title="Python Retry Decorator",
        prompt=(
            "Write a Python decorator `retry(max_attempts=3, delay=0.01)` that retries a decorated function "
            "if it raises an Exception, up to max_attempts times, before re-raising. Return ONLY executable python code."
        ),
    ),
    BenchmarkTask(
        id="task_schema_checker",
        category="code",
        title="User Record Validator",
        prompt=(
            "Write a Python function `def validate_user(user: dict) -> bool:` that returns True if user has: "
            "1. 'username' (non-empty string)\n"
            "2. 'age' (integer >= 18)\n"
            "3. 'email' (string containing '@')\n"
            "4. 'roles' (non-empty list of strings)\n"
            "Return False otherwise. Return ONLY executable python code."
        ),
    ),
    BenchmarkTask(
        id="task_sql_window",
        category="code",
        title="SQL Window Function",
        prompt=(
            "Write an SQL query for table 'employees' (id, name, department_id, salary). "
            "Rank employees within each department by salary descending, producing columns "
            "(name, department_id, salary, dept_rank) using the ROW_NUMBER() window function. "
            "Return ONLY the SQL query."
        ),
    ),
]

def validate_retry_decorator(text: str) -> bool:
    """Validates retry decorator functionality."""
    code = extract_code_block(text)
    scope: Dict[str, Any] = {}
    try:
        exec(code, scope, scope)
        retry_fn = scope.get("retry")
        if not callable(retry_fn):
            return False
        
        attempts = 0
        @retry_fn(max_attempts=3, delay=0.001)
        def flaky_success():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ValueError("temporary error")
            return "success"
        
        assert flaky_success() == "success"
        assert attempts == 3

        fail_attempts = 0
        @retry_fn(max_attempts=2, delay=0.001)
        def always_fail():
            nonlocal fail_attempts
            fail_attempts += 1
            raise RuntimeError("permanent error")
        
        try:
            always_fail()
            return False
        except RuntimeError:
            assert fail_attempts == 2
            return True
    except Exception:
        return False

def validate_schema_checker(text: str) -> bool:
    """Validates user record validator."""
    code = extract_code_block(text)
    scope: Dict[str, Any] = {}
    try:
        exec(code, scope, scope)
        fn = scope.get("validate_user")
        if not callable(fn):
            return False
        valid = {"username": "neo", "age": 28, "email": "neo@matrix.org", "roles": ["admin"]}
        invalid_age = {"username": "neo", "age": 16, "email": "neo@matrix.org", "roles": ["admin"]}
        invalid_email = {"username": "neo", "age": 20, "email": "invalid_email", "roles": ["user"]}
        invalid_roles = {"username": "neo", "age": 20, "email": "neo@test.com", "roles": []}
        assert fn(valid) is True
        assert fn(invalid_age) is False
        assert fn(invalid_email) is False
        assert fn(invalid_roles) is False
        return True
    except Exception:
        return False

def validate_sql_window(text: str) -> bool:
    """Validates SQL Window ranking query."""
    clean = extract_code_block(text).upper()
    return (
        "SELECT" in clean and
        "FROM EMPLOYEES" in clean and
        "ROW_NUMBER()" in clean and
        "OVER" in clean and
        "PARTITION BY" in clean and
        "DEPARTMENT_ID" in clean and
        "ORDER BY" in clean and
        "SALARY" in clean
    )

TASK_VALIDATORS: Dict[str, Callable[[str], bool]] = {
    "task_palindrome": validate_palindrome,
    "task_lru": validate_lru_cache,
    "task_flatten": validate_flatten_dict,
    "task_extract_json": validate_contact_extraction,
    "task_sql_query": validate_sql_query,
    "task_parse_log": validate_log_parsing,
    "task_retry_decorator": validate_retry_decorator,
    "task_schema_checker": validate_schema_checker,
    "task_sql_window": validate_sql_window,
}
