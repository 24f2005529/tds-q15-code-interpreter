import os
import re
import sys
import traceback
from io import StringIO
from typing import List

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        exec(code)
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}

    except Exception:
        output = traceback.format_exc()
        return {"success": False, "output": output}

    finally:
        sys.stdout = old_stdout


def fallback_error_lines(tb: str) -> List[int]:
    matches = re.findall(r'File "<string>", line (\d+)', tb)

    if matches:
        return [int(matches[-1])]

    return []


def analyze_error_with_ai(code: str, tb: str) -> List[int]:
    token = os.environ.get("AIPIPE_TOKEN")

    if not token:
        return fallback_error_lines(tb)

    try:
        response = httpx.post(
            "https://aipipe.org/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            'Identify the exact line number(s) where the Python '
                            'error occurred. Return ONLY JSON like '
                            '{"error_lines":[3]}.'
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"CODE:\n{code}\n\n"
                            f"TRACEBACK:\n{tb}"
                        ),
                    },
                ],
                "response_format": {
                    "type": "json_object"
                },
            },
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        result = ErrorAnalysis.model_validate_json(content)

        if result.error_lines:
            return result.error_lines

    except Exception:
        pass

    return fallback_error_lines(tb)


@app.post("/code-interpreter")
async def code_interpreter(request: CodeRequest):
    execution = execute_python_code(request.code)

    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"],
        }

    return {
        "error": analyze_error_with_ai(
            request.code,
            execution["output"]
        ),
        "result": execution["output"],
    }
