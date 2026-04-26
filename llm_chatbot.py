import asyncio
import json
import os
import re
from urllib.parse import parse_qs, urlparse
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from google import genai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

APP_SCHEMA_CONTEXT = """
Tables and purpose:
- profiles(id, role, household_id)
- accounts(id, user_id, name, type, balance)
- categories(id, user_id, name, type)
- budgets(id, user_id, category_id, amount_limit, month, year)
- goals(id, user_id, name, target_amount, current_amount, deadline)
- transactions(id, user_id, household_id, account_id, category_id, amount, type, description, transaction_date)
- households(id, name, join_code, created_by)
- shared_expenses(id, household_id, created_by, name, amount, split_type, description)
- privacy_rules(id, household_id, role, resource_type, can_view, can_edit)
""".strip()

SYSTEM_PROMPT = (
    "You are an intelligent financial assistant integrated into a role-based "
    "household budgeting platform. Use SQL-derived context provided by the backend "
    "when available, and never invent specific balances, totals, or counts. "
    "If data is missing, say what is missing and what to do next. Respect role-based "
    "privacy and never reveal unauthorized data. "
    "The response format must be concise bullet points only, with no headings and no paragraphs."
)

PLANNER_PROMPT_TEMPLATE = """
You are a SQL planning assistant for a household budgeting app.

Database context:
{schema_context}

User id for scoping: {user_id}
User question: {user_message}

Return ONLY valid JSON with this shape:
{{
  "action": "execute_sql" | "no_sql",
  "query": "SQL query string or empty",
  "needs_write": true | false,
  "reason": "short reason"
}}

Rules:
- Prefer read queries for analytics and insights.
- Use user_id filter whenever the question is user-specific.
- Use household_id joins only when needed for shared data.
- If question is generic advice and does not need database values, use no_sql.
- Keep query compact.
- Do not include markdown or code fences.
""".strip()


def _extract_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def _safe_sql_allowed(query: str, allow_writes: bool) -> bool:
    first = query.strip().lower()
    read_prefixes = ("select", "with", "show", "explain")

    if first.startswith(read_prefixes):
        return True

    if allow_writes and (
        first.startswith("insert")
        or first.startswith("update")
        or first.startswith("delete")
    ):
        return True

    return False


def _extract_tool_text(tool_result: Any) -> str:
    content = getattr(tool_result, "content", None)
    if not content:
        return str(tool_result)

    parts = []
    for item in content:
        text = getattr(item, "text", None)
        if text:
            parts.append(text)

    if parts:
        return "\n".join(parts)

    return str(content)


def _build_stdio_server_params() -> StdioServerParameters:
    access_token = os.getenv("SUPABASE_ACCESS_TOKEN")
    if not access_token:
        raise ValueError("SUPABASE_ACCESS_TOKEN is not set in the environment.")

    command = os.getenv("SUPABASE_MCP_COMMAND", "npx")
    package_name = os.getenv(
        "SUPABASE_MCP_PACKAGE", "@supabase/mcp-server-supabase@latest"
    )
    project_ref = os.getenv("SUPABASE_MCP_PROJECT_REF") or os.getenv("SUPABASE_PROJECT_REF")
    features = os.getenv("SUPABASE_MCP_FEATURES", "database,docs")
    read_only = os.getenv("SUPABASE_MCP_READ_ONLY", "true").lower() == "true"
    extra_packages_raw = os.getenv("SUPABASE_MCP_NPX_PACKAGES", "").strip()
    extra_packages = [
        pkg.strip() for pkg in extra_packages_raw.split(",") if pkg.strip()
    ]

    args = ["-y"]
    executable = os.getenv("SUPABASE_MCP_EXECUTABLE", "mcp-server-supabase")

    if command == "npx":
        # Use npm exec-style package injection to avoid shell resolution issues.
        args.extend(["--package", package_name])
        for pkg in extra_packages:
            args.extend(["--package", pkg])
        args.extend(["--", executable])
    else:
        args.append(package_name)

    args.extend(["--access-token", access_token, "--features", features])
    if project_ref:
        args.extend(["--project-ref", project_ref])
    if read_only:
        args.append("--read-only")

    env = {
        "SUPABASE_ACCESS_TOKEN": access_token,
    }

    return StdioServerParameters(command=command, args=args, env=env)


async def _execute_sql_with_mcp(query: str) -> str:
    server_params = _build_stdio_server_params()

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool("execute_sql", arguments={"query": query})
            return _extract_tool_text(result)


def generate_gemini_response(user_message: str, user_id: Optional[str] = None) -> str:
    try:
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in the environment.")

        allow_writes = os.getenv("ALLOW_MCP_WRITES", "false").lower() == "true"
        scope_user_id = user_id or "unknown"

        client = genai.Client(api_key=api_key)

        planner_prompt = PLANNER_PROMPT_TEMPLATE.format(
            schema_context=APP_SCHEMA_CONTEXT,
            user_id=scope_user_id,
            user_message=user_message,
        )

        plan_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=planner_prompt,
            config={
                "temperature": 0.1,
                "max_output_tokens": 1000,
            },
        )

        plan = _extract_json_object(getattr(plan_response, "text", ""))

        action = str(plan.get("action", "no_sql")).strip().lower()
        query = str(plan.get("query", "")).strip()
        sql_output = "No database query executed."

        if action == "execute_sql" and query:
            if not _safe_sql_allowed(query, allow_writes=allow_writes):
                sql_output = (
                    "Skipped SQL execution for safety. Query type is blocked by current policy. "
                    "Set ALLOW_MCP_WRITES=true only in demo/dev if you need writes."
                )
            else:
                try:
                    sql_output = asyncio.run(_execute_sql_with_mcp(query))
                except Exception as mcp_error:
                    sql_output = f"MCP execute_sql failed: {str(mcp_error)}"

        final_prompt = (
            f"User question:\n{user_message}\n\n"
            f"User id context:\n{scope_user_id}\n\n"
            f"Planner action:\n{action}\n\n"
            f"SQL query used (if any):\n{query or 'N/A'}\n\n"
            f"SQL/MCP output:\n{sql_output}\n\n"
            "Answer using concise bullet points only."
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=final_prompt,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "temperature": 0.4,
                "max_output_tokens": 2000,
            },
        )

        return (response.text or "").strip()

    except Exception as e:
        raise Exception(f"Gemini/MCP error: {str(e)}")