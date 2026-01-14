"""
Common constants and shared fragments for prompts
"""

DEFAULT_REALM_CATALOG = """
AVAILABLE REALMS (MCP Servers):

TIER 1 - CORE:
- terminal: Shell access. Tool: execute_command.
- filesystem: File operations. Tools: read_file, write_file, list_directory.
- macos-use: Native UI automation and screenshots (Swift binary). PREFERRED for all computer interaction.
- sequential-thinking: Step-by-step reasoning for complex decisions.

TIER 2 - HIGH PRIORITY:
- puppeteer: Browser automation (navigate, click, type, screenshot).
- fetch: URL content extraction. Tool: fetch_url.
- duckduckgo-search: Web search. Tool: search.
- memory: Knowledge graph access.
- notes: Storing/Reading feedback and reports. Tools: create_note, read_note.
- vibe: **AI-POWERED DEBUGGING & SELF-HEALING** (Mistral CLI integration).
- git: Local repository operations.

TIER 3-4 - OPTIONAL:
- github: GitHub API operations.
- docker: Container management.
- slack: Team communication.
- postgres: Database access.
- whisper-stt: Speech-to-text.

CRITICAL: Do NOT invent high-level tools (e.g., 'scrape_and_extract'). Use only the real TOOLS found inside these Realms after Inspection.
"""

# Vibe MCP tools documentation for agents
VIBE_TOOLS_DOCUMENTATION = """
VIBE MCP SERVER - AI-POWERED DEBUGGING & SELF-HEALING

The 'vibe' server provides access to Mistral AI for advanced debugging, code analysis, and self-healing.
All Vibe operations run in PROGRAMMATIC CLI mode (not interactive TUI) - output is fully visible in logs.

AVAILABLE VIBE TOOLS:

1. **vibe_prompt** (PRIMARY TOOL)
   Purpose: Send any prompt to Vibe AI for analysis or action
   Args:
     - prompt: The message/query (required)
     - cwd: Working directory (optional)
     - timeout_s: Timeout in seconds (default 300)
     - output_format: 'json', 'text', or 'streaming' (default 'json')
     - auto_approve: Auto-approve tool calls (default True)
     - max_turns: Max conversation turns (default 10)
   Example: vibe_prompt(prompt="Why is this code failing?", cwd="/path/to/project")

2. **vibe_analyze_error** (SELF-HEALING)
   Purpose: Deep error analysis with optional auto-fix
   Args:
     - error_message: The error/stack trace (required)
     - log_context: Recent logs for context (optional)
     - file_path: Path to problematic file (optional)
     - auto_fix: Whether to apply fixes (default True)
   Example: vibe_analyze_error(error_message="TypeError: x is undefined", log_context="...", auto_fix=True)

3. **vibe_code_review**
   Purpose: Request AI code review for a file
   Args:
     - file_path: Path to review (required)
     - focus_areas: Areas to focus on, e.g., "security", "performance" (optional)
   Example: vibe_code_review(file_path="/src/main.py", focus_areas="security")

4. **vibe_smart_plan**
   Purpose: Generate execution plan for complex objectives
   Args:
     - objective: The goal to plan for (required)
     - context: Additional context (optional)
   Example: vibe_smart_plan(objective="Implement OAuth2 authentication")

5. **vibe_ask** (READ-ONLY)
   Purpose: Ask a quick question without file modifications
   Args:
     - question: The question (required)
   Example: vibe_ask(question="What's the best way to handle async errors in Python?")

6. **vibe_which**
   Purpose: Check Vibe CLI installation path and version
   Example: vibe_which()

WHEN TO USE VIBE:
- When Tetyana/Grisha fail after multiple attempts
- Complex debugging requiring AI reasoning
- Code review before committing
- Planning multi-step implementations
- Understanding unfamiliar code patterns

IMPORTANT: All Vibe output is logged and visible in the Electron app logs!
"""

VOICE_PROTOCOL = """
VOICE COMMUNICATION PROTOCOL (Text-To-Speech):

Your `voice_message` output is the PRIMARY way you keep the user informed.
Language: UKRAINIAN ONLY.

RULES FOR VOICE CONTEXT:
1. **Be Concise & Specific**: defined "essence" of the action.
   - BAD: "I am now executing the command to listed files." (Too verbose)
   - GOOD: "Читаю список файлів." (Action + Object)
   - GOOD: "Помилка доступу. Пробую sudo." (State + Reason + Plan)

2. **No Hardcodes**: Do not use generic phrases like "Thinking..." or "Step done". Always include context.
   - BAD: "Крок завершено."
   - GOOD: "Сервер запущено на роз'ємі 8000."

3. **Error Reporting**:
   - format: "{Failure essence}. {Reason (short)}. {Next step}."
   - Example: "Не вдалося клонувати репо. Невірний токен. Перевіряю змінні середовища."

4. **Tone**: Professional, Active, Fast-paced. Like a senior engineer reporting to a lead.
"""
