# Vibe MCP Server - Reengineered Architecture

## 📋 Overview

The Vibe MCP Server has been completely reengineered to follow proper MCP (Model Context Protocol) patterns and architectural best practices. This document describes the new design, key improvements, and usage patterns.

**Version**: 2.0 (Reengineered)  
**Date**: 2026-01-18  
**Status**: Production Ready

---

## 🎯 Key Improvements

### 1. **Proper FastMCP Integration**
- ✅ Correct async/await patterns throughout
- ✅ Proper Context object handling
- ✅ Streaming output with real-time notifications
- ✅ Clean error handling and recovery

### 2. **Simplified Architecture**
- **Removed**: Complex PTY wrapper (`vibe_runner.py`) - now using standard async subprocess
- **Removed**: Redundant logging layers - consolidated to single logger
- **Improved**: Direct subprocess communication with proper stream reading
- **Kept**: All critical functionality from original design

### 3. **Better Error Handling**
- Explicit timeout handling with graceful shutdown
- Proper resource cleanup (temporary files, processes)
- Comprehensive error messages with context
- Non-blocking operations throughout

### 4. **Production-Grade Logging**
- Structured logging to file and stderr
- Debug level logging for troubleshooting
- Real-time streaming of important events
- Session tracking and metrics

### 5. **Configuration-Driven**
- Loads Vibe binary path from config
- Configurable workspace and timeout
- Environment variable support
- Backward compatible with existing setup

---

## 🔧 Architecture Components

### Core Functions

#### `run_vibe_subprocess()`
**Purpose**: Low-level subprocess execution  
**Responsibilities**:
- Launch Vibe CLI with proper environment
- Read stdout/stderr concurrently
- Handle timeouts gracefully
- Strip ANSI codes and truncate output
- Return structured result

**Key Features**:
```python
- Async/await throughout
- Timeout protection (timeout_s + 10s buffer)
- Streaming output collection
- Proper error handling
- Resource cleanup
```

#### `handle_long_prompt()`
**Purpose**: Handle prompts that exceed CLI argument limits  
**Mechanism**:
- Stores prompts > 2000 chars in temporary file
- Returns reference to file + path to cleanup
- Automatic cleanup after execution

**Returns**: `(final_prompt, file_path_to_cleanup)`

#### `resolve_vibe_binary()`
**Purpose**: Locate Vibe CLI executable  
**Search Order**:
1. `~/.local/bin/vibe` (common location)
2. Absolute path from config
3. PATH environment variable
4. Returns None if not found

#### `prepare_workspace_and_instructions()`
**Purpose**: Ensure workspace directories exist  
**Creates**:
- Vibe workspace directory
- Instructions subdirectory
- Logs directory (handled by root config)

---

## 📦 MCP Tools (12 Total)

### 1. **vibe_which()**
Locate and report Vibe CLI version.

```python
# Returns
{
    "success": True,
    "binary": "/path/to/vibe",
    "version": "0.x.x"
}
```

### 2. **vibe_prompt()** ⭐ PRIMARY TOOL
Send a prompt to Vibe AI in programmatic mode.

**Parameters**:
- `prompt`: Message for Vibe AI
- `cwd`: Working directory
- `timeout_s`: Timeout in seconds (default: 300)
- `auto_approve`: Auto-approve tool calls
- `max_turns`: Conversation turn limit
- `max_price`: Cost limit in dollars

**Returns**:
```python
{
    "success": True,
    "stdout": "command output",
    "stderr": "error output",
    "returncode": 0,
    "parsed_response": {...}  # if JSON
}
```

**Example**:
```python
result = await vibe_prompt(
    ctx=ctx,
    prompt="Analyze the error in main.py and fix it",
    cwd="/path/to/project",
    timeout_s=120,
    auto_approve=True
)
```

### 3. **vibe_analyze_error()**
Deep error analysis with optional auto-fix.

**Parameters**:
- `error_message`: Error message or stack trace
- `file_path`: Path to file with error
- `log_context`: Recent logs for context
- `recovery_history`: Past recovery attempts
- `auto_fix`: Automatically apply fixes (default: True)

**Use Case**: System encounters an error it cannot handle internally

### 4. **vibe_implement_feature()**
Deep coding: Implement a complex feature.

**Parameters**:
- `goal`: High-level objective
- `context_files`: Relevant file paths
- `constraints`: Technical constraints
- `timeout_s`: Extended timeout (default: 1200s)

**Use Case**: Architecture-scale feature development

### 5. **vibe_code_review()**
Request code review for a file.

**Parameters**:
- `file_path`: File to review
- `focus_areas`: "security", "performance", etc.
- `cwd`: Working directory

**Returns**: Code review analysis with suggestions

### 6. **vibe_smart_plan()**
Generate structured execution plan.

**Parameters**:
- `objective`: Goal to plan for
- `context`: Additional context

**Returns**: Step-by-step plan with verification criteria

### 7. **vibe_ask()**
Quick question without file modifications.

**Parameters**:
- `question`: Question for Vibe
- `cwd`: Working directory

**Returns**: AI response (read-only mode)

### 8. **vibe_execute_subcommand()**
Execute utility Vibe subcommands.

**Allowed Subcommands**:
```
list-editors, list-modules, run, enable, disable,
install, smart-plan, ask, agent-reset, agent-on,
agent-off, vibe-status, vibe-continue, vibe-cancel,
vibe-help, eternal-engine, screenshots
```

**Blocked** (use `vibe_prompt` instead):
```
tui, agent-chat, self-healing-status, self-healing-scan
```

### 9. **vibe_list_sessions()**
List recent Vibe session logs.

**Parameters**:
- `limit`: Number of sessions to return (default: 10)

**Returns**: Session list with metadata and metrics

### 10. **vibe_session_details()**
Get full details of a specific session.

**Parameters**:
- `session_id_or_file`: Session ID or filename

**Returns**: Complete session data with history

### 11. **vibe_check_db()**
Execute read-only SQL queries against the database.

**Schema**:
- `sessions`: id, started_at, ended_at
- `tasks`: id, session_id, goal, status, created_at
- `task_steps`: id, task_id, action, tool, status, error_message
- `tool_executions`: id, step_id, server_name, tool_name, arguments, result
- `logs`: timestamp, level, source, message

**Restrictions**: SELECT queries only (no mutations)

### 12. **vibe_get_system_context()**
Retrieve current operational context.

**Returns**:
```python
{
    "current_session_id": "...",
    "recent_tasks": [...],
    "recent_errors": [...],
    "system_root": "...",
    "project_root": "..."
}
```

---

## 🔄 Execution Flow

### Typical Vibe Prompt Execution

```
1. Input Validation
   ├─ Check Vibe binary exists
   ├─ Ensure workspace directory exists
   └─ Handle large prompts (> 2000 chars)

2. Command Construction
   ├─ Build argv: ["vibe", "-p", prompt, "--output", "streaming"]
   ├─ Add optional flags (auto-approve, max-turns, etc.)
   └─ Log command for debugging

3. Subprocess Execution
   ├─ Launch Vibe with proper environment
   ├─ Set TERM=dumb, PAGER=cat (disable interactive)
   ├─ Disable colors and buffering
   └─ Prepare environment vars

4. Stream Processing
   ├─ Read stdout and stderr concurrently
   ├─ Strip ANSI escape codes
   ├─ Parse JSON for structured logging
   ├─ Log important events in real-time
   └─ Handle timeout gracefully

5. Response Parsing
   ├─ Truncate output if > 500KB
   ├─ Attempt to parse as JSON
   ├─ Extract from streaming format if needed
   └─ Return structured result

6. Cleanup
   ├─ Remove temporary prompt file
   ├─ Ensure process terminated
   └─ Log completion
```

---

## 🚀 Usage Examples

### Example 1: Simple Prompt
```python
result = await vibe_prompt(
    ctx=ctx,
    prompt="What is the current date?",
    timeout_s=30
)

if result["success"]:
    print(result["parsed_response"])
```

### Example 2: Error Analysis and Fix
```python
result = await vibe_analyze_error(
    ctx=ctx,
    error_message="TypeError: cannot unpack non-iterable NoneType",
    file_path="/path/to/module.py",
    auto_fix=True,
    timeout_s=180
)
```

### Example 3: Feature Implementation
```python
result = await vibe_implement_feature(
    ctx=ctx,
    goal="Add user authentication with JWT tokens",
    context_files=[
        "/path/to/main.py",
        "/path/to/models.py",
        "/path/to/config.yaml"
    ],
    timeout_s=1800  # 30 minutes for complex work
)
```

### Example 4: Database Query for Debugging
```python
result = await vibe_check_db(
    ctx=ctx,
    query="SELECT * FROM logs WHERE level='ERROR' ORDER BY timestamp DESC LIMIT 5"
)

for error in result["data"]:
    print(f"{error['timestamp']}: {error['message']}")
```

---

## 📊 Configuration

### config.yaml Integration

```yaml
mcp:
  vibe:
    binary: "vibe"  # or /path/to/vibe
    timeout_s: 300
    max_output_chars: 500000
    workspace: "${CONFIG_ROOT}/vibe_workspace"
```

### Environment Variables

```bash
# Override defaults
export VIBE_BINARY="/custom/path/to/vibe"
export VIBE_WORKSPACE="/custom/workspace"
export DATABASE_URL="postgresql://..."
```

---

## 🛡️ Error Handling

### Common Errors and Solutions

#### 1. **Vibe binary not found**
```
Error: Vibe CLI not found on PATH
```
**Solution**:
- Install Vibe: `pip install vibe-cli`
- Or set config: `mcp.vibe.binary: /path/to/vibe`

#### 2. **Timeout**
```
Error: Vibe execution timed out after 300s
```
**Solution**:
- Increase timeout: `timeout_s=600`
- Break task into smaller steps
- Check Vibe is not stuck

#### 3. **Database connection failed**
```
Error: Failed to get system context
```
**Solution**:
- Check DATABASE_URL environment variable
- Verify PostgreSQL is running
- Check network connectivity

#### 4. **Large output truncated**
```
[TRUNCATED: Output exceeded 500000 chars]
```
**Solution**:
- Break task into smaller steps
- Request summary instead of full output
- Use streaming output format

---

## 📈 Performance Considerations

### Timeouts
- **Default**: 300s (5 minutes)
- **Quick queries**: 30-60s
- **Deep analysis**: 600s+ (10+ minutes)
- **Feature implementation**: 1200s+ (20+ minutes)

### Resource Limits
- **Max output**: 500KB (configurable)
- **Max prompt file**: Unlimited (stored on disk)
- **Concurrent sessions**: Depends on Vibe installation

### Optimization Tips
1. Use specific, concise prompts
2. Provide relevant context files
3. Break large tasks into smaller steps
4. Use `vibe_ask()` for read-only operations
5. Cache session IDs for continuation

---

## 🔐 Security

### Input Validation
- Database queries: SELECT-only (prevents mutations)
- Subcommands: Whitelist-based validation
- File paths: Must exist or be creatable
- Prompts: Sanitized for shell execution

### Resource Limits
- Output truncation prevents memory exhaustion
- Timeout protection prevents infinite loops
- Temporary files auto-cleanup after execution
- Process termination on timeout

### Logging
- All operations logged to file
- Error messages logged but secrets hidden
- Session history persisted for audit trail

---

## 📝 Logging

### Log Levels
- **DEBUG**: Low-level operations, stream processing
- **INFO**: Tool execution, important events
- **WARNING**: Configuration issues, cleanup failures
- **ERROR**: Exceptions, command failures

### Log Location
```
~/.config/atlastrinity/logs/vibe_server.log
```

### Sample Log Output
```
2026-01-18 14:23:45 [INFO] [vibe_mcp] [VIBE] Executing prompt: Analyze error... (timeout=300s)
2026-01-18 14:23:46 [DEBUG] [vibe_mcp] [VIBE] Found binary at: /usr/local/bin/vibe
2026-01-18 14:24:12 [INFO] [vibe_mcp] [VIBE] Process completed with exit code: 0
```

---

## 🔄 Migration Guide (From Old to New)

### What Changed
| Aspect | Before | After |
|--------|--------|-------|
| PTY wrapper | Complex vibe_runner.py | Direct async subprocess |
| Logging | Multi-layer | Single consolidated logger |
| Error handling | Try/except scattered | Centralized in run_vibe_subprocess() |
| Context handling | Optional/inconsistent | Required Context parameter |
| Timeouts | Implicit | Explicit with buffer |
| Resource cleanup | Manual | Automatic with finally blocks |

### Migration Steps
1. Replace `src/mcp_server/vibe_server.py` with new version ✅
2. Remove `src/mcp_server/vibe_runner.py` ✅
3. No changes needed to consumers (same API)
4. Test existing tools with new implementation
5. Verify logging in `~/.config/atlastrinity/logs/vibe_server.log`

---

## 🧪 Testing

### Test Checklist
```
[ ] vibe_which() - binary location
[ ] vibe_prompt() - simple prompt
[ ] vibe_ask() - read-only question
[ ] vibe_analyze_error() - error handling
[ ] vibe_implement_feature() - feature development
[ ] vibe_code_review() - code analysis
[ ] vibe_smart_plan() - planning
[ ] vibe_execute_subcommand() - utility commands
[ ] vibe_list_sessions() - session listing
[ ] vibe_session_details() - session details
[ ] vibe_check_db() - database queries
[ ] vibe_get_system_context() - system state
```

### Quick Test
```python
import asyncio
from src.mcp_server.vibe_server import server

async def test():
    result = await vibe_which(Context())
    print(result)

asyncio.run(test())
```

---

## 📞 Support & Debugging

### Enable Debug Logging
```python
import logging
logging.getLogger("vibe_mcp").setLevel(logging.DEBUG)
```

### Check Vibe Installation
```bash
vibe --version
vibe --help
```

### Monitor Logs in Real-Time
```bash
tail -f ~/.config/atlastrinity/logs/vibe_server.log
```

### Check Session History
```bash
ls -lh ~/.vibe/logs/session/
cat ~/.vibe/logs/session/session_*.json | jq .
```

---

## 🎓 Architecture Lessons

### Design Principles Applied
1. **Separation of Concerns**: Subprocess execution ≠ tool logic
2. **Explicit Over Implicit**: Clear error messages, explicit timeouts
3. **Async-First**: All I/O is non-blocking
4. **Resource Safety**: Cleanup in finally blocks
5. **Configuration-Driven**: External settings override defaults
6. **Logging Over Debugging**: Comprehensive logging instead of print()

### Common Pitfalls Avoided
1. ~~Complex PTY handling~~ → Simple async subprocess
2. ~~Implicit timeouts~~ → Explicit timeout with graceful shutdown
3. ~~Scattered error handling~~ → Centralized error handling
4. ~~Memory issues from large outputs~~ → Output truncation
5. ~~Resource leaks~~ → Automatic cleanup

---

## 📚 Related Documents
- [MCP_ARCHITECTURE.md](./MCP_ARCHITECTURE.md) - Overall MCP design
- [MCP_AUDIT_REPORT.md](./MCP_AUDIT_REPORT.md) - System audit
- [MCP_SUMMARY.md](./MCP_SUMMARY.md) - Executive summary

---

## 📋 Changelog

### Version 2.0 (2026-01-18)
- ✅ Complete architectural reengineering
- ✅ Proper FastMCP integration
- ✅ Removed vibe_runner.py (now using native async subprocess)
- ✅ Improved error handling and logging
- ✅ Configuration-driven settings
- ✅ Production-grade implementation

### Version 1.x (Legacy)
- Original implementation with PTY wrapper
- Complex logging layers
- Manual resource management

---

**Status**: ✅ Production Ready | **Maintainer**: AtlasTrinity Team | **Last Updated**: 2026-01-18
