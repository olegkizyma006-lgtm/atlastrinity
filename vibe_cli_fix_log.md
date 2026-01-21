Vibe CLI Fix Log - 2026-01-21T14:40:56.279572

Component: Vibe CLI Integration
Issue: Unsupported --model argument in Vibe CLI invocations
Fix Description: Removed --model argument from vibe_config.py to_cli_args() method as it is not supported by current Vibe CLI. Updated function signature and tests accordingly.
Files Modified: src/mcp_server/vibe_config.py, tests/test_vibe_server.py
Test Results: All regression tests passed. --model argument no longer appears in generated CLI commands.
Environment: development, production

Technical Details:
- Removed model parameter from to_cli_args() function signature
- Removed code that added --model argument to CLI args
- Updated test to verify --model is not present in generated args
- Verified all other CLI arguments still work correctly
