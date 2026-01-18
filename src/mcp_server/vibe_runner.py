
import os
import sys
import pty
import subprocess
import threading
import json
import asyncio
import re

# ANSI escape code regex
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub('', text)

def main():
    # Helper to clean and print JSON
    def process_output(data):
        try:
            text = data.decode(errors='replace')
        except:
            return

        # Split into lines to find JSON
        # Note: We might split a JSON blob if it contains newlines? 
        # But Vibe streaming usually sends one JSON object per line or block.
        # We will just print raw chunks but try to clean TUI?
        # Actually, best to just pass valid JSON lines.
        
        # Simple heuristic: If line looks like JSON, print it.
        # If line contains JSON, extract it.
        
        lines = text.splitlines()
        for line in lines:
            stripped = strip_ansi(line).strip()
            # Check for JSON start
            idx = stripped.find("{")
            if idx != -1:
                potential_json = stripped[idx:]
                try:
                    # Verify it's valid JSON
                    json.loads(potential_json)
                    print(potential_json, flush=True)
                except:
                    # Partial JSON or invalid?
                    # If we are streaming, we might get partials.
                    # But Vibe usually flushes complete objects in streaming mode.
                    pass
            elif stripped:
                 # Pass through non-JSON text that looks important?
                 # E.g. errors?
                 pass

    # Arguments: [python, script, binary, arg1, arg2...]
    # We expect sys.argv[1] to be the binary path?
    # No, vibe_server passes [binary, args...] as argv to subprocess.
    # So if we replace binary with [python, runner, binary], then argv[1] is binary.
    
    if len(sys.argv) < 2:
        sys.exit(1)

    debug_mode = os.environ.get("VIBE_DEBUG_RAW", "true").lower() == "true"

    def process_output(data):
        try:
            text = data.decode(errors='replace')
        except:
            return

        lines = text.splitlines()
        for line in lines:
            stripped = strip_ansi(line).strip()
            # Check for JSON start
            idx = stripped.find("{")
            if idx != -1:
                potential_json = stripped[idx:]
                try:
                    # Verify it's valid JSON
                    json.loads(potential_json)
                    print(potential_json, flush=True)
                    return # Handled
                except:
                    pass
            
            # If not handled as JSON, print to stderr for debugging
            if stripped and debug_mode:
                 # Prefix to identify it in logs
                 print(f"[VIBE_RAW] {stripped}", file=sys.stderr, flush=True)

    target_cmd = sys.argv[1:]


    
    # Run in PTY
    master, slave = pty.openpty()
    
    try:
        # Use simple env (inherit)
        env = os.environ.copy()
        # env["TERM"] = "dumb" # Let's NOT force dumb if xterm worked better?
        # Actually, let's stick to what works. test_pty_threaded used inherited env.
        
        process = subprocess.Popen(
            target_cmd,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            env=env,
            close_fds=True
        )
        os.close(slave)
        
        # Keep master open. Send \n just in case.
        try:
            os.write(master, b"\n")
        except: pass
        
        # Reading loop
        while True:
            try:
                data = os.read(master, 4096)
            except OSError:
                break
            
            if not data:
                break
                
            process_output(data)
            
        process.wait()
        sys.exit(process.returncode)
        
    except Exception as e:
        print(f"{{\"error\": \"Wrapper failed: {e}\"}}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
