
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
                    pass
            elif stripped:
                 # Pass through non-JSON text that looks important?
                 pass

    if len(sys.argv) < 2:
        sys.exit(1)

    debug_mode = os.environ.get("VIBE_DEBUG_RAW", "false").lower() == "true"

    line_buffer = ""

    def process_output(data):
        nonlocal line_buffer
        try:
            text = data.decode(errors='replace')
        except:
            return

        line_buffer += text
        if "\n" not in line_buffer:
            # If line is getting too long without newline, it might be a TUI prompt
            if len(line_buffer) > 100:
                # Check for known prompts in the fragment
                tui_artifacts = ["Press Enter", "Welcome to Mistral", "│", "─", "╭", "╮", "╰", "╯"]
                if any(art in line_buffer for art in tui_artifacts):
                    line_buffer = "" # Clear if it's spam
                return
            return

        lines = line_buffer.split("\n")
        line_buffer = lines.pop() # Keep the last fragment
        
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
                    continue # Handled
                except:
                    pass
            
            # If not handled as JSON, print to stderr for debugging
            if stripped and debug_mode:
                 # Filtering TUI noise
                 tui_artifacts = ["Press Enter", "Welcome to Mistral", "│", "─", "╭", "╮", "╰", "╯"]
                 if any(art in stripped for art in tui_artifacts):
                     continue
                 
                 # Prefix to identify it in logs
                 print(f"[VIBE_RAW] {stripped}", file=sys.stderr, flush=True)

    target_cmd = sys.argv[1:]

    # Run in PTY
    master, slave = pty.openpty()
    
    try:
        # Use simple env (inherit)
        env = os.environ.copy()
        env["TERM"] = "dumb"
        env["PAGER"] = "cat"
        env["NO_COLOR"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
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
