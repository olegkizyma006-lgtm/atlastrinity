
import subprocess
import time
import os
import signal
import psutil

def find_procs_by_name(names):
    "Return a list of processes matching 'names'."
    ls = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = " ".join(p.info['cmdline'] or [])
            if any(name in cmdline for name in names):
                ls.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return ls

def test_shutdown():
    print("Step 1: Spawning brain.server...")
    # Use the same python as current
    python_path = "python3"
    
    # Spawn as a process group to mimic Electron's behavior if needed, 
    # but we want to see if our internal shutdown works
    proc = subprocess.Popen(
        [python_path, "-m", "src.brain.server"],
        cwd=os.getcwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    print(f"Spawned server with PID: {proc.pid}")
    time.sleep(10) # Wait for it to initialize and maybe spawn child MCPs
    
    # Check what's running now
    targets = ["mcp-server", "macos-use", "vibe_server", "brain.server"]
    running_before = find_procs_by_name(targets)
    print(f"Processes running before shutdown: {[p.pid for p in running_before]}")
    
    print("Step 2: Sending SIGTERM to brain.server...")
    proc.send_signal(signal.SIGTERM)
    
    # Wait for completion
    try:
        proc.wait(timeout=15)
        print("Brain server exited accurately.")
    except subprocess.TimeoutExpired:
        print("Brain server did not exit in time, forcing kill.")
        proc.kill()

    time.sleep(2) # Wait for children to cleanup
    
    print("Step 3: Verifying all target processes are gone...")
    running_after = find_procs_by_name(targets)
    
    if not running_after:
        print("SUCCESS: All processes terminated correctly.")
    else:
        print(f"FAILURE: Following processes still running: {[p.info['cmdline'] for p in running_after]}")
        # Cleanup for safety
        for p in running_after:
            try: p.kill()
            except: pass

if __name__ == "__main__":
    test_shutdown()
