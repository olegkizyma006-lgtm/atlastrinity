import asyncio
import os
import pty
import subprocess

VIBE_BINARY = os.path.expanduser("~/.local/bin/vibe")
VIBE_WORKSPACE = os.path.expanduser("~/.config/atlastrinity/vibe_workspace")


async def run_vibe_pty():
    prompt = "Create a file called '/tmp/hello_vibe_test.py' with a simple hello world script that prints 'Hello from Vibe MCP!'."
    argv = [
        VIBE_BINARY,
        "-p",
        prompt,
        "--output",
        "streaming",
        "--auto-approve",
        "--max-turns",
        "5",
    ]

    cwd = VIBE_WORKSPACE
    os.makedirs(cwd, exist_ok=True)

    # Create PTY
    master, slave = pty.openpty()

    env = os.environ.copy()
    env["TERM"] = "dumb"
    env["NO_COLOR"] = "1"

    print(f"Running via PTY: {argv}")

    process = None
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdout=slave,
            stderr=slave,
            stdin=slave,
            text=True,
            preexec_fn=os.setsid,
        )
        os.close(slave)

        loop = asyncio.get_event_loop()
        chunks = []

        async def read_loop():
            try:
                while True:
                    data = await loop.run_in_executor(None, os.read, master, 1024)
                    if not data:
                        break
                    text = data.decode(errors="replace")
                    chunks.append(text)
                    print(f"[PTY] {text}", end="")
            except Exception as e:
                print(f"Read error: {e}")

        await asyncio.wait_for(
            asyncio.gather(read_loop(), loop.run_in_executor(None, process.wait)),
            timeout=10.0,
        )

        print(f"\nExit code: {process.returncode}")

    except Exception as e:
        print(f"\nError: {e}")
        if process:
            try:
                process.terminate()
            except:
                pass
    finally:
        if "master" in locals():
            try:
                os.close(master)
            except:
                pass


if __name__ == "__main__":
    asyncio.run(run_vibe_pty())
