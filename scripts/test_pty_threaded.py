import asyncio
import os
import pty
import threading

VIBE_BINARY = os.path.expanduser("~/.local/bin/vibe")
VIBE_WORKSPACE = os.path.expanduser("~/.config/atlastrinity/vibe_workspace")


def read_master(master_fd, loop, callback):
    """Read from PTY master in a thread."""
    try:
        while True:
            # Blocking read
            try:
                data = os.read(master_fd, 1024)
            except OSError:
                break

            if not data:
                break

            text = data.decode(errors="replace")
            # Schedule callback in loop
            asyncio.run_coroutine_threadsafe(callback(text), loop)
    except Exception as e:
        print(f"Read error: {e}")
    finally:
        try:
            os.close(master_fd)
        except:
            pass


async def run_vibe_pty_threaded():
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

    # Define callback
    chunks = []

    async def on_output(text):
        chunks.append(text)
        print(f"[PTY] {text}", end="")

    print(f"Running via PTY (Threaded): {argv}")

    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        stdout=slave,
        stderr=slave,
        stdin=slave,
        close_fds=True,
    )
    os.close(slave)

    # Start reader thread
    loop = asyncio.get_running_loop()
    reader_thread = threading.Thread(target=read_master, args=(master, loop, on_output))
    reader_thread.start()

    # Wait for process
    try:
        await asyncio.wait_for(process.wait(), timeout=15.0)
    except TimeoutError:
        print("TIMED OUT")
        process.terminate()

    # Wait for thread usually? It ends when master closes (process exit closes slave -> master EOF)
    reader_thread.join(timeout=1.0)
    print(f"Exit code: {process.returncode}")


if __name__ == "__main__":
    asyncio.run(run_vibe_pty_threaded())
