#!/usr/bin/env python3
"""
AtlasTrinity MCP Server Health Check

Enhanced CLI tool for checking MCP server status with:
- Colored terminal output (green/yellow/red)
- Tier information for each server
- Response time and tool count
- JSON output for automation (--json flag)
"""

import argparse
import asyncio
import json
import os
import sys

# Add src to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, PROJECT_ROOT) # Ensure src.brain etc works if imported as src.brain


class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ENDC = "\033[0m"


async def check_mcp(output_json: bool = False):
    """Run MCP health checks for all servers."""
    from brain.config import ensure_dirs
    from brain.mcp_manager import mcp_manager

    ensure_dirs()

    servers = mcp_manager.config.get("mcpServers", {})
    results = {}

    # Filter out comment keys and disabled servers
    active_servers = [
        (name, cfg)
        for name, cfg in servers.items()
        if not name.startswith("_") and not cfg.get("disabled", False)
    ]

    if not output_json:
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.CYAN}  🔌 MCP Server Status Report{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.ENDC}\n")
        print(
            f"  {Colors.BOLD}{'Server':<24} {'Tier':^6} {'Status':^12} {'Tools':^6} {'Time':^10}{Colors.ENDC}"
        )
        print(f"  {'-' * 64}")

    for server_name, server_config in active_servers:
        tier = server_config.get("tier", 4)

        try:
            import time

            start = time.time()

            # list_tools will automatically call get_session and connect if needed
            tools = await asyncio.wait_for(mcp_manager.list_tools(server_name), timeout=30.0)

            elapsed = (time.time() - start) * 1000  # ms

            if tools:
                results[server_name] = {
                    "status": "online",
                    "tier": tier,
                    "tools_count": len(tools),
                    "response_time_ms": round(elapsed, 1),
                }
                if not output_json:
                    print(
                        f"  {Colors.GREEN}✓{Colors.ENDC} {server_name:<22} "
                        f"{Colors.BLUE}T{tier}{Colors.ENDC}     "
                        f"{Colors.GREEN}ONLINE{Colors.ENDC}      "
                        f"{len(tools):^6} "
                        f"{elapsed:>6.0f}ms"
                    )
            # check if it's connected
            elif server_name in mcp_manager.sessions:
                results[server_name] = {
                    "status": "degraded",
                    "tier": tier,
                    "tools_count": 0,
                    "response_time_ms": round(elapsed, 1),
                    "note": "Connected but no tools",
                }
                if not output_json:
                    print(
                        f"  {Colors.YELLOW}⚠{Colors.ENDC} {server_name:<22} "
                        f"{Colors.BLUE}T{tier}{Colors.ENDC}     "
                        f"{Colors.YELLOW}DEGRADED{Colors.ENDC}    "
                        f"{0:^6} "
                        f"{elapsed:>6.0f}ms"
                    )
            else:
                results[server_name] = {
                    "status": "offline",
                    "tier": tier,
                    "error": "Failed to get session",
                }
                if not output_json:
                    print(
                        f"  {Colors.RED}✗{Colors.ENDC} {server_name:<22} "
                        f"{Colors.BLUE}T{tier}{Colors.ENDC}     "
                        f"{Colors.RED}OFFLINE{Colors.ENDC}     "
                        f"{'—':^6} "
                        f"{'—':^10}"
                    )

        except TimeoutError:
            results[server_name] = {
                "status": "offline",
                "tier": tier,
                "error": "Connection timeout (30s)",
            }
            if not output_json:
                print(
                    f"  {Colors.RED}✗{Colors.ENDC} {server_name:<22} "
                    f"{Colors.BLUE}T{tier}{Colors.ENDC}     "
                    f"{Colors.RED}TIMEOUT{Colors.ENDC}     "
                    f"{'—':^6} "
                    f"{Colors.DIM}>30s{Colors.ENDC}"
                )

        except Exception as e:
            results[server_name] = {
                "status": "offline",
                "tier": tier,
                "error": str(e)[:100],
            }
            if not output_json:
                error_short = str(e)[:40]
                print(
                    f"  {Colors.RED}✗{Colors.ENDC} {server_name:<22} "
                    f"{Colors.BLUE}T{tier}{Colors.ENDC}     "
                    f"{Colors.RED}ERROR{Colors.ENDC}       "
                    f"{'—':^6} "
                    f"{Colors.DIM}{error_short}{Colors.ENDC}"
                )

    if output_json:
        # JSON output for automation
        output = {
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "total_servers": len(results),
            "online": sum(1 for r in results.values() if r["status"] == "online"),
            "offline": sum(1 for r in results.values() if r["status"] == "offline"),
            "degraded": sum(1 for r in results.values() if r["status"] == "degraded"),
            "servers": results,
        }
        print(json.dumps(output, indent=2))
    else:
        # Summary
        online = sum(1 for r in results.values() if r["status"] == "online")
        offline = sum(1 for r in results.values() if r["status"] == "offline")
        degraded = sum(1 for r in results.values() if r["status"] == "degraded")
        total = len(results)

        print(f"\n  {'-' * 64}")

        if offline == 0:
            status_color = Colors.GREEN
            status_icon = "✓"
        elif online > offline:
            status_color = Colors.YELLOW
            status_icon = "⚠"
        else:
            status_color = Colors.RED
            status_icon = "✗"

        print(
            f"\n  {status_color}{status_icon}{Colors.ENDC} "
            f"{Colors.BOLD}Summary:{Colors.ENDC} "
            f"{Colors.GREEN}{online}{Colors.ENDC} online, "
            f"{Colors.YELLOW}{degraded}{Colors.ENDC} degraded, "
            f"{Colors.RED}{offline}{Colors.ENDC} offline "
            f"(of {total} total)"
        )

        health_pct = (online / total * 100) if total > 0 else 0
        print(
            f"  {status_color}{status_icon}{Colors.ENDC} "
            f"{Colors.BOLD}Health:{Colors.ENDC} {health_pct:.0f}%"
        )

        print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.ENDC}\n")


def main():
    parser = argparse.ArgumentParser(description="Check MCP server health")
    parser.add_argument("--json", action="store_true", help="Output in JSON format for automation")
    args = parser.parse_args()

    asyncio.run(check_mcp(output_json=args.json))


if __name__ == "__main__":
    main()
