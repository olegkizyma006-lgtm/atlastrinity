"""
AtlasTrinity MCP Health Dashboard

Centralized health monitoring for all MCP servers with:
- Parallel async health checks
- Server metrics (response time, failure count)
- Startup diagnostics with colored output
- Tier-based categorization
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .logger import logger


@dataclass
class ServerStatus:
    """Status information for a single MCP server."""

    name: str
    tier: int
    status: str  # "online", "offline", "degraded", "unknown"
    tools_count: int = 0
    response_time_ms: float = 0.0
    last_check: datetime | None = None
    last_error: str | None = None
    failure_count: int = 0
    description: str = ""


@dataclass
class HealthCheckResult:
    """Result of a complete health check cycle."""

    timestamp: datetime
    total_servers: int
    online_count: int
    offline_count: int
    degraded_count: int
    servers: dict[str, ServerStatus] = field(default_factory=dict)

    @property
    def health_percentage(self) -> float:
        if self.total_servers == 0:
            return 0.0
        return (self.online_count / self.total_servers) * 100


class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


class MCPHealthDashboard:
    """
    Centralized health monitoring dashboard for MCP servers.

    Features:
    - Async parallel health checks for all servers
    - Detailed metrics per server (response time, failures)
    - Startup diagnostics with colored terminal output
    - Server categorization by tier and status
    """

    def __init__(self, mcp_manager=None):
        """
        Initialize dashboard.

        Args:
            mcp_manager: Optional MCPManager instance. If None, imports lazily.
        """
        self._mcp_manager = mcp_manager
        self._last_result: HealthCheckResult | None = None
        self._server_metrics: dict[str, ServerStatus] = {}

    @property
    def mcp_manager(self):
        """Lazy import of MCPManager to avoid circular imports."""
        if self._mcp_manager is None:
            from .mcp_manager import mcp_manager

            self._mcp_manager = mcp_manager
        return self._mcp_manager

    def _get_server_tier(self, server_name: str) -> int:
        """Get tier level for a server from config."""
        config = self.mcp_manager.config.get("mcpServers", {})
        server_config = config.get(server_name, {})
        return server_config.get("tier", 4)  # Default to Tier 4

    def _get_server_description(self, server_name: str) -> str:
        """Get description for a server from config."""
        config = self.mcp_manager.config.get("mcpServers", {})
        server_config = config.get(server_name, {})
        return server_config.get("description", "")

    async def _check_single_server(self, server_name: str) -> ServerStatus:
        """
        Check health of a single server.

        Returns ServerStatus with timing and status information.
        """
        tier = self._get_server_tier(server_name)
        description = self._get_server_description(server_name)

        start_time = datetime.now()
        status = ServerStatus(
            name=server_name,
            tier=tier,
            status="unknown",
            last_check=start_time,
            description=description,
        )

        try:
            # Time the health check
            check_start = asyncio.get_event_loop().time()

            # Try to list tools (this validates connection)
            tools = await asyncio.wait_for(self.mcp_manager.list_tools(server_name), timeout=30.0)

            check_end = asyncio.get_event_loop().time()
            response_time = (check_end - check_start) * 1000  # Convert to ms

            if tools:
                status.status = "online"
                status.tools_count = len(tools)
                status.response_time_ms = response_time
                status.failure_count = 0
            else:
                # Connected but no tools - degraded
                status.status = "degraded"
                status.last_error = "Connected but no tools available"
                status.response_time_ms = response_time

        except TimeoutError:
            status.status = "offline"
            status.last_error = "Connection timeout (30s)"
            status.failure_count = self._server_metrics.get(server_name, status).failure_count + 1

        except Exception as e:
            status.status = "offline"
            status.last_error = str(e)[:200]
            status.failure_count = self._server_metrics.get(server_name, status).failure_count + 1

        # Update stored metrics
        self._server_metrics[server_name] = status
        return status

    async def run_all_checks(self) -> HealthCheckResult:
        """
        Run parallel health checks for all configured servers.

        Returns HealthCheckResult with aggregate statistics and per-server status.
        """
        config = self.mcp_manager.config.get("mcpServers", {})

        # Filter out comment keys and disabled servers
        server_names = [
            name
            for name, cfg in config.items()
            if not name.startswith("_") and not cfg.get("disabled", False)
        ]

        # Run all checks in parallel
        tasks = [self._check_single_server(name) for name in server_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build result
        servers: dict[str, ServerStatus] = {}
        online = offline = degraded = 0

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"[HealthDashboard] Unexpected error: {result}")
                continue

            # Type narrowing for Pyrefly
            if not isinstance(result, ServerStatus):
                continue

            servers[result.name] = result

            if result.status == "online":
                online += 1
            elif result.status == "offline":
                offline += 1
            elif result.status == "degraded":
                degraded += 1

        self._last_result = HealthCheckResult(
            timestamp=datetime.now(),
            total_servers=len(server_names),
            online_count=online,
            offline_count=offline,
            degraded_count=degraded,
            servers=servers,
        )

        return self._last_result

    def format_startup_diagnostics(self, result: HealthCheckResult) -> str:
        """
        Format health check result as colored terminal output.

        Returns multi-line string with ANSI color codes.
        """
        lines = []
        c = Colors

        # Header
        lines.append(f"\n{c.BOLD}{c.CYAN}{'=' * 60}{c.ENDC}")
        lines.append(f"{c.BOLD}{c.CYAN}  🔌 MCP Server Health Dashboard{c.ENDC}")
        lines.append(f"{c.BOLD}{c.CYAN}{'=' * 60}{c.ENDC}\n")

        # Summary
        health_color = (
            c.GREEN
            if result.health_percentage >= 80
            else (c.YELLOW if result.health_percentage >= 50 else c.RED)
        )
        lines.append(
            f"  {c.BOLD}Status:{c.ENDC} {health_color}{result.online_count}/{result.total_servers} online ({result.health_percentage:.0f}%){c.ENDC}"
        )
        lines.append(f"  {c.BOLD}Time:{c.ENDC} {result.timestamp.strftime('%H:%M:%S')}\n")

        # Group by tier
        tiers: dict[int, list[ServerStatus]] = {1: [], 2: [], 3: [], 4: []}
        for status in result.servers.values():
            tiers.setdefault(status.tier, []).append(status)

        tier_names = {
            1: "CORE (Must-Have)",
            2: "HIGH (Recommended)",
            3: "MEDIUM (Optional)",
            4: "OPTIONAL (On-Demand)",
        }

        for tier_num in sorted(tiers.keys()):
            servers_in_tier = tiers[tier_num]
            if not servers_in_tier:
                continue

            lines.append(
                f"  {c.BOLD}{c.BLUE}Tier {tier_num}: {tier_names.get(tier_num, '')}{c.ENDC}"
            )

            for srv in sorted(servers_in_tier, key=lambda x: x.name):
                if srv.status == "online":
                    icon = f"{c.GREEN}✓{c.ENDC}"
                    status_text = f"{c.GREEN}ONLINE{c.ENDC}"
                    extra = f" ({srv.tools_count} tools, {srv.response_time_ms:.0f}ms)"
                elif srv.status == "degraded":
                    icon = f"{c.YELLOW}⚠{c.ENDC}"
                    status_text = f"{c.YELLOW}DEGRADED{c.ENDC}"
                    extra = f" - {srv.last_error}"
                else:
                    icon = f"{c.RED}✗{c.ENDC}"
                    status_text = f"{c.RED}OFFLINE{c.ENDC}"
                    extra = f" - {srv.last_error}" if srv.last_error else ""

                lines.append(f"    {icon} {srv.name:20} {status_text}{c.DIM}{extra}{c.ENDC}")

            lines.append("")

        lines.append(f"{c.BOLD}{c.CYAN}{'=' * 60}{c.ENDC}\n")

        return "\n".join(lines)

    async def startup_diagnostics(self) -> str:
        """
        Run health checks and return formatted diagnostics string.

        This is the main entry point for startup diagnostics.
        """
        logger.info("[HealthDashboard] Running startup health checks...")

        result = await self.run_all_checks()
        output = self.format_startup_diagnostics(result)

        # Log summary
        if result.offline_count > 0:
            logger.warning(
                f"[HealthDashboard] {result.offline_count} servers offline: "
                f"{[n for n, s in result.servers.items() if s.status == 'offline']}"
            )
        else:
            logger.info(f"[HealthDashboard] All {result.online_count} servers online")

        return output

    def get_summary(self) -> dict[str, Any]:
        """
        Get last health check result as dictionary for API responses.

        Returns empty dict if no checks have been run yet.
        """
        if not self._last_result:
            return {"status": "no_checks_run"}

        return {
            "timestamp": self._last_result.timestamp.isoformat(),
            "total_servers": self._last_result.total_servers,
            "online": self._last_result.online_count,
            "offline": self._last_result.offline_count,
            "degraded": self._last_result.degraded_count,
            "health_percentage": self._last_result.health_percentage,
            "servers": {
                name: {
                    "tier": s.tier,
                    "status": s.status,
                    "tools_count": s.tools_count,
                    "response_time_ms": s.response_time_ms,
                    "last_error": s.last_error,
                }
                for name, s in self._last_result.servers.items()
            },
        }

    def get_server_status(self, server_name: str) -> ServerStatus | None:
        """Get cached status for a specific server."""
        return self._server_metrics.get(server_name)


# Global instance
health_dashboard = MCPHealthDashboard()
