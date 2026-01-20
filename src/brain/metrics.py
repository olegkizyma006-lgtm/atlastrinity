import time

import psutil


class MetricsCollector:
    def __init__(self):
        self.start_time = time.time()
        self._last_net_io = psutil.net_io_counters()
        self._last_net_time = time.time()

    def _format_speed(self, bytes_per_sec):
        if bytes_per_sec < 1024:
            return f"{bytes_per_sec:.0f}", "B/S"
        elif bytes_per_sec < 1024**2:
            return f"{bytes_per_sec / 1024:.1f}", "K/S"
        elif bytes_per_sec < 1024**3:
            return f"{bytes_per_sec / 1024**2:.1f}", "M/S"
        else:
            return f"{bytes_per_sec / 1024**3:.1f}", "G/S"

    def get_metrics(self):
        # CPU
        cpu_percent = psutil.cpu_percent(interval=None)

        # Memory
        mem = psutil.virtual_memory()
        mem_used_gb = mem.used / (1024**3)

        # Network
        current_net_io = psutil.net_io_counters()
        current_time = time.time()
        time_delta = current_time - self._last_net_time

        if time_delta <= 0:
            time_delta = 0.1  # Prevent div by zero

        bytes_sent = (current_net_io.bytes_sent - self._last_net_io.bytes_sent) / time_delta
        bytes_recv = (current_net_io.bytes_recv - self._last_net_io.bytes_recv) / time_delta

        self._last_net_io = current_net_io
        self._last_net_time = current_time

        up_val, up_unit = self._format_speed(bytes_sent)
        down_val, down_unit = self._format_speed(bytes_recv)

        return {
            "cpu": f"{int(cpu_percent)}%",
            "memory": f"{mem_used_gb:.1f}GB",
            "net_up_val": up_val,
            "net_up_unit": up_unit,
            "net_down_val": down_val,
            "net_down_unit": down_unit,
        }

    def record(self, name, value, tags=None):
        """Record a custom metric (stub for now to prevent crashes)."""
        pass


metrics_collector = MetricsCollector()
