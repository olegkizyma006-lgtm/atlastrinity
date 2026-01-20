"""
AtlasTrinity Notifications

Human-in-the-Loop system for:
- Alerting user when system is stuck
- Requesting human decisions
- Showing progress notifications
"""

import os
import subprocess
from datetime import datetime

from .logger import logger


class NotificationManager:
    """
    Manages user notifications via macOS notification center.

    Used for Human-in-the-Loop interactions:
    - Alert when stuck after max retries
    - Request approval for dangerous actions
    - Show progress updates
    """

    def __init__(self):
        self.pending_decisions: list[dict] = []
        self.notification_history: list[dict] = []

    def send_notification(
        self, title: str, message: str, sound: bool = True, subtitle: str = ""
    ) -> bool:
        """
        Send macOS notification.

        Args:
            title: Notification title
            message: Main message body
            sound: Play notification sound
            subtitle: Optional subtitle

        Returns:
            True if sent successfully
        """
        try:
            # Build AppleScript

            script = """
            display notification "{message}" with title "{title}" {subtitle_cmd} {sound_cmd}
            """

            subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)

            # Log to history
            self.notification_history.append(
                {
                    "title": title,
                    "message": message,
                    "timestamp": datetime.now().isoformat(),
                    "type": "notification",
                }
            )

            logger.info(f"[NOTIFY] Sent: {title}")
            return True

        except Exception as e:
            logger.error(f"[NOTIFY] Failed to send notification: {e}")
            return False

    def send_stuck_alert(self, step_id: int, error: str, attempts: int = 0) -> bool:
        """
        Alert user that system is stuck and needs help.

        Args:
            step_id: Which step failed
            error: Error description
            attempts: Number of attempts made
        """
        title = "AtlasTrinity застряла"
        message = f"Крок {step_id} не вдався після {attempts} спроб"
        subtitle = error[:50] + "..." if len(error) > 50 else error

        return self.send_notification(title=title, message=message, subtitle=subtitle, sound=True)

    def request_approval(self, action: str, risk_level: str = "medium") -> bool:
        """
        Request user approval for potentially dangerous action.
        Shows a dialog and waits for response.

        Args:
            action: Description of action
            risk_level: low/medium/high/critical

        Returns:
            True if approved, False if rejected
        """
        # Check for auto-approve environment variable
        auto_approve = os.getenv("TRINITY_AUTO_APPROVE", "false").lower() == "true"
        if auto_approve:
            logger.info(f"[NOTIFY] Auto-approving action: {action}")
            self.notification_history.append(
                {
                    "action": action,
                    "risk_level": risk_level,
                    "approved": True,
                    "timestamp": datetime.now().isoformat(),
                    "type": "approval_request",
                }
            )
            return True

        try:
            {
                "low": "note",
                "medium": "caution",
                "high": "stop",
                "critical": "stop",
            }.get(risk_level, "note")

            script = """
            display dialog "{action}" with title "AtlasTrinity - Підтвердження" buttons {{"Відмовити", "Дозволити"}} default button "Відмовити" with icon {icon}
            """

            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            approved = "Дозволити" in result.stdout

            self.notification_history.append(
                {
                    "action": action,
                    "risk_level": risk_level,
                    "approved": approved,
                    "timestamp": datetime.now().isoformat(),
                    "type": "approval_request",
                }
            )

            logger.info(
                f"[NOTIFY] Approval request: {action} -> {'Approved' if approved else 'Rejected'}"
            )
            return approved

        except subprocess.TimeoutExpired:
            logger.warning("[NOTIFY] Approval request timed out")
            return False
        except Exception as e:
            logger.error(f"[NOTIFY] Failed to request approval: {e}")
            return False

    def show_progress(self, current_step: int, total_steps: int, description: str = "") -> bool:
        """
        Show progress notification.

        Args:
            current_step: Current step number
            total_steps: Total number of steps
            description: What's being done
        """
        progress = int((current_step / total_steps) * 100) if total_steps > 0 else 0

        title = f"AtlasTrinity ({progress}%)"
        message = f"Крок {current_step}/{total_steps}"

        if description:
            message += f": {description[:30]}"

        # Only show at key milestones (25%, 50%, 75%, 100%)
        if progress in [25, 50, 75, 100] or current_step == 1:
            return self.send_notification(title, message, sound=False)

        return True

    def show_completion(self, task: str, success: bool, duration_seconds: float = 0) -> bool:
        """
        Show task completion notification.

        Args:
            task: Task description
            success: Whether task succeeded
            duration_seconds: How long it took
        """
        if success:
            title = "✅ AtlasTrinity Завершила"
            message = f"{task[:50]}"
        else:
            title = "❌ AtlasTrinity Не Вдалось"
            message = f"Не вдалося: {task[:40]}"

        if duration_seconds > 0:
            mins = int(duration_seconds // 60)
            secs = int(duration_seconds % 60)
            message += f" ({mins}м {secs}с)"

        return self.send_notification(title, message, sound=success)

    def get_history(self, limit: int = 10) -> list[dict]:
        """Get recent notification history."""
        return self.notification_history[-limit:]


# Singleton instance
notifications = NotificationManager()
