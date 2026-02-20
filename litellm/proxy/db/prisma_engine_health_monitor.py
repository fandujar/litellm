"""
Prisma Engine Health Monitor

This module provides a background health monitor for the Prisma query engine subprocess.
It periodically checks if the Prisma engine is running properly and not in a zombie state,
and can restart it if necessary.
"""

import asyncio
import os
import subprocess
import sys
from typing import Any, Callable, List, Optional, Tuple

from litellm._logging import verbose_proxy_logger


def _log_info(msg: str):
    """Log info message to both verbose logger and stderr for visibility."""
    verbose_proxy_logger.info(msg)
    print(f"[PrismaHealthMonitor] INFO: {msg}", file=sys.stderr, flush=True)


def _log_warning(msg: str):
    """Log warning message to both verbose logger and stderr for visibility."""
    verbose_proxy_logger.warning(msg)
    print(f"[PrismaHealthMonitor] WARNING: {msg}", file=sys.stderr, flush=True)


def _log_error(msg: str):
    """Log error message to both verbose logger and stderr for visibility."""
    verbose_proxy_logger.error(msg)
    print(f"[PrismaHealthMonitor] ERROR: {msg}", file=sys.stderr, flush=True)


def _log_debug(msg: str):
    """Log debug message."""
    verbose_proxy_logger.debug(msg)


class PrismaEngineHealthMonitor:
    """
    Background monitor that ensures the Prisma query engine subprocess is healthy.

    The monitor runs as a background task that:
    1. Periodically checks if the Prisma engine process is running
    2. Detects zombie processes (defunct state)
    3. Attempts to reconnect the Prisma client if the engine is unhealthy
    4. Logs health status for observability

    Usage:
        monitor = PrismaEngineHealthMonitor(
            check_interval_seconds=30,
            prisma_client=prisma_client
        )
        await monitor.start()
        # ... application runs ...
        await monitor.stop()
    """

    # Default interval between health checks (in seconds)
    DEFAULT_CHECK_INTERVAL_SECONDS = 30

    # Maximum consecutive failures before triggering reconnection
    MAX_CONSECUTIVE_FAILURES = 1  # React immediately on zombie detection

    # Process names to look for (Prisma query engine binary names)
    PRISMA_PROCESS_PATTERNS = [
        "query-engine",
        "prisma-query-engine",
        "query-engine-de",  # Debian variant
        "query-engine-linux",
    ]

    def __init__(
        self,
        check_interval_seconds: int = DEFAULT_CHECK_INTERVAL_SECONDS,
        prisma_client: Optional[Any] = None,
        on_unhealthy_callback: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize the Prisma engine health monitor.

        Args:
            check_interval_seconds: How often to check the engine health (default: 30s)
            prisma_client: The Prisma client instance to reconnect if needed
            on_unhealthy_callback: Optional callback to invoke when engine is unhealthy
        """
        self._check_interval = check_interval_seconds
        self._prisma_client = prisma_client
        self._on_unhealthy_callback = on_unhealthy_callback
        self._monitor_task: Optional[asyncio.Task] = None
        self._consecutive_failures = 0
        self._is_running = False
        self._prisma_engine_pid: Optional[int] = None
        self._reconnection_lock = asyncio.Lock()
        self._last_healthy_pid: Optional[int] = None

    def set_prisma_client(self, prisma_client: Any):
        """
        Set or update the Prisma client reference.

        Args:
            prisma_client: The Prisma client instance
        """
        self._prisma_client = prisma_client
        _log_info(f"Prisma client reference set: {type(prisma_client).__name__}")

    def _find_prisma_engine_processes(self) -> List[Tuple[int, str, str]]:
        """
        Find all Prisma query engine processes.

        Returns:
            List of tuples (pid, state, command) for matching processes.
        """
        processes = []
        try:
            # Use ps to get PID, state, and command
            result = subprocess.run(
                ["ps", "-eo", "pid,stat,comm,args"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                # Skip header line
                for line in lines[1:]:
                    parts = line.split(None, 3)
                    if len(parts) >= 3:
                        try:
                            pid = int(parts[0])
                            state = parts[1]
                            comm = parts[2]
                            args = parts[3] if len(parts) > 3 else comm

                            # Check if this is a Prisma query engine process
                            for pattern in self.PRISMA_PROCESS_PATTERNS:
                                if pattern in comm.lower() or pattern in args.lower():
                                    processes.append((pid, state, args))
                                    break
                        except (ValueError, IndexError):
                            continue
        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            _log_debug(f"Error finding Prisma engine processes: {e}")
        except FileNotFoundError:
            _log_debug("ps command not found")

        return processes

    def _is_zombie_state(self, state: str) -> bool:
        """
        Check if a process state indicates zombie.

        Args:
            state: Process state string from ps (e.g., "Z", "Zs", "Z+")

        Returns:
            True if the state indicates a zombie process.
        """
        # Z = zombie on both Linux and macOS
        return "Z" in state.upper()

    def _reap_zombie_processes(self) -> int:
        """
        Attempt to reap any zombie child processes.

        This is necessary because the Prisma query engine is spawned as a child
        of the Python process, and zombies need to be reaped by waitpid.

        Returns:
            Number of zombie processes reaped.
        """
        reaped_count = 0
        try:
            # Try to reap any zombie children with WNOHANG (non-blocking)
            while True:
                try:
                    pid, status = os.waitpid(-1, os.WNOHANG)
                    if pid == 0:
                        # No more zombies to reap
                        break
                    reaped_count += 1
                    _log_info(f"Reaped zombie child process (PID: {pid}, status: {status})")
                except ChildProcessError:
                    # No child processes
                    break
        except Exception as e:
            _log_debug(f"Error reaping zombie processes: {e}")

        return reaped_count

    async def _check_prisma_health(self) -> Tuple[bool, Optional[int]]:
        """
        Check if the Prisma engine is healthy.

        Returns:
            Tuple of (is_healthy, zombie_pid or None)
        """
        # First, try to reap any zombie children
        reaped = self._reap_zombie_processes()
        if reaped > 0:
            _log_info(f"Reaped {reaped} zombie process(es)")

        # Find all Prisma engine processes
        processes = self._find_prisma_engine_processes()

        if not processes:
            _log_debug("No Prisma engine process found in process list")
            # Check if we previously had a healthy engine - if so, it's gone
            if self._last_healthy_pid is not None:
                _log_warning(
                    f"Prisma engine (previously PID: {self._last_healthy_pid}) "
                    "is no longer running!"
                )
                self._last_healthy_pid = None
                return False, None
            # No previous engine known, might be okay
            return True, None

        # Check each process for zombie state
        for pid, state, command in processes:
            self._prisma_engine_pid = pid

            if self._is_zombie_state(state):
                _log_warning(
                    f"ZOMBIE DETECTED! Prisma engine process (PID: {pid}) is in zombie state. "
                    f"State: {state}, Command: {command}"
                )
                return False, pid

            _log_debug(f"Prisma engine process (PID: {pid}) is healthy. State: {state}")
            self._last_healthy_pid = pid

        return True, None

    async def _reconnect_prisma_client(self):
        """
        Attempt to reconnect the Prisma client.

        This will disconnect and reconnect the client, which should spawn
        a new query engine process.
        """
        if self._prisma_client is None:
            _log_error("Cannot reconnect Prisma client: no client reference set")
            return

        async with self._reconnection_lock:
            _log_info("=" * 60)
            _log_info("STARTING PRISMA CLIENT RECONNECTION")
            _log_info("=" * 60)

            try:
                # Access the underlying Prisma client
                # The prisma_client is PrismaClient class from litellm which has a 'db' attribute
                # that is a PrismaWrapper wrapping the actual Prisma client
                db_wrapper = getattr(self._prisma_client, "db", None)
                _log_info(f"prisma_client type: {type(self._prisma_client).__name__}")
                _log_info(f"db_wrapper type: {type(db_wrapper).__name__ if db_wrapper else 'None'}")

                if db_wrapper is None:
                    _log_error("No db attribute found on prisma_client")
                    return

                # Get the original Prisma client from the wrapper
                if hasattr(db_wrapper, "_original_prisma"):
                    original_prisma = db_wrapper._original_prisma
                    _log_info(f"Got _original_prisma: {type(original_prisma).__name__}")
                else:
                    original_prisma = db_wrapper
                    _log_info(f"Using db_wrapper directly: {type(original_prisma).__name__}")

                # Log current state
                _log_info(f"Prisma client is_connected: {getattr(original_prisma, 'is_connected', lambda: 'unknown')()}")

                # Step 1: Disconnect the current client
                _log_info("Step 1: Disconnecting Prisma client...")
                try:
                    if hasattr(original_prisma, "disconnect"):
                        await original_prisma.disconnect()
                        _log_info("Prisma client disconnected successfully")
                    else:
                        _log_warning("No disconnect method found on Prisma client")
                except Exception as e:
                    _log_warning(f"Error during disconnect (may be expected): {e}")

                # Step 2: Reap any zombie processes
                _log_info("Step 2: Reaping zombie processes...")
                reaped = self._reap_zombie_processes()
                _log_info(f"Reaped {reaped} zombie process(es)")

                # Step 3: Wait for cleanup
                _log_info("Step 3: Waiting for cleanup (2 seconds)...")
                await asyncio.sleep(2)

                # Check process state after disconnect
                processes_after_disconnect = self._find_prisma_engine_processes()
                _log_info(f"Processes after disconnect: {processes_after_disconnect}")

                # Step 4: Reconnect
                _log_info("Step 4: Reconnecting Prisma client...")
                try:
                    if hasattr(original_prisma, "connect"):
                        await original_prisma.connect()
                        _log_info("Prisma client reconnected successfully!")
                    else:
                        _log_error("No connect method found on Prisma client")
                        raise AttributeError("No connect method")
                except Exception as e:
                    _log_error(f"Error during reconnect: {e}")
                    _log_info("Attempting to create a completely new Prisma client...")
                    await self._create_new_prisma_client(db_wrapper)

                # Step 5: Verify new engine is running
                _log_info("Step 5: Verifying new engine process...")
                await asyncio.sleep(1)
                processes_after_reconnect = self._find_prisma_engine_processes()
                _log_info(f"Processes after reconnect: {processes_after_reconnect}")

                if processes_after_reconnect:
                    for pid, state, cmd in processes_after_reconnect:
                        if not self._is_zombie_state(state):
                            _log_info(f"SUCCESS! New healthy Prisma engine running (PID: {pid})")
                            self._last_healthy_pid = pid
                            break
                    else:
                        _log_error("All found processes are still zombies!")
                else:
                    _log_warning("No Prisma engine process found after reconnect")

                _log_info("=" * 60)
                _log_info("PRISMA CLIENT RECONNECTION COMPLETE")
                _log_info("=" * 60)

            except Exception as e:
                _log_error(f"Failed to reconnect Prisma client: {e}")
                import traceback
                _log_error(traceback.format_exc())

    async def _create_new_prisma_client(self, db_wrapper: Any):
        """
        Create a completely new Prisma client instance.

        This is a fallback when simple reconnect fails.

        Args:
            db_wrapper: The PrismaWrapper instance to update
        """
        try:
            from prisma import Prisma  # type: ignore

            _log_info("Creating completely new Prisma client instance...")

            # Create new Prisma instance
            new_prisma = Prisma()
            _log_info("New Prisma instance created, connecting...")

            await new_prisma.connect()
            _log_info("New Prisma instance connected!")

            # Update the wrapper's reference
            if hasattr(db_wrapper, "_original_prisma"):
                old_prisma = db_wrapper._original_prisma
                db_wrapper._original_prisma = new_prisma
                _log_info("Updated wrapper with new Prisma client")

                # Try to clean up old instance
                try:
                    if hasattr(old_prisma, "disconnect"):
                        await old_prisma.disconnect()
                except Exception:
                    pass
            else:
                _log_warning("Could not update wrapper - no _original_prisma attribute")

        except Exception as e:
            _log_error(f"Failed to create new Prisma client: {e}")
            import traceback
            _log_error(traceback.format_exc())

    async def _handle_unhealthy_engine(self, zombie_pid: Optional[int] = None):
        """
        Handle an unhealthy Prisma engine by attempting recovery.

        Args:
            zombie_pid: The PID of the zombie process, if known
        """
        self._consecutive_failures += 1
        _log_warning(
            f"Prisma engine unhealthy! Consecutive failures: {self._consecutive_failures}"
        )

        if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
            _log_error(
                f"Prisma engine has been unhealthy for {self._consecutive_failures} "
                f"consecutive check(s). TRIGGERING RECOVERY..."
            )

            # Reap the zombie process first
            if zombie_pid is not None:
                _log_info(f"Attempting to reap zombie PID: {zombie_pid}")
                self._reap_zombie_processes()

            # Attempt to reconnect the Prisma client
            await self._reconnect_prisma_client()

            # Invoke the callback if provided
            if self._on_unhealthy_callback:
                try:
                    _log_info("Invoking unhealthy callback...")
                    result = self._on_unhealthy_callback()
                    # Handle async callbacks
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    _log_error(f"Error in unhealthy callback: {e}")

            # Reset consecutive failures after recovery attempt
            self._consecutive_failures = 0

    async def _health_check_loop(self):
        """
        Main health check loop that runs in the background.
        """
        _log_info(
            f"Prisma engine health monitor STARTED. "
            f"Check interval: {self._check_interval}s"
        )

        # Initial check
        processes = self._find_prisma_engine_processes()
        if processes:
            for pid, state, cmd in processes:
                _log_info(f"Initial Prisma engine found: PID={pid}, State={state}")
                if not self._is_zombie_state(state):
                    self._last_healthy_pid = pid
        else:
            _log_info("No Prisma engine process found on startup")

        while self._is_running:
            try:
                is_healthy, zombie_pid = await self._check_prisma_health()

                if is_healthy:
                    self._consecutive_failures = 0
                else:
                    await self._handle_unhealthy_engine(zombie_pid)

            except asyncio.CancelledError:
                _log_info("Prisma engine health monitor cancelled")
                break
            except Exception as e:
                _log_error(f"Error in Prisma engine health check: {e}")
                import traceback
                _log_error(traceback.format_exc())

            # Wait for the next check interval
            try:
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break

        _log_info("Prisma engine health monitor STOPPED")

    async def start(self):
        """
        Start the background health monitoring task.
        """
        if self._is_running:
            _log_debug("Prisma engine health monitor is already running")
            return

        self._is_running = True
        self._monitor_task = asyncio.create_task(self._health_check_loop())
        _log_info("Prisma engine health monitor task CREATED and RUNNING")

    async def stop(self):
        """
        Stop the background health monitoring task gracefully.
        """
        if not self._is_running:
            return

        self._is_running = False

        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        _log_info("Prisma engine health monitor stopped")

    @property
    def is_running(self) -> bool:
        """Check if the monitor is currently running."""
        return self._is_running

    @property
    def consecutive_failures(self) -> int:
        """Get the current count of consecutive health check failures."""
        return self._consecutive_failures

    @property
    def prisma_engine_pid(self) -> Optional[int]:
        """Get the last known PID of the Prisma engine process."""
        return self._prisma_engine_pid


# Global instance for use in the proxy
_prisma_health_monitor: Optional[PrismaEngineHealthMonitor] = None


async def start_prisma_health_monitor(
    check_interval_seconds: int = PrismaEngineHealthMonitor.DEFAULT_CHECK_INTERVAL_SECONDS,
    prisma_client: Optional[Any] = None,
    on_unhealthy_callback: Optional[Callable[[], None]] = None,
) -> PrismaEngineHealthMonitor:
    """
    Start the global Prisma engine health monitor.

    Args:
        check_interval_seconds: How often to check the engine health
        prisma_client: The Prisma client instance to reconnect if needed
        on_unhealthy_callback: Optional callback to invoke when engine is unhealthy

    Returns:
        The started PrismaEngineHealthMonitor instance
    """
    global _prisma_health_monitor

    if _prisma_health_monitor is not None and _prisma_health_monitor.is_running:
        _log_debug("Prisma health monitor already running, returning existing instance")
        # Update the prisma client reference if provided
        if prisma_client is not None:
            _prisma_health_monitor.set_prisma_client(prisma_client)
        return _prisma_health_monitor

    _log_info(f"Creating Prisma health monitor with interval={check_interval_seconds}s")
    _prisma_health_monitor = PrismaEngineHealthMonitor(
        check_interval_seconds=check_interval_seconds,
        prisma_client=prisma_client,
        on_unhealthy_callback=on_unhealthy_callback,
    )
    await _prisma_health_monitor.start()
    return _prisma_health_monitor


async def stop_prisma_health_monitor():
    """
    Stop the global Prisma engine health monitor.
    """
    global _prisma_health_monitor

    if _prisma_health_monitor is not None:
        await _prisma_health_monitor.stop()
        _prisma_health_monitor = None


def get_prisma_health_monitor() -> Optional[PrismaEngineHealthMonitor]:
    """
    Get the global Prisma engine health monitor instance.

    Returns:
        The PrismaEngineHealthMonitor instance, or None if not started.
    """
    return _prisma_health_monitor
