"""Keyboard input providers and shared listener."""

from __future__ import annotations

import sys
import threading
from collections import deque

import numpy as np
from sshkeyboard import listen_keyboard

from holosoma_inference.inputs.api.base import InputProvider
from holosoma_inference.inputs.api.commands import StateCommand, VelCmd

# ---------------------------------------------------------------------------
# Keyboard command mappings (discrete commands)
# ---------------------------------------------------------------------------

KEYBOARD_COMMANDS: dict[str, StateCommand] = {
    "]": StateCommand.START,
    "o": StateCommand.STOP,
    "i": StateCommand.INIT,
    "v": StateCommand.KP_DOWN_FINE,
    "b": StateCommand.KP_UP_FINE,
    "f": StateCommand.KP_DOWN,
    "g": StateCommand.KP_UP,
    "r": StateCommand.KP_RESET,
    "=": StateCommand.STAND_TOGGLE,
    "z": StateCommand.ZERO_VELOCITY,
    "m": StateCommand.START_MOTION_CLIP,
    "c": StateCommand.TOGGLE_RECORDING,
    "x": StateCommand.SWITCH_MODE,
    **{str(n): StateCommand[f"SWITCH_POLICY_{n}"] for n in range(1, 10)},
}

# ---------------------------------------------------------------------------
# Keyboard velocity mappings (continuous velocity increments)
#
# Each entry maps a keycode to (array_index, column, delta):
#   array_index 0 = lin_vel, 1 = ang_vel
#   column = which element within that array
#   delta = increment per keypress
# ---------------------------------------------------------------------------

KEYBOARD_VELOCITY_LOCOMOTION: dict[str, tuple[int, int, float]] = {
    "w": (0, 0, +0.1),  # lin_vel[0, 0] += 0.1
    "s": (0, 0, -0.1),  # lin_vel[0, 0] -= 0.1
    "a": (0, 1, +0.1),  # lin_vel[0, 1] += 0.1
    "d": (0, 1, -0.1),  # lin_vel[0, 1] -= 0.1
    "q": (1, 0, -0.1),  # ang_vel[0, 0] -= 0.1
    "e": (1, 0, +0.1),  # ang_vel[0, 0] += 0.1
}

# ---------------------------------------------------------------------------
# Keyboard direction mappings (discrete, absolute headings)
#
# For policies trained on a direction *class* instead of a velocity vector.
# Each press selects a heading outright, so one tap responds immediately —
# unlike the velocity map above, where a key nudges an accumulator and a
# direction reversal takes several presses to cross zero.
# ---------------------------------------------------------------------------

KEYBOARD_DIRECTION_COMMANDS: dict[str, StateCommand] = {
    "w": StateCommand.MOVE_FORWARD,
    "s": StateCommand.MOVE_BACKWARD,
    "a": StateCommand.MOVE_LEFT_45,
    "d": StateCommand.MOVE_RIGHT_45,
    "q": StateCommand.MOVE_LEFT_90,
    "e": StateCommand.MOVE_RIGHT_90,
}


# Key-repeat delays for a held key, in seconds. sshkeyboard's own defaults
# (0.75 / 0.05) put a long stall before the first repeat, which feels like
# dropped input when steering a robot.
REPEAT_DELAY_FIRST = 0.15
REPEAT_DELAY_REST = 0.05


class _KeyboardListenerThread(threading.Thread):
    """Daemon thread that broadcasts keypresses to subscriber queues.

    ``start()`` is idempotent and returns whether the listener is active.
    """

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._subscribers: list[deque[str]] = []

    def subscribe(self) -> deque[str]:
        q: deque[str] = deque()
        self._subscribers.append(q)
        return q

    def start(self) -> bool:
        """Start the thread if not already running. Returns True if active."""
        if self.is_alive():
            return True
        if not sys.stdin.isatty():
            return False
        super().start()
        return True

    def run(self) -> None:
        def on_press(keycode):
            for q in self._subscribers:
                q.append(keycode)

        try:
            # sshkeyboard defaults to delay_second_char=0.75, so holding a key
            # emits one press then stalls for 3/4 of a second before repeating —
            # which reads as the robot ignoring input. Shorten both repeat delays
            # so a held key streams smoothly; discrete taps are unaffected.
            listener = listen_keyboard(
                on_press=on_press,
                delay_second_char=REPEAT_DELAY_FIRST,
                delay_other_chars=REPEAT_DELAY_REST,
            )
            listener.start()
            listener.join()
        except OSError:
            pass
        except TypeError:
            # Older sshkeyboard without the delay kwargs: fall back to defaults
            # rather than losing keyboard control entirely.
            listener = listen_keyboard(on_press=on_press)
            listener.start()
            listener.join()


# Module-level singleton — one listener thread shared across all KeyboardInput instances.
_listener: _KeyboardListenerThread | None = None


def get_keyboard_listener() -> _KeyboardListenerThread:
    """Return the module-level keyboard listener, creating it on first call."""
    global _listener  # noqa: PLW0603
    if _listener is None:
        _listener = _KeyboardListenerThread()
    return _listener


class KeyboardInput(InputProvider):
    """Unified keyboard device implementing both velocity and command protocols.

    Subscribes to a single keyboard queue. ``poll_velocity()`` drains the queue,
    applies velocity key increments, and buffers any command matches.
    ``poll_commands()`` returns the buffered commands.

    If no velocity_keys mapping is provided, ``poll_velocity()`` returns None
    but still drains the queue and buffers commands.
    """

    def __init__(
        self,
        queue: deque[str],
        velocity_keys: dict[str, tuple[int, int, float]] | None = None,
        direction_keys: dict[str, StateCommand] | None = None,
    ) -> None:
        self._mapping = dict(KEYBOARD_COMMANDS)
        # Direction keys shadow the shared command table so a policy can rebind
        # w/a/s/d/q/e without those keys also firing their default meaning.
        if direction_keys:
            self._mapping.update(direction_keys)
        self._queue = queue
        # A key cannot be both a velocity increment and a direction command; when
        # a policy asks for directions, they win and the accumulator is unused.
        self._velocity_keys = None if direction_keys else velocity_keys
        self._lin_vel = np.zeros((1, 2))
        self._ang_vel = np.zeros((1, 1))
        self._pending_commands: list[StateCommand] = []

    @classmethod
    def create(
        cls,
        velocity_keys: dict[str, tuple[int, int, float]] | None = None,
        direction_keys: dict[str, StateCommand] | None = None,
    ) -> KeyboardInput:
        """Create a KeyboardInput subscribed to the module-level keyboard listener."""
        listener = get_keyboard_listener()
        queue = listener.subscribe()
        return cls(queue, velocity_keys, direction_keys)

    def start(self) -> None:
        pass  # Listener already started by factory / create()

    def _drain_queue(self) -> None:
        """Process all pending keypresses into velocity state and command buffer."""
        while True:
            try:
                keycode = self._queue.popleft()
            except IndexError:
                break
            action = self._velocity_keys.get(keycode) if self._velocity_keys else None
            if action is not None:
                array_idx, col, delta = action
                if array_idx == 0:
                    self._lin_vel[0, col] += delta
                else:
                    self._ang_vel[0, col] += delta
                continue
            cmd = self._mapping.get(keycode)
            if cmd is not None:
                self._pending_commands.append(cmd)

    def poll_velocity(self) -> VelCmd | None:
        self._drain_queue()
        if not self._velocity_keys:
            return None
        return VelCmd(
            (float(self._lin_vel[0, 0]), float(self._lin_vel[0, 1])),
            float(self._ang_vel[0, 0]),
        )

    def zero(self) -> None:
        """Reset velocity state to zero."""
        self._lin_vel[:] = 0.0
        self._ang_vel[:] = 0.0

    def poll_commands(self) -> list[StateCommand]:
        self._drain_queue()
        commands = self._pending_commands
        self._pending_commands = []
        return commands
