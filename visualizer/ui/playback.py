from __future__ import annotations

from dataclasses import dataclass, replace

from visualizer.model import FrameBundle


@dataclass(frozen=True)
class PlaybackState:
    """Dependency-free playback state shared by future UI widgets."""

    current_round: int = 0
    is_playing: bool = False
    speed: float = 1.0
    loop_enabled: bool = False


class PlaybackController:
    """Small playback coordinator independent of the GUI toolkit."""

    def __init__(self) -> None:
        self._bundles: tuple[FrameBundle, ...] = ()
        self._state = PlaybackState()

    def load_bundles(self, bundles: tuple[FrameBundle, ...]) -> None:
        self._bundles = bundles
        self._state = replace(self._state, current_round=0, is_playing=False)

    def bundles(self) -> tuple[FrameBundle, ...]:
        return self._bundles

    def state(self) -> PlaybackState:
        return self._state

    def current_bundle(self) -> FrameBundle | None:
        if not self._bundles:
            return None
        return self._bundles[self._state.current_round]

    def set_round(self, round_index: int) -> None:
        if not self._bundles:
            return
        current_round = max(0, min(int(round_index), len(self._bundles) - 1))
        self._state = replace(self._state, current_round=current_round)

    def step_forward(self) -> None:
        if not self._bundles:
            return
        if self._state.current_round >= len(self._bundles) - 1:
            if self._state.loop_enabled:
                self.set_round(0)
            else:
                self.pause()
            return
        self.set_round(self._state.current_round + 1)

    def step_backward(self) -> None:
        self.set_round(self._state.current_round - 1)

    def play(self) -> None:
        if self._bundles:
            self._state = replace(self._state, is_playing=True)

    def pause(self) -> None:
        self._state = replace(self._state, is_playing=False)

    def toggle_play(self) -> None:
        if self._state.is_playing:
            self.pause()
        else:
            self.play()

    def set_speed(self, speed: float) -> None:
        self._state = replace(self._state, speed=max(0.25, float(speed)))

    def set_loop_enabled(self, enabled: bool) -> None:
        self._state = replace(self._state, loop_enabled=bool(enabled))

