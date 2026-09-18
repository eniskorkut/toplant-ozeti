import { fireEvent, render, screen } from "@testing-library/react";
import { useRef } from "react";
import { describe, expect, it, vi } from "vitest";

import { AudioPlayer } from "@/components/audio-player";

function Harness() {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  return <AudioPlayer audioRef={audioRef} src="blob:test" />;
}

function audioElement(): HTMLAudioElement {
  return document.querySelector("audio") as HTMLAudioElement;
}

describe("AudioPlayer", () => {
  it("plays and pauses through one accessible button", () => {
    render(<Harness />);
    const play = vi.mocked(window.HTMLMediaElement.prototype.play);
    const pause = vi.mocked(window.HTMLMediaElement.prototype.pause);
    play.mockClear();
    pause.mockClear();

    fireEvent.click(screen.getByRole("button", { name: "Oynat" }));
    expect(play).toHaveBeenCalled();

    fireEvent.play(audioElement());
    expect(screen.getByRole("button", { name: "Duraklat" })).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Duraklat" }));
    expect(pause).toHaveBeenCalled();
  });

  it("shows current and total time from the underlying audio element", () => {
    render(<Harness />);
    const audio = audioElement();
    Object.defineProperty(audio, "duration", { configurable: true, value: 42.5 });
    fireEvent.loadedMetadata(audio);
    audio.currentTime = 12.35;
    fireEvent.timeUpdate(audio);

    expect(screen.getByText("0:12 / 0:42")).toBeTruthy();
  });

  it("seeks through the position slider", () => {
    render(<Harness />);
    const audio = audioElement();
    Object.defineProperty(audio, "duration", { configurable: true, value: 42.5 });
    fireEvent.loadedMetadata(audio);

    fireEvent.change(screen.getByRole("slider", { name: "Ses konumu" }), {
      target: { value: "30" },
    });

    expect(audio.currentTime).toBeCloseTo(30, 2);
  });

  it("adjusts volume through the volume slider", () => {
    render(<Harness />);
    const audio = audioElement();

    fireEvent.change(screen.getByRole("slider", { name: "Ses düzeyi" }), {
      target: { value: "0.4" },
    });

    expect(audio.volume).toBeCloseTo(0.4, 2);
  });
});
