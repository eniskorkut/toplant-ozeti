import "@testing-library/jest-dom/vitest";

import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

afterEach(() => {
  cleanup();
});

// jsdom does not implement media playback: provide deterministic stand-ins so the
// seek behaviour can be asserted.
const mediaState = new WeakMap<HTMLMediaElement, number>();

Object.defineProperty(window.HTMLMediaElement.prototype, "currentTime", {
  configurable: true,
  get(this: HTMLMediaElement) {
    return mediaState.get(this) ?? 0;
  },
  set(this: HTMLMediaElement, value: number) {
    mediaState.set(this, value);
  },
});

Object.defineProperty(window.HTMLMediaElement.prototype, "play", {
  configurable: true,
  value: vi.fn().mockResolvedValue(undefined),
});

Object.defineProperty(window.HTMLMediaElement.prototype, "pause", {
  configurable: true,
  value: vi.fn(),
});
