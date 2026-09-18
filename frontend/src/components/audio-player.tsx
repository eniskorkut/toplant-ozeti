"use client";

import { type RefObject, useState } from "react";

import { formatDuration } from "@/lib/format";
import { PauseIcon, PlayIcon, VolumeIcon } from "@/lib/icons";

type AudioPlayerProps = {
  audioRef: RefObject<HTMLAudioElement | null>;
  src: string;
  label?: string;
};

/**
 * Clean controls around a real HTMLAudioElement. The parent owns the ref, so
 * transcript timestamp seeking keeps using the exact same audio element.
 */
export function AudioPlayer({ audioRef, src, label = "Toplantı kaydı" }: AudioPlayerProps) {
  const [playing, setPlaying] = useState(false);
  const [current, setCurrent] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);

  const toggle = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (playing) {
      audio.pause();
    } else {
      void audio.play().catch(() => undefined);
    }
  };

  return (
    <div className="rounded-xl bg-zinc-500/5 p-2.5">
      <audio
        ref={audioRef}
        src={src}
        preload="metadata"
        className="hidden"
        aria-label={label}
        onLoadedMetadata={(event) => {
          const value = event.currentTarget.duration;
          setDuration(Number.isFinite(value) ? value : 0);
        }}
        onTimeUpdate={(event) => setCurrent(event.currentTarget.currentTime)}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
      />

      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={toggle}
          aria-label={playing ? "Duraklat" : "Oynat"}
          className="grid size-10 shrink-0 place-items-center rounded-full bg-zinc-900 text-white transition-transform duration-160 ease-out active:scale-[0.96] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-zinc-500 dark:bg-zinc-100 dark:text-zinc-900"
        >
          {playing ? <PauseIcon className="size-4" /> : <PlayIcon className="ml-0.5 size-4" />}
        </button>

        <span className="shrink-0 font-mono text-xs tabular-nums text-zinc-600 dark:text-zinc-400">
          {formatDuration(current)} / {formatDuration(duration)}
        </span>

        <input
          type="range"
          min={0}
          max={duration || 0}
          step={0.1}
          value={Math.min(current, duration || 0)}
          onChange={(event) => {
            const audio = audioRef.current;
            if (!audio) return;
            const next = Number(event.target.value);
            audio.currentTime = next;
            setCurrent(next);
          }}
          aria-label="Ses konumu"
          className="h-1.5 min-w-0 flex-1 cursor-pointer appearance-none rounded-full bg-zinc-500/20 accent-zinc-900 dark:accent-zinc-100"
        />

        <span className="hidden shrink-0 items-center gap-2 sm:flex">
          <VolumeIcon className="size-4 text-zinc-500 dark:text-zinc-400" />
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={volume}
            onChange={(event) => {
              const audio = audioRef.current;
              const next = Number(event.target.value);
              setVolume(next);
              if (audio) audio.volume = next;
            }}
            aria-label="Ses düzeyi"
            className="h-1.5 w-20 cursor-pointer appearance-none rounded-full bg-zinc-500/20 accent-zinc-900 dark:accent-zinc-100"
          />
        </span>
      </div>
    </div>
  );
}
