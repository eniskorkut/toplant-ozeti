import { BrowserCapabilities } from "@/components/browser-capabilities";
import { Pipeline } from "@/components/pipeline";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col items-center bg-background px-6 py-16 sm:py-24">
      <main className="flex w-full max-w-2xl flex-col gap-10">
        <header className="enter">
          <span className="inline-flex items-center gap-2 rounded-full bg-zinc-500/10 px-3 py-1 text-xs text-zinc-600 dark:text-zinc-400">
            <span className="size-1.5 rounded-full bg-amber-500" aria-hidden="true" />
            Scaffolding — recording not implemented yet
          </span>

          <h1 className="mt-5 text-4xl font-semibold tracking-tight text-zinc-900 sm:text-5xl dark:text-zinc-50">
            Meeting Intelligence
          </h1>
          <p className="mt-3 text-lg text-zinc-600 dark:text-zinc-400">
            Local-first meeting transcription
          </p>
        </header>

        <BrowserCapabilities />

        <Pipeline />

        <footer className="enter enter-4 border-t border-zinc-950/5 pt-4 dark:border-white/10">
          <p className="text-xs text-zinc-500 dark:text-zinc-400">
            Runs entirely in the browser. Microphone access requires a secure context —{" "}
            <span className="font-mono">localhost</span> during development, HTTPS in production.
          </p>
        </footer>
      </main>
    </div>
  );
}
