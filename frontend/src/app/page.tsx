import { BrowserCapabilities } from "@/components/browser-capabilities";
import { MeetingHistory } from "@/components/meeting-history";
import { Pipeline } from "@/components/pipeline";
import { RecordingPanel } from "@/components/recording-panel";
import { ChevronDownIcon } from "@/lib/icons";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col items-center bg-background px-4 py-10 sm:px-6 sm:py-14">
      <main className="flex w-full max-w-2xl flex-col gap-8">
        <header className="enter">
          <h1 className="text-2xl font-semibold tracking-tight text-zinc-900 sm:text-3xl dark:text-zinc-50">
            Toplantılar
          </h1>
          <p className="mt-1.5 text-sm text-zinc-600 dark:text-zinc-400">
            Toplantılarınızı kaydedin, konuşmacılara ayırın ve transkripte dönüştürün.
          </p>
        </header>

        <RecordingPanel />

        <MeetingHistory />

        <section className="enter enter-4">
          <details className="surface group rounded-2xl px-4 py-3">
            <summary className="flex cursor-pointer list-none items-center justify-between gap-3 text-sm font-medium tracking-tight text-zinc-900 transition-colors duration-150 ease-out hover:text-zinc-950 dark:text-zinc-100 dark:hover:text-zinc-50">
              Nasıl çalışır ve tarayıcı desteği
              <ChevronDownIcon className="size-4 text-zinc-500 transition-transform duration-150 ease-out group-open:rotate-180 dark:text-zinc-400" />
            </summary>
            <div className="mt-4 space-y-6 pb-2">
              <BrowserCapabilities />
              <Pipeline />
            </div>
          </details>
        </section>

        <footer className="enter enter-4 border-t border-zinc-950/5 pt-4 dark:border-white/10">
          <p className="text-xs text-zinc-500 dark:text-zinc-400">
            Kayıtlar bu cihazda saklanır. Yerel yöntemde ses cihazdan çıkmaz; ElevenLabs
            seçilirse ses yalnızca transkripsiyon için ElevenLabs&apos;a gönderilir. Mikrofon
            erişimi güvenli bağlam gerektirir: geliştirmede{" "}
            <span className="font-mono">localhost</span>, üretimde HTTPS.
          </p>
        </footer>
      </main>
    </div>
  );
}
