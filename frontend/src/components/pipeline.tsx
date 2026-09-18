type Step = {
  name: string;
  detail: string;
};

const browserSteps: Step[] = [
  { name: "Mikrofon", detail: "MediaDevices.getUserMedia" },
  { name: "MediaRecorder", detail: "tarayıcıda ses kodlama, 1 sn'lik parçalar" },
  { name: "Ses yükleme", detail: "API'ye multipart yükleme" },
];

const backendSteps: Step[] = [
  { name: "FastAPI", detail: "servis + POST /api/recordings" },
  { name: "ffmpeg", detail: "MP3 + WAV dönüşümü (tek geçiş)" },
  { name: "whisper.cpp", detail: "yerel konuşma → metin" },
  { name: "sherpa-onnx", detail: "yerel konuşmacı ayrımı" },
  { name: "ElevenLabs Scribe v2", detail: "isteğe bağlı bulut transkripsiyonu" },
  { name: "LLM analizi", detail: "yapılandırıldıysa özet, karar ve aksiyonlar" },
];

function StepList({ title, steps }: { title: string; steps: Step[] }) {
  return (
    <div className="flex-1 rounded-xl bg-zinc-500/5 p-2">
      <div className="px-3 pt-2 pb-1">
        <h3 className="text-sm font-medium tracking-tight text-zinc-900 dark:text-zinc-100">
          {title}
        </h3>
      </div>
      <ol className="px-3 pb-2">
        {steps.map((step, index) => (
          <li key={step.name} className="flex gap-3">
            <span className="flex flex-col items-center" aria-hidden="true">
              <span className="mt-2.5 size-1.5 rounded-full bg-zinc-300 dark:bg-zinc-700" />
              {index < steps.length - 1 ? (
                <span className="w-px flex-1 bg-zinc-950/5 dark:bg-white/10" />
              ) : null}
            </span>
            <span className="flex min-w-0 flex-1 items-baseline justify-between gap-3 py-1.5">
              <span className="min-w-0">
                <span className="block truncate text-sm text-zinc-900 dark:text-zinc-100">
                  {step.name}
                </span>
                <span className="block truncate text-xs text-zinc-500 dark:text-zinc-400">
                  {step.detail}
                </span>
              </span>
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}

export function Pipeline() {
  return (
    <section aria-labelledby="pipeline">
      <h2
        id="pipeline"
        className="mb-3 text-sm font-medium tracking-tight text-zinc-900 dark:text-zinc-100"
      >
        İşleyiş
      </h2>
      <div className="flex flex-col gap-4 sm:flex-row">
        <StepList title="Tarayıcı" steps={browserSteps} />
        <StepList title="Sunucu" steps={backendSteps} />
      </div>
    </section>
  );
}
