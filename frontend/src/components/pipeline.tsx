type Step = {
  name: string;
  detail: string;
  status: "ready" | "planned";
};

const browserSteps: Step[] = [
  { name: "Microphone", detail: "MediaDevices.getUserMedia", status: "planned" },
  { name: "MediaRecorder", detail: "browser audio encoding", status: "planned" },
  { name: "Audio upload", detail: "multipart upload to the API", status: "planned" },
];

const backendSteps: Step[] = [
  { name: "FastAPI", detail: "service skeleton + /health", status: "ready" },
  { name: "ffmpeg", detail: "availability check only", status: "ready" },
  { name: "whisper.cpp", detail: "CPU speech-to-text", status: "planned" },
  { name: "sherpa-onnx", detail: "CPU speaker diarization", status: "planned" },
  { name: "Speaker-labelled transcript", detail: "merged output", status: "planned" },
  { name: "External LLM", detail: "OpenAI-compatible analysis API", status: "planned" },
];

function StatusChip({ status }: { status: Step["status"] }) {
  if (status === "ready") {
    return (
      <span className="rounded-full bg-emerald-500/10 px-2 py-0.5 text-xs text-emerald-700 dark:text-emerald-400">
        scaffolded
      </span>
    );
  }

  return (
    <span className="rounded-full bg-zinc-500/10 px-2 py-0.5 text-xs text-zinc-600 dark:text-zinc-400">
      planned
    </span>
  );
}

function StepList({ title, steps }: { title: string; steps: Step[] }) {
  return (
    <div className="surface flex-1 rounded-2xl p-2">
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
              <StatusChip status={step.status} />
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}

export function Pipeline() {
  return (
    <section aria-labelledby="pipeline" className="enter enter-4">
      <h2
        id="pipeline"
        className="mb-3 text-sm font-medium tracking-tight text-zinc-900 dark:text-zinc-100"
      >
        Pipeline
      </h2>
      <div className="flex flex-col gap-4 sm:flex-row">
        <StepList title="Browser" steps={browserSteps} />
        <StepList title="Backend" steps={backendSteps} />
      </div>
    </section>
  );
}
