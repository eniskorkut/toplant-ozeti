# Meeting Intelligence — Frontend

Next.js (App Router) web client for the Meeting Intelligence application.

- TypeScript, Tailwind CSS, ESLint
- `src/app` — routes; `src/lib` — browser helpers (see `src/lib/browser-media.ts`)
- Microphone capture uses standard browser APIs only (`MediaDevices`, `MediaRecorder`).
  Recording is **not implemented yet**; the current page only reports capability detection.

## Development

```bash
npm install
npm run dev     # http://localhost:3000
npm run lint
npm run build
npx tsc --noEmit
```

The backend runs separately on `http://127.0.0.1:8000` (see `../backend/README.md`).
See the repository root `README.md` for the architecture overview.
