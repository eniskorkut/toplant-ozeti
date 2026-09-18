import Link from "next/link";

import { MeetingDetail } from "@/components/meeting-detail";

export default async function MeetingPage({ params }: PageProps<"/meetings/[id]">) {
  const { id } = await params;

  return (
    <div className="flex min-h-screen flex-col items-center bg-background px-6 py-12">
      <main className="w-full max-w-3xl">
        <Link
          href="/"
          className="text-xs text-zinc-500 underline underline-offset-4 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100"
        >
          ← Yeni kayıt
        </Link>
        <div className="mt-4">
          <MeetingDetail meetingId={id} />
        </div>
      </main>
    </div>
  );
}
