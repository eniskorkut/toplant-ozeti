"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import {
  ApiError,
  askMeetingQuestion,
  clearMeetingChat,
  getMeetingChat,
  type MeetingChatMessage,
} from "@/lib/api";

const UNAVAILABLE_MESSAGE =
  "Soru-cevap için uygun bir LLM sağlayıcısı yapılandırılmamış.";

function scrollToBottom(node: HTMLDivElement | null) {
  if (!node) return;
  node.scrollTop = node.scrollHeight;
}

/**
 * Grounded meeting Q&A: questions are answered by the configured LLM using only
 * the meeting transcript (and its analysis summary). The conversation is persisted
 * by the backend, so it is reloaded on mount and survives a page refresh.
 */
export function MeetingChat({ meetingId }: { meetingId: string }) {
  const [messages, setMessages] = useState<MeetingChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [clearing, setClearing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const listRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    getMeetingChat(meetingId)
      .then((conversation) => {
        if (!cancelled) setMessages(conversation.messages);
      })
      .catch(() => {
        // An empty history is a normal state; the panel stays usable.
        if (!cancelled) setMessages([]);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [meetingId]);

  useEffect(() => {
    scrollToBottom(listRef.current);
  }, [messages, busy]);

  const send = useCallback(async () => {
    const question = input.trim();
    if (!question || busy) return;

    setInput("");
    setBusy(true);
    setError(null);
    try {
      const result = await askMeetingQuestion(meetingId, { question });
      setMessages(result.messages);
    } catch (failure) {
      if (failure instanceof ApiError && failure.status === 503) {
        setError(UNAVAILABLE_MESSAGE);
      } else {
        setError(failure instanceof Error ? failure.message : "Cevap alınamadı.");
      }
    } finally {
      setBusy(false);
    }
  }, [busy, input, meetingId]);

  const clear = useCallback(async () => {
    if (clearing) return;
    setClearing(true);
    setError(null);
    try {
      await clearMeetingChat(meetingId);
      setMessages([]);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Sohbet temizlenemedi.");
    } finally {
      setClearing(false);
    }
  }, [clearing, meetingId]);

  return (
    <div className="flex flex-col">
      <div
        ref={listRef}
        aria-live="polite"
        className="max-h-[28rem] min-h-[10rem] space-y-3 overflow-y-auto pr-1"
      >
        {loading ? (
          <p className="text-xs text-zinc-500 dark:text-zinc-400">Sohbet yükleniyor…</p>
        ) : messages.length === 0 ? (
          <p className="text-xs text-zinc-500 dark:text-zinc-400">
            Toplantıyla ilgili sorular sorun. Yanıtlar yalnızca bu toplantının
            transkriptine dayanır; bilgi yoksa model bunu açıkça söyler.
          </p>
        ) : (
          messages.map((message, index) => (
            <div
              key={index}
              className={message.role === "user" ? "flex justify-end" : "flex justify-start"}
            >
              <p
                className={
                  message.role === "user"
                    ? "max-w-[85%] rounded-2xl rounded-br-sm bg-zinc-900 px-3 py-2 text-sm whitespace-pre-wrap text-white dark:bg-zinc-100 dark:text-zinc-900"
                    : "max-w-[85%] rounded-2xl rounded-bl-sm bg-zinc-500/10 px-3 py-2 text-sm whitespace-pre-wrap text-zinc-900 dark:text-zinc-100"
                }
              >
                {message.content}
              </p>
            </div>
          ))
        )}
        {busy ? (
          <p className="text-xs text-zinc-500 dark:text-zinc-400">Yanıt hazırlanıyor…</p>
        ) : null}
      </div>

      {error ? (
        <p
          role="alert"
          className="mt-3 rounded-xl bg-amber-500/10 px-3 py-2 text-xs text-amber-800 dark:text-amber-300"
        >
          {error}
        </p>
      ) : null}

      <form
        onSubmit={(event) => {
          event.preventDefault();
          void send();
        }}
        className="mt-3 flex items-end gap-2"
      >
        <textarea
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              void send();
            }
          }}
          rows={2}
          placeholder="Bir soru yazın…"
          aria-label="Toplantıyla ilgili soru"
          className="min-h-0 flex-1 resize-none rounded-xl border border-zinc-950/10 bg-transparent px-3 py-2 text-sm text-zinc-900 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-zinc-500 dark:border-white/15 dark:text-zinc-100"
        />
        <button
          type="submit"
          disabled={busy || input.trim().length === 0}
          className="inline-flex shrink-0 items-center rounded-xl bg-zinc-900 px-3.5 py-2 text-sm font-medium text-white transition-transform duration-160 ease-out hover:bg-zinc-800 active:scale-[0.96] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-zinc-500 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-white"
        >
          {busy ? "Gönderiliyor…" : "Gönder"}
        </button>
      </form>

      {messages.length > 0 ? (
        <button
          type="button"
          onClick={() => void clear()}
          disabled={clearing}
          className="mt-2 self-start text-xs text-zinc-500 underline decoration-dotted underline-offset-4 transition-colors duration-150 ease-out hover:text-zinc-900 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-zinc-500 disabled:cursor-not-allowed disabled:opacity-60 dark:text-zinc-400 dark:hover:text-zinc-100"
        >
          {clearing ? "Temizleniyor…" : "Sohbeti temizle"}
        </button>
      ) : null}
    </div>
  );
}
