import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { LiveTranscript, type LiveLine } from "@/components/live-transcript";

function line(overrides: Partial<LiveLine>): LiveLine {
  return {
    id: 1,
    startSeconds: 3,
    endSeconds: 5,
    text: "Merhaba",
    canonical: null,
    provisional: false,
    ...overrides,
  };
}

describe("LiveTranscript", () => {
  it("shows provisional text subdued with a placeholder label", () => {
    render(
      <LiveTranscript
        lines={[line({ provisional: true, canonical: null })]}
        aliases={{}}
        status="Metin işleniyor"
        warning={null}
        onRenameSpeaker={() => undefined}
      />,
    );

    expect(screen.getByText("Konuşmacı belirleniyor")).toBeTruthy();
    expect(screen.getByText("Merhaba")).toBeTruthy();
    expect(screen.getByText("Metin işleniyor")).toBeTruthy();
  });

  it("renders the canonical label once the rolling result arrives", () => {
    const { rerender } = render(
      <LiveTranscript
        lines={[line({})]}
        aliases={{}}
        status="Konuşmacılar eşleştiriliyor"
        warning={null}
        onRenameSpeaker={() => undefined}
      />,
    );
    expect(screen.getByText("Konuşmacı belirleniyor")).toBeTruthy();

    rerender(
      <LiveTranscript
        lines={[line({ canonical: "Kişi 1" })]}
        aliases={{}}
        status="Dinleniyor"
        warning={null}
        onRenameSpeaker={() => undefined}
      />,
    );
    expect(screen.queryByText("Konuşmacı belirleniyor")).toBeNull();
    expect(screen.getByText("Kişi 1")).toBeTruthy();
  });

  it("applies a meeting-local alias to every existing line", () => {
    render(
      <LiveTranscript
        lines={[
          line({ id: 1, canonical: "Kişi 1" }),
          line({ id: 2, canonical: "Kişi 1", text: "ikinci" }),
          line({ id: 3, canonical: "Kişi 2", text: "başka" }),
        ]}
        aliases={{ "Kişi 1": "Ahmet" }}
        status="Dinleniyor"
        warning={null}
        onRenameSpeaker={() => undefined}
      />,
    );

    expect(screen.getAllByText("Ahmet")).toHaveLength(2);
    expect(screen.getByText("Kişi 2")).toBeTruthy();
  });

  it("opens the rename dialog from the speaker badge", () => {
    const onRename = vi.fn();
    render(
      <LiveTranscript
        lines={[line({ canonical: "Kişi 1" })]}
        aliases={{}}
        status="Dinleniyor"
        warning={null}
        onRenameSpeaker={onRename}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /Kişi 1 · Konuşmacı adını düzenle/ }));
    expect(onRename).toHaveBeenCalledWith("Kişi 1");
  });

  it("shows a safe warning without hiding the transcript", () => {
    render(
      <LiveTranscript
        lines={[line({ canonical: "Kişi 1" })]}
        aliases={{}}
        status="Canlı transkript bağlantısı kesildi"
        warning="Canlı transkript bağlantısı kesildi — kayıt devam ediyor."
        onRenameSpeaker={() => undefined}
      />,
    );

    expect(screen.getByText(/kayıt devam ediyor/)).toBeTruthy();
    expect(screen.getByText("Merhaba")).toBeTruthy();
  });

  it("auto-scrolls only while the user is near the bottom", () => {
    const { rerender } = render(
      <LiveTranscript
        lines={[line({ id: 1 })]}
        aliases={{}}
        status="Dinleniyor"
        warning={null}
        onRenameSpeaker={() => undefined}
      />,
    );
    const list = screen.getByRole("list");

    const scrollState = { scrollTop: 100, scrollHeight: 500, clientHeight: 400 };
    Object.defineProperty(list, "scrollHeight", { configurable: true, value: scrollState.scrollHeight });
    Object.defineProperty(list, "clientHeight", { configurable: true, value: scrollState.clientHeight });
    Object.defineProperty(list, "scrollTop", {
      configurable: true,
      get: () => scrollState.scrollTop,
      set: (value: number) => {
        scrollState.scrollTop = value;
      },
    });

    // Near the bottom (100 >= 500-400-48): the next line scrolls the list down.
    fireEvent.scroll(list);
    rerender(
      <LiveTranscript
        lines={[line({ id: 1 }), line({ id: 2, text: "yeni" })]}
        aliases={{}}
        status="Dinleniyor"
        warning={null}
        onRenameSpeaker={() => undefined}
      />,
    );
    expect(scrollState.scrollTop).toBe(scrollState.scrollHeight);

    // Scrolled up: content updates must not force the view down.
    scrollState.scrollTop = 0;
    fireEvent.scroll(list);
    const before = scrollState.scrollTop;
    rerender(
      <LiveTranscript
        lines={[line({ id: 1 }), line({ id: 2, text: "yeni" }), line({ id: 3, text: "daha" })]}
        aliases={{}}
        status="Dinleniyor"
        warning={null}
        onRenameSpeaker={() => undefined}
      />,
    );
    expect(scrollState.scrollTop).toBe(before);
  });
});
