import { describe, expect, it } from "vitest";

import {
  applyAlias,
  canonicalSpeakerIndex,
  isCanonicalSpeaker,
  resolveSpeakerDisplay,
} from "@/lib/speakers";

describe("speaker display resolver", () => {
  it("resolves an alias for its meeting only", () => {
    expect(resolveSpeakerDisplay("Kişi 1", { "Kişi 1": "Ahmet" })).toBe("Ahmet");
    expect(resolveSpeakerDisplay("Kişi 2", { "Kişi 1": "Ahmet" })).toBe("Kişi 2");
    expect(resolveSpeakerDisplay("Kişi 1", {})).toBe("Kişi 1");
    expect(resolveSpeakerDisplay("Kişi 1", null)).toBe("Kişi 1");
  });

  it("never aliases the unresolved speaker", () => {
    expect(resolveSpeakerDisplay("Bilinmeyen", { Bilinmeyen: "Ahmet" })).toBe("Bilinmeyen");
  });

  it("caps speaker tones deterministically", () => {
    expect(canonicalSpeakerIndex("Kişi 1")).toBe(0);
    expect(canonicalSpeakerIndex("Kişi 12")).toBe(11);
    expect(canonicalSpeakerIndex("Bilinmeyen")).toBe(-1);
    expect(canonicalSpeakerIndex("Ahmet")).toBe(-1);
  });

  it("applies and clears aliases immutably", () => {
    const original = { "Kişi 1": "Ahmet" };
    const renamed = applyAlias(original, "Kişi 1", "Mehmet");
    const reset = applyAlias(renamed, "Kişi 1", null);
    expect(original).toEqual({ "Kişi 1": "Ahmet" });
    expect(renamed).toEqual({ "Kişi 1": "Mehmet" });
    expect(reset).toEqual({});
    expect(isCanonicalSpeaker("Kişi 2")).toBe(true);
    expect(isCanonicalSpeaker("Konuşmacı belirleniyor")).toBe(false);
  });
});
