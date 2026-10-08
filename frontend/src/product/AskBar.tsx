"use client";

// Ask bar (P4.4): "/" focuses it, Enter sends, ↑/↓ walk earlier questions,
// the microphone records a question and posts it to /api/voice (browser speech
// recognition when the server cannot transcribe, e.g. on-prem mode).
import { useRef, useState } from "react";
import { ApiError, endpoints } from "@/lib/api/client";
import { useEvora } from "./store";

type SpeechCtor = new () => {
  lang: string;
  interimResults: boolean;
  onresult: (e: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void;
  onerror: () => void;
  onend: () => void;
  start: () => void;
};

export function AskBar() {
  const ask = useEvora((s) => s.ask);
  const history = useEvora((s) => s.history);
  const busy = useEvora((s) => s.cases.some((c) => c.status === "planning" || c.status === "searching"));
  const [text, setText] = useState("");
  const [cursor, setCursor] = useState(-1);
  const [voice, setVoice] = useState<"idle" | "recording" | "transcribing">("idle");
  const [hint, setHint] = useState<string | null>(null);
  const recorder = useRef<MediaRecorder | null>(null);

  const submit = () => {
    if (!text.trim()) return;
    ask(text);
    setText("");
    setCursor(-1);
    setHint(null);
  };

  const browserSpeech = () => {
    const w = window as unknown as { SpeechRecognition?: SpeechCtor; webkitSpeechRecognition?: SpeechCtor };
    const Ctor = w.SpeechRecognition ?? w.webkitSpeechRecognition;
    if (!Ctor) {
      setHint("Voice is not available here. Type the question.");
      setVoice("idle");
      return;
    }
    const rec = new Ctor();
    rec.lang = "en";
    rec.interimResults = false;
    rec.onresult = (e) => setText(e.results[0]?.[0]?.transcript ?? "");
    rec.onerror = () => setHint("Did not catch that. Try again or type.");
    rec.onend = () => setVoice("idle");
    setVoice("recording");
    rec.start();
  };

  const toggleVoice = async () => {
    if (voice === "recording") {
      recorder.current?.stop();
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) return browserSpeech();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const chunks: Blob[] = [];
      const mr = new MediaRecorder(stream);
      recorder.current = mr;
      mr.ondataavailable = (e) => chunks.push(e.data);
      mr.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        setVoice("transcribing");
        try {
          const { text: heard } = await endpoints.voice(new Blob(chunks, { type: mr.mimeType }));
          setText(heard);
          setVoice("idle");
          document.getElementById("ask-input")?.focus();
        } catch (e) {
          if (e instanceof ApiError && e.status === 503) {
            setHint("Transcribing on this machine is off. Listening in the browser instead.");
            browserSpeech();
          } else {
            setHint(e instanceof Error ? e.message : "Voice failed. Type the question.");
            setVoice("idle");
          }
        }
      };
      mr.start();
      setVoice("recording");
      window.setTimeout(() => mr.state === "recording" && mr.stop(), 12000);
    } catch {
      setHint("The microphone is blocked. Type the question.");
    }
  };

  return (
    <form
      className="lt-ask"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <label htmlFor="ask-input" className="sr-only">
        Ask about the footage
      </label>
      <input
        id="ask-input"
        value={text}
        autoComplete="off"
        placeholder="Ask about the footage"
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "ArrowUp" && history.length) {
            e.preventDefault();
            const i = Math.min(history.length - 1, cursor + 1);
            setCursor(i);
            setText(history[i]);
          } else if (e.key === "ArrowDown" && cursor >= 0) {
            e.preventDefault();
            const i = cursor - 1;
            setCursor(i);
            setText(i >= 0 ? history[i] : "");
          }
        }}
      />
      <button type="button" className={`lt-mic${voice !== "idle" ? " is-on" : ""}`} onClick={() => void toggleVoice()} aria-pressed={voice === "recording"} aria-label={voice === "recording" ? "Stop recording" : "Ask by voice"}>
        <svg viewBox="0 0 20 20" aria-hidden="true">
          <rect x="7" y="2.5" width="6" height="10" rx="3" />
          <path d="M4.5 9.5a5.5 5.5 0 0 0 11 0M10 15v2.5" />
        </svg>
      </button>
      <button type="submit" className="lt-send" disabled={!text.trim()}>
        {busy ? "Asking…" : "Ask"}
      </button>
      <p className="lt-ask-hint" aria-live="polite">
        {voice === "recording" ? "Listening. Press the microphone again to stop." : voice === "transcribing" ? "Turning that into text…" : hint}
      </p>
    </form>
  );
}
