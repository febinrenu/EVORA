import type { Metadata } from "next";
import Link from "next/link";
import "./app.css";

export const metadata: Metadata = {
  title: "EVORA — case log",
};

// The product shell on the light table (PLAN §10). The working views (ask bar,
// case log, evidence sheet, timeline, site plan) land here in P4.4 onward.
export default function AppShell() {
  return (
    <div className="lt">
      <header className="lt-top">
        <Link href="/" className="lt-mark">
          EVORA
        </Link>
        <span className="lt-site">Own campus</span>
        <span className="lt-clock">Footage clock 09:14:23</span>
        <span className="lt-onprem">On this machine only</span>
      </header>
      <aside className="lt-rail" aria-label="Cameras">
        <h2>Cameras</h2>
        <p>Drop footage to begin.</p>
      </aside>
      <main className="lt-case">
        <p className="lt-empty">Ask about the footage. Try: did anyone carry a large bag through the lobby?</p>
        <form className="lt-ask" action="#" aria-label="Ask about the footage">
          <input placeholder="Ask about the footage" aria-label="Question" />
        </form>
      </main>
      <aside className="lt-side" aria-label="Site plan and known places">
        <h2>Site plan</h2>
        <h2>Known places</h2>
      </aside>
      <footer className="lt-timeline" aria-label="Timeline" />
    </div>
  );
}
