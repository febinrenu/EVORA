// Optional sound, synthesised with Web Audio (no files). Off until the visitor
// turns it on; every cue is rate-limited so fast scrolling never machine-guns.

type Cue = "key" | "tick" | "pulse" | "open" | "whoosh" | "confirm";

class AudioEngine {
  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  private hum: { stop: () => void } | null = null;
  private last = new Map<Cue, number>();
  enabled = false;
  private listeners = new Set<(on: boolean) => void>();

  subscribe(fn: (on: boolean) => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  async setEnabled(on: boolean): Promise<void> {
    this.enabled = on;
    if (on) {
      if (!this.ctx) {
        this.ctx = new AudioContext();
        this.master = this.ctx.createGain();
        this.master.gain.value = 0;
        this.master.connect(this.ctx.destination);
      }
      await this.ctx.resume();
      this.master?.gain.setTargetAtTime(0.5, this.ctx.currentTime, 0.4);
      this.startHum();
    } else if (this.ctx && this.master) {
      this.master.gain.setTargetAtTime(0, this.ctx.currentTime, 0.15);
      const ctx = this.ctx;
      window.setTimeout(() => {
        if (!this.enabled) void ctx.suspend();
      }, 600);
    }
    this.listeners.forEach((l) => l(on));
  }

  /** A low room tone: filtered noise and a 50 Hz mains hum, barely there. */
  private startHum(): void {
    if (this.hum || !this.ctx || !this.master) return;
    const ctx = this.ctx;
    const len = ctx.sampleRate * 2;
    const buf = ctx.createBuffer(1, len, ctx.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < len; i++) data[i] = Math.random() * 2 - 1;
    const noise = ctx.createBufferSource();
    noise.buffer = buf;
    noise.loop = true;
    const lp = ctx.createBiquadFilter();
    lp.type = "lowpass";
    lp.frequency.value = 180;
    const ng = ctx.createGain();
    ng.gain.value = 0.05;
    noise.connect(lp).connect(ng).connect(this.master);
    const osc = ctx.createOscillator();
    osc.frequency.value = 50;
    const og = ctx.createGain();
    og.gain.value = 0.012;
    osc.connect(og).connect(this.master);
    noise.start();
    osc.start();
    this.hum = {
      stop: () => {
        noise.stop();
        osc.stop();
      },
    };
  }

  play(cue: Cue): void {
    if (!this.enabled || !this.ctx || !this.master || this.ctx.state !== "running") return;
    const now = performance.now();
    const gap = cue === "key" ? 45 : 140;
    if (now - (this.last.get(cue) ?? 0) < gap) return;
    this.last.set(cue, now);
    const ctx = this.ctx;
    const t = ctx.currentTime;
    const out = ctx.createGain();
    out.connect(this.master);
    const env = (peak: number, attack: number, decay: number) => {
      out.gain.setValueAtTime(0, t);
      out.gain.linearRampToValueAtTime(peak, t + attack);
      out.gain.exponentialRampToValueAtTime(0.0001, t + attack + decay);
    };
    const osc = (type: OscillatorType, f0: number, f1: number, length: number) => {
      const o = ctx.createOscillator();
      o.type = type;
      o.frequency.setValueAtTime(f0, t);
      o.frequency.exponentialRampToValueAtTime(f1, t + length);
      o.connect(out);
      o.start(t);
      o.stop(t + length + 0.05);
    };
    switch (cue) {
      case "key":
        env(0.05, 0.002, 0.03);
        osc("square", 2400 + Math.random() * 400, 1800, 0.03);
        break;
      case "tick":
        env(0.08, 0.002, 0.06);
        osc("sine", 1320, 990, 0.06);
        break;
      case "pulse":
        env(0.22, 0.01, 0.5);
        osc("sine", 82, 46, 0.5);
        break;
      case "open":
        env(0.1, 0.05, 0.6);
        osc("triangle", 220, 440, 0.6);
        break;
      case "whoosh": {
        env(0.12, 0.15, 0.6);
        const len = Math.floor(ctx.sampleRate * 0.8);
        const buf = ctx.createBuffer(1, len, ctx.sampleRate);
        const d = buf.getChannelData(0);
        for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.sin((i / len) * Math.PI);
        const src = ctx.createBufferSource();
        src.buffer = buf;
        const bp = ctx.createBiquadFilter();
        bp.type = "bandpass";
        bp.frequency.setValueAtTime(400, t);
        bp.frequency.exponentialRampToValueAtTime(1800, t + 0.7);
        const pan = ctx.createStereoPanner();
        pan.pan.setValueAtTime(-0.7, t);
        pan.pan.linearRampToValueAtTime(0.7, t + 0.7);
        src.connect(bp).connect(pan).connect(out);
        src.start(t);
        break;
      }
      case "confirm":
        env(0.12, 0.01, 0.5);
        osc("sine", 660, 660, 0.18);
        window.setTimeout(() => this.enabled && osc("sine", 990, 990, 0.3), 110);
        break;
    }
  }
}

export const audio = new AudioEngine();
export const sfx = (cue: Cue): void => audio.play(cue);
