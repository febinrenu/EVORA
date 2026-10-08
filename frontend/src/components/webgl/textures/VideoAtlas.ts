// One <video> drives every feed on the page. The footage script tiles nine
// EPFL views into a 3x3 atlas, so 24 sleeves cost one decoder and one texture.
import { LinearFilter, SRGBColorSpace, VideoTexture, type Texture, DataTexture, RGBAFormat } from "three";

interface Manifest {
  atlas: string;
  cols: number;
  rows: number;
}

export class VideoAtlas {
  texture: Texture;
  ready = false;
  private video: HTMLVideoElement | null = null;
  private wanted = false;

  constructor() {
    // 1x1 placeholder until (or unless) footage exists; shaders switch to the procedural feed.
    const t = new DataTexture(new Uint8Array([8, 10, 12, 255]), 1, 1, RGBAFormat);
    t.needsUpdate = true;
    this.texture = t;
  }

  async load(): Promise<void> {
    let manifest: Manifest;
    try {
      const res = await fetch("/footage/manifest.json", { cache: "force-cache" });
      if (!res.ok) return;
      manifest = (await res.json()) as Manifest;
    } catch {
      return;
    }
    const v = document.createElement("video");
    v.src = manifest.atlas;
    v.muted = true;
    v.loop = true;
    v.playsInline = true;
    v.preload = "auto";
    v.crossOrigin = "anonymous";
    v.setAttribute("aria-hidden", "true");
    await new Promise<void>((resolve) => {
      const done = () => resolve();
      v.addEventListener("loadeddata", done, { once: true });
      v.addEventListener("error", done, { once: true });
      v.load();
    });
    if (v.readyState < 2) return;
    const tex = new VideoTexture(v);
    tex.colorSpace = SRGBColorSpace;
    tex.minFilter = LinearFilter;
    tex.magFilter = LinearFilter;
    tex.generateMipmaps = false;
    this.texture.dispose();
    this.texture = tex;
    this.video = v;
    this.ready = true;
    if (this.wanted) void v.play().catch(() => undefined);
  }

  /** Decode only while a feed is on screen. */
  setPlaying(on: boolean): void {
    if (on === this.wanted) return;
    this.wanted = on;
    const v = this.video;
    if (!v) return;
    if (on) void v.play().catch(() => undefined);
    else v.pause();
  }

  dispose(): void {
    this.video?.pause();
    if (this.video) {
      this.video.removeAttribute("src");
      this.video.load();
    }
    this.texture.dispose();
  }
}
