import type { Metadata, Viewport } from "next";
import localFont from "next/font/local";
import "@/styles/tokens.css";
import "./globals.css";

// Self-hosted (the demo must run with Wi-Fi off). Archivo keeps its width axis.
const archivo = localFont({
  src: "../fonts/archivo-latin-wdth.woff2",
  variable: "--font-archivo",
  weight: "100 900",
  style: "normal",
  display: "swap",
  declarations: [{ prop: "font-stretch", value: "62% 125%" }],
});

const geistMono = localFont({
  src: "../fonts/geist-mono.woff2",
  variable: "--font-geist-mono",
  weight: "100 900",
  display: "swap",
});

export const metadata: Metadata = {
  title: "EVORA — search the memory of a place",
  description: "Multi-camera video intelligence. Ask in plain words; get the camera, the moment and the frame, circled.",
};

export const viewport: Viewport = {
  themeColor: "#06080a",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${archivo.variable} ${geistMono.variable}`}>
      <body>{children}</body>
    </html>
  );
}
