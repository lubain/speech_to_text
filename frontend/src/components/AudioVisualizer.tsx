import type { RefObject } from "react";

type Props = {
  canvasRef: RefObject<HTMLCanvasElement | null>;
};

export function AudioVisualizer({ canvasRef }: Props) {
  return (
    <div
      className="w-full h-28 bg-stone-50 dark:bg-stone-800/60 border border-stone-100
        dark:border-stone-700 rounded-xl overflow-hidden flex items-end px-2 pb-2"
    >
      <canvas
        ref={canvasRef}
        width={560}
        height={96}
        className="w-full h-full"
        style={{ display: "block" }}
      />
    </div>
  );
}
