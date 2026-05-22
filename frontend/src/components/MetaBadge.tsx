type Props = { label: string; value: string };

export function MetaBadge({ label, value }: Props) {
  return (
    <span
      className="inline-flex items-center gap-1 text-[11px] px-2 py-0.5 rounded-md
        bg-stone-100 dark:bg-stone-800 text-stone-500 dark:text-stone-400"
      style={{ fontFamily: "'Space Mono', monospace" }}
    >
      <span className="text-stone-300 dark:text-stone-600">{label}</span>
      {value}
    </span>
  );
}
