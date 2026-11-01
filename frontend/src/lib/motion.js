import { useEffect, useRef, useState } from "react";

const reduced = () =>
  typeof window !== "undefined" &&
  window.matchMedia &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Reveal an element when it scrolls into view. Returns [ref, shown].
// Respects reduced-motion (shows immediately) and degrades safely.
export function useReveal(options = {}) {
  const ref = useRef(null);
  const [shown, setShown] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (reduced() || typeof IntersectionObserver === "undefined") {
      setShown(true);
      return;
    }
    const io = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          setShown(true);
          io.disconnect();
        }
      },
      { threshold: 0.12, rootMargin: "0px 0px -8% 0px", ...options }
    );
    io.observe(el);
    return () => io.disconnect();
  }, []); // eslint-disable-line
  return [ref, shown];
}

// Animate a number from 0 → target (cubic ease-out) once `start` is true.
export function useCountUp(target, { duration = 1100, start = true } = {}) {
  const [val, setVal] = useState(0);
  const raf = useRef();
  useEffect(() => {
    if (!start) return;
    const goal = Number(target) || 0;
    if (reduced()) {
      setVal(goal);
      return;
    }
    const t0 = performance.now();
    const tick = (t) => {
      const p = Math.min(1, (t - t0) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      setVal(goal * eased);
      if (p < 1) raf.current = requestAnimationFrame(tick);
      else setVal(goal);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [target, start, duration]);
  return val;
}
