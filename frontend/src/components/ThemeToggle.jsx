import React, { useState } from "react";
import { Sun, Moon } from "lucide-react";
import { getTheme, toggleTheme } from "../lib/theme.js";

export default function ThemeToggle({ className = "" }) {
  const [t, setT] = useState(getTheme());
  return (
    <button type="button" className={`btn btn-ghost btn-icon ${className}`} onClick={() => setT(toggleTheme())}
            aria-label={t === "dark" ? "Switch to light theme" : "Switch to dark theme"} title="Toggle theme">
      {t === "dark" ? <Sun size={16} /> : <Moon size={16} />}
    </button>
  );
}
