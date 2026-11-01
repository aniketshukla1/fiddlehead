import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import { initTheme } from "./lib/theme.js";
import "./styles.css";   // studio instrument panels
import "./ui.css";       // design system + app pages (wins on equal specificity)

initTheme();

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
