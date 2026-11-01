import React from "react";
import { BRAND } from "../lib/brand.js";

// The mark: a fern fiddlehead — a stem curling into a coil. It is also a violin's
// scroll (the fern's namesake) and an optimizer spiralling into its minimum.
export const SPIRAL =
  "M8.8 21.2C7.3 18.4 6.6 14.6 6.6 9.6L7.0 8.4L7.6 7.4L8.3 6.5L9.2 5.8L10.1 5.3L11.1 5.1L12.1 5.0" +
  "L13.0 5.1L13.8 5.4L14.5 5.8L15.2 6.3L15.6 6.9L16.0 7.6L16.2 8.2L16.2 8.9L16.2 9.6L16.0 10.1" +
  "L15.7 10.7L15.3 11.1L14.9 11.4L14.4 11.7L14.0 11.8L13.5 11.9L13.0 11.8L12.6 11.7L12.3 11.5" +
  "L12.0 11.2L11.7 10.9L11.6 10.6L11.5 10.3L11.4 10.0L11.4 9.6L11.5 9.4L11.7 9.1L11.8 8.9L12.1 8.7";

export function Mark({ size = 22, className = "" }) {
  return (
    <svg className={`mark ${className}`} width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <rect x="1" y="1" width="22" height="22" rx="6.5" className="mark-bg" />
      <path d={SPIRAL} className="mark-coil" pathLength="1" />
    </svg>
  );
}

export default function Logo({ size = 22, className = "", onClick, href }) {
  const body = (
    <>
      <Mark size={size} />
      <span className="logo-word">{BRAND.toLowerCase()}</span>
    </>
  );
  if (href) return <a className={`logo ${className}`} href={href} aria-label={`${BRAND} home`}>{body}</a>;
  if (onClick) return <button type="button" className={`logo ${className}`} onClick={onClick} aria-label={`${BRAND} home`}>{body}</button>;
  return <span className={`logo ${className}`}>{body}</span>;
}
