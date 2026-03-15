// =====================
// DOM Ready
// =====================
document.addEventListener('DOMContentLoaded', () => {
  console.log('Page loaded and ready.');
  init();
});

// =====================
// Init
// =====================
function init() {
  // Entry point — set up event listeners and initial state here
}

// =====================
// Helpers
// =====================

/**
 * Select a single DOM element.
 * @param {string} selector - CSS selector
 * @returns {Element|null}
 */
function $(selector) {
  return document.querySelector(selector);
}

/**
 * Select multiple DOM elements.
 * @param {string} selector - CSS selector
 * @returns {NodeList}
 */
function $$(selector) {
  return document.querySelectorAll(selector);
}

/**
 * Fetch JSON from a URL.
 * @param {string} url
 * @returns {Promise<any>}
 */
async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${url}`);
  return res.json();
}
