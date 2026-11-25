/**
 * markdown.js - a tiny, dependency-free Markdown renderer for chat answers.   (NEW)
 *
 * Language models like to answer with **bold**, `code`, bullet lists and
 * numbered lists. This file turns that text into safe HTML.
 *
 * SAFETY: every piece of text is HTML-escaped FIRST, and only afterwards do we
 * add our own tags. Model output (or a video transcript echoed by the model)
 * can therefore never inject HTML or scripts into the page.
 *
 * Supported: paragraphs, "# headings" (shown as bold lines), "-", "*" and "1."
 * lists, ``` fenced code, **bold**, *italic*, `inline code`, and [m:ss] / [h:mm:ss]
 * timestamps, which become links that open the video at that moment.
 *
 * Usage:  element.innerHTML = renderMarkdown(text, { videoId: "dQw4w9WgXcQ" });
 */
(function () {
  "use strict";

  /** Escape the five characters that matter in HTML. */
  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  /** "[1:02:05]" parts -> seconds. */
  function toSeconds(h, m, s) {
    return (Number(h || 0) * 3600) + (Number(m) * 60) + Number(s);
  }

  /**
   * Inline formatting for ONE already-escaped line.
   * `code` spans are split off first so their contents are left untouched.
   */
  function inline(escaped, videoId) {
    return escaped
      .split(/(`[^`]+`)/)
      .map((part, i) => {
        if (i % 2 === 1) return "<code>" + part.slice(1, -1) + "</code>";
        let out = part
          .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
          .replace(/(^|[^*])\*([^*\s][^*]*?)\*(?!\*)/g, "$1<em>$2</em>");
        if (videoId) {
          out = out.replace(/\[(?:(\d+):)?(\d{1,2}):(\d{2})\]/g, (all, h, m, s) =>
            '<a class="ts" target="_blank" rel="noopener" href="https://www.youtube.com/watch?v=' +
            videoId + "&amp;t=" + toSeconds(h, m, s) + 's">' + all + "</a>");
        }
        return out;
      })
      .join("");
  }

  /** Convert Markdown text into an HTML string (safe to assign to innerHTML). */
  function renderMarkdown(text, opts) {
    const videoId = opts && /^[\w-]{11}$/.test(opts.videoId || "") ? opts.videoId : "";
    const lines = String(text).replace(/\r\n/g, "\n").split("\n");
    const html = [];
    let para = [];       // lines of the paragraph being built
    let list = null;     // { tag: "ul" | "ol", items: [html, ...] }
    let code = null;     // raw lines of a ``` block, or null when not inside one

    const flushPara = () => {
      if (para.length) html.push("<p>" + para.join("<br>") + "</p>");
      para = [];
    };
    const flushList = () => {
      if (list) {
        html.push("<" + list.tag + ">" + list.items.map((i) => "<li>" + i + "</li>").join("") + "</" + list.tag + ">");
      }
      list = null;
    };

    for (const line of lines) {
      // Fenced code block: collect raw lines until the closing fence.
      if (/^\s*```/.test(line)) {
        if (code === null) { flushPara(); flushList(); code = []; }
        else { html.push("<pre><code>" + escapeHtml(code.join("\n")) + "</code></pre>"); code = null; }
        continue;
      }
      if (code !== null) { code.push(line); continue; }

      const esc = escapeHtml(line);
      let m;
      if (!line.trim()) {                                   // blank line ends blocks
        flushPara(); flushList();
      } else if ((m = esc.match(/^\s*#{1,6}\s+(.*)$/))) {   // heading -> bold line
        flushPara(); flushList();
        html.push("<p><strong>" + inline(m[1], videoId) + "</strong></p>");
      } else if ((m = esc.match(/^\s*[-*\u2022]\s+(.*)$/))) { // bullet item
        flushPara();
        if (!list || list.tag !== "ul") { flushList(); list = { tag: "ul", items: [] }; }
        list.items.push(inline(m[1], videoId));
      } else if ((m = esc.match(/^\s*\d+[.)]\s+(.*)$/))) {  // numbered item
        flushPara();
        if (!list || list.tag !== "ol") { flushList(); list = { tag: "ol", items: [] }; }
        list.items.push(inline(m[1], videoId));
      } else {                                              // ordinary text
        flushList();
        para.push(inline(esc, videoId));
      }
    }
    if (code !== null) html.push("<pre><code>" + escapeHtml(code.join("\n")) + "</code></pre>"); // unclosed fence
    flushPara(); flushList();
    return html.join("");
  }

  // Browser: expose globally. Node (used by tests): export as a module.
  if (typeof module !== "undefined" && module.exports) module.exports = { renderMarkdown, escapeHtml };
  else window.renderMarkdown = renderMarkdown;
})();
