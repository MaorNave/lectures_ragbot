/*!
 * SmartUp Assistant — embeddable chat widget (vanilla JS, no dependencies, no build step).
 *
 * Embed:
 *   <script src="https://YOUR-API-HOST/widget/widget.js"
 *           data-api-base="https://YOUR-API-HOST"      (default: origin of this script)
 *           data-title="SmartUp Assistant"
 *           data-lang="auto"                            (auto | en | he — UI language / direction)
 *           data-position="right"                       (right | left)
 *           data-primary="#2b4acb"                      (optional theme colour; or set CSS vars on .su-widget)
 *           data-theme="light"                          (light | dark | auto — auto follows the visitor's OS)
 *           data-welcome="..."                          (optional custom welcome message)
 *           data-suggestions='["…", "…"]'              (optional: questions offered before the first message)
 *           data-timing="off"                           (optional: hide the answer stopwatches)
 *           data-no-css                                 (optional: do not auto-load widget.css)
 *           defer></script>
 *
 * Talks to:  POST {apiBase}/chat/stream  (Server-Sent Events: token* → final | error)
 *            POST {apiBase}/chat         (fallback when streaming is unavailable)
 *
 * Public API: window.SmartUpWidget.{open, close, toggle, send(text), reset, setSuggestions([...]),
 *             setTheme({'--su-primary': '#...'})}
 */
(function () {
  'use strict';
  if (window.SmartUpWidget && window.SmartUpWidget._mounted) return;

  // ---------------------------------------------------------------------------------------------
  // configuration
  // ---------------------------------------------------------------------------------------------
  var script = document.currentScript || (function () {
    var all = document.getElementsByTagName('script');
    for (var i = all.length - 1; i >= 0; i--) {
      if (all[i].src && /widget\.js(\?|#|$)/.test(all[i].src)) return all[i];
    }
    return null;
  })();
  var ds = (script && script.dataset) || {};
  var scriptUrl = script && script.src ? new URL(script.src, location.href) : null;

  function parseList(raw) {
    if (!raw) return [];
    try { var v = JSON.parse(raw); return Array.isArray(v) ? v.map(String).filter(Boolean) : []; } catch (e) { return []; }
  }

  var config = {
    apiBase: (ds.apiBase || (scriptUrl ? scriptUrl.origin : location.origin)).replace(/\/+$/, ''),
    title: ds.title || 'SmartUp Assistant',
    lang: (ds.lang || 'auto').toLowerCase(),        // 'auto' | 'en' | 'he'
    position: ds.position === 'left' ? 'left' : 'right',
    primary: ds.primary || null,
    welcome: ds.welcome || null,
    loadCss: !('noCss' in ds),
    cssHref: ds.css || (scriptUrl ? new URL('widget.css', scriptUrl).href : null),
    storageKey: 'smartup-widget:' + (ds.storageKey || 'session'),
    mode: ds.mode === 'inline' ? 'inline' : 'float',   // inline = full-page chat box rendered into data-target
    target: ds.target || null,
    timing: ds.timing !== 'off',                       // stopwatches under each answer
    theme: ds.theme === 'dark' || ds.theme === 'auto' ? ds.theme : 'light',
    suggestions: parseList(ds.suggestions),            // questions offered in the empty conversation
    maxChars: 2000
  };

  // ---------------------------------------------------------------------------------------------
  // i18n
  // ---------------------------------------------------------------------------------------------
  var STR = {
    en: {
      subtitle: 'Answers grounded in Yonatan Stern’s lectures',
      heroTitle: 'Ask the SmartUp lectures',
      heroText: 'Every answer comes from Yonatan Stern’s lectures — the v2 course first, then his TEDx talk, the v1 course and other lectures — and cites the lecture it is drawn from.',
      heroLabel: 'Try asking',
      copy: 'Copy', copied: 'Copied',
      placeholder: 'Ask about the lectures…',
      send: 'Send', open: 'Open chat', close: 'Close chat', newChat: 'New conversation',
      thinking: 'Thinking…', searching: 'Searching the lectures…',
      sources: 'Sources', source: 'Source',
      welcome: 'Hi! I answer questions about the SmartUp Academy lectures by Yonatan Stern — in English or Hebrew. What would you like to know?',
      busy: 'The assistant is helping someone else right now. Please try again in a moment.',
      error: 'Something went wrong while contacting the assistant. Please try again.',
      interrupted: 'The connection was interrupted — this answer may be incomplete.',
      fallbackNote: 'v1 material used',
      fallbackNotes: { tedx: 'From Yonatan\u2019s TEDx talk', v1: 'v1 material used', outsource: 'From lectures outside the course' },
      footer: 'Answers come only from the lecture transcripts.',
      footerInline: 'Answers come only from the lecture transcripts · Enter to send, Shift+Enter for a new line',
      you: 'You', bot: 'Assistant', chars: 'characters',
      firstWords: 'First words', fullAnswer: 'Full answer', secs: 's',
      firstWordsHint: 'Time from sending the question until the answer started to appear',
      fullAnswerHint: 'Time from sending the question until the answer was complete',
      trace: 'trace ↗', traceHint: 'Open this answer on the pipeline dashboard: query, passages, grading, generation'
    },
    he: {
      subtitle: 'תשובות שמבוססות על ההרצאות של יונתן שטרן',
      heroTitle: 'שאלו את הרצאות SmartUp',
      heroText: 'כל תשובה מבוססת על ההרצאות של יונתן שטרן — קודם גרסה v2 של הקורס, אחר כך הרצאת ה-TEDx, גרסה v1 והרצאות נוספות — ומציינת את ההרצאה שממנה היא לקוחה.',
      heroLabel: 'אפשר לשאול למשל',
      copy: 'העתקה', copied: 'הועתק',
      placeholder: 'שאל/י על ההרצאות…',
      send: 'שליחה', open: 'פתיחת צ׳אט', close: 'סגירת צ׳אט', newChat: 'שיחה חדשה',
      thinking: 'חושב…', searching: 'מחפש בהרצאות…',
      sources: 'מקורות', source: 'מקור',
      welcome: 'שלום! אני עונה על שאלות לגבי הרצאות SmartUp Academy של יונתן שטרן — בעברית או באנגלית. במה אפשר לעזור?',
      busy: 'העוזר עסוק כרגע בשיחה אחרת. נסו שוב בעוד רגע.',
      error: 'משהו השתבש בפנייה לעוזר. נסו שוב.',
      interrupted: 'החיבור נותק — התשובה עשויה להיות חלקית.',
      fallbackNote: 'נעשה שימוש בחומר v1',
      fallbackNotes: { tedx: 'מתוך הרצאת ה-TEDx של יונתן', v1: 'נעשה שימוש בחומר v1', outsource: 'מתוך הרצאות מחוץ לקורס' },
      footer: 'התשובות מבוססות על תמלילי ההרצאות בלבד.',
      footerInline: 'התשובות מבוססות על תמלילי ההרצאות בלבד · Enter לשליחה, Shift+Enter לשורה חדשה',
      you: 'את/ה', bot: 'העוזר', chars: 'תווים',
      firstWords: 'מילים ראשונות', fullAnswer: 'תשובה מלאה', secs: 'שנ׳',
      firstWordsHint: 'הזמן משליחת השאלה ועד שהתשובה התחילה להופיע',
      fullAnswerHint: 'הזמן משליחת השאלה ועד שהתשובה הושלמה',
      trace: 'מעקב ↗', traceHint: 'פתיחת התשובה בלוח הבקרה: שאילתה, קטעים, דירוג ויצירה'
    }
  };

  // ---------------------------------------------------------------------------------------------
  // helpers
  // ---------------------------------------------------------------------------------------------
  var HEBREW_RE = /[֐-׿]/g;
  var LATIN_RE = /[A-Za-z]/g;

  /** Hebrew when Hebrew letters dominate (same rule as the server's lang.py). */
  function isHebrew(text) {
    if (!text) return false;
    var he = (text.match(HEBREW_RE) || []).length;
    if (!he) return false;
    var la = (text.match(LATIN_RE) || []).length;
    if (!la) return true;
    return he >= la * 0.6;
  }
  function dirOf(text) { return isHebrew(text) ? 'rtl' : 'ltr'; }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  /** Inline markdown on already-escaped text: code, bold, italic, citation highlighting. */
  function inlineMd(s) {
    s = s.replace(/`([^`\n]+)`/g, '<code>$1</code>');
    s = s.replace(/\*\*([^*\n]+?)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/__([^_\n]+?)__/g, '<strong>$1</strong>');
    s = s.replace(/(^|[\s(א-ת])\*([^*\n]+?)\*(?=[\s).,;:!?]|$)/g, '$1<em>$2</em>');
    s = s.replace(/\((v[12]),\s*Lecture\s+\d+(?:\s+Part\s+\d+)?\s*[—–-]+\s*[^()]*\)/gi, function (m, version) {
      return '<span class="su-cite su-cite-' + version.toLowerCase() + '" dir="ltr">' + m + '</span>';
    });
    // outside sources: (TEDx — "…"), (Talk — "…"), (Digital Innovation IDC, Lecture 3 — "…"); quotes are escaped here
    s = s.replace(/\((TEDx|Talk|Digital Innovation IDC)(?:,\s*Lecture\s+\d+)?\s*[—–-]+\s*(?:&quot;|“)[^()]*\)/gi, function (m, label) {
      return '<span class="su-cite su-cite-' + (/tedx/i.test(label) ? 'tedx' : 'outsource') + '" dir="ltr">' + m + '</span>';
    });
    return s;
  }

  /**
   * Minimal, safe Markdown → HTML. Escapes HTML first, then handles headers, bullet/numbered
   * lists, blockquotes, horizontal rules, paragraphs and line breaks. No raw HTML passes through.
   */
  function renderMarkdown(text) {
    var lines = escapeHtml(String(text || '').replace(/\r\n?/g, '\n')).split('\n');
    var html = '', para = [], list = null, quote = [];
    function flushPara() { if (para.length) { html += '<p>' + para.map(inlineMd).join('<br>') + '</p>'; para = []; } }
    function flushList() {
      if (!list) return;
      var start = list.type === 'ol' && list.start !== 1 ? ' start="' + list.start + '"' : '';
      html += '<' + list.type + start + '>' + list.items.map(function (i) { return '<li>' + inlineMd(i) + '</li>'; }).join('') +
        '</' + list.type + '>';
      list = null;
    }
    function flushQuote() { if (quote.length) { html += '<blockquote>' + quote.map(inlineMd).join('<br>') + '</blockquote>'; quote = []; } }
    function flushAll() { flushPara(); flushList(); flushQuote(); }

    for (var i = 0; i < lines.length; i++) {
      var line = lines[i].replace(/\s+$/, '');
      var m;
      if (!line.trim()) { flushAll(); continue; }
      if ((m = /^(#{1,4})\s+(.*)$/.exec(line))) {
        flushAll(); var lvl = m[1].length; html += '<h' + lvl + '>' + inlineMd(m[2]) + '</h' + lvl + '>'; continue;
      }
      if ((m = /^&gt;\s?(.*)$/.exec(line))) { flushPara(); flushList(); quote.push(m[1]); continue; }
      if ((m = /^\s*[-*•]\s+(.*)$/.exec(line))) {
        flushPara(); flushQuote();
        if (!list || list.type !== 'ul') { flushList(); list = { type: 'ul', items: [], start: 1 }; }
        list.items.push(m[1]); continue;
      }
      if ((m = /^\s*(\d{1,3})[.)]\s+(.*)$/.exec(line))) {
        flushPara(); flushQuote();
        if (!list || list.type !== 'ol') { flushList(); list = { type: 'ol', items: [], start: parseInt(m[1], 10) || 1 }; }
        list.items.push(m[2]); continue;
      }
      if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) { flushAll(); html += '<hr>'; continue; }
      flushList(); flushQuote(); para.push(line.trim());
    }
    flushAll();
    return html;
  }

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        if (k === 'class') node.className = attrs[k];
        else if (k === 'html') node.innerHTML = attrs[k];
        else if (k === 'text') node.textContent = attrs[k];
        else if (k.indexOf('on') === 0 && typeof attrs[k] === 'function') node.addEventListener(k.slice(2), attrs[k]);
        else if (attrs[k] !== null && attrs[k] !== undefined) node.setAttribute(k, attrs[k]);
      });
    }
    (children || []).forEach(function (c) { if (c) node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c); });
    return node;
  }

  var ICONS = {
    chat: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M21 12a8 8 0 0 1-8 8H7l-4 3v-5.5A8 8 0 1 1 21 12z" stroke-linejoin="round"/><path d="M8 11h8M8 14h5" stroke-linecap="round"/></svg>',
    close: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18" stroke-linecap="round"/></svg>',
    send: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 12h14M13 6l6 6-6 6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    refresh: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 12a8 8 0 1 1-2.34-5.66M20 4v5h-5" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    timer: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="13.5" r="7.5"/><path d="M12 9.5v4l2.5 2M9.5 2.5h5" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    copy: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="9" width="11" height="11" rx="2.5"/><path d="M5 15V6a2 2 0 0 1 2-2h9" stroke-linecap="round"/></svg>',
    check: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    ask: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3l1.9 4.6L18.5 9.5l-4.6 1.9L12 16l-1.9-4.6L5.5 9.5l4.6-1.9z" stroke-linejoin="round"/><path d="M18 15l.8 2.2L21 18l-2.2.8L18 21l-.8-2.2L15 18l2.2-.8z" stroke-linejoin="round"/></svg>'
  };

  // ---------------------------------------------------------------------------------------------
  // answer stopwatches: question sent → first words of the answer, and question sent → full answer
  // ---------------------------------------------------------------------------------------------
  var now = window.performance && performance.now ? function () { return performance.now(); } : Date.now;

  function fmtSecs(ms, lang) { return (ms / 1000).toFixed(1) + ' ' + (STR[lang] || STR.en).secs; }

  /** The two stopwatch chips. With `timing` ({firstMs, totalMs}) they show final values; without, blanks to fill live. */
  function timingChips(lang, timing) {
    var s = STR[lang] || STR.en;
    function chip(label, hint, ms) {
      var val = el('span', { class: 'su-sw-val', text: ms == null ? '' : fmtSecs(ms, lang) });
      return { node: el('span', { class: 'su-sw', title: hint, html: ICONS.timer }, [el('span', { text: label }), val]), val: val };
    }
    var first = chip(s.firstWords, s.firstWordsHint, timing && timing.firstMs);
    var total = chip(s.fullAnswer, s.fullAnswerHint, timing && timing.totalMs);
    return { node: el('div', { class: 'su-timing', dir: lang === 'he' ? 'rtl' : 'ltr' }, [first.node, total.node]), first: first, total: total };
  }

  /** Live stopwatches for one question: `markFirst()` when the first word streams in, `stop()` when the answer is done. */
  function startStopwatch(lang) {
    var t0 = now(), firstMs = null, chips = timingChips(lang, null);
    chips.node.setAttribute('aria-hidden', 'true');  // ticking text is noise for screen readers; the final values come with the answer
    chips.first.node.classList.add('su-running');
    chips.total.node.classList.add('su-running');
    function tick() {
      var ms = now() - t0;
      if (firstMs === null) chips.first.val.textContent = fmtSecs(ms, lang);
      chips.total.val.textContent = fmtSecs(ms, lang);
    }
    tick();
    var timer = setInterval(tick, 100);
    return {
      node: chips.node,
      markFirst: function () {
        if (firstMs !== null) return;
        firstMs = now() - t0;
        chips.first.val.textContent = fmtSecs(firstMs, lang);
        chips.first.node.classList.remove('su-running');
      },
      stop: function () {
        clearInterval(timer);
        var totalMs = now() - t0;
        if (firstMs === null) firstMs = totalMs;  // nothing was streamed: the whole answer appeared at once
        return { firstMs: Math.round(firstMs), totalMs: Math.round(totalMs) };
      },
      cancel: function () { clearInterval(timer); }
    };
  }

  // ---------------------------------------------------------------------------------------------
  // persistence (sessionStorage — per tab; thread_id links the tab to the server-side memory)
  // ---------------------------------------------------------------------------------------------
  var store = {
    load: function () {
      try {
        var raw = sessionStorage.getItem(config.storageKey);
        var v = raw ? JSON.parse(raw) : null;
        return v && typeof v === 'object' ? v : {};
      } catch (e) { return {}; }
    },
    save: function (state) {
      try {
        sessionStorage.setItem(config.storageKey, JSON.stringify({
          threadId: state.threadId || null,
          uiLang: state.uiLang,
          open: !!state.open,
          messages: (state.messages || []).slice(-40)
        }));
      } catch (e) { /* storage unavailable: nothing to do */ }
    },
    clear: function () { try { sessionStorage.removeItem(config.storageKey); } catch (e) { /* ignore */ } }
  };

  // ---------------------------------------------------------------------------------------------
  // state
  // ---------------------------------------------------------------------------------------------
  var saved = store.load();
  var pageLang = (document.documentElement.getAttribute('lang') || '').toLowerCase();
  var state = {
    threadId: saved.threadId || null,
    uiLang: config.lang === 'he' || config.lang === 'en' ? config.lang
      : (saved.uiLang || (pageLang.indexOf('he') === 0 ? 'he' : 'en')),
    open: !!saved.open,
    messages: Array.isArray(saved.messages) ? saved.messages : [],  // {role, text, status, citations, usedFallback, fallbackTier, elapsedMs, timing, traceId}
    busy: false,
    unread: 0
  };
  function t(key) { return (STR[state.uiLang] || STR.en)[key]; }

  // ---------------------------------------------------------------------------------------------
  // DOM
  // ---------------------------------------------------------------------------------------------
  var root, panel, messagesEl, inputEl, sendBtn, launcherWrap, badgeEl, titleEl, subtitleEl, footerEl, newBtn, closeBtn, launcher;

  function build() {
    if (config.loadCss && config.cssHref && !document.querySelector('link[data-smartup-widget-css]')) {
      var link = el('link', { rel: 'stylesheet', href: config.cssHref, 'data-smartup-widget-css': '1' });
      document.head.appendChild(link);
    }
    var inline = config.mode === 'inline';
    root = el('div', {
      class: 'su-widget su-theme-' + config.theme + (inline ? ' su-inline su-open' : ''),
      'data-position': config.position, 'data-mode': config.mode,
      dir: state.uiLang === 'he' ? 'rtl' : 'ltr'
    });
    if (config.primary) root.style.setProperty('--su-primary', config.primary);

    // panel
    titleEl = el('div', { class: 'su-title', text: config.title });
    subtitleEl = el('div', { class: 'su-subtitle', text: t('subtitle') });
    newBtn = el('button', { class: 'su-iconbtn', type: 'button', title: t('newChat'), 'aria-label': t('newChat'), html: ICONS.refresh, onclick: reset });
    closeBtn = el('button', { class: 'su-iconbtn', type: 'button', title: t('close'), 'aria-label': t('close'), html: ICONS.close, onclick: close });
    var header = el('div', { class: 'su-header' }, [
      el('div', { class: 'su-avatar', text: 'SU', 'aria-hidden': 'true' }),
      el('div', { class: 'su-title-wrap' }, [titleEl, subtitleEl]),
      newBtn, inline ? null : closeBtn
    ]);
    messagesEl = el('div', { class: 'su-messages', role: 'log', 'aria-live': 'polite', 'aria-relevant': 'additions' });
    inputEl = el('textarea', { class: 'su-input', rows: '1', placeholder: t('placeholder'), maxlength: String(config.maxChars), 'aria-label': t('placeholder') });
    sendBtn = el('button', { class: 'su-send', type: 'button', title: t('send'), 'aria-label': t('send'), html: ICONS.send, onclick: function () { send(); } });
    var composer = el('div', { class: 'su-composer' }, [el('div', { class: 'su-composer-box' }, [inputEl, sendBtn])]);
    footerEl = el('div', { class: 'su-footer', text: t(inline ? 'footerInline' : 'footer') });
    panel = el('div', { class: 'su-panel', role: 'dialog', 'aria-modal': 'false', 'aria-label': config.title }, [header, messagesEl, composer, footerEl]);

    // launcher
    launcher = el('button', { class: 'su-launcher', type: 'button', 'aria-label': t('open'), 'aria-expanded': 'false', onclick: toggle, html:
      '<span class="su-ico-chat">' + ICONS.chat + '</span><span class="su-ico-close">' + ICONS.close + '</span>' });
    badgeEl = el('span', { class: 'su-launcher-badge', 'aria-hidden': 'true' });
    launcherWrap = el('div', { class: 'su-launcher-wrap' }, [launcher, badgeEl]);

    root.appendChild(panel);
    if (!inline) root.appendChild(launcherWrap);
    var host = inline && config.target ? document.querySelector(config.target) : null;
    (host || document.body).appendChild(root);

    inputEl.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
      else if (e.key === 'Escape' && !inline) { close(); }
    });
    inputEl.addEventListener('input', function () {
      autosize();
      inputEl.setAttribute('dir', inputEl.value ? dirOf(inputEl.value) : (state.uiLang === 'he' ? 'rtl' : 'ltr'));
    });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && state.open && !inline) close(); });

    renderAll();
    if (inline) { state.open = true; open(true); setTimeout(function () { inputEl.focus(); }, 50); }
    else if (state.open) open(true);
  }

  function autosize() {
    inputEl.style.height = 'auto';
    inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
  }

  function applyLang() {
    root.setAttribute('dir', state.uiLang === 'he' ? 'rtl' : 'ltr');
    subtitleEl.textContent = t('subtitle');
    inputEl.placeholder = t('placeholder');
    inputEl.setAttribute('aria-label', t('placeholder'));
    sendBtn.title = t('send'); sendBtn.setAttribute('aria-label', t('send'));
    newBtn.title = t('newChat'); newBtn.setAttribute('aria-label', t('newChat'));
    closeBtn.title = t('close'); closeBtn.setAttribute('aria-label', t('close'));
    launcher.setAttribute('aria-label', state.open ? t('close') : t('open'));
    footerEl.textContent = t(config.mode === 'inline' ? 'footerInline' : 'footer');
    if (!inputEl.value) inputEl.setAttribute('dir', state.uiLang === 'he' ? 'rtl' : 'ltr');
  }

  // ---------------------------------------------------------------------------------------------
  // rendering
  // ---------------------------------------------------------------------------------------------
  function nearBottom() { return messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight < 80; }
  function scrollToBottom(force) { if (force || nearBottom()) messagesEl.scrollTop = messagesEl.scrollHeight; }

  var TIERS = { v2: true, v1: true, tedx: true, outsource: true };
  // only a trailing heading followed by nothing but list items (what the server appends) — a "Sources" heading
  // followed by prose is part of the answer and stays
  var SOURCES_SECTION_RE = /\n+[ \t]*#{1,4}[ \t]*(?:Sources|מקורות)[ \t]*:?[ \t]*(?:\n[ \t]*(?:[-*+•][^\n]*)?)+$/;

  function sourcesBlock(citations, lang) {
    if (!citations || !citations.length) return null;
    var label = (STR[lang] || STR.en)[citations.length > 1 ? 'sources' : 'source'] + ' · ' + citations.length;
    var items = citations.map(function (c) {
      var head = c.display === undefined || c.display === null ? ('Lecture ' + c.lecture_number) : c.display;  // '' = single talk
      var title = c.lecture_title ? '“' + c.lecture_title + '”' : '';
      var tier = TIERS[c.tier] ? c.tier : (c.course_version === 'v1' ? 'v1' : 'v2');  // saved before tiers existed
      return el('li', null, [
        el('span', { class: 'su-ver su-tier-' + tier, text: c.source || c.course_version || '' }),
        el('span', { text: head && title ? head + ' — ' + title : (head || title) })
      ]);
    });
    var details = el('details', { class: 'su-sources', dir: 'ltr' }, [el('summary', { text: label }), el('ul', null, items)]);
    if (citations.length <= 4) details.open = true;
    return details;
  }

  function copyText(text, btn, lang) {
    var s = STR[lang] || STR.en;
    function label(icon, text_) { btn.innerHTML = icon; btn.appendChild(el('span', { text: text_ })); }
    function done() {
      btn.classList.add('su-done'); label(ICONS.check, s.copied);
      clearTimeout(btn._suReset);  // a second click restarts the "Copied" period instead of cutting it short
      btn._suReset = setTimeout(function () { btn.classList.remove('su-done'); label(ICONS.copy, s.copy); }, 1600);
    }
    function legacyCopy() {  // http:// pages have no Clipboard API; a denied permission rejects it
      var ta = el('textarea', { readonly: '', 'aria-hidden': 'true' });
      ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
      document.body.appendChild(ta); ta.select();
      try { if (document.execCommand('copy')) done(); } catch (e) { /* copying is a convenience */ }
      document.body.removeChild(ta);
    }
    if (navigator.clipboard && window.isSecureContext) { navigator.clipboard.writeText(text).then(done, legacyCopy); return; }
    legacyCopy();
  }

  /** Create the DOM for one message; returns {wrap, bubble}. An answer is a card (text + sources) followed by its
   *  notes (the source that answered) and actions (stopwatches, copy, trace). */
  function messageNode(msg) {
    var lang = isHebrew(msg.text) ? 'he' : (msg.role === 'user' ? (state.uiLang) : (msg.language || state.uiLang));
    var wrap = el('div', { class: 'su-msg su-' + (msg.role === 'user' ? 'user' : 'bot') + (msg.status ? ' su-status-' + msg.status : '') });
    var bubble = el('div', { class: 'su-bubble', dir: dirOf(msg.text || '') });
    if (msg.role === 'user') {
      bubble.textContent = msg.text;
      wrap.appendChild(bubble);
      return { wrap: wrap, bubble: bubble };
    }
    // the server appends a "## Sources" list to the answer text (for API and CLI readers); the card shows the same
    // citations as its own sources block, so the text version is left out here
    var hasSources = !!(msg.citations && msg.citations.length);
    bubble.innerHTML = renderMarkdown(hasSources ? msg.text.replace(SOURCES_SECTION_RE, '') : msg.text);
    var card = el('div', { class: 'su-card' }, [bubble]);
    var src = sourcesBlock(msg.citations, lang);
    if (src) card.appendChild(src);
    wrap.appendChild(card);

    var strs = STR[lang] || STR.en;
    var metaBits = [];
    // the source that answered: TEDx / v1 / outside lectures (older saved messages only know "fallback used")
    if (msg.fallbackTier && strs.fallbackNotes[msg.fallbackTier]) metaBits.push(strs.fallbackNotes[msg.fallbackTier]);
    else if (msg.usedFallback && msg.fallbackTier === undefined) metaBits.push(strs.fallbackNote);  // saved before tiers
    if (msg.note) metaBits.push(msg.note);
    if (metaBits.length) wrap.appendChild(el('div', { class: 'su-meta' }, metaBits.map(function (b) { return el('span', { text: b }); })));

    var actions = [];
    if (msg.timing && config.timing) actions.push(timingChips(lang, msg.timing).node);
    if (msg.text && msg.status === 'answered') {
      var copyBtn = el('button', { class: 'su-copy', type: 'button', title: strs.copy, html: ICONS.copy });
      copyBtn.appendChild(el('span', { text: strs.copy }));
      copyBtn.addEventListener('click', function () { copyText(msg.text, copyBtn, lang); });
      actions.push(copyBtn);
    }
    if (msg.traceId) {  // only present when the API shares traces with this browser (DEBUG_DASHBOARD)
      actions.push(el('a', { class: 'su-trace-link', href: config.apiBase + '/dashboard#' + encodeURIComponent(msg.traceId),
        target: '_blank', rel: 'noopener', title: strs.traceHint, text: strs.trace }));
    }
    if (actions.length) wrap.appendChild(el('div', { class: 'su-actions' }, actions));
    return { wrap: wrap, bubble: bubble };
  }

  function suggestionButton(question) {
    var b = el('button', { class: 'su-suggestion', type: 'button', dir: dirOf(question), html: ICONS.ask });
    b.appendChild(el('span', { text: question }));
    b.addEventListener('click', function () { send(question); });
    return b;
  }

  /** Before the first question: a welcome screen with suggested questions (inline), or a welcome message (bubble). */
  function emptyState() {
    if (config.mode === 'inline') {
      var hero = el('div', { class: 'su-hero su-empty' }, [
        el('div', { class: 'su-hero-mark', text: 'SU', 'aria-hidden': 'true' }),
        el('h2', { class: 'su-hero-title', text: t('heroTitle') }),
        el('p', { class: 'su-hero-text', text: config.welcome || t('heroText') })
      ]);
      if (config.suggestions.length) {
        hero.appendChild(el('div', { class: 'su-hero-label', text: t('heroLabel') }));
        hero.appendChild(el('div', { class: 'su-suggest-grid' }, config.suggestions.map(suggestionButton)));
      }
      return hero;
    }
    var node = messageNode({ role: 'bot', text: config.welcome || t('welcome'), status: 'welcome' });
    node.wrap.classList.add('su-empty');
    if (config.suggestions.length) {
      node.wrap.appendChild(el('div', { class: 'su-suggest' }, config.suggestions.slice(0, 4).map(suggestionButton)));
    }
    return node.wrap;
  }

  function renderAll() {
    messagesEl.innerHTML = '';
    applyLang();
    if (!state.messages.length) messagesEl.appendChild(emptyState());
    else state.messages.forEach(function (m) { messagesEl.appendChild(messageNode(m).wrap); });
    scrollToBottom(true);
  }

  function pushMessage(msg) {
    var empty = messagesEl.querySelector('.su-empty');  // the welcome screen gives way to the conversation
    if (empty && empty.parentNode) empty.parentNode.removeChild(empty);
    state.messages.push(msg);
    store.save(state);
    var node = messageNode(msg);
    messagesEl.appendChild(node.wrap);
    scrollToBottom(true);
    return node;
  }

  function typingNode(label) {
    var wrap = el('div', { class: 'su-msg su-bot su-typing-wrap' }, [
      el('div', { class: 'su-card' }, [
        el('div', { class: 'su-bubble su-typing', 'aria-label': label }, [el('i'), el('i'), el('i'), el('span', { class: 'su-typing-label', text: label })])
      ])
    ]);
    messagesEl.appendChild(wrap);
    scrollToBottom(true);
    return wrap;
  }

  function setBusy(b) {
    state.busy = b;
    inputEl.disabled = b;
    sendBtn.disabled = b;
    if (!b) { inputEl.focus(); }
  }

  function setUnread(n) {
    state.unread = n;
    badgeEl.textContent = n > 9 ? '9+' : String(n);
    launcherWrap.classList.toggle('su-has-badge', n > 0 && !state.open);
  }

  // ---------------------------------------------------------------------------------------------
  // transport
  // ---------------------------------------------------------------------------------------------
  function BusyError() { this.name = 'BusyError'; this.message = 'busy'; }
  BusyError.prototype = Object.create(Error.prototype);

  function requestBody(text) {
    var body = { message: text };
    if (state.threadId) body.thread_id = state.threadId;
    return JSON.stringify(body);
  }

  /** Parse one SSE frame ("event: x\ndata: {...}") → {event, data}. */
  function parseFrame(frame) {
    var event = 'message', data = [];
    frame.split('\n').forEach(function (line) {
      if (!line || line.charAt(0) === ':') return;
      var idx = line.indexOf(':');
      var field = idx < 0 ? line : line.slice(0, idx);
      var value = idx < 0 ? '' : line.slice(idx + 1).replace(/^ /, '');
      if (field === 'event') event = value;
      else if (field === 'data') data.push(value);
    });
    if (!data.length) return null;
    var joined = data.join('\n');
    try { return { event: event, data: JSON.parse(joined) }; } catch (e) { return { event: event, data: { text: joined } }; }
  }

  /**
   * POST /chat/stream and dispatch SSE events. Resolves with the `final` payload, or null when
   * the stream ended without one. Rejects with BusyError on 429 and Error on other failures.
   */
  function streamChat(text, onToken, onError, onStatus) {
    return fetch(config.apiBase + '/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
      body: requestBody(text)
    }).then(function (res) {
      if (res.status === 429) throw new BusyError();
      if (!res.ok) throw new Error('HTTP ' + res.status);
      if (!res.body || !res.body.getReader) throw new Error('streaming unsupported');
      var reader = res.body.getReader();
      var decoder = new TextDecoder('utf-8');
      var buf = '', finalPayload = null;
      function handle(frame) {
        var ev = parseFrame(frame);
        if (!ev) return;
        if (ev.event === 'token') onToken(ev.data.text || '');
        else if (ev.event === 'status') { if (onStatus) onStatus(ev.data.text || ''); }  // fallback-chain progress
        else if (ev.event === 'final') finalPayload = ev.data;
        else if (ev.event === 'error') onError(ev.data && ev.data.message);
      }
      function pump() {
        return reader.read().then(function (r) {
          if (r.done) {
            if (buf.trim()) handle(buf);
            return finalPayload;
          }
          buf += decoder.decode(r.value, { stream: true }).replace(/\r\n?/g, '\n');
          var idx;
          while ((idx = buf.indexOf('\n\n')) >= 0) {
            handle(buf.slice(0, idx));
            buf = buf.slice(idx + 2);
          }
          return pump();
        });
      }
      return pump();
    });
  }

  function postChat(text) {
    return fetch(config.apiBase + '/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: requestBody(text)
    }).then(function (res) {
      if (res.status === 429) throw new BusyError();
      if (!res.ok) throw new Error('HTTP ' + res.status);
      return res.json();
    });
  }

  // ---------------------------------------------------------------------------------------------
  // conversation flow
  // ---------------------------------------------------------------------------------------------
  function send(textArg) {
    var text = (typeof textArg === 'string' ? textArg : inputEl.value).trim();
    if (!text || state.busy) return;
    if (text.length > config.maxChars) text = text.slice(0, config.maxChars);
    inputEl.value = ''; autosize();

    if (config.lang === 'auto') {
      var wanted = isHebrew(text) ? 'he' : 'en';
      if (wanted !== state.uiLang) { state.uiLang = wanted; applyLang(); store.save(state); }
    }
    pushMessage({ role: 'user', text: text });
    setBusy(true);

    var watch = config.timing ? startStopwatch(state.uiLang) : null;  // starts as the question is sent
    var typing = typingNode(t('searching'));
    if (watch) { typing.appendChild(watch.node); scrollToBottom(true); }
    var draft = '', tokens = 0, streamNode = null, rafPending = false, serverError = null;

    function ensureStreamNode() {
      if (streamNode) return;
      if (typing.parentNode) typing.parentNode.removeChild(typing);
      streamNode = messageNode({ role: 'bot', text: '' });
      streamNode.bubble.classList.add('su-streaming');
      if (watch) { watch.markFirst(); streamNode.wrap.appendChild(watch.node); }
      messagesEl.appendChild(streamNode.wrap);
    }
    function paint() {
      rafPending = false;
      if (!streamNode) return;
      streamNode.bubble.setAttribute('dir', dirOf(draft));
      streamNode.bubble.innerHTML = renderMarkdown(draft);
      scrollToBottom(false);
    }
    function onStatus(msg) {
      // "Not found in the v2 course — checking Yonatan's TEDx talk…": shown in the typing bubble until the answer starts
      if (!msg || streamNode) return;
      var label = typing.querySelector('.su-typing-label');
      if (label) { label.textContent = msg; label.setAttribute('dir', dirOf(msg)); }
      var bubble = typing.querySelector('.su-typing');
      if (bubble) bubble.setAttribute('aria-label', msg);
      scrollToBottom(false);
    }
    function onToken(tok) {
      if (!tok) return;
      tokens++; draft += tok;
      ensureStreamNode();
      if (!rafPending) { rafPending = true; (window.requestAnimationFrame || setTimeout)(paint); }
    }
    function cleanupTransient() {
      if (typing.parentNode) typing.parentNode.removeChild(typing);
      if (streamNode && streamNode.wrap.parentNode) streamNode.wrap.parentNode.removeChild(streamNode.wrap);
    }
    function finish(payload, note) {
      var timing = watch ? watch.stop() : null;
      cleanupTransient();
      if (payload && payload.thread_id) state.threadId = payload.thread_id;
      pushMessage({
        role: 'bot',
        text: payload.answer || '',
        status: payload.status || 'answered',
        language: payload.language || null,
        citations: payload.citations || [],
        usedFallback: !!payload.used_fallback,
        fallbackTier: payload.fallback_tier || null,
        elapsedMs: payload.elapsed_ms || 0,
        timing: timing,
        traceId: payload.trace_id || null,
        note: note || null
      });
      store.save(state);
      if (!state.open) setUnread(state.unread + 1);
      setBusy(false);
    }
    function fail(kind) {
      if (watch) watch.cancel();
      cleanupTransient();
      var msg = kind === 'busy' ? t('busy') : t('error');
      var node = messageNode({ role: 'bot', text: msg, status: 'error' });
      messagesEl.appendChild(node.wrap);
      scrollToBottom(true);
      setBusy(false);
    }
    function fallbackToPost(reason) {
      if (typing.parentNode === null && !streamNode) messagesEl.appendChild(typing);
      var label = typing.querySelector('.su-typing-label'); if (label) label.textContent = t('thinking');
      return postChat(text).then(function (payload) { finish(payload, null); })
        .catch(function (err) { fail(err instanceof BusyError ? 'busy' : 'error'); });
    }

    streamChat(text, onToken, function (message) { serverError = message || 'error'; }, onStatus)
      .then(function (finalPayload) {
        if (finalPayload) return finish(finalPayload, null);
        if (serverError) return fail('error');
        if (tokens > 0) {
          // stream broke after tokens arrived: keep what we have rather than re-running the model
          return finish({ answer: draft, status: 'answered', thread_id: state.threadId }, t('interrupted'));
        }
        return fallbackToPost('no-final');
      })
      .catch(function (err) {
        if (err instanceof BusyError) return fail('busy');
        if (tokens > 0) return finish({ answer: draft, status: 'answered', thread_id: state.threadId }, t('interrupted'));
        return fallbackToPost(String(err && err.message || err));
      });
  }

  function open(silent) {
    state.open = true;
    root.classList.add('su-open');
    launcher.setAttribute('aria-expanded', 'true');
    launcher.setAttribute('aria-label', t('close'));
    setUnread(0);
    store.save(state);
    scrollToBottom(true);
    if (!silent) setTimeout(function () { inputEl.focus(); }, 50);
  }
  function close() {
    if (config.mode === 'inline') return;  // the full-page chat box cannot be closed
    state.open = false;
    root.classList.remove('su-open');
    launcher.setAttribute('aria-expanded', 'false');
    launcher.setAttribute('aria-label', t('open'));
    store.save(state);
  }
  function toggle() { if (state.open) close(); else open(); }
  function reset() {
    if (state.busy) return;
    state.threadId = null;
    state.messages = [];
    store.save(state);
    renderAll();
    inputEl.focus();
  }
  function setTheme(vars) {
    Object.keys(vars || {}).forEach(function (k) { root.style.setProperty(k, vars[k]); });
  }
  function setSuggestions(list) {
    config.suggestions = (Array.isArray(list) ? list : []).map(String).filter(Boolean);
    if (root && !state.messages.length) renderAll();  // only the empty conversation shows them
  }

  // ---------------------------------------------------------------------------------------------
  // mount
  // ---------------------------------------------------------------------------------------------
  function mount() {
    if (document.body) build();
    else document.addEventListener('DOMContentLoaded', build);
  }
  window.SmartUpWidget = {
    _mounted: true, config: config,
    open: function () { open(); }, close: close, toggle: toggle, send: send, reset: reset, setTheme: setTheme,
    setSuggestions: setSuggestions,
    renderMarkdown: renderMarkdown, isHebrew: isHebrew
  };
  mount();
})();
