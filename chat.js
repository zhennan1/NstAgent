"use strict";
// Floating chat about the paper, shown on the home page only.
const Chat = {
  el: null, open: false, busy: false, tipTimer: null,
  msgs: (() => { try { return JSON.parse(sessionStorage.getItem("nst_chat") || "[]"); } catch { return []; } })(),
};

function chatSave() { try { sessionStorage.setItem("nst_chat", JSON.stringify(Chat.msgs.slice(-30))); } catch { } }

// Minimal Markdown: paragraphs, lists, **bold**, *italic*, `code`, links. Input is escaped first.
function chatMarkdown(text) {
  const inline = s => esc(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<i>$2</i>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
    .replace(/(^|[\s(])(https?:\/\/[^\s)<]+)/g, '$1<a href="$2" target="_blank" rel="noopener">$2</a>');
  const out = [];
  let list = null, table = null;
  const flushTable = () => {
    if (!table) return;
    const rows = table.filter(r => !/^\s*\|?\s*:?-{2,}/.test(r));
    const cells = r => r.trim().replace(/^\||\|$/g, "").split("|").map(c => inline(c.trim()));
    const [head, ...rest] = rows;
    out.push(`<div class="tbl"><table><thead><tr>${cells(head).map(c => `<th>${c}</th>`).join("")}</tr></thead><tbody>${rest.map(r => `<tr>${cells(r).map(c => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
    table = null;
  };
  for (const raw of String(text).split("\n")) {
    const line = raw.trimEnd();
    if (/^\s*\|.*\|\s*$/.test(line)) { if (list) { out.push(list.close); list = null; } (table = table || []).push(line); continue; }
    flushTable();
    const m = /^\s*(?:[-*•]|\d+[.)])\s+(.*)$/.exec(line);
    if (m) {
      const ordered = /^\s*\d/.test(line);
      if (!list || list.ordered !== ordered) { if (list) out.push(list.close); list = { ordered, close: ordered ? "</ol>" : "</ul>" }; out.push(ordered ? "<ol>" : "<ul>"); }
      out.push(`<li>${inline(m[1])}</li>`);
      continue;
    }
    if (list) { out.push(list.close); list = null; }
    if (!line.trim()) continue;
    const h = /^#{1,4}\s+(.*)$/.exec(line);
    out.push(h ? `<p><b>${inline(h[1])}</b></p>` : `<p>${inline(line)}</p>`);
  }
  flushTable();
  if (list) out.push(list.close);
  return out.join("");
}

function chatBuild() {
  const el = document.createElement("div");
  el.id = "chat";
  el.className = "chat";
  el.innerHTML = `
    <div class="chat-tip" role="status"></div>
    <button class="chat-fab" aria-label="Chat">
      <svg viewBox="0 0 24 24" width="26" height="26" aria-hidden="true"><path fill="currentColor" d="M12 3C6.5 3 2 6.9 2 11.7c0 2.6 1.3 4.9 3.4 6.5-.1 1.2-.6 2.6-1.7 3.6 2.1-.1 3.9-.9 5.1-1.9 1 .3 2.1.4 3.2.4 5.5 0 10-3.9 10-8.6S17.5 3 12 3z"/></svg>
    </button>
    <section class="chat-panel" aria-live="polite">
      <header><div><b class="chat-title"></b><div class="chat-sub"></div></div><button class="chat-x" aria-label="Close">×</button></header>
      <div class="chat-body"></div>
      <form class="chat-form"><textarea rows="1" maxlength="2000"></textarea><button type="submit" class="chat-send" aria-label="Send">
        <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path fill="currentColor" d="M3 20.5 21 12 3 3.5l.02 6.6L15 12 3.02 13.9z"/></svg></button></form>
    </section>`;
  document.body.appendChild(el);
  Chat.el = el;
  $(".chat-fab", el).onclick = () => chatToggle(true);
  $(".chat-x", el).onclick = () => chatToggle(false);
  const ta = $("textarea", el);
  ta.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); chatSend(); } });
  ta.addEventListener("input", () => { ta.style.height = "auto"; ta.style.height = Math.min(ta.scrollHeight, 120) + "px"; });
  $(".chat-form", el).onsubmit = e => { e.preventDefault(); chatSend(); };
}

function chatText() {
  if (!Chat.el) return;
  $(".chat-tip", Chat.el).textContent = t("chat.tip");
  $(".chat-title", Chat.el).textContent = t("chat.title");
  $(".chat-sub", Chat.el).textContent = t("chat.sub");
  $("textarea", Chat.el).placeholder = t("chat.placeholder");
  chatRender();
}

function chatRender() {
  const body = $(".chat-body", Chat.el);
  let html = `<div class="msg bot"><div class="bubble">${esc(t("chat.hello"))}</div></div>`;
  if (!Chat.msgs.length) {
    html += `<div class="chat-sugs">${t("chat.sugs").split("|").map(q => `<button class="chip">${esc(q)}</button>`).join("")}</div>`;
  }
  html += Chat.msgs.map(m => `<div class="msg ${m.role === "user" ? "me" : "bot"}${m.error ? " err" : ""}"><div class="bubble">${m.role === "user" ? esc(m.content).replace(/\n/g, "<br>") : chatMarkdown(m.content)}</div></div>`).join("");
  if (Chat.busy) html += `<div class="msg bot"><div class="bubble typing"><i></i><i></i><i></i></div></div>`;
  body.innerHTML = html;
  $$(".chat-sugs .chip", body).forEach(b => b.onclick = () => { $("textarea", Chat.el).value = b.textContent; chatSend(); });
  body.scrollTop = body.scrollHeight;
}

function chatToggle(open) {
  Chat.open = open;
  Chat.el.classList.toggle("open", open);
  Chat.el.classList.remove("tip");
  if (open) { chatRender(); setTimeout(() => $("textarea", Chat.el).focus(), 200); }
}

async function chatSend() {
  const ta = $("textarea", Chat.el);
  const q = ta.value.trim();
  if (!q || Chat.busy) return;
  ta.value = ""; ta.style.height = "auto";
  Chat.msgs = Chat.msgs.filter(m => !m.error);
  Chat.msgs.push({ role: "user", content: q });
  Chat.busy = true; chatSave(); chatRender();
  let reply, error = false;
  try {
    const r = await fetch(api("api/chat"), { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: Chat.msgs.map(({ role, content }) => ({ role, content })) }) });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error);
    reply = j.reply;
  } catch { reply = t("chat.error"); error = true; }
  Chat.busy = false;
  const msg = { role: "assistant", content: "", error };
  Chat.msgs.push(msg);
  // Reveal the answer progressively.
  const step = Math.max(2, Math.ceil(reply.length / 90));
  for (let i = 0; i < reply.length; i += step) {
    msg.content = reply.slice(0, i + step);
    chatRender();
    await new Promise(r => setTimeout(r, 16));
  }
  msg.content = reply;
  if (error) Chat.msgs.pop(), Chat.msgs.push({ role: "assistant", content: reply, error: true });
  chatSave(); chatRender();
}

// Show the floating button on the home page only, and pop the hint now and then.
function chatShow(visible) {
  if (!Chat.el) { chatBuild(); chatText(); }
  Chat.el.classList.toggle("shown", visible);
  clearTimeout(Chat.tipTimer);
  if (!visible) { chatToggle(false); return; }
  const cycle = (delay) => {
    Chat.tipTimer = setTimeout(() => {
      if (!Chat.open && Chat.el.classList.contains("shown")) {
        Chat.el.classList.add("tip");
        setTimeout(() => Chat.el.classList.remove("tip"), 5000);
      }
      cycle(30000);
    }, delay);
  };
  cycle(2500);
}
