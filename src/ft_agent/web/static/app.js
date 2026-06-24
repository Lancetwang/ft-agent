const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const messages = document.querySelector("#messages");
const activity = document.querySelector("#activity");
const flowPanel = document.querySelector("#flow-panel");
const reportButton = document.querySelector("#preview-report");
const reportPreview = document.querySelector("#report-preview");

let state = {};
let currentAssistant = null;
let reportPath = null;

document.querySelector("#toggle-flow").addEventListener("click", () => {
  flowPanel.classList.toggle("hidden");
});

document.querySelector("#close-flow").addEventListener("click", () => {
  flowPanel.classList.add("hidden");
});

reportButton.addEventListener("click", async () => {
  if (!reportPath) return;
  const response = await fetch(`/api/report?path=${encodeURIComponent(reportPath)}`);
  const data = await response.json();
  reportPreview.innerHTML = renderMarkdown(data.content || "");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  appendMessage("YOU", text);
  currentAssistant = appendMessage("FT-AGENT", "");
  clearActiveNodes();

  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: text, state }),
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (line.trim()) handleEvent(JSON.parse(line));
    }
  }
});

function handleEvent(event) {
  if (event.type === "answer_delta") {
    currentAssistant.querySelector(".content").textContent += event.delta;
  } else if (event.type === "node") {
    setActiveNode(event.node);
  } else if (event.type === "activity") {
    addActivity(`${event.category || "event"} - ${event.event}`);
  } else if (event.type === "final") {
    state = event.state || {};
    reportPath = event.report_path || null;
    reportButton.disabled = !reportPath;
    reportButton.textContent = reportPath ? `Preview ${reportPath}` : "No report yet";
    addActivity(`complete - ${Math.round(event.metrics?.elapsed_ms || 0)} ms`);
  } else if (event.type === "error") {
    currentAssistant.querySelector(".content").textContent = event.message;
  }
}

function appendMessage(role, content) {
  const node = document.createElement("article");
  node.className = "message";
  node.innerHTML = `<div class="role">${role}</div><div class="content"></div>`;
  node.querySelector(".content").textContent = content;
  messages.appendChild(node);
  node.scrollIntoView({ block: "end", behavior: "smooth" });
  return node;
}

function setActiveNode(name) {
  clearActiveNodes();
  const node = document.querySelector(`[data-node="${name}"]`);
  if (node) node.classList.add("active");
}

function clearActiveNodes() {
  document.querySelectorAll(".node.active").forEach((node) => node.classList.remove("active"));
}

function addActivity(text) {
  const item = document.createElement("div");
  item.textContent = text;
  activity.prepend(item);
}

function renderMarkdown(markdown) {
  const lines = markdown.split(/\r?\n/);
  const html = [];
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    if (isTableStart(lines, index)) {
      const rows = [];
      while (index < lines.length && isTableRow(lines[index])) {
        if (!isSeparatorRow(lines[index])) rows.push(cells(lines[index]));
        index += 1;
      }
      index -= 1;
      html.push(table(rows));
    } else if (line.startsWith("### ")) {
      html.push(`<h3>${inline(line.slice(4))}</h3>`);
    } else if (line.startsWith("## ")) {
      html.push(`<h2>${inline(line.slice(3))}</h2>`);
    } else if (line.startsWith("# ")) {
      html.push(`<h1>${inline(line.slice(2))}</h1>`);
    } else if (line.trim()) {
      html.push(`<p>${inline(line)}</p>`);
    }
  }
  return html.join("");
}

function isTableStart(lines, index) {
  return isTableRow(lines[index]) && isSeparatorRow(lines[index + 1] || "");
}

function isTableRow(line) {
  return /^\s*\|.+\|\s*$/.test(line);
}

function isSeparatorRow(line) {
  return /^\s*\|?[\s:-]+\|[\s|:-]*$/.test(line);
}

function cells(line) {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => inline(cell.trim()));
}

function table(rows) {
  if (!rows.length) return "";
  const [head, ...body] = rows;
  const header = head.map((cell) => `<th>${cell}</th>`).join("");
  const bodyRows = body
    .map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`)
    .join("");
  return `<table><thead><tr>${header}</tr></thead><tbody>${bodyRows}</tbody></table>`;
}

function inline(text) {
  return escapeHtml(text).replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
