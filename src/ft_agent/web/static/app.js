const body = document.body;
const chatPane = document.querySelector(".chat-pane");
const composer = document.querySelector("#composer");
const input = document.querySelector("#messageInput");
const sendButton = document.querySelector("#sendButton");
const messages = document.querySelector("#messages");
const intro = document.querySelector("#intro");
const activityLog = document.querySelector("#activityLog");
const clearLog = document.querySelector("#clearLog");
const flowToggle = document.querySelector("#flowToggle");
const closeFlow = document.querySelector("#closeFlow");
const artifactPath = document.querySelector("#artifactPath");
const artifactSummary = document.querySelector("#artifactSummary");
const previewReport = document.querySelector("#previewReport");
const reportPreview = document.querySelector("#reportPreview");
const closePreview = document.querySelector("#closePreview");
const previewTitle = document.querySelector("#previewTitle");
const previewContent = document.querySelector("#previewContent");

let conversationState = null;
let running = false;
let streamingAssistant = null;
let lastRouterAction = null;
let currentReportPath = "";
let planCard = null;
let planStepItems = new Map();
let activePlanCapability = "";

if (window.matchMedia("(max-width: 1100px)").matches) {
  setFlowOpen(false);
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  submitMessage();
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 140)}px`;
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    submitMessage();
  }
});

flowToggle.addEventListener("click", () => {
  setFlowOpen(body.classList.contains("flow-collapsed"));
});

closeFlow.addEventListener("click", () => {
  setFlowOpen(false);
});

clearLog.addEventListener("click", () => {
  activityLog.replaceChildren();
});

previewReport.addEventListener("click", () => {
  openReportPreview();
});

closePreview.addEventListener("click", () => {
  reportPreview.hidden = true;
});

async function submitMessage() {
  const text = input.value.trim();
  if (!text || running) return;

  running = true;
  lastRouterAction = null;
  setInputEnabled(false);
  resetFlow();
  addMessage("user", text);
  streamingAssistant = null;
  input.value = "";
  input.style.height = "44px";
  intro.hidden = true;
  chatPane.classList.add("has-chat");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, state: conversationState }),
    });

    if (!response.ok || !response.body) {
      throw new Error(`Request failed: ${response.status}`);
    }

    await readNdjson(response.body);
  } catch (error) {
    addMessage("assistant", `Something went wrong: ${error.message}`);
    addActivity("error", error.message);
  } finally {
    running = false;
    streamingAssistant = null;
    setInputEnabled(true);
    input.focus();
  }
}

async function readNdjson(stream) {
  const reader = stream.getReader();
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

  if (buffer.trim()) {
    handleEvent(JSON.parse(buffer));
  }
}

function handleEvent(event) {
  if (event.type === "node") {
    handleNodeEvent(event);
    return;
  }

  if (event.type === "activity") {
    handleActivityEvent(event);
    return;
  }

  if (event.type === "plan") {
    showPlanCard(event.data ? event.data.plan : null);
    return;
  }

  if (event.type === "answer_delta") {
    appendAssistantDelta(event.delta || "");
    return;
  }

  if (event.type === "final") {
    conversationState = event.state || null;
    const currentAnswer = streamingAssistant ? streamingAssistant.dataset.raw || "" : "";
    let answerContent = streamingAssistant;
    if (event.answer && (!streamingAssistant || currentAnswer.trim() !== event.answer.trim())) {
      if (streamingAssistant) {
        setMessageContent(streamingAssistant, event.answer, true);
      } else {
        answerContent = addMessage("assistant", event.answer);
      }
    }
    addMessageMeta(answerContent, event.metrics);
    updateArtifact(event.answer || "", event.report_path || "");
    addActivity("complete", `Path: ${(event.path || []).join(" -> ")}`);
    markUnvisitedAsSkipped();
    completePlanCard();
    return;
  }

  if (event.type === "error") {
    const errorContent = addMessage("assistant", event.message || "Unknown error");
    addMessageMeta(errorContent, event.metrics);
    addActivity("error", event.message || "Unknown error");
    completePlanCard();
  }
}

function handleNodeEvent(event) {
  if (event.node === "RouterNode" && event.event === "node.end") {
    lastRouterAction = event.action || null;
    if (event.action === "clarify") {
      markAlias("clarify", "done", "asked");
    }
  }

  const cards = cardsForNode(event.node);
  if (!cards.length) return;

  if (event.event === "node.start") {
    for (const card of cards) {
      card.classList.remove("done", "skipped");
      card.classList.add("active");
      setCardStatus(card, "running");
    }
    addActivity("node", `${shortNode(event.node)} started`);
    return;
  }

  if (event.event === "node.end") {
    for (const card of cards) {
      card.classList.remove("active");
      card.classList.add("done");
      setCardStatus(card, event.action || "done");
    }
    addActivity("node", `${shortNode(event.node)} finished: ${event.action || "done"}`);
    if (event.node === "WriterNode") {
      completePlanCard();
    }
  }
}

function handleActivityEvent(event) {
  const data = event.data || {};
  if (event.event === "tool.call") {
    const name = data.name || "tool";
    markResource(name);
    activePlanCapability = planCapabilityForTool(name);
    if (activePlanCapability) {
      markPlanStep(activePlanCapability, "active");
    }
    addActivity("tool", `${name} called`);
  } else if (event.event === "tool.result") {
    if (activePlanCapability) {
      markPlanStep(activePlanCapability, "done");
      activePlanCapability = "";
    }
    addActivity("tool", `tool result${data.is_error ? " error" : ""}`);
  }
}

function cardsForNode(node) {
  if (node !== "FinalAnswerNode") {
    return [...document.querySelectorAll(`[data-node="${node}"]`)];
  }

  if (lastRouterAction === "irrelevant") {
    return [...document.querySelectorAll('[data-alias="irrelevant"]')];
  }
  if (lastRouterAction === "clarify") {
    return [...document.querySelectorAll('[data-alias="clarify"]')];
  }
  return [...document.querySelectorAll('[data-alias="report"]')];
}

function addMessage(role, text) {
  const wrapper = document.createElement("article");
  wrapper.className = `message ${role}`;

  const label = document.createElement("p");
  label.className = "message-label";
  label.textContent = role === "user" ? "You" : "ft-agent";

  const content = document.createElement("div");
  content.className = "message-content";
  if (role === "assistant") {
    content.classList.add("markdown-body");
  }
  setMessageContent(content, text, role === "assistant");

  wrapper.append(label, content);
  messages.append(wrapper);
  messages.scrollTop = messages.scrollHeight;
  return content;
}

function addMessageMeta(content, metrics) {
  if (!content || !metrics) return;
  const wrapper = content.closest(".message");
  if (!wrapper) return;
  const existing = wrapper.querySelector(".message-meta");
  if (existing) existing.remove();

  const elapsed = formatElapsed(metrics.elapsed_ms);
  const totalTokens = Number(metrics.total_tokens || 0);
  const calls = Number(metrics.llm_calls || 0);
  const promptTokens = Number(metrics.prompt_tokens || 0);
  const completionTokens = Number(metrics.completion_tokens || 0);
  const tokenText = totalTokens > 0
    ? `${totalTokens.toLocaleString()} tokens`
    : "tokens unavailable";
  const detail = totalTokens > 0
    ? `prompt ${promptTokens.toLocaleString()} · completion ${completionTokens.toLocaleString()}`
    : `${calls.toLocaleString()} LLM calls`;

  const meta = document.createElement("div");
  meta.className = "message-meta";
  meta.textContent = `${tokenText} · ${elapsed} · ${detail}`;
  wrapper.append(meta);
}

function appendAssistantDelta(delta) {
  if (!streamingAssistant) {
    streamingAssistant = addMessage("assistant", "");
  }
  const nextText = (streamingAssistant.dataset.raw || "") + delta;
  setMessageContent(streamingAssistant, nextText, true);
  messages.scrollTop = messages.scrollHeight;
}

function addActivity(kind, text) {
  const item = document.createElement("div");
  item.className = "activity-item";
  const label = kind === "node" ? "node" : kind === "tool" ? "tool" : kind;
  item.innerHTML = `<strong>${escapeHtml(label)}</strong> ${escapeHtml(text)}`;
  activityLog.append(item);
  activityLog.scrollTop = activityLog.scrollHeight;
}

function resetFlow() {
  completePlanCard();
  activePlanCapability = "";
  for (const card of document.querySelectorAll(".flow-card")) {
    card.classList.remove("active", "done", "skipped", "used");
    setCardStatus(card, "waiting");
  }
}

function markUnvisitedAsSkipped() {
  for (const card of document.querySelectorAll(".flow-card[data-node], .flow-card[data-alias]")) {
    if (!card.classList.contains("done") && !card.classList.contains("active")) {
      card.classList.add("skipped");
      setCardStatus(card, "skipped");
    }
  }
}

function markAlias(alias, className, status) {
  for (const card of document.querySelectorAll(`[data-alias="${alias}"]`)) {
    card.classList.remove("active", "skipped");
    card.classList.add(className);
    setCardStatus(card, status);
  }
}

function markResource(toolName) {
  const map = {
    search_science_knowledge_base: "science",
    search_template_knowledge_base: "template",
    write_file: "artifact",
    read_file: "artifact",
    edit_file: "artifact",
  };
  const resource = map[toolName];
  if (!resource) return;
  const card = document.querySelector(`[data-resource="${resource}"]`);
  if (card) card.classList.add("used");
}

function updateArtifact(answer, reportPath = "") {
  const match = answer.match(/Report path:\s*([^\n]+)/);
  currentReportPath = reportPath || (match ? match[1].trim() : "");
  const text = currentReportPath || "No report yet";
  artifactPath.textContent = text;
  artifactSummary.textContent = text;
  previewReport.hidden = !currentReportPath;
}

function showPlanCard(plan) {
  if (!plan || !Array.isArray(plan.steps) || !plan.steps.length) return;
  completePlanCard();

  const wrapper = document.createElement("article");
  wrapper.className = "message assistant plan-progress-message";

  const label = document.createElement("p");
  label.className = "message-label";
  label.textContent = "plan";

  const card = document.createElement("div");
  card.className = "plan-card";

  const title = document.createElement("div");
  title.className = "plan-card-title";
  title.innerHTML = `<strong>Execution plan</strong><span>tracking writer work</span>`;
  card.append(title);

  if (plan.summary) {
    const summary = document.createElement("p");
    summary.className = "plan-summary";
    summary.textContent = plan.summary;
    card.append(summary);
  }

  const list = document.createElement("ol");
  list.className = "plan-steps";
  planStepItems = new Map();
  for (const step of plan.steps) {
    const item = document.createElement("li");
    item.className = "plan-step";
    item.dataset.capability = step.capability || "";
    item.innerHTML = `
      <span class="plan-step-dot"></span>
      <div>
        <strong>${escapeHtml(step.id || "step")} · ${escapeHtml(step.capability || "task")}</strong>
        <p>${escapeHtml(step.instruction || step.expected_output || "")}</p>
      </div>
    `;
    list.append(item);
    if (step.capability && !planStepItems.has(step.capability)) {
      planStepItems.set(step.capability, []);
    }
    if (step.capability) {
      planStepItems.get(step.capability).push(item);
    }
  }
  card.append(list);

  wrapper.append(label, card);
  messages.append(wrapper);
  planCard = wrapper;
  messages.scrollTop = messages.scrollHeight;
}

function markPlanStep(capability, status) {
  const items = planStepItems.get(capability) || [];
  if (!items.length) return;
  const target = items.find((item) => !item.classList.contains("done")) || items[items.length - 1];
  if (status === "active") {
    target.classList.add("active");
    target.classList.remove("done");
  }
  if (status === "done") {
    target.classList.remove("active");
    target.classList.add("done");
  }
}

function completePlanCard() {
  if (planCard) {
    planCard.remove();
  }
  planCard = null;
  planStepItems = new Map();
  activePlanCapability = "";
}

function planCapabilityForTool(toolName) {
  const map = {
    search_science_knowledge_base: "search_science_knowledge_base",
    search_template_knowledge_base: "search_template_knowledge_base",
    write_file: "write_experimental_report",
    edit_file: "write_experimental_report",
  };
  return map[toolName] || "";
}

async function openReportPreview() {
  if (!currentReportPath) return;
  previewReport.disabled = true;
  previewReport.textContent = "Loading...";
  try {
    const response = await fetch(`/api/report?path=${encodeURIComponent(currentReportPath)}`);
    if (!response.ok) {
      throw new Error(`Unable to load report: ${response.status}`);
    }
    const data = await response.json();
    previewTitle.textContent = data.path || currentReportPath;
    previewContent.dataset.raw = data.content || "";
    previewContent.innerHTML = renderMarkdown(data.content || "");
    reportPreview.hidden = false;
  } catch (error) {
    addActivity("error", error.message);
  } finally {
    previewReport.disabled = false;
    previewReport.textContent = "Preview Markdown";
  }
}

function setCardStatus(card, status) {
  const statusEl = card.querySelector(".node-status");
  if (statusEl) statusEl.textContent = status;
}

function setInputEnabled(enabled) {
  input.disabled = !enabled;
  sendButton.disabled = !enabled;
}

function setFlowOpen(open) {
  body.classList.toggle("flow-collapsed", !open);
  flowToggle.setAttribute("aria-expanded", String(open));
  flowToggle.textContent = open ? "Flow" : "Open flow";
}

function shortNode(node) {
  return node.replace("Node", "");
}

function setMessageContent(element, text, markdown) {
  element.dataset.raw = text || "";
  if (markdown) {
    element.innerHTML = renderMarkdown(text || "");
  } else {
    element.textContent = text || "";
  }
}

function renderMarkdown(value) {
  const lines = String(value || "").replace(/\r\n?/g, "\n").split("\n");
  const html = [];
  let paragraph = [];
  let listType = "";
  let inCode = false;
  let codeLines = [];
  let index = 0;

  const flushParagraph = () => {
    if (!paragraph.length) return;
    html.push(`<p>${renderInline(paragraph.join(" "))}</p>`);
    paragraph = [];
  };

  const flushList = () => {
    if (!listType) return;
    html.push(`</${listType}>`);
    listType = "";
  };

  const openList = (type) => {
    if (listType === type) return;
    flushParagraph();
    flushList();
    listType = type;
    html.push(`<${type}>`);
  };

  while (index < lines.length) {
    const line = lines[index];
    const fence = line.match(/^```[\w-]*\s*$/);
    if (inCode) {
      if (fence) {
        html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
        inCode = false;
        codeLines = [];
      } else {
        codeLines.push(line);
      }
      index += 1;
      continue;
    }

    if (fence) {
      flushParagraph();
      flushList();
      inCode = true;
      codeLines = [];
      index += 1;
      continue;
    }

    if (!line.trim()) {
      flushParagraph();
      flushList();
      index += 1;
      continue;
    }

    const table = collectTable(lines, index);
    if (table) {
      flushParagraph();
      flushList();
      html.push(renderTable(table.headers, table.rows));
      index = table.nextIndex;
      continue;
    }

    if (/^\s*---+\s*$/.test(line)) {
      flushParagraph();
      flushList();
      html.push("<hr>");
      index += 1;
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      flushList();
      const level = Math.min(heading[1].length, 4);
      html.push(`<h${level}>${renderInline(heading[2].trim())}</h${level}>`);
      index += 1;
      continue;
    }

    const unordered = line.match(/^\s*[-*+]\s+(.+)$/);
    if (unordered) {
      openList("ul");
      html.push(`<li>${renderInline(unordered[1].trim())}</li>`);
      index += 1;
      continue;
    }

    const ordered = line.match(/^\s*\d+\.\s+(.+)$/);
    if (ordered) {
      openList("ol");
      html.push(`<li>${renderInline(ordered[1].trim())}</li>`);
      index += 1;
      continue;
    }

    flushList();
    paragraph.push(line.trim());
    index += 1;
  }

  if (inCode) {
    html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
  }
  flushParagraph();
  flushList();
  return html.join("");
}

function collectTable(lines, startIndex) {
  const rows = [];
  let index = startIndex;

  while (index < lines.length) {
    const line = lines[index].trim();
    if (!line.startsWith("|") || !line.includes("|")) break;
    const parsedRows = parsePipeRows(line);
    if (!parsedRows.length) break;
    rows.push(...parsedRows);
    index += 1;
  }

  const separatorIndex = rows.findIndex((row) => (
    row.length > 0 && row.every((cell) => /^:?-{3,}:?$/.test(cell.replace(/\s/g, "")))
  ));
  if (separatorIndex < 1) return null;

  const headers = rows[separatorIndex - 1];
  const bodyRows = rows
    .slice(separatorIndex + 1)
    .filter((row) => row.length === headers.length);
  if (!headers.length || !bodyRows.length) return null;

  return {
    headers,
    rows: bodyRows,
    nextIndex: index,
  };
}

function parsePipeRows(line) {
  const cells = line.split("|").map((cell) => cell.trim());
  const rows = [];
  let row = [];

  for (const cell of cells) {
    if (!cell) {
      if (row.length) {
        rows.push(row);
        row = [];
      }
      continue;
    }
    row.push(cell);
  }
  if (row.length) {
    rows.push(row);
  }
  return rows;
}

function renderTable(headers, rows) {
  const head = headers
    .map((cell) => `<th>${renderInline(cell)}</th>`)
    .join("");
  const body = rows
    .map((row) => `<tr>${row.map((cell) => `<td>${renderInline(cell)}</td>`).join("")}</tr>`)
    .join("");
  return `<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function formatElapsed(milliseconds) {
  const value = Number(milliseconds || 0);
  if (!Number.isFinite(value) || value <= 0) return "time unavailable";
  if (value < 1000) return `${Math.round(value)} ms`;
  return `${(value / 1000).toFixed(value < 10000 ? 1 : 0)} s`;
}

function renderInline(value) {
  return escapeHtml(value)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_match, label, url) => {
      const safe = safeLink(url);
      if (!safe) return label;
      return `<a href="${escapeHtml(safe)}" target="_blank" rel="noreferrer">${label}</a>`;
    });
}

function safeLink(value) {
  try {
    if (String(value).startsWith("#")) return value;
    const url = new URL(value, window.location.origin);
    if (["http:", "https:", "mailto:"].includes(url.protocol)) return url.href;
  } catch (_error) {
    return "";
  }
  return "";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

window.renderMarkdown = renderMarkdown;
