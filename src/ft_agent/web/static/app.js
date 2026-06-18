const body = document.body;
const chatPane = document.querySelector(".chat-pane");
const composer = document.querySelector("#composer");
const input = document.querySelector("#messageInput");
const sendButton = document.querySelector("#sendButton");
const messages = document.querySelector("#messages");
const intro = document.querySelector("#intro");
const runState = document.querySelector("#runState");
const activityLog = document.querySelector("#activityLog");
const clearLog = document.querySelector("#clearLog");
const flowToggle = document.querySelector("#flowToggle");
const closeFlow = document.querySelector("#closeFlow");
const artifactPath = document.querySelector("#artifactPath");
const artifactSummary = document.querySelector("#artifactSummary");

let conversationState = null;
let running = false;
let streamingAssistant = null;
let lastRouterAction = null;

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

async function submitMessage() {
  const text = input.value.trim();
  if (!text || running) return;

  running = true;
  lastRouterAction = null;
  setRunState("Working", true);
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
    setRunState("Idle", false);
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

  if (event.type === "answer_delta") {
    appendAssistantDelta(event.delta || "");
    return;
  }

  if (event.type === "final") {
    conversationState = event.state || null;
    if (event.answer && (!streamingAssistant || streamingAssistant.textContent.trim() !== event.answer.trim())) {
      if (streamingAssistant) {
        streamingAssistant.textContent = event.answer;
      } else {
        addMessage("assistant", event.answer);
      }
    }
    updateArtifact(event.answer || "");
    addActivity("complete", `Path: ${(event.path || []).join(" -> ")}`);
    markUnvisitedAsSkipped();
    return;
  }

  if (event.type === "error") {
    addMessage("assistant", event.message || "Unknown error");
    addActivity("error", event.message || "Unknown error");
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
  }
}

function handleActivityEvent(event) {
  const data = event.data || {};
  if (event.event === "tool.call") {
    const name = data.name || "tool";
    markResource(name);
    addActivity("tool", `${name} called`);
  } else if (event.event === "tool.result") {
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

  const content = document.createElement("p");
  content.className = "message-content";
  content.textContent = text;

  wrapper.append(label, content);
  messages.append(wrapper);
  messages.scrollTop = messages.scrollHeight;
  return content;
}

function appendAssistantDelta(delta) {
  if (!streamingAssistant) {
    streamingAssistant = addMessage("assistant", "");
  }
  streamingAssistant.textContent += delta;
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

function updateArtifact(answer) {
  const match = answer.match(/Report path:\s*([^\n]+)/);
  const text = match ? match[1].trim() : "No report yet";
  artifactPath.textContent = text;
  artifactSummary.textContent = text;
}

function setCardStatus(card, status) {
  const statusEl = card.querySelector(".node-status");
  if (statusEl) statusEl.textContent = status;
}

function setRunState(text, busy) {
  runState.textContent = text;
  runState.classList.toggle("busy", busy);
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

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
