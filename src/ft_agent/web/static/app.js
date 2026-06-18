const nodes = ["RouterNode", "PlannerNode", "WriterNode", "SupervisorNode", "FinalAnswerNode"];

const composer = document.querySelector("#composer");
const input = document.querySelector("#messageInput");
const sendButton = document.querySelector("#sendButton");
const messages = document.querySelector("#messages");
const intro = document.querySelector("#intro");
const runState = document.querySelector("#runState");
const activityLog = document.querySelector("#activityLog");
const clearLog = document.querySelector("#clearLog");
const artifactPath = document.querySelector("#artifactPath");

let conversationState = null;
let running = false;
let streamingAssistant = null;

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  submitMessage();
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    submitMessage();
  }
});

clearLog.addEventListener("click", () => {
  activityLog.replaceChildren();
});

async function submitMessage() {
  const text = input.value.trim();
  if (!text || running) return;

  running = true;
  setRunState("Working", true);
  setInputEnabled(false);
  resetNodes();
  addMessage("user", text);
  streamingAssistant = null;
  input.value = "";
  input.style.height = "auto";
  intro.hidden = true;

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
    addActivity("complete", `Path: ${(event.path || []).join(" → ")}`);
    markUnvisitedAsSkipped();
    return;
  }

  if (event.type === "error") {
    addMessage("assistant", event.message || "Unknown error");
    addActivity("error", event.message || "Unknown error");
  }
}

function handleNodeEvent(event) {
  const row = document.querySelector(`[data-node="${event.node}"]`);
  if (!row) return;
  const status = row.querySelector(".node-status");

  if (event.event === "node.start") {
    row.classList.remove("done", "skipped");
    row.classList.add("active");
    status.textContent = "running";
    addActivity("node", `${shortNode(event.node)} started`);
    return;
  }

  if (event.event === "node.end") {
    row.classList.remove("active");
    row.classList.add("done");
    status.textContent = event.action || "done";
    addActivity("node", `${shortNode(event.node)} finished: ${event.action || "done"}`);
  }
}

function handleActivityEvent(event) {
  const data = event.data || {};
  if (event.event === "tool.call") {
    addActivity("tool", `${data.name || "tool"} called`);
  } else if (event.event === "tool.result") {
    addActivity("tool", `tool result${data.is_error ? " error" : ""}`);
  }
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

function resetNodes() {
  for (const node of nodes) {
    const row = document.querySelector(`[data-node="${node}"]`);
    row.classList.remove("active", "done", "skipped");
    row.querySelector(".node-status").textContent = "waiting";
  }
}

function markUnvisitedAsSkipped() {
  for (const node of nodes) {
    const row = document.querySelector(`[data-node="${node}"]`);
    if (!row.classList.contains("done")) {
      row.classList.add("skipped");
      row.querySelector(".node-status").textContent = "skipped";
    }
  }
}

function updateArtifact(answer) {
  const match = answer.match(/Report path:\s*([^\n]+)/);
  artifactPath.textContent = match ? match[1].trim() : "No report yet";
}

function setRunState(text, busy) {
  runState.textContent = text;
  runState.classList.toggle("busy", busy);
}

function setInputEnabled(enabled) {
  input.disabled = !enabled;
  sendButton.disabled = !enabled;
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
