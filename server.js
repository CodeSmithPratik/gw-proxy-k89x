/**
 * =============================================================================
 * GPU Gateway — Render Free Tier
 * =============================================================================
 *
 * Lightweight Express server that:
 *  1. Serves the dashboard UI for model management
 *  2. Receives client requests for LLM / ASR inference
 *  3. If GPU notebook is offline → triggers it via Kaggle API
 *  4. Notebook boots, loads models, opens tunnel, registers URL here
 *  5. Gateway proxies all subsequent requests to the notebook's tunnel
 *  6. Returns notebook responses to clients
 *
 * Endpoints:
 *  DASHBOARD:
 *    GET  /                    → Dashboard UI
 *
 *  PUBLIC (client-facing):
 *    GET  /health              → Gateway health
 *    GET  /api/status          → Notebook status + tunnel URL
 *    GET  /api/models          → Available model catalog
 *    POST /api/deploy          → Deploy with chosen models
 *    POST /api/terminate       → Terminate running notebook
 *    POST /api/trigger         → Manually trigger notebook (legacy)
 *    POST /api/asr             → Proxy: audio transcription
 *    POST /api/llm             → Proxy: text completion
 *    POST /api/llm/chat        → Proxy: chat completion
 *    GET  /api/notebook/health → Proxy: notebook health details
 *    GET  /api/keep-alive      → Cron endpoint: check + re-trigger if dead
 *
 *  INTERNAL (notebook → gateway):
 *    POST /webhook/register    → Notebook registers its tunnel URL
 *    POST /webhook/heartbeat   → Notebook keepalive ping
 * =============================================================================
 */

const express = require("express");
const multer = require("multer");
const fetch = require("node-fetch");
const FormData = require("form-data");
const fs = require("fs");
const path = require("path");

// ─── Configuration ───────────────────────────────────────────
if (process.loadEnvFile && fs.existsSync(path.join(__dirname, ".env"))) {
  process.loadEnvFile(path.join(__dirname, ".env"));
}

const PORT = process.env.PORT || 3000;
const SHARED_SECRET = process.env.SHARED_SECRET || "change-me-in-production";
const RENDER_EXTERNAL_URL = process.env.RENDER_EXTERNAL_URL;

// Dual Kaggle Accounts (Primary + Fallback)
const KAGGLE_ACCOUNTS = [];
if (process.env.KAGGLE_USERNAME && process.env.KAGGLE_KEY) {
  KAGGLE_ACCOUNTS.push({
    name: "Primary (" + process.env.KAGGLE_USERNAME + ")",
    username: process.env.KAGGLE_USERNAME,
    key: process.env.KAGGLE_KEY,
    slug: process.env.KAGGLE_KERNEL_SLUG || "gpu-inference-server",
  });
}
if (process.env.KAGGLE2_USERNAME && process.env.KAGGLE2_KEY) {
  KAGGLE_ACCOUNTS.push({
    name: "Fallback 2 (" + process.env.KAGGLE2_USERNAME + ")",
    username: process.env.KAGGLE2_USERNAME,
    key: process.env.KAGGLE2_KEY,
    slug: process.env.KAGGLE2_KERNEL_SLUG || "gpu-inference-server",
  });
}

// Groq API Fallback (Zero Cold-Start Inference while Kaggle boots)
const GROQ_API_KEY = process.env.GROQ_API_KEY || null;
const GROQ_WHISPER_MODEL = process.env.GROQ_WHISPER_MODEL || "whisper-large-v3";
const GROQ_LLM_MODEL = process.env.GROQ_LLM_MODEL || "llama-3.1-8b-instant";

// How long to wait before considering notebook dead (ms)
const HEARTBEAT_TIMEOUT = 10 * 60 * 1000; // 10 minutes
// Minimum gap between trigger attempts (ms)
const TRIGGER_COOLDOWN = 5 * 60 * 1000; // 5 minutes

// ─── Model Catalog ───────────────────────────────────────────
const MODEL_CATALOG = {
  llm: [
    {
      id: "phi3-mini",
      name: "Phi-3 Mini 3.8B",
      description: "Microsoft's compact yet powerful model. Best quality/size ratio.",
      repo_id: "bartowski/Phi-3-mini-4k-instruct-GGUF",
      filename: "Phi-3-mini-4k-instruct-Q4_K_M.gguf",
      vram_gb: 2.5,
      context_length: 4096,
      chat_template: "phi3",
    },
    {
      id: "llama3-8b",
      name: "Llama 3.1 8B",
      description: "Meta's latest. Strong reasoning and instruction following.",
      repo_id: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
      filename: "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
      vram_gb: 4.9,
      context_length: 8192,
      chat_template: "llama3",
    },
    {
      id: "mistral-7b",
      name: "Mistral 7B v0.2",
      description: "Excellent general-purpose. Great at code and reasoning.",
      repo_id: "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
      filename: "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
      vram_gb: 4.4,
      context_length: 8192,
      chat_template: "mistral",
    },
    {
      id: "gemma2-2b",
      name: "Gemma 2 2B",
      description: "Google's lightweight model. Fast with good quality.",
      repo_id: "bartowski/gemma-2-2b-it-GGUF",
      filename: "gemma-2-2b-it-Q4_K_M.gguf",
      vram_gb: 1.5,
      context_length: 8192,
      chat_template: "gemma",
    },
    {
      id: "tinyllama",
      name: "TinyLlama 1.1B",
      description: "Ultra-lightweight. Fastest inference, lowest VRAM.",
      repo_id: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
      filename: "tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
      vram_gb: 1.2,
      context_length: 2048,
      chat_template: "tinyllama",
    },
  ],
  asr: [
    {
      id: "large-v3",
      name: "Whisper Large V3",
      description: "Highest accuracy. 99+ languages supported.",
      model_size: "large-v3",
      vram_gb: 1.5,
    },
    {
      id: "medium",
      name: "Whisper Medium",
      description: "Good accuracy, significantly faster than Large.",
      model_size: "medium",
      vram_gb: 0.8,
    },
    {
      id: "small",
      name: "Whisper Small",
      description: "Balanced speed and accuracy for most tasks.",
      model_size: "small",
      vram_gb: 0.5,
    },
    {
      id: "base",
      name: "Whisper Base",
      description: "Fast with basic accuracy. Good for clear audio.",
      model_size: "base",
      vram_gb: 0.15,
    },
    {
      id: "tiny",
      name: "Whisper Tiny",
      description: "Ultra-fast. Minimal VRAM. Best for quick tests.",
      model_size: "tiny",
      vram_gb: 0.08,
    },
  ],
};

// ─── State ───────────────────────────────────────────────────
const state = {
  status: "offline", // offline | starting | online
  tunnelUrl: null,
  lastHeartbeat: null,
  lastTrigger: null,
  deployedModels: { llm: null, asr: null },
};

// ─── Express Setup ───────────────────────────────────────────
const app = express();
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ extended: true, limit: "50mb" }));

// Serve dashboard UI
app.use(express.static(path.join(__dirname, "public")));

// Multer for file uploads (ASR audio)
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 25 * 1024 * 1024 }, // 25 MB max
});

// CORS
app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.header("Access-Control-Allow-Headers", "Content-Type, Authorization");
  if (req.method === "OPTIONS") return res.sendStatus(200);
  next();
});

// Request logging
app.use((req, res, next) => {
  if (req.path !== "/health" && req.path !== "/api/status") {
    const ts = new Date().toISOString();
    console.log(`[${ts}] ${req.method} ${req.path}`);
  }
  next();
});

// ─── Helpers ─────────────────────────────────────────────────

function isNotebookHealthy() {
  if (state.status !== "online" || !state.tunnelUrl) return false;
  if (!state.lastHeartbeat) return false;
  return Date.now() - state.lastHeartbeat < HEARTBEAT_TIMEOUT;
}

/**
 * Trigger Kaggle notebook with specific model configuration.
 */
async function triggerKaggleNotebook(llmId, asrId) {
  if (KAGGLE_ACCOUNTS.length === 0) {
    throw new Error(
      "No Kaggle accounts configured (KAGGLE_USERNAME and KAGGLE_KEY environment variables required)"
    );
  }

  // Cooldown check
  if (state.lastTrigger && Date.now() - state.lastTrigger < TRIGGER_COOLDOWN) {
    const waitSec = Math.ceil(
      (TRIGGER_COOLDOWN - (Date.now() - state.lastTrigger)) / 1000
    );
    console.log(`⏳ Trigger cooldown, ${waitSec}s remaining`);
    return { already_triggering: true, retry_after_seconds: waitSec };
  }

  // Look up models
  const llmModel = llmId
    ? MODEL_CATALOG.llm.find((m) => m.id === llmId)
    : MODEL_CATALOG.llm[0];
  const asrModel = asrId
    ? MODEL_CATALOG.asr.find((m) => m.id === asrId)
    : MODEL_CATALOG.asr[0];

  if (!llmModel) throw new Error(`Unknown LLM model: ${llmId}`);
  if (!asrModel) throw new Error(`Unknown ASR model: ${asrId}`);

  // Read notebook script template
  let scriptContent;
  const scriptPath = path.join(__dirname, "notebook-script.py");
  try {
    scriptContent = fs.readFileSync(scriptPath, "utf-8");
  } catch {
    throw new Error("notebook-script.py not found in deployment");
  }

  // Inject configuration
  const renderUrl = RENDER_EXTERNAL_URL || `http://localhost:${PORT}`;
  scriptContent = scriptContent
    .replace(/\{\{RENDER_URL\}\}/g, renderUrl)
    .replace(/\{\{SHARED_SECRET\}\}/g, SHARED_SECRET)
    .replace(/\{\{LLM_REPO_ID\}\}/g, llmModel.repo_id)
    .replace(/\{\{LLM_FILENAME\}\}/g, llmModel.filename)
    .replace(/\{\{LLM_CONTEXT_LENGTH\}\}/g, String(llmModel.context_length))
    .replace(/\{\{LLM_CHAT_TEMPLATE\}\}/g, llmModel.chat_template)
    .replace(/\{\{LLM_NAME\}\}/g, llmModel.name)
    .replace(/\{\{WHISPER_MODEL_SIZE\}\}/g, asrModel.model_size)
    .replace(/\{\{ASR_NAME\}\}/g, asrModel.name);

  // Attempt triggering across configured Kaggle accounts (Primary with Fallback)
  let lastError = null;

  for (const account of KAGGLE_ACCOUNTS) {
    try {
      const kernelPushBody = {
        slug: `${account.username}/${account.slug}`,
        newTitle: "GPU Inference Server",
        text: scriptContent,
        language: "python",
        kernelType: "script",
        isPrivate: "true",
        enableGpu: "true",
        enableInternet: "true",
        datasetDataSources: [],
        competitionDataSources: [],
        kernelDataSources: [],
        categoryIds: [],
      };

      const authHeader = account.key.startsWith("KGAT_")
        ? `Bearer ${account.key}`
        : "Basic " + Buffer.from(`${account.username}:${account.key}`).toString("base64");

      console.log(
        `🚀 Triggering on account [${account.name}] — LLM: ${llmModel.name}, ASR: ${asrModel.name}`
      );

      const resp = await fetch("https://www.kaggle.com/api/v1/kernels/push", {
        method: "POST",
        headers: {
          Authorization: authHeader,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(kernelPushBody),
      });

      const respBody = await resp.json().catch(() => ({}));

      if (!resp.ok) {
        console.warn(`⚠️ Kaggle API push failed for ${account.name} (Status: ${resp.status}):`, respBody);
        lastError = new Error(`Kaggle API returned ${resp.status}: ${JSON.stringify(respBody)}`);
        continue; // Try next fallback account!
      }

      state.status = "starting";
      state.lastTrigger = Date.now();
      state.tunnelUrl = null;
      state.lastHeartbeat = null;
      state.deployedModels = { llm: llmModel.name, asr: asrModel.name };
      state.activeAccount = account.name;

      console.log(`✅ Kaggle notebook triggered on [${account.name}]:`, respBody);
      return {
        triggered: true,
        account: account.name,
        models: { llm: llmModel.name, asr: asrModel.name },
        kaggle_response: respBody,
      };
    } catch (err) {
      console.warn(`⚠️ Exception triggering ${account.name}:`, err.message);
      lastError = err;
    }
  }

  // If all accounts failed
  console.error("❌ All Kaggle accounts failed to trigger!");
  throw lastError || new Error("No Kaggle accounts available to trigger.");
}

/**
 * Proxy a JSON request to the notebook.
 */
/**
 * Auth headers from the API caller, forwarded to the notebook so its own
 * API key enforcement (fail-closed since server v2.1) can validate the
 * original client instead of trusting everything from the gateway.
 */
function clientAuthHeaders(req) {
  const h = {};
  if (req.headers.authorization) h["Authorization"] = req.headers.authorization;
  if (req.headers["x-api-key"]) h["X-API-Key"] = req.headers["x-api-key"];
  return h;
}

async function proxyJsonRequest(notebookPath, body, timeout = 120000, extraHeaders = {}) {
  const url = `${state.tunnelUrl}${notebookPath}`;
  console.log(`  ↳ Proxy → ${url}`);

  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...extraHeaders },
    body: JSON.stringify(body),
    timeout,
  });

  if (!resp.ok) {
    const errText = await resp.text().catch(() => "Unknown error");
    throw new Error(`Notebook returned ${resp.status}: ${errText}`);
  }
  return resp.json();
}

/**
 * Proxy a GET request to the notebook.
 */
async function proxyGetRequest(notebookPath, timeout = 10000, extraHeaders = {}) {
  const url = `${state.tunnelUrl}${notebookPath}`;
  const resp = await fetch(url, { method: "GET", timeout, headers: extraHeaders });
  if (!resp.ok) {
    const errText = await resp.text().catch(() => "Unknown error");
    throw new Error(`Notebook returned ${resp.status}: ${errText}`);
  }
  return resp.json();
}

/**
 * Proxy a file upload (ASR) to the notebook.
 */
async function proxyFileUpload(
  notebookPath,
  file,
  extraFields = {},
  timeout = 180000,
  extraHeaders = {}
) {
  const url = `${state.tunnelUrl}${notebookPath}`;
  console.log(`  ↳ Proxy file → ${url}`);

  const form = new FormData();
  form.append("file", file.buffer, {
    filename: file.originalname,
    contentType: file.mimetype,
  });
  for (const [key, val] of Object.entries(extraFields)) {
    if (val !== undefined && val !== null && val !== "") {
      form.append(key, String(val));
    }
  }

  const resp = await fetch(url, {
    method: "POST",
    body: form,
    headers: { ...form.getHeaders(), ...extraHeaders },
    timeout,
  });

  if (!resp.ok) {
    const errText = await resp.text().catch(() => "Unknown error");
    throw new Error(`Notebook returned ${resp.status}: ${errText}`);
  }
  return resp.json();
}

/**
 * Standard response when notebook is not ready.
 */
function notReadyResponse(res, triggerResult = null) {
  const response = {
    status: state.status,
    message:
      state.status === "starting"
        ? "GPU notebook is booting up. Models are loading. Please retry in 3-5 minutes."
        : "GPU notebook is offline. It has been triggered and will be ready in 3-5 minutes.",
    retry_after_seconds: 180,
  };
  if (triggerResult) response.trigger = triggerResult;
  return res.status(503).json(response);
}

/**
 * Handle proxy errors — mark notebook offline if connection fails.
 */
function handleProxyError(err, res) {
  console.error("Proxy error:", err.message);
  if (
    err.message.includes("ECONNREFUSED") ||
    err.message.includes("ENOTFOUND") ||
    err.message.includes("network") ||
    err.message.includes("socket hang up") ||
    err.message.includes("ETIMEDOUT")
  ) {
    state.status = "offline";
    state.tunnelUrl = null;
  }
  res.status(502).json({ error: "Notebook error", detail: err.message });
}

// ─── Groq Fallback Helpers ────────────────────────────────────

/**
 * Fallback to Groq API for speech recognition (Whisper Large V3).
 * Zero cold-start latency (~200ms).
 */
async function fallbackGroqASR(file, language) {
  if (!GROQ_API_KEY) {
    throw new Error("GROQ_API_KEY not configured");
  }

  const form = new FormData();
  form.append("file", file.buffer, {
    filename: file.originalname || "audio.wav",
    contentType: file.mimetype || "audio/wav",
  });
  form.append("model", GROQ_WHISPER_MODEL);
  form.append("response_format", "json");
  if (language && language !== "auto") {
    form.append("language", language);
  }

  console.log(`  ⚡ [Groq Fallback] Transcribing ${file.size} bytes with ${GROQ_WHISPER_MODEL}...`);
  const resp = await fetch("https://api.groq.com/openai/v1/audio/transcriptions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${GROQ_API_KEY}`,
      ...form.getHeaders(),
    },
    body: form,
    timeout: 30000,
  });

  if (!resp.ok) {
    const errText = await resp.text().catch(() => "Unknown error");
    throw new Error(`Groq ASR API error ${resp.status}: ${errText}`);
  }

  const data = await resp.json();
  return data;
}

/**
 * Fallback to Groq API for chat completion (Llama 3.1 8B Instant).
 */
async function fallbackGroqChat(messages, maxTokens = 512, temperature = 0.7) {
  if (!GROQ_API_KEY) {
    throw new Error("GROQ_API_KEY not configured");
  }

  console.log(`  ⚡ [Groq Fallback] Chat completion with ${GROQ_LLM_MODEL}...`);
  const resp = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${GROQ_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: GROQ_LLM_MODEL,
      messages,
      max_tokens: maxTokens,
      temperature,
    }),
    timeout: 30000,
  });

  if (!resp.ok) {
    const errText = await resp.text().catch(() => "Unknown error");
    throw new Error(`Groq Chat API error ${resp.status}: ${errText}`);
  }

  const data = await resp.json();
  return data;
}

/**
 * Fallback to Groq API for text completion.
 */
async function fallbackGroqCompletion(prompt, maxTokens = 512, temperature = 0.7) {
  const data = await fallbackGroqChat(
    [{ role: "user", content: prompt }],
    maxTokens,
    temperature
  );
  const text =
    data.choices && data.choices[0] && data.choices[0].message
      ? data.choices[0].message.content
      : "";
  return {
    text,
    content: text,
    model: data.model || GROQ_LLM_MODEL,
    fallback: "groq",
  };
}

// ─── Webhook Endpoints (Notebook → Gateway) ─────────────────

/**
 * POST /webhook/register & POST /api/register
 * Notebook calls this after tunnel is up to register its URL.
 */
app.post(["/webhook/register", "/api/register"], (req, res) => {
  const { tunnel_url, secret } = req.body;
  if (secret && SHARED_SECRET && secret !== SHARED_SECRET && SHARED_SECRET !== "change-me-in-production") {
    return res.status(403).json({ error: "Invalid secret" });
  }
  if (!tunnel_url) {
    return res.status(400).json({ error: "tunnel_url required" });
  }

  state.status = "online";
  state.tunnelUrl = tunnel_url;
  state.lastHeartbeat = Date.now();

  console.log(`\n${"=".repeat(60)}`);
  console.log(`🟢 NOTEBOOK REGISTERED — ${tunnel_url}`);
  console.log(`${"=".repeat(60)}\n`);

  res.json({ status: "registered" });
});

/**
 * POST /webhook/heartbeat & POST /api/heartbeat
 * Notebook pings this every 5 minutes to confirm it's alive.
 */
app.post(["/webhook/heartbeat", "/api/heartbeat"], (req, res) => {
  const { tunnel_url, secret } = req.body;
  if (secret && SHARED_SECRET && secret !== SHARED_SECRET && SHARED_SECRET !== "change-me-in-production") {
    return res.status(403).json({ error: "Invalid secret" });
  }

  state.lastHeartbeat = Date.now();
  if (tunnel_url && tunnel_url !== state.tunnelUrl) {
    console.log(`🔄 Tunnel URL updated: ${tunnel_url}`);
    state.tunnelUrl = tunnel_url;
  }
  state.status = "online";
  res.json({ status: "ok" });
});

// ─── Public API ──────────────────────────────────────────────

/** GET /health */
app.get("/health", (req, res) => {
  res.json({
    gateway: "healthy",
    notebook: {
      status: state.status,
      tunnel_url: state.tunnelUrl,
      last_heartbeat: state.lastHeartbeat
        ? new Date(state.lastHeartbeat).toISOString()
        : null,
    },
    groq_fallback: {
      enabled: !!GROQ_API_KEY,
      whisper_model: GROQ_WHISPER_MODEL,
      llm_model: GROQ_LLM_MODEL,
    },
  });
});

/** GET /api/status */
app.get("/api/status", (req, res) => {
  res.json({
    status: state.status,
    is_healthy: isNotebookHealthy(),
    tunnel_url: isNotebookHealthy() ? state.tunnelUrl : null,
    deployed_models: state.deployedModels,
    last_heartbeat: state.lastHeartbeat
      ? new Date(state.lastHeartbeat).toISOString()
      : null,
    uptime_since_trigger: state.lastTrigger
      ? Math.floor((Date.now() - state.lastTrigger) / 1000)
      : null,
    has_groq_fallback: !!GROQ_API_KEY,
    groq_models: {
      asr: GROQ_WHISPER_MODEL,
      llm: GROQ_LLM_MODEL,
    },
  });
});

/** GET /api/models — Model catalog for the dashboard */
/** GET /api/models — live catalog from the notebook; static fallback when offline */
app.get("/api/models", async (req, res) => {
  if (isNotebookHealthy()) {
    try {
      const data = await proxyGetRequest("/api/models", 10000, clientAuthHeaders(req));
      return res.json(data);
    } catch (err) {
      console.error("Live model catalog fetch failed, serving static:", err.message);
    }
  }
  res.json(MODEL_CATALOG);
});

/** GET /api/capabilities — proxy the notebook's versioned manifest (dynamic
 * model list + accepted wire formats) so clients negotiate through the
 * permanent gateway URL exactly as they would against the tunnel directly. */
app.get("/api/capabilities", async (req, res) => {
  if (!isNotebookHealthy()) {
    return res.status(503).json({ error: "Notebook offline — no capabilities available." });
  }
  try {
    const data = await proxyGetRequest("/api/capabilities", 10000, clientAuthHeaders(req));
    res.json(data);
  } catch (err) {
    handleProxyError(err, res);
  }
});

/** POST /api/deploy — Deploy notebook with chosen models */
app.post("/api/deploy", async (req, res) => {
  try {
    const { llm_id, asr_id } = req.body;

    if (isNotebookHealthy()) {
      return res.status(409).json({
        error: "Notebook is already running. Terminate first or use Redeploy.",
        status: "already_online",
      });
    }

    const result = await triggerKaggleNotebook(llm_id, asr_id);
    res.json({
      status: "triggered",
      message:
        "Notebook is starting with selected models. Ready in 3-5 minutes.",
      ...result,
    });
  } catch (err) {
    console.error("Deploy error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

/** POST /api/terminate — Terminate running notebook */
app.post("/api/terminate", async (req, res) => {
  if (!state.tunnelUrl && state.status === "offline") {
    return res.json({ status: "already_offline", message: "Nothing to terminate." });
  }

  // Try graceful shutdown via notebook endpoint
  if (state.tunnelUrl) {
    try {
      await fetch(`${state.tunnelUrl}/shutdown`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ secret: SHARED_SECRET }),
        timeout: 10000,
      });
      console.log("🛑 Shutdown signal sent to notebook");
    } catch (e) {
      console.log("⚠️ Could not reach notebook for graceful shutdown:", e.message);
    }
  }

  // Reset state
  state.status = "offline";
  state.tunnelUrl = null;
  state.lastHeartbeat = null;

  res.json({ status: "terminated", message: "Notebook terminated." });
});

/** POST /api/trigger — Legacy trigger (no model selection) */
app.post("/api/trigger", async (req, res) => {
  try {
    if (isNotebookHealthy()) {
      return res.json({
        status: "already_online",
        tunnel_url: state.tunnelUrl,
      });
    }
    const result = await triggerKaggleNotebook();
    res.json({ status: "triggered", ...result });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/** GET /api/keep-alive — Cron endpoint */
app.get("/api/keep-alive", async (req, res) => {
  console.log("🔍 Keep-alive check...");
  if (isNotebookHealthy()) {
    return res.json({ status: "healthy", action: "none" });
  }

  console.log("  ⚠️ Notebook appears dead, re-triggering...");
  try {
    const llmId = state.deployedModels.llm ? undefined : undefined; // use defaults
    const result = await triggerKaggleNotebook(llmId);
    res.json({ status: "retriggered", ...result });
  } catch (err) {
    console.error("  ❌ Re-trigger failed:", err.message);
    res.status(500).json({ status: "error", error: err.message });
  }
});

/** GET /api/notebook/health — Proxy to notebook's health endpoint */
app.get("/api/notebook/health", async (req, res) => {
  if (!isNotebookHealthy()) {
    return res.status(503).json({ error: "Notebook is not online" });
  }
  try {
    const data = await proxyGetRequest("/health", undefined, clientAuthHeaders(req));
    res.json(data);
  } catch (err) {
    handleProxyError(err, res);
  }
});

/** POST /api/asr, /asr, /v1/audio/transcriptions — Proxy: audio transcription with instant Groq fallback */
app.post(["/api/asr", "/asr", "/v1/audio/transcriptions"], upload.single("file"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No audio file. Use 'file' field." });
  }

  if (!isNotebookHealthy()) {
    // Auto-trigger Kaggle GPU in background so it warms up
    try {
      if (state.status !== "starting") triggerKaggleNotebook().catch(() => {});
    } catch (e) {
      console.error("Auto-trigger error:", e.message);
    }

    // If Groq Fallback is configured, fulfill immediately with 0ms cold-start!
    if (GROQ_API_KEY) {
      try {
        const groqResult = await fallbackGroqASR(req.file, req.body.language);
        groqResult.fallback = "groq";
        groqResult.gpu_status = state.status;
        return res.json(groqResult);
      } catch (err) {
        console.error("Groq fallback ASR error:", err.message);
      }
    }

    return notReadyResponse(res);
  }

  try {
    const result = await proxyFileUpload("/asr", req.file, {
      language: req.body.language,
      task: req.body.task,
    }, undefined, clientAuthHeaders(req));
    res.json(result);
  } catch (err) {
    if (GROQ_API_KEY) {
      console.log("⚡ Notebook proxy failed — trying Groq fallback ASR...");
      try {
        const groqResult = await fallbackGroqASR(req.file, req.body.language);
        groqResult.fallback = "groq";
        return res.json(groqResult);
      } catch (fallbackErr) {
        console.error("Groq fallback also failed:", fallbackErr.message);
      }
    }
    handleProxyError(err, res);
  }
});

/** POST /api/llm, /v1/completions — Proxy: text completion with instant Groq fallback */
app.post(["/api/llm", "/v1/completions"], async (req, res) => {
  if (!isNotebookHealthy()) {
    // Auto-trigger Kaggle in background
    try {
      if (state.status !== "starting") triggerKaggleNotebook().catch(() => {});
    } catch (e) {
      console.error("Auto-trigger error:", e.message);
    }

    if (GROQ_API_KEY) {
      try {
        const groqResult = await fallbackGroqCompletion(
          req.body.prompt,
          req.body.max_tokens || 512,
          req.body.temperature || 0.7
        );
        groqResult.gpu_status = state.status;
        return res.json(groqResult);
      } catch (err) {
        console.error("Groq fallback LLM error:", err.message);
      }
    }

    return notReadyResponse(res);
  }

  try {
    const result = await proxyJsonRequest("/llm", {
      prompt: req.body.prompt,
      max_tokens: req.body.max_tokens || 512,
      temperature: req.body.temperature || 0.7,
      stop: req.body.stop || null,
    }, undefined, clientAuthHeaders(req));
    res.json(result);
  } catch (err) {
    if (GROQ_API_KEY) {
      console.log("⚡ Notebook proxy failed — trying Groq fallback LLM...");
      try {
        const groqResult = await fallbackGroqCompletion(
          req.body.prompt,
          req.body.max_tokens || 512,
          req.body.temperature || 0.7
        );
        return res.json(groqResult);
      } catch (fallbackErr) {
        console.error("Groq fallback also failed:", fallbackErr.message);
      }
    }
    handleProxyError(err, res);
  }
});

/** POST /api/llm/chat, /v1/chat/completions — Proxy: chat completion with instant Groq fallback */
app.post(["/api/llm/chat", "/v1/chat/completions"], async (req, res) => {
  if (!isNotebookHealthy()) {
    // Auto-trigger Kaggle in background
    try {
      if (state.status !== "starting") triggerKaggleNotebook().catch(() => {});
    } catch (e) {
      console.error("Auto-trigger error:", e.message);
    }

    if (GROQ_API_KEY) {
      try {
        const groqResult = await fallbackGroqChat(
          req.body.messages,
          req.body.max_tokens || 512,
          req.body.temperature || 0.7
        );
        groqResult.fallback = "groq";
        groqResult.gpu_status = state.status;
        return res.json(groqResult);
      } catch (err) {
        console.error("Groq fallback chat error:", err.message);
      }
    }

    return notReadyResponse(res);
  }

  try {
    const result = await proxyJsonRequest("/v1/chat/completions", {
      messages: req.body.messages,
      max_tokens: req.body.max_tokens || 512,
      temperature: req.body.temperature || 0.7,
    }, undefined, clientAuthHeaders(req));
    res.json(result);
  } catch (err) {
    if (GROQ_API_KEY) {
      console.log("⚡ Notebook proxy failed — trying Groq fallback chat...");
      try {
        const groqResult = await fallbackGroqChat(
          req.body.messages,
          req.body.max_tokens || 512,
          req.body.temperature || 0.7
        );
        groqResult.fallback = "groq";
        return res.json(groqResult);
      } catch (fallbackErr) {
        console.error("Groq fallback also failed:", fallbackErr.message);
      }
    }
    handleProxyError(err, res);
  }
});

// ─── Start Server ────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n${"=".repeat(60)}`);
  console.log(`🌐 GPU Gateway running on port ${PORT}`);
  console.log(`${"=".repeat(60)}`);
  console.log(`  Dashboard:  http://localhost:${PORT}`);
  console.log(`  Health:     GET  /health`);
  console.log(`  Status:     GET  /api/status`);
  console.log(`  Models:     GET  /api/models`);
  console.log(`  Deploy:     POST /api/deploy`);
  console.log(`  Terminate:  POST /api/terminate`);
  console.log(`  ASR:        POST /api/asr`);
  console.log(`  LLM:        POST /api/llm`);
  console.log(`  LLM Chat:   POST /api/llm/chat`);
  console.log(`  Keep-alive: GET  /api/keep-alive`);
  console.log(`  Groq Fallback: ${GROQ_API_KEY ? "⚡ ENABLED (" + GROQ_WHISPER_MODEL + " / " + GROQ_LLM_MODEL + ")" : "DISABLED (Set GROQ_API_KEY for 0-cold-start)"}`);
  console.log(`${"=".repeat(60)}\n`);
});
