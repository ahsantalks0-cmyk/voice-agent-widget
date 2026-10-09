/* ============================================================
   Shine Dental Test — Voice Widget (TEST version)
   Connects to Gemini Live API via ephemeral token.
   Live voice conversation + CRM booking via function calling
   (check_slots, book_appointment → voice-agent-backend /api/*).
   ============================================================ */

(function () {
  "use strict";

  // ── Config ────────────────────────────────────────────────
  const API_BASE = "https://voice-agent-backend.ahsanvoice.workers.dev";
  const TOKEN_URL = API_BASE + "/get-token";
  const CHECK_SLOTS_URL = API_BASE + "/api/check-slots";
  const BOOK_URL = API_BASE + "/api/book";
  const WS_BASE = "wss://generativelanguage.googleapis.com/ws";
  const WS_PATH = "/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContentConstrained";
  const MODEL = "models/gemini-3.8-live";
  // Turn-taking speed. "fast" = 300ms silence commit (snappy, default).
  // Add ?vad=default to test.html to compare against Google's default.
  const VAD_PARAM = new URLSearchParams(location.search).get("vad") || "fast";
  // turn-end silence threshold; "default" = leave it to Google
  const SILENCE_MS = VAD_PARAM === "fast" ? 300 : parseInt(VAD_PARAM, 10) || 0;
  const INPUT_RATE = 16000; // Hz — Gemini expects 16kHz input
  const OUTPUT_RATE = 24000; // Hz — Gemini sends 24kHz output

  // ── State ─────────────────────────────────────────────────
  let state = "idle"; // idle | connecting | listening | speaking
  let micStream = null;
  let audioCtx = null;
  let sourceNode = null;
  let scriptNode = null;
  let analyserNode = null;
  let ws = null;
  let outputQueue = [];
  let outputPlaying = false;
  let clientId = null;
  let utteranceStartTime = 0;
  let setupCompleteReceived = false;
  let audioChunkCount = 0;
  let micLevelTimer = null;
  let bargeInTimer = null;
  let bargeInHits = 0;
  let nextPlayTime = 0;
  let activeSources = new Set();
  let playbackGeneration = 0;
  let playbackEndTimer = null;
  let lastPlaybackStartAt = 0;
  let micSuppressed = false; // used by the deterministic voice test hook
  let injection = null; // deterministic voice test: PCM injected on the audio clock
  let lastSpeechSentAt = 0; // performance.now() when we last sent USER speech PCM
  let lastReplyDelayMs = null; // user stopped talking → first reply audio
  let lastSpeechEndedAt = 0; // when the injected user speech actually ended
  // Pre-roll: while the agent speaks the mic is muted to the server (echo fix), so
  // we keep the last 600ms of user audio here. On an interruption we send it
  // FIRST — the server then hears the START of the user's sentence, not just
  // its tail. A truncated tail is what made her "answer late" deep into a call.
  const PREROLL_BYTES = Math.floor(INPUT_RATE * 2 * 0.6); // 600ms of 16kHz 16-bit
  let preRoll = [];
  let preRollBytes = 0;
  let lastTurnLatencyMs = null; // speech end → first reply audio (the real metric)
  let micLogTick = 0;
  let lastUserTranscript = "";
  let lastModelTranscript = "";
  let wsMsgCount = 0;
  let lastSentInfo = "(none)";
  let lastToolCalls = []; // { name, args, result } — used by tests + console logs

  // ── Today's date/weekday in Asia/Karachi (UTC+5, no DST) ───
  function pktClock() {
    const now = new Date();
    try {
      const parts = {};
      for (const p of new Intl.DateTimeFormat("en-GB", {
        timeZone: "Asia/Karachi",
        weekday: "long",
        day: "numeric",
        month: "long",
        year: "numeric",
      }).formatToParts(now)) {
        parts[p.type] = p.value;
      }
      const iso = new Intl.DateTimeFormat("en-CA", {
        timeZone: "Asia/Karachi",
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
      }).format(now); // en-CA → YYYY-MM-DD
      return { weekday: parts.weekday, pretty: `${parts.day} ${parts.month} ${parts.year}`, iso };
    } catch (e) {
      // Deterministic fallback: shift +5h and read the UTC fields.
      const p = new Date(now.getTime() + 5 * 60 * 60 * 1000);
      const WD = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
      const MO = ["January", "February", "March", "April", "May", "June", "July",
        "August", "September", "October", "November", "December"];
      const padNum = (n) => String(n).padStart(2, "0");
      return {
        weekday: WD[p.getUTCDay()],
        pretty: `${p.getUTCDate()} ${MO[p.getUTCMonth()]} ${p.getUTCFullYear()}`,
        iso: `${p.getUTCFullYear()}-${padNum(p.getUTCMonth() + 1)}-${padNum(p.getUTCDate())}`,
      };
    }
  }

  const PKT_TODAY = pktClock();

  // ── Agent brain (downloaded at session start) ───────────────────
  // The clinic edits its knowledge base / personality in the dashboard;
  // GET /api/config hands it back. DEFAULT_BRAIN is the safety net: if the
  // fetch fails or is slow the agent still boots with a working brain.
  const DEFAULT_BRAIN = {
    agent_name: "Aashi",
    tone: "warm, friendly and caring",
    greeting: "",
    business_info: "",
    booking_rules: "",
    custom_instructions: "",
  };
  let brain = DEFAULT_BRAIN;

  async function loadBrain(id) {
    if (!id) {
      console.log("[VoiceWidget] No data-client — using default brain");
      return { ...DEFAULT_BRAIN };
    }
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 3500); // slow = fall back
      const resp = await fetch(
        `${API_BASE}/api/config?client=${encodeURIComponent(id)}`,
        { signal: ctrl.signal }
      );
      clearTimeout(timer);
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      const cfg = await resp.json();
      if (!cfg || typeof cfg !== "object" || Array.isArray(cfg)) {
        throw new Error("bad config shape");
      }
      return { ...DEFAULT_BRAIN, ...cfg };
    } catch (err) {
      console.log("[VoiceWidget] Brain fetch failed — using default brain:",
        err && err.message);
      return { ...DEFAULT_BRAIN };
    }
  }

  // PROTECTED base template: the clinic's words get merged in, the booking
  // flow stays fixed. Weekday/date are computed fresh at every session start.
  function buildSystemInstruction(cfg) {
    const b = cfg && typeof cfg === "object" ? cfg : DEFAULT_BRAIN;
    const now = pktClock();
    const name = String(b.agent_name || DEFAULT_BRAIN.agent_name).trim();
    const tone = String(b.tone || DEFAULT_BRAIN.tone).trim();
    const greeting =
      String(b.greeting || "").trim() ||
      `Assalam o Alaikum! Main ${name} hoon. Kya madad kar sakti hoon?`;

    const lines = [
      `You are ${name}, the friendly voice assistant of the clinic. ALWAYS speak Pakistani Urdu. Tone: ${tone}. Be warm and brief.`,
      `Today is ${now.weekday}, ${now.pretty} (Asia/Karachi time)`,
    ];
    const business = String(b.business_info || "").trim();
    if (business) lines.push(`Business info: ${business}`);
    lines.push(
      "",
      "BOOKING FLOW — follow strictly:",
      "1. When the user wants an appointment, ask which day if not given. Convert 'kal' and 'parson' into real dates using today's date.",
      "2. Call check_slots. Offer ONLY the slots it returns, spoken naturally in Urdu. NEVER invent slots. If multiple doctors come back, ask which doctor first, then offer that doctor's slots. If zero slots, apologize and suggest another day.",
      "3. Ask the customer's name and phone number.",
      "4. Read back everything for confirmation.",
      "5. Only after an explicit yes, call book_appointment with doctor_name and slot in ISO with +05:00 offset.",
      "6. If slot_taken: apologize, call check_slots again, offer fresh options.",
      "7. On success: confirm warmly — repeat name, doctor, day and time."
    );
    const rules = String(b.booking_rules || "").trim();
    if (rules) lines.push(`Clinic booking rules (follow strictly): ${rules}`);
    lines.push("");
    const extra = String(b.custom_instructions || "").trim();
    if (extra) {
      lines.push(
        `Extra instructions from the clinic — follow them even if written in Urdu or English: ${extra}`
      );
      lines.push("");
    }
    lines.push(`Greet the user like this when the conversation starts: ${greeting}`);
    lines.push("Urdu time words: 'shaam' = 17:00-19:00, 'raat' = 19:00-21:00.");
    return lines.join("\n");
  }

  // ── Function declarations the model can call ──
  const TOOLS = [
    {
      functionDeclarations: [
        {
          name: "check_slots",
          description:
            "Get the free appointment slots for one day at this clinic. Always call this before offering any time to the user.",
          parameters: {
            type: "OBJECT",
            properties: {
              date: {
                type: "STRING",
                description: "The day to check, in YYYY-MM-DD format (Asia/Karachi).",
              },
            },
            required: ["date"],
          },
        },
        {
          name: "book_appointment",
          description:
            "Book the appointment once the customer has confirmed name, phone, doctor and time out loud.",
          parameters: {
            type: "OBJECT",
            properties: {
              doctor_name: {
                type: "STRING",
                description:
                  "Doctor chosen by the user, exactly as returned by check_slots (e.g. 'Dr. Ayesha').",
              },
              name: { type: "STRING", description: "Customer's name." },
              phone: { type: "STRING", description: "Customer's phone number." },
              slot_iso: {
                type: "STRING",
                description:
                  "Chosen slot as ISO 8601 with the +05:00 offset, e.g. 2026-10-09T18:30:00+05:00.",
              },
              service: {
                type: "STRING",
                description: "Service requested, if mentioned (e.g. checkup, cleaning).",
              },
              notes: { type: "STRING", description: "Any extra note from the customer." },
            },
            required: ["doctor_name", "name", "phone", "slot_iso"],
          },
        },
      ],
    },
  ];

  // ── DOM refs ──────────────────────────────────────────────
  const btn = document.createElement("button");
  btn.className = "ast-voice-btn";
  btn.innerHTML = `<span class="ast-mic-icon">&#x1F3A4;</span>`;
  btn.setAttribute("aria-label", "Voice assistant");
  // document.body is null when this script runs from <head> — which is exactly
  // what WordPress and many CMSes do. Mount on <html> instead of crashing so
  // the widget embeds cleanly on any site.
  const mountAt = document.body || document.documentElement;
  console.log("[VoiceWidget] mounting on", mountAt === document.body ? "body" : "html");
  mountAt.appendChild(btn);

  const panel = document.createElement("div");
  panel.className = "ast-voice-panel";
  panel.style.display = "none";
  panel.innerHTML = `
    <div class="ast-status"></div>
    <button class="ast-end-btn" aria-label="End call">End</button>
  `;
  mountAt.appendChild(panel);

  const statusEl = panel.querySelector(".ast-status");
  const endBtn = panel.querySelector(".ast-end-btn");

  // ── Position panel above the button, within viewport ──────
  function positionPanel() {
    const r = btn.getBoundingClientRect();
    const panelW = panel.offsetWidth || 220;
    const panelH = panel.offsetHeight || 80;
    let left = r.left + r.width / 2 - panelW / 2;
    let top = r.top - panelH - 14;
    // Keep within viewport
    left = Math.max(10, Math.min(left, window.innerWidth - panelW - 10));
    top = Math.max(10, top);
    panel.style.left = left + "px";
    panel.style.top = top + "px";
  }
  window.addEventListener("resize", positionPanel);
  window.addEventListener("scroll", positionPanel);
  window.addEventListener("load", positionPanel);

  // Delay first position until DOM is ready
  setTimeout(positionPanel, 50);

  // ── Styling ───────────────────────────────────────────────
  const style = document.createElement("style");
  style.textContent = `
    .ast-voice-btn {
      position: fixed;
      bottom: 24px;
      right: 24px;
      width: 60px;
      height: 60px;
      border-radius: 50%;
      border: none;
      background: #1a1a2e;
      color: #fff;
      font-size: 26px;
      cursor: pointer;
      box-shadow: 0 4px 16px rgba(0,0,0,0.35);
      display: flex;
      align-items: center;
      justify-content: center;
      transition: transform 0.15s, background 0.15s;
      z-index: 999999;
    }
    .ast-voice-btn:hover { transform: scale(1.06); }
    .ast-voice-btn:active { transform: scale(0.95); }
    .ast-voice-btn.listening { background: #e63946; }
    .ast-voice-btn.speaking { background: #2a9d8f; }

    .ast-voice-panel {
      position: fixed;
      background: #16213e;
      color: #e0e0e0;
      padding: 14px 18px;
      border-radius: 14px;
      box-shadow: 0 6px 24px rgba(0,0,0,0.45);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 14px;
      z-index: 999999;
      min-width: 200px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .ast-status { line-height: 1.4; }
    .ast-end-btn {
      background: #e63946;
      color: #fff;
      border: none;
      border-radius: 8px;
      padding: 6px 14px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      align-self: flex-start;
    }
    .ast-end-btn:hover { background: #c1121f; }
  `;
  document.head.appendChild(style);

  // ── UI helpers ────────────────────────────────────────────
  function setStatus(text, mode) {
    statusEl.textContent = text;
    btn.classList.remove("listening", "speaking");
    if (mode) btn.classList.add(mode);
    state = mode || "idle";
  }

  function showPanel(show) {
    panel.style.display = show ? "flex" : "none";
    if (show) positionPanel();
  }

  function updateMicIcon(mode) {
    const icon = btn.querySelector(".ast-mic-icon");
    if (mode === "listening") icon.textContent = "\u{1F50A}"; // 🎤
    else if (mode === "speaking") icon.textContent = "\u{1F509}"; // 🔔
    else icon.textContent = "\u{1F3A4}"; // 🎙️
  }

  // ── Mic + token + WS flow ────────────────────────────────
  btn.addEventListener("click", async () => {
    if (state === "listening" || state === "speaking") {
      endSession();
      return;
    }
    showPanel(true);
    setStatus("Connecting...", "listening");
    updateMicIcon("listening");
    await startSession();
  });

  endBtn.addEventListener("click", endSession);

  async function startSession() {
    try {
      // 1. Read clientId from script tag
      clientId = document.querySelector('script[data-client]')?.getAttribute("data-client");
      console.log("[VoiceWidget] clientId:", clientId);

      // 1b. Download the agent's brain BEFORE connecting to Gemini. Failures
      // and timeouts fall back inside loadBrain, so this can never break.
      brain = await loadBrain(clientId);
      console.log("Brain loaded:", brain);

      // 2. Request mic
      setStatus("Requesting mic...", "listening");
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      console.log("[VoiceWidget] Mic granted");

      // 3. Get ephemeral token
      setStatus("Getting token...", "listening");
      const tokenResp = await fetch(TOKEN_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ uses: 1 }),
      });
      if (!tokenResp.ok) {
        const errText = await tokenResp.text();
        throw new Error(`Token request failed: ${tokenResp.status} ${errText}`);
      }
      const tokenData = await tokenResp.json();
      const token = tokenData.token;
      if (!token) throw new Error("No token in response");
      console.log("[VoiceWidget] Token received:", token.slice(0, 20) + "...");

      // 4. Setup audio graph (mic capture + analysis)
      setStatus("Setting up audio...", "listening");
      await setupAudioGraph(micStream);

      // 5. Start barge-in detection (poll mic level while Gemini speaks)
      startBargeInDetection();

      // 5. Open WebSocket
      setStatus("Connecting to Gemini...", "listening");
      const wsUrl = WS_BASE + WS_PATH + "?access_token=" + encodeURIComponent(token);
      ws = new WebSocket(wsUrl);

      ws.onopen = onWsOpen;
      ws.onmessage = onWsMessage;
      ws.onerror = onWsError;
      ws.onclose = onWsClose;

    } catch (err) {
      console.error("[VoiceWidget] Failed to start session:", err);
      setStatus("Error: " + err.message, "listening");
      updateMicIcon();
      btn.classList.remove("listening");
    }
  }

  function endSession() {
    console.log("[VoiceWidget] Ending session");
    if (micLevelTimer) { clearInterval(micLevelTimer); micLevelTimer = null; }
    if (bargeInTimer) { clearInterval(bargeInTimer); bargeInTimer = null; }
    if (playbackEndTimer) { clearTimeout(playbackEndTimer); playbackEndTimer = null; }
    bargeInHits = 0;
    setupCompleteReceived = false;
    if (ws) { ws.close(); ws = null; }
    if (micStream) { micStream.getTracks().forEach((t) => t.stop()); micStream = null; }
    if (audioCtx && audioCtx.state !== "closed") audioCtx.close();
    audioCtx = null;
    scriptNode = null;
    analyserNode = null;
    sourceNode = null;
    outputQueue = [];
    stopOutputPlayback();
    showPanel(false);
    setStatus("", "");
    updateMicIcon();
    state = "idle";
  }

  // ── Audio graph setup ─────────────────────────────────────
  async function setupAudioGraph(stream) {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      if (audioCtx.state === "suspended") await audioCtx.resume();
    }

    sourceNode = audioCtx.createMediaStreamSource(stream);

    // ScriptProcessorNode for capturing mic audio and sending to Gemini
    // 1024 samples per callback ≈ 21ms at 48kHz (low capture latency)
    scriptNode = audioCtx.createScriptProcessor(1024, 1, 1);
    sourceNode.connect(scriptNode);
    // Pull the node through a zero-gain node so onaudioprocess fires,
    // but mic audio can NEVER leak to the speakers (no echo).
    const mute = audioCtx.createGain();
    mute.gain.value = 0;
    scriptNode.connect(mute);
    mute.connect(audioCtx.destination);

    scriptNode.onaudioprocess = (e) => {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      if (!setupCompleteReceived) return; // docs: wait for setupComplete first
      if (injection) { pumpInjection(); return; } // real-time injected test audio
      if (micSuppressed) return; // test hook is injecting PCM instead
      const input = e.inputBuffer.getChannelData(0);
      // Resample from context rate (usually 48000) → 16000 Hz
      const resampled = resampleLinear(input, audioCtx.sampleRate, INPUT_RATE);
      if (resampled.length === 0) return;
      // Convert Float32 → 16-bit PCM
      const pcm = new Int16Array(resampled.length);
      let rms = 0;
      for (let i = 0; i < resampled.length; i++) {
        const s = Math.max(-1, Math.min(1, resampled[i]));
        pcm[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        rms += s * s;
      }
      rms = Math.sqrt(rms / resampled.length);
      audioChunkCount++;
      if (audioChunkCount <= 3 || audioChunkCount % 100 === 0) {
        console.log(
          "[VoiceWidget] Audio OUT #" + audioChunkCount,
          "srcRate:", audioCtx.sampleRate,
          "→ sent samples:", resampled.length,
          "@16kHz, RMS:", rms.toFixed(4)
        );
      }
      const bytes = new Uint8Array(pcm.buffer);
      if (outputPlaying) {
        // half-duplex: don't stream while the agent speaks (kills the echo loop),
        // but remember it so an interruption isn't missing its first words
        pushPreRoll(bytes);
        return;
      }
      sendPcmBytes(bytes, true);
    };

    // Analyser for barge-in detection
    analyserNode = audioCtx.createAnalyser();
    analyserNode.fftSize = 256;
    sourceNode.connect(analyserNode);

    console.log("[VoiceWidget] Audio graph ready — context rate:", audioCtx.sampleRate, "Hz → resampling to", INPUT_RATE, "Hz");

    // Log mic level every second so we can verify capture works
    if (micLevelTimer) clearInterval(micLevelTimer);
    micLevelTimer = setInterval(() => {
      if (!analyserNode || !ws || ws.readyState !== WebSocket.OPEN) return;
      const data = new Uint8Array(analyserNode.frequencyBinCount);
      analyserNode.getByteTimeDomainData(data);
      let peak = 0;
      for (let i = 0; i < data.length; i++) {
        const v = Math.abs(data[i] - 128) / 128;
        if (v > peak) peak = v;
      }
      if (++micLogTick % 5 === 0) {
        console.log("[VoiceWidget] Mic level peak:", peak.toFixed(3), "| sent chunks:", audioChunkCount);
      }
    }, 1000);
  }

  // Linear-interpolation resampler: src float32 array → dst rate
  function resampleLinear(input, srcRate, dstRate) {
    if (srcRate === dstRate || input.length === 0) return input;
    const ratio = srcRate / dstRate;
    const newLen = Math.floor(input.length / ratio);
    if (newLen <= 0) return new Float32Array(0);
    const out = new Float32Array(newLen);
    for (let i = 0; i < newLen; i++) {
      const pos = i * ratio;
      const idx = Math.floor(pos);
      const frac = pos - idx;
      const a = input[idx];
      const b = idx + 1 < input.length ? input[idx + 1] : a;
      out[i] = a + (b - a) * frac;
    }
    return out;
  }

  // Generic raw-PCM sender (used by the deterministic voice test hook)
  function sendPcmBytes(bytes, isSpeech) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    let bin = "";
    for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
    const payload = JSON.stringify({
      realtimeInput: { audio: { data: btoa(bin), mimeType: "audio/pcm;rate=" + INPUT_RATE } },
    });
    ws.send(payload);
    wsMsgCount++;
    lastSentInfo = (isSpeech ? "speech" : "silence") + " pcm bytes=" + bytes.length + " json=" + payload.length;
    if (isSpeech) lastSpeechSentAt = performance.now();
  }

  // ── Deterministic voice test: inject PCM paced by the AUDIO clock ──
  // (background-tab timer throttling cannot distort the stream this way)
  const INJECT_CHUNK_BYTES = Math.floor(INPUT_RATE * 2 * 0.1); // 100ms

  function startInjection(buf, speakEndOffset) {
    injection = {
      buf,
      offset: 0,
      speakEndOffset,
      startCtxTime: null,
      doneAt: 0,
    };
  }

  function pumpInjection() {
    const inj = injection;
    if (!inj || !audioCtx) return;
    if (inj.startCtxTime == null) inj.startCtxTime = audioCtx.currentTime;
    const elapsed = Math.max(0, audioCtx.currentTime - inj.startCtxTime);
    const targetBytes = Math.floor(elapsed * INPUT_RATE * 2);
    const limit = Math.min(targetBytes, inj.buf.length);
    const MIN_CHUNK = 320; // 10ms — don't spam tiny messages
    while (limit - inj.offset >= MIN_CHUNK) {
      let end = Math.min(inj.offset + INJECT_CHUNK_BYTES, limit);
      if ((end - inj.offset) % 2) end -= 1; // 16-bit PCM must stay sample-aligned
      if (end <= inj.offset) break;
      const silent = inj.offset >= inj.speakEndOffset;
      sendPcmBytes(inj.buf.subarray(inj.offset, end), !silent);
      inj.offset = end;
      if (!inj.speechEndedAt && inj.offset >= inj.speakEndOffset) {
        inj.speechEndedAt = performance.now();
        lastSpeechEndedAt = inj.speechEndedAt;
      }
    }
    // final aligned flush once the whole buffer has been scheduled
    if (targetBytes >= inj.buf.length - 1) {
      let end = inj.buf.length;
      if ((end - inj.offset) % 2) end -= 1;
      if (end > inj.offset) {
        const silent = inj.offset >= inj.speakEndOffset;
        sendPcmBytes(inj.buf.subarray(inj.offset, end), !silent);
        inj.offset = end;
      }
    }
    if (inj.offset >= inj.buf.length - 1) {
      inj.doneAt = performance.now();
      injection = null;
      micSuppressed = false; // hand the mic back to the live stream
      console.log("[VoiceWidget] Test injection finished");
    }
  }

  // ── Pre-roll buffer helpers ───────────────────────────────
  function pushPreRoll(bytes) {
    const copy = bytes.slice();
    preRoll.push(copy);
    preRollBytes += copy.length;
    while (preRollBytes > PREROLL_BYTES && preRoll.length > 1) {
      preRollBytes -= preRoll.shift().length;
    }
  }

  function sendPreRoll() {
    if (!preRoll.length) return;
    const ms = Math.round((preRollBytes / (INPUT_RATE * 2)) * 1000);
    console.log(
      "[VoiceWidget] Flushing pre-roll",
      preRollBytes,
      "bytes (~" + ms + "ms) so the server hears the START of the user's speech"
    );
    for (const chunk of preRoll) sendPcmBytes(chunk, true);
    preRoll = [];
    preRollBytes = 0;
  }

  function clearPreRoll() {
    preRoll = [];
    preRollBytes = 0;
  }

  function currentMicPeak() {
    if (!analyserNode) return 0;
    const data = new Uint8Array(analyserNode.frequencyBinCount);
    analyserNode.getByteTimeDomainData(data);
    let peak = 0;
    for (let i = 0; i < data.length; i++) {
      const v = Math.abs(data[i] - 128) / 128;
      if (v > peak) peak = v;
    }
    return peak;
  }

  function maybeSendAudioStreamEnd() {
    // Only flush the server's cached audio when the user is NOT mid-sentence,
    // otherwise we would throw away the start of their question.
    const peak = currentMicPeak();
    if (peak > 0.15) {
      console.log("[VoiceWidget] audioStreamEnd skipped — user is speaking (peak", peak.toFixed(2) + ")");
      return;
    }
    sendAudioStreamEnd();
  }

  // ── Barge-in: sustained speech above threshold ──
  function startBargeInDetection() {
    if (!micStream || !audioCtx || !analyserNode) return;
    if (bargeInTimer) clearInterval(bargeInTimer);
    bargeInHits = 0;
    bargeInTimer = setInterval(() => {
      if (!outputPlaying || !analyserNode) { bargeInHits = 0; return; }
      const peak = currentMicPeak();
      // Require sustained speech: 3 consecutive polls (240ms) above 0.18
      // — fast enough to feel instant, still filters single noise spikes
      if (peak > 0.18) {
        bargeInHits++;
        if (bargeInHits >= 3) {
          console.log("[VoiceWidget] Barge-in (sustained speech, peak", peak.toFixed(2) + ") — stopping playback");
          bargeInHits = 0;
          stopOutputPlayback();
          sendPreRoll(); // hand the user's opening words back to the server
          setStatus("Listening...", "listening");
          updateMicIcon("listening");
        }
      } else {
        bargeInHits = 0;
      }
    }, 80);
  }

  // ── WebSocket handlers ────────────────────────────────────
  function onWsOpen() {
    console.log("[VoiceWidget] WebSocket connected");
    setupCompleteReceived = false;
    audioChunkCount = 0;
    setStatus("Connecting...", "listening");

    // Send setup message
    const setupMsg = {
      setup: {
        model: MODEL,
        generationConfig: {
          responseModalities: ["AUDIO"],
        },
        systemInstruction: {
          parts: [{ text: buildSystemInstruction(brain) }],
        },
        tools: TOOLS,
        // ── VAD: the documented latency lever ──
        // Docs: "silenceDurationMs ... the larger this value ... the more the
        // model's latency". Default is long (~800ms) which makes replies feel
        // late, so we commit end-of-speech after 300ms of silence, and keep
        // HIGH sensitivity on both ends for snappy turn-taking.
        ...(SILENCE_MS > 0
          ? {
              realtimeInputConfig: {
                automaticActivityDetection: {
                  prefixPaddingMs: 60,
                  silenceDurationMs: SILENCE_MS,
                  startOfSpeechSensitivity: "START_SENSITIVITY_HIGH",
                  endOfSpeechSensitivity: "END_SENSITIVITY_HIGH",
                },
              },
            }
          : {}),
        // transcripts — handy for verifying language mirroring while testing
        inputAudioTranscription: {},
        outputAudioTranscription: {},
      },
    };
    ws.send(JSON.stringify(setupMsg));      console.log("[VoiceWidget] Setup sent:", JSON.stringify(setupMsg).slice(0, 120) + "...");
    setStatus("Listening...", "listening");
    updateMicIcon("listening");
    utteranceStartTime = Date.now();
    console.log("[VoiceWidget] Session started at", new Date(utteranceStartTime).toISOString());
  }

  function onWsMessage(event) {
    const data = event.data;
    // Server may send String OR Blob OR ArrayBuffer
    if (typeof data === "string") {
      handleWsText(data);
    } else if (typeof Blob !== "undefined" && data instanceof Blob) {
      data.text().then(handleWsText).catch((err) => {
        console.warn("[VoiceWidget] Blob read error:", err);
      });
    } else if (data instanceof ArrayBuffer) {
      handleWsText(new TextDecoder().decode(data));
    } else {
      console.warn("[VoiceWidget] Unknown message type:", typeof data);
    }
  }

  function handleWsText(text) {
    let msg;
    try { msg = JSON.parse(text); } catch (e) {
      console.warn("[VoiceWidget] Unparseable message:", String(text).slice(0, 300));
      return;
    }

    // ── Debug: log EVERY message from server ──
    const msgKeys = Object.keys(msg);
    console.log("[VoiceWidget] WS message keys:", msgKeys.join(", "));
    if (msg.error) {
      console.error("[VoiceWidget] SERVER ERROR:", JSON.stringify(msg.error, null, 2));
    }
    if (msg.setupComplete) {
      setupCompleteReceived = true;
      console.log("[VoiceWidget] ✅ setupComplete received — session READY, audio streaming starts now");
      setStatus("Listening... say something!", "listening");
    }

    // ── Server audio (model speaking) ──
    if (msg.serverContent?.modelTurn?.parts) {
      for (const part of msg.serverContent.modelTurn.parts) {
        if (part.inlineData && part.inlineData.data) {
          const mimeType = part.inlineData.mimeType || "audio/pcm;rate=24000";
          const rate = parseInt(mimeType.match(/rate=(\d+)/)?.[1] || OUTPUT_RATE, 10);
          enqueueOutputAudio(part.inlineData.data, rate);
        }
      }
      if (outputPlaying) drainOutputQueue(playbackGeneration);
      else if (outputQueue.length > 0) startOutputPlayback();
    }

    // ── Model was interrupted — stop local playback ──
    if (msg.serverContent?.interrupted) {
      console.log("[VoiceWidget] Model interrupted — clearing playback");
      stopOutputPlayback();
      sendPreRoll(); // the user's opening words must not be lost
      setStatus("Listening...", "listening");
      updateMicIcon("listening");
    }

    // ── Transcriptions (debug) ──
    if (msg.serverContent?.inputTranscription) {
      lastUserTranscript += msg.serverContent.inputTranscription.text;
      console.log("[VoiceWidget] User said:", msg.serverContent.inputTranscription.text);
    }
    if (msg.serverContent?.outputTranscription) {
      lastModelTranscript += msg.serverContent.outputTranscription.text;
      console.log("[VoiceWidget] Gemini said:", msg.serverContent.outputTranscription.text);
    }

    // ── Function calls from the model (check_slots / book_appointment) ──
    if (msg.toolCall?.functionCalls?.length) {
      handleToolCall(msg.toolCall.functionCalls);
    }
    if (msg.toolCallCancellation) {
      console.warn("[VoiceWidget] toolCallCancellation:", JSON.stringify(msg.toolCallCancellation));
    }

    // ── Barge-in is handled by startBargeInDetection() interval ──
  }

  // ── Function calling → CRM backend ────────────────────────
  const p2 = (n) => String(n).padStart(2, "0");

  // The model may hand us something that isn't strictly YYYY-MM-DD.
  function normalizeDate(input) {
    const s = String(input ?? "").trim();
    if (/^\d{4}-\d{2}-\d{2}$/.test(s)) return s;
    const ms = Date.parse(s);
    if (!Number.isNaN(ms)) {
      const p = new Date(ms + 5 * 60 * 60 * 1000); // PKT wall clock
      return `${p.getUTCFullYear()}-${p2(p.getUTCMonth() + 1)}-${p2(p.getUTCDate())}`;
    }
    return null;
  }

  async function callCheckSlots(dateArg) {
    const date = normalizeDate(dateArg);
    if (!date) {
      return {
        error: "invalid_date",
        detail: "Pass date as YYYY-MM-DD (Asia/Karachi), e.g. " + PKT_TODAY.iso,
      };
    }
    const url =
      CHECK_SLOTS_URL +
      "?client=" + encodeURIComponent(clientId || "") +
      "&date=" + encodeURIComponent(date);
    console.log("[VoiceWidget] → GET", url);
    const res = await fetch(url);
    const text = await res.text();
    if (!res.ok) throw new Error(`check-slots HTTP ${res.status}: ${text.slice(0, 200)}`);
    return JSON.parse(text);
  }

  async function callBookAppointment(args) {
    const body = {
      client: clientId,
      doctor_name: args.doctor_name,
      name: args.name,
      phone: args.phone,
      slot_iso: args.slot_iso,
      service: args.service ?? null,
      notes: args.notes ?? null,
    };
    console.log("[VoiceWidget] → POST", BOOK_URL, JSON.stringify(body));
    const res = await fetch(BOOK_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    if (!res.ok) throw new Error(`book HTTP ${res.status}: ${text.slice(0, 200)}`);
    return JSON.parse(text);
  }

  async function handleToolCall(functionCalls) {
    const responses = [];
    for (const fc of functionCalls) {
      const args = fc.args || {};
      console.log("[VoiceWidget] 🔧 toolCall:", fc.name, JSON.stringify(args));
      let payload;
      try {
        if (fc.name === "check_slots") payload = await callCheckSlots(args.date);
        else if (fc.name === "book_appointment") payload = await callBookAppointment(args);
        else payload = { error: "unknown_function", detail: String(fc.name) };
      } catch (err) {
        payload = { error: String(err && err.message ? err.message : err) };
      }
      console.log("[VoiceWidget] 🔧 toolResult:", fc.name, JSON.stringify(payload));
      lastToolCalls.push({ name: fc.name, args, result: payload });
      responses.push({ id: fc.id, name: fc.name, response: payload });
    }
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ toolResponse: { functionResponses: responses } }));
      wsMsgCount++;
      lastSentInfo = "toolResponse x" + responses.length;
      console.log("[VoiceWidget] 🔧 toolResponse sent:", responses.length);
    } else {
      console.warn("[VoiceWidget] 🔧 cannot send toolResponse — socket not open");
    }
  }

  function onWsError(err) {
    console.error("[VoiceWidget] WebSocket ERROR event:", err);
    setStatus("Connection error");
  }

  function onWsClose(event) {
    console.log(
      "[VoiceWidget] WebSocket CLOSED — code:",
      event ? event.code : "?",
      "reason:",
      event && event.reason ? event.reason : "(empty)",
      "wasClean:",
      event ? event.wasClean : "?",
      "| msgs sent:",
      wsMsgCount,
      "| last sent:",
      lastSentInfo
    );
    stopOutputPlayback();
    // Real fix: when the socket dies the UI must not keep pretending to
    // listen — reset to idle so the next tap starts a fresh session.
    if (state === "listening" || state === "speaking") {
      setStatus("Disconnected — tap to start again", "");
      updateMicIcon();
      if (micLevelTimer) { clearInterval(micLevelTimer); micLevelTimer = null; }
      if (bargeInTimer) { clearInterval(bargeInTimer); bargeInTimer = null; }
      if (micStream) { micStream.getTracks().forEach((t) => t.stop()); micStream = null; }
      state = "idle";
    }
  }

  // ── Output audio playback ─────────────────────────────────
  function enqueueOutputAudio(base64Data, rate) {
    outputQueue.push({ data: base64Data, rate });
    if (outputQueue.length <= 1 || outputQueue.length % 10 === 0) {
      console.log("[VoiceWidget] Audio chunk queued:", base64Data.length, "chars @", rate, "Hz | queue:", outputQueue.length);
    }
  }

  // Decode base64 16-bit PCM (little-endian) → AudioBuffer
  function decodePcmChunk(chunk) {
    const binary = atob(chunk.data);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const numSamples = bytes.length >> 1;
    const audioBuffer = audioCtx.createBuffer(1, numSamples, chunk.rate);
    const channelData = audioBuffer.getChannelData(0);
    const dv = new DataView(bytes.buffer);
    for (let i = 0; i < numSamples; i++) {
      channelData[i] = dv.getInt16(i * 2, true) / 0x8000;
    }
    return audioBuffer;
  }

  // Gapless scheduled playback: every chunk lines up on the audio clock,
  // so there are no gaps between chunks and no overlapping sources.
  function startOutputPlayback() {
    if (outputQueue.length === 0) return;
    if (!outputPlaying) {
      outputPlaying = true;
      playbackGeneration++;
      nextPlayTime = audioCtx.currentTime;
      setStatus("Speaking...", "speaking");
      updateMicIcon("speaking");
      lastPlaybackStartAt = performance.now();
      if (lastSpeechSentAt) {
        lastReplyDelayMs = Math.round(lastPlaybackStartAt - lastSpeechSentAt);
      }
      if (lastSpeechEndedAt) {
        lastTurnLatencyMs = Math.round(lastPlaybackStartAt - lastSpeechEndedAt);
        console.log("[VoiceWidget] ⏱ TURN LATENCY (user stopped → reply audio):", lastTurnLatencyMs, "ms");
      }
      // Docs: when the mic stream pauses (>1s) we MUST flush cached server
      // audio with audioStreamEnd — but never while the user is mid-sentence.
      maybeSendAudioStreamEnd();
      console.log("[VoiceWidget] ▶ Playback started (gen", playbackGeneration + ")");
    }
    drainOutputQueue(playbackGeneration);
  }

  function drainOutputQueue(gen) {
    while (outputQueue.length > 0 && gen === playbackGeneration && outputPlaying) {
      const chunk = outputQueue.shift();
      try {
        const buffer = decodePcmChunk(chunk);
        const src = audioCtx.createBufferSource();
        src.buffer = buffer;
        src.connect(audioCtx.destination);
        // tiny lead only — keeps start-of-speech latency minimal
        const startAt = Math.max(nextPlayTime, audioCtx.currentTime + 0.01);
        src.start(startAt);
        nextPlayTime = startAt + buffer.duration;
        activeSources.add(src);
        src.onended = () => activeSources.delete(src);
      } catch (err) {
        console.error("[VoiceWidget] Playback error:", err);
      }
    }
    schedulePlaybackEnd(gen);
  }

  function schedulePlaybackEnd(gen) {
    if (playbackEndTimer) clearTimeout(playbackEndTimer);
    if (gen !== playbackGeneration || !outputPlaying) return;
    const remainingMs = Math.max(0, (nextPlayTime - audioCtx.currentTime) * 1000) + 150;
    playbackEndTimer = setTimeout(() => {
      if (gen !== playbackGeneration || !outputPlaying) return;
      if (outputQueue.length > 0) { drainOutputQueue(gen); return; }
      outputPlaying = false;
      activeSources.forEach((s) => { try { s.stop(); } catch (e) {} });
      activeSources.clear();
      // If the user was already talking when the agent stopped, keep their words.
      if (currentMicPeak() > 0.15) sendPreRoll();
      else clearPreRoll();
      setStatus("Listening...", "listening");
      updateMicIcon("listening");
      console.log("[VoiceWidget] ▶ Playback finished — back to listening");
    }, remainingMs);
  }

  // Docs: flush cached server audio when the mic stream pauses (>1s).
  function sendAudioStreamEnd() {
    if (ws && ws.readyState === WebSocket.OPEN && setupCompleteReceived) {
      try {
        ws.send(JSON.stringify({ realtimeInput: { audioStreamEnd: true } }));
        wsMsgCount++;
        lastSentInfo = "audioStreamEnd";
        console.log("[VoiceWidget] audioStreamEnd sent (mic paused → flush server buffer)");
      } catch (e) { /* ignore */ }
    }
  }

  function stopOutputPlayback() {
    playbackGeneration++; // invalidate pending timers & stale handlers
    if (playbackEndTimer) { clearTimeout(playbackEndTimer); playbackEndTimer = null; }
    activeSources.forEach((s) => { try { s.stop(); } catch (e) {} });
    activeSources.clear();
    outputPlaying = false;
    outputQueue = [];
    nextPlayTime = 0;
  }



  // ── Cleanup on page unload ────────────────────────────────
  window.addEventListener("beforeunload", () => {
    endSession();
  });

  console.log("[VoiceWidget] Widget loaded");

  // ── Debug/testing hook (no secrets exposed) ──────────────
  window.__vwTest = {
    sendText(text) {
      if (ws && ws.readyState === WebSocket.OPEN) {
        // clientContent with turnComplete triggers an immediate model
        // response (realtimeInput.text alone never ends the turn).
        ws.send(JSON.stringify({
          clientContent: {
            turns: [{ role: "user", parts: [{ text }] }],
            turnComplete: true,
          },
        }));
        console.log("[VoiceWidget] Test text sent:", text);
        return true;
      }
      console.warn("[VoiceWidget] WS not open");
      return false;
    },
    model: MODEL,
    // brain plumbing — exposed so the deployed artifact can be verified
    get brain() { return brain; },
    buildPrompt: buildSystemInstruction,
    loadBrain,
    get lastPlaybackStartAt() { return lastPlaybackStartAt; },
    // Deterministic voice-path test: inject a real 16kHz PCM speech sample
    // straight over the websocket (real-time paced), then trailing silence,
    // and report when the user stopped talking so latency can be measured.
    async playUserAudio(url, trailMs = 2000) {
      if (!ws || ws.readyState !== WebSocket.OPEN || !setupCompleteReceived) {
        return { error: "ws not ready" };
      }
      if (injection) return { error: "injection already running" };
      const speech = new Uint8Array(await (await fetch(url)).arrayBuffer());
      const total = new Uint8Array(speech.length + Math.floor(INPUT_RATE * 2 * (trailMs / 1000)));
      total.set(speech, 0); // trailing silence = the user's natural pause
      micSuppressed = true;
      lastSpeechSentAt = 0;
      lastReplyDelayMs = null;
      lastUserTranscript = "";
      lastModelTranscript = "";
      lastTurnLatencyMs = null;
      lastSpeechEndedAt = 0;
      startInjection(total, speech.length);
      return {
        sampleMs: Math.round((speech.length / (INPUT_RATE * 2)) * 1000),
        trailMs,
        paced: "audio-clock",
      };
    },
    // Simulate the user interrupting the agent (for verifying the pre-roll path)
    simulateBargeIn() {
      const before = wsMsgCount;
      stopOutputPlayback();
      sendPreRoll();
      return { sentWhileMuted: before, msgsAfterFlush: wsMsgCount, preRollBytes };
    },
    // Wait until a new reply starts (or timeout) and return the status
    async waitForReply(afterGen, timeoutMs = 12000) {
      const t0 = performance.now();
      while (performance.now() - t0 < timeoutMs) {
        await new Promise((r) => setTimeout(r, 100));
        if (this.status().playbackGeneration > afterGen) return this.status();
      }
      return this.status();
    },
    // Multi-turn voice test: injects the sample again and again and records
    // the turn latency of EVERY turn — this is how we catch slowdowns that
    // only show up deeper into a conversation.
    async runTurns(url, turns = 6, trailMs = 2000, gapMs = 1000) {
      const results = [];
      window.__multi = results;
      window.__multiDone = false;
      for (let i = 0; i < turns; i++) {
        const before = this.status().playbackGeneration;
        await this.playUserAudio(url, trailMs);
        await this.waitForReply(before, 12000);
        const s = this.status();
        results.push({
          turn: i + 1,
          turnLatencyMs: s.lastTurnLatencyMs,
          replyDelayMs: s.lastReplyDelayMs,
          heard: s.lastUserTranscript,
          said: s.lastModelTranscript.slice(0, 70),
          wsState: s.wsState,
        });
        // let playback drain before the next turn
        const t1 = performance.now();
        while (performance.now() - t1 < 25000) {
          await new Promise((r) => setTimeout(r, 200));
          const st = this.status();
          if (!st.outputPlaying && !st.injectionActive) break;
        }
        await new Promise((r) => setTimeout(r, gapMs));
      }
      window.__multiDone = true;
      return results;
    },
    status() {
      return {
        model: MODEL,
        lastPlaybackStartAt,
        state,
        setupCompleteReceived,
        audioChunkCount,
        wsState: ws ? ws.readyState : -1,
        outputQueueLen: outputQueue.length,
        outputPlaying,
        activeSources: activeSources.size,
        playbackGeneration,
        injectionActive: !!injection,
        injectionOffsetBytes: injection ? injection.offset : null,
        preRollBytes,
        micPeak: currentMicPeak(),
        lastReplyDelayMs,
        vadMode: VAD_PARAM,
        silenceMs: SILENCE_MS,
        wsMsgCount,
        lastSentInfo,
        lastTurnLatencyMs,
        toolsEnabled: true,
        toolCallCount: lastToolCalls.length,
        lastToolCalls: lastToolCalls.slice(-3),
        pktToday: PKT_TODAY,
        lastUserTranscript,
        lastModelTranscript,
      };
    },
  };
})();
