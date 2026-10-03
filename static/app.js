/* AI Video Maker — frontend controller */
(() => {
  "use strict";

  const state = {
    meta: null,
    ratio: "auto",
    count: 1,
    images: [], // {url, name}
    polling: new Set(),
  };

  const $ = (id) => document.getElementById(id);

  /* ---------------- Init ---------------- */
  async function init() {
    const [meta, health] = await Promise.all([
      fetch("/api/meta").then((r) => r.json()),
      fetch("/api/health").then((r) => r.json()),
    ]);
    state.meta = meta;

    $("badge-model").textContent = `model: ${meta.model_version}`;
    const engine = $("badge-engine");
    engine.textContent = health.ai_provider_available
      ? "engine: ModelArk (AI asli)"
      : `engine: ${health.render_engine}`;
    engine.classList.add(health.ai_provider_available ? "ok" : "warn");

    const aiToggle = $("use-ai");
    aiToggle.disabled = !health.ai_provider_available;
    if (!health.ai_provider_available) $("ai-toggle-wrap").classList.add("disabled");

    $("cta").textContent = meta.branding.cta;
    renderRatioChips(meta.ratios, meta.default_ratio);
    renderCountChips(meta.max_count);
    bindEvents();
    refreshJobs();
  }

  function renderRatioChips(ratios, defaultRatio) {
    const wrap = $("ratio-chips");
    wrap.innerHTML = "";
    const auto = document.createElement("div");
    auto.className = "chip active";
    auto.dataset.ratio = "auto";
    auto.textContent = "Otomatis";
    wrap.appendChild(auto);
    ratios.forEach((r) => {
      const c = document.createElement("div");
      c.className = "chip";
      c.dataset.ratio = r;
      c.textContent = r;
      if (r === defaultRatio) c.dataset.default = "1";
      wrap.appendChild(c);
    });
    wrap.addEventListener("click", (e) => {
      const chip = e.target.closest(".chip");
      if (!chip) return;
      wrap.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      state.ratio = chip.dataset.ratio;
    });
  }

  function renderCountChips(max) {
    const wrap = $("count-chips");
    wrap.innerHTML = "";
    for (let i = 1; i <= max; i++) {
      const c = document.createElement("div");
      c.className = "chip" + (i === 1 ? " active" : "");
      c.dataset.count = i;
      c.textContent = i;
      wrap.appendChild(c);
    }
    wrap.addEventListener("click", (e) => {
      const chip = e.target.closest(".chip");
      if (!chip) return;
      wrap.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      state.count = Number(chip.dataset.count);
    });
  }

  function bindEvents() {
    $("duration").addEventListener("input", (e) => {
      $("duration-val").textContent = e.target.value;
    });

    const dz = $("dropzone");
    const input = $("file-input");
    dz.addEventListener("click", () => input.click());
    input.addEventListener("change", () => handleFiles(input.files));
    ["dragenter", "dragover"].forEach((ev) =>
      dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); })
    );
    ["dragleave", "drop"].forEach((ev) =>
      dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); })
    );
    dz.addEventListener("drop", (e) => handleFiles(e.dataTransfer.files));

    $("generate").addEventListener("click", generate);
  }

  /* ---------------- Uploads ---------------- */
  async function handleFiles(files) {
    for (const file of Array.from(files || [])) {
      if (!file.type.startsWith("image/")) continue;
      const fd = new FormData();
      fd.append("file", file);
      try {
        const res = await fetch("/api/upload", { method: "POST", body: fd });
        if (!res.ok) throw new Error((await res.json()).detail || "upload gagal");
        const data = await res.json();
        state.images.push({ url: data.url, name: data.name });
      } catch (err) {
        showNotice(`Gagal mengunggah ${file.name}: ${err.message}`, true);
      }
    }
    renderThumbs();
  }

  function renderThumbs() {
    const wrap = $("thumbs");
    wrap.innerHTML = "";
    state.images.forEach((img, i) => {
      const div = document.createElement("div");
      div.className = "thumb";
      div.innerHTML = `<img src="${img.url}" alt="${img.name}" />
        <button title="Hapus" data-i="${i}">✕</button>`;
      wrap.appendChild(div);
    });
    wrap.querySelectorAll("button").forEach((b) =>
      b.addEventListener("click", () => {
        state.images.splice(Number(b.dataset.i), 1);
        renderThumbs();
      })
    );
  }

  /* ---------------- Generate ---------------- */
  async function generate() {
    const prompt = $("prompt").value.trim();
    if (!prompt) return showNotice("Deskripsi video tidak boleh kosong.", true);

    const btn = $("generate");
    btn.disabled = true;
    btn.textContent = "⏳ Mengirim ke pipeline…";
    hideNotice();

    const body = {
      prompt,
      duration: Number($("duration").value),
      ratio: state.ratio === "auto" ? null : state.ratio,
      count: state.count,
      images: state.images.map((i) => i.url),
      use_ai: $("use-ai").checked,
    };

    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "gagal membuat job");

      const notes = [...(data.notes || [])];
      notes.push(`Mode: ${data.mode} · rasio ${data.ratio} · durasi ${data.duration}s`);
      showNotice(notes.join(" "), false);

      data.jobs.forEach((j) => addPlaceholder(j.job_id, j.filename, data));
      data.jobs.forEach((j) => pollJob(j.job_id));
    } catch (err) {
      showNotice(`Error: ${err.message}`, true);
    } finally {
      btn.disabled = false;
      btn.textContent = "✨ Generate Video";
    }
  }

  function addPlaceholder(jobId, filename, data) {
    const results = $("results");
    const empty = results.querySelector(".empty");
    if (empty) empty.remove();

    const card = document.createElement("div");
    card.className = "card";
    card.id = `card-${jobId}`;
    card.innerHTML = `
      <div class="skeleton"></div>
      <div class="card-body">
        <div class="card-name">${filename}</div>
        <div class="card-meta">
          <span class="tag running">${jobId} · memproses</span>
          <span class="tag">${data.duration}s</span>
          <span class="tag">${data.ratio}</span>
        </div>
      </div>`;
    results.prepend(card);
    setPipelineLive(true);
  }

  async function pollJob(jobId) {
    if (state.polling.has(jobId)) return;
    state.polling.add(jobId);
    for (let i = 0; i < 240; i++) {
      let job;
      try {
        job = await fetch(`/api/jobs/${jobId}`).then((r) => r.json());
      } catch { /* transient */ }
      if (job && (job.status === "done" || job.status === "failed")) {
        updateCard(jobId, job);
        state.polling.delete(jobId);
        if (!state.polling.size) setPipelineLive(false);
        return;
      }
      if (job) updateCard(jobId, job);
      await sleep(1500);
    }
    state.polling.delete(jobId);
  }

  function updateCard(jobId, job) {
    const card = $(`card-${jobId}`);
    if (!card) return;
    const filename = job.filename || jobId;
    const statusClass = job.status === "done" ? "done"
      : job.status === "failed" ? "failed" : "running";

    if (job.status === "done" && job.url) {
      card.innerHTML = `
        <video src="${job.url}" controls preload="metadata" playsinline></video>
        <div class="card-body">
          <div class="card-name">${filename}</div>
          <div class="card-meta">
            <span class="tag done">selesai</span>
            <span class="tag">${job.duration_clamped}s</span>
            <span class="tag">${job.ratio}</span>
            <span class="tag">${job.mode}</span>
          </div>
          <div class="card-actions">
            <a href="${job.url}" download="${filename}">⬇ Unduh</a>
            <button data-copy="${job.url}">🔗 Salin URL</button>
          </div>
        </div>`;
      card.querySelector("[data-copy]").addEventListener("click", (e) => {
        navigator.clipboard.writeText(location.origin + e.target.dataset.copy);
        e.target.textContent = "✓ Tersalin";
      });
    } else {
      const meta = card.querySelector(".card-meta");
      if (meta) {
        meta.innerHTML = `
          <span class="tag ${statusClass}">${jobId} · ${job.status}</span>
          <span class="tag">${job.duration_clamped}s</span>
          <span class="tag">${job.ratio}</span>
          ${job.attempts ? `<span class="tag">percobaan ${job.attempts}</span>` : ""}`;
      }
      if (job.status === "failed") {
        if (job.error) {
          card.querySelector(".card-body").insertAdjacentHTML(
            "beforeend",
            `<div class="hint" style="color:var(--err)">${job.error}</div>`
          );
        }
        card.querySelector(".card-body").insertAdjacentHTML(
          "beforeend",
          `<div class="card-actions"><button data-retry="${jobId}">↻ Coba lagi</button></div>`
        );
        card.querySelector("[data-retry]").addEventListener("click", async (e) => {
          e.target.disabled = true;
          e.target.textContent = "⏳ mengulang…";
          await fetch(`/api/jobs/${jobId}/retry`, { method: "POST" });
          pollJob(jobId);
        });
      }
    }
  }

  /* ---------------- Pipeline status ---------------- */
  function setPipelineLive(live) {
    $("pipeline").querySelector(".dot").classList.toggle("live", live);
    $("pipeline-text").textContent = live ? "pipeline berjalan…" : "pipeline siap";
  }

  async function refreshJobs() {
    try {
      const { jobs } = await fetch("/api/jobs").then((r) => r.json());
      jobs.filter((j) => j.status === "done" && j.url)
        .slice(0, 12)
        .forEach((j) => {
          if (!$(`card-${j.id}`)) {
            addPlaceholder(j.id, j.filename || j.id, {
              duration: j.duration_clamped, ratio: j.ratio,
            });
            updateCard(j.id, j);
          }
        });
    } catch { /* ignore */ }
  }

  /* ---------------- Helpers ---------------- */
  function showNotice(text, isError) {
    const n = $("notice");
    n.textContent = text;
    n.classList.remove("hidden");
    n.classList.toggle("err", !!isError);
  }
  function hideNotice() { $("notice").classList.add("hidden"); }
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  init();
})();
