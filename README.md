# 🎬 AI Video Maker

Aplikasi web AI video maker yang dibangun berdasarkan skill
[`antono4/bbuseedance`](https://github.com/antono4/bbuseedance)
(**Seedane 2.5 30s** / Dreamina-Seedance-2.5).

Aplikasi ini mengimplementasikan seluruh kebijakan inti skill tersebut menjadi
sebuah layanan yang bisa langsung dipakai:

- **Built-in `text_to_video` & `image_to_video`** — tanpa API key eksternal.
- **Model dipaksa** `model_version = seedance_2.5`.
- **Durasi asli model `[4, 30]` detik** — cap lama 15 detik tidak dipakai.
- **Inferensi rasio otomatis** (TikTok → 9:16, YouTube → 16:9, dst).
- **Pipeline berkelanjutan** dengan auto-retry (maks 3x) dan pemulihan job
  yang terputus saat server restart.
- **Branding BBU CHANNEL wajib** — nama file `bbuchannel_*.mp4`, watermark
  overlay, dan blok CTA di setiap delivery.

Setiap permintaan menghasilkan **file MP4 nyata** (H.264 + AAC) yang bisa
diputar dan diunduh langsung dari browser.

---

## 🚀 Menjalankan

```bash
pip install -r requirements.txt
./run.sh                 # default port 8000
PORT=12000 ./run.sh      # port lain
```

Buka `http://localhost:8000`.

> **Catatan:** rendering memakai `ffmpeg` dari sistem. Pada Debian/Ubuntu:
> `sudo apt-get install -y ffmpeg`.

---

## 🧠 Cara Kerja

### Mesin render

| Engine | Kapan aktif | Keterangan |
|---|---|---|
| `procedural` (default) | selalu | Merender video animasi nyata via `ffmpeg` — deterministik dari prompt, bekerja offline. |
| `ModelArk` | `MODELARK_API_KEY` di-set **dan** `RENDER_ENGINE=auto` | Mengarahkan generation ke model Seedance 2.5 asli. |

Aplikasi tetap berfungsi penuh tanpa API key apa pun. Untuk memakai model AI
asli:

```bash
export MODELARK_API_KEY=...          # BytePlus ModelArk
export RENDER_ENGINE=auto
./run.sh
```

### Kemampuan AI gratis (dari MarbelAIv2.1)

Selain engine di atas, aplikasi ini mengadopsi pendekatan
[**MarbelAIv2.1**](https://antono4.github.io/MarbelAIv2.1/) — provider
OpenAI-compatible gratis yang **tidak butuh API key** — untuk dua hal:

1. **Perkuat prompt** — deskripsi Anda ditulis ulang menjadi prompt video
   bahasa Inggris yang lebih kaya (subjek, aksi, latar, cahaya, gerakan kamera).
2. **Gambar referensi AI** — sebuah gambar nyata dibuat dari prompt, lalu
   dianimasikan oleh renderer `ffmpeg` (efek Ken Burns). Inilah kemampuan
   "AI image → video" tanpa kredensial.

Provider dicoba berurutan (failover): `hermes.ai.unturf.com`,
`qwen.ai.unturf.com`, `text.pollinations.ai`, `api.free.ai`. Model yang
tersedia: `qwen3.8-27b` (default), `gpt-oss-20b`, `qwen3-8b`. Gambar diambil
dari `api.a0.dev` lalu `image.pollinations.ai`.

Semua langkah ini **opsional dan non-fatal**: bila jaringan mati, aplikasi
otomatis kembali ke prompt asli + render prosedural. Matikan dengan
`FREE_AI_ENABLED=0` untuk mode sepenuhnya offline.

```bash
# default: AI gratis aktif
FREE_AI_ENABLED=1 FREE_AI_CHAT_MODEL=qwen3.8-27b ./run.sh
FREE_AI_ENABLED=0 ./run.sh            # matikan AI gratis
FREE_AI_TIMEOUT=30 ./run.sh           # timeout per provider (detik)
```

Di UI, centang **"AI gratis (MarbelAI)"** dan pilih modelnya. Di API, kirim
`use_free_ai` dan `ai_model` pada `POST /api/generate`.

### Alur permintaan

```
prompt (ID) ──► clamp durasi [4,30] ──► inferensi rasio ──► terjemah ke EN
     │
     ├─► [AI gratis] perkuat prompt (LLM gratis) ──► gambar referensi AI
     │
     └──► pipeline.enqueue(job) ──► worker (auto-retry 3x) ──► MP4 + branding
                                          │
                                          └──► GET /api/jobs  (polling status)
```

---

## 🔌 API

| Method | Endpoint | Fungsi |
|---|---|---|
| `GET` | `/api/health` | Status server, model, rentang durasi, engine aktif |
| `GET` | `/api/meta` | Rasio, durasi, mode, konfigurasi branding |
| `POST` | `/api/upload` | Upload gambar referensi (PNG/JPG/WEBP, maks 25MB) |
| `POST` | `/api/generate` | Buat 1–5 job generation |
| `GET` | `/api/jobs` | Daftar semua job + ringkasan status |
| `GET` | `/api/jobs/{id}` | Status satu job |
| `POST` | `/api/jobs/{id}/retry` | Ulangi job yang gagal |
| `GET` | `/api/pipeline` | Ringkasan pipeline + video terkirim |
| `GET` | `/media/{file}` | Streaming/unduh MP4 hasil |
| `GET` | `/uploads/{file}` | Gambar referensi yang diunggah |

### Contoh

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"kucing berlari di pantai saat senja, video tiktok","duration":60,"count":2}'
```

```json
{
  "jobs": [{"job_id": "job_002", "filename": "bbuchannel_kucing_..._01.mp4"}],
  "duration": 30,
  "duration_requested": 60,
  "ratio": "9:16",
  "mode": "text_to_video",
  "english_prompt": "Cinematic scene: running, beach, sunset, cat.",
  "notes": ["Durasi 60s dipotong ke maksimum model 30s."]
}
```

---

## 📐 Aturan Durasi (sesuai skill)

| Permintaan | Nilai dikirim | Alasan |
|---|---|---|
| 30 detik | `30` | ✅ Maksimum asli model |
| 20 detik | `20` | ✅ Dalam rentang [4, 30] |
| 60 detik | `30` + pemberitahuan | ⚠️ Batas asli model 30s |
| 2 detik | `4` + pemberitahuan | ⚠️ Minimum asli model 4s |
| Tidak diisi | `30` | ✅ Default = kapabilitas maksimum |

## 🖼️ Rasio

| Kata kunci | Rasio |
|---|---|
| TikTok, Reels, Shorts, Story, vertical, HP | `9:16` |
| YouTube, landscape, cinema, presentasi | `16:9` |
| Instagram, square | `1:1` |
| Tidak ada | `16:9` |

Rasio eksplisit dari pengguna selalu menang.
Didukung: `21:9, 16:9, 4:3, 1:1, 3:4, 9:16`.

---

## 📁 Struktur

```
app/
  config.py         konfigurasi + kebijakan skill (model, durasi, rasio, branding)
  prompt_utils.py   clamp durasi, inferensi rasio, terjemah prompt, slug
  renderer.py       mesin render ffmpeg (text_to_video & image_to_video)
  pipeline.py       bookkeeping job (port dari scripts/pipeline_tracker.py)
  jobs.py           worker background + auto-retry
  ai_provider.py    adapter opsional ke BytePlus ModelArk (Seedance 2.5)
  free_ai.py        kemampuan AI gratis dari MarbelAIv2.1 (chat + gambar, failover)
  main.py           aplikasi FastAPI + endpoint
static/             UI (HTML/CSS/JS, Bahasa Indonesia)
data/outputs/       MP4 hasil
data/uploads/       gambar referensi
tests/              pengujian (kebijakan, pipeline, render, AI gratis)
```

---

## 🏷️ Branding BBU CHANNEL

- **Nama file:** `bbuchannel_[deskripsi_singkat]_[nomor].mp4`
- **Watermark:** overlay teks `BBU CHANNEL | youtube.com/@BBUchannel13`
  di pojok kiri bawah video.
- **CTA:** ditampilkan otomatis di footer UI dan pada setiap respons API.

---

## ⚠️ Catatan tentang skill asal

Skill `seedane-2.5-30s` di repo sumber mendokumentasikan kapabilitas model
Seedance 2.5 (durasi 4–30s, "30s continuous straight out") dan menargetkan
pemanggilan tool bawaan agen. Aplikasi ini mengimplementasikan logika
parameter dan alur pipeline-nya sebagai layanan mandiri. Klaim kapabilitas
model mengikuti dokumentasi yang dirujuk skill tersebut; verifikasi ulang
terhadap dokumentasi resmi BytePlus ModelArk sangat disarankan sebelum
memakai jalur `ModelArk` di produksi.
