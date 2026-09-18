# M12 — Fast & Lightweight Chat CLI & Native OpenAI-Compatible API

> Proyek: `disk-streaming-moe-engine`. Fase: **Production Delivery & Interface**. Index: `../README.md`.
> Landasan Protokol: OpenAI Chat Completions API reference (skema payload chat),
> WHATWG HTML Living Standard § Server-sent events (framing/encoding SSE normatif),
> Qwen ChatML Prompt Format.
> **Scope M12 (M12-2)**: milestone ini mengimplementasikan **subset text-only**
> dari template Qwen3.6 (§2.1) — BUKAN kompatibilitas template penuh.
> Klaim "standar Qwen3.6" tanpa kualifikasi subset DILARANG di milestone ini.
> Prasyarat: M10 hijau (Konsolidasi DISMOEN & Qwen 3.6 SSOT) + M11 hijau (Multi-Core Scaling F16, Async I/O F18, RAM Budgeting).
> Next: M13 (Multi-Token Prediction / MTP — Speculative Decoding).

| Field           | Nilai                                                                                                |
| :-------------- | :--------------------------------------------------------------------------------------------------- |
| **Deliverable** | CLI interaktif `dismoen chat` dan HTTP micro-daemon `dismoen serve` (OpenAI-compatible)              |
| **Komponen**    | Terminal REPL, ChatML Formatter, SSE Streaming Engine, KMSS v1 Multi-Turn Session State, HTTP Router |
| **Integrasi**   | Direct zero-harness connect: Aider, Cline, Continue.dev, Open WebUI, Python `openai` SDK, `curl`     |
| **Prasyarat**   | M11 hijau (Engine konkurensi $c^*$ stabil, latency hiding $\mathcal{E}_{overlap} \ge 80\%$)          |
| **Gate**        | G-M12-1..G-M12-5                                                                                     |

---

## 1. Latar Belakang & Sasaran Utama

Hingga milestone M11, engine `dismoen` beroperasi sebagai kernel komputasi tingkat rendah (_low-level engine_) yang menerima input via berkas/argumen dan memuntahkan output array token ID mentah.

Milestone M12 mentransformasi `dismoen` menjadi **produk inferensi mandiri (_self-contained engine_)** yang siap pakai untuk interaksi manusia maupun integrasi perkakas AI, tanpa memerlukan lapisan perantara (_no third-party harness needed_):

### Dua Mode Antarmuka Utama:

1. **`dismoen chat` (Fast & Lightweight Terminal REPL)**:
   - Antarmuka terminal interaktif langsung di shell bash.
   - Pembangkitan token di-stream secara instan ke _stdout_ karakter per karakter (_zero latency feeling_).
   - Penanganan otomatis template ChatML Qwen3.6 **subset text-only** — role
   `system`/`user`/`assistant` dengan konten string saja
   (ID di-resolve runtime dari metadata tokenizer, tanpa hardcode — §2.1).
   - Melanjutkan konteks multi-turn (_session continuation_) dengan memanfaatkan KMSS v1 (Kimo MoE Session State) tanpa perlu menghitung ulang (_prefill_) riwayat percakapan sebelumnya.

2. **`dismoen serve` (Native OpenAI-Compatible Micro HTTP Server)**:
   - Server HTTP ringan (_low footprint_) yang berjalan sebagai daemon lokal (default port `8000`).
   - Menyediakan endpoint REST standar OpenAI:
     - `GET /v1/models` $\to$ daftar model yang aktif (`qwen3.6-35b-a3b`).
     - `POST /v1/chat/completions` $\to$ melayani permintaan obrolan dengan dukungan _Server-Sent Events_ (SSE) streaming (`stream: true`) maupun respons JSON blok tunggal (`stream: false`).
    - **Zero-Harness Direct Compatibility (subset text-only)**: Perkakas AI seperti **Aider** (terminal coding agent), **Cline** (VSCode extension agent), **Continue.dev**, dan **Open WebUI** dapat langsung diarahkan ke `http://localhost:8000/v1` tanpa adaptor atau wrapper tambahan — **selama request berada dalam subset §2.1** (chat teks biasa). Request dengan fitur di luar subset (tool use, konten multimodal) ditolak eksplisit dengan error terstruktur, bukan di-render salah (§2.1).

---

## 2. Arsitektur Komponen M12

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                            PILIHAN ANTARMUKA USER                           │
│                                                                             │
│   [A] Terminal Interaktif                 [B] External Tools (Aider / Cline)│
│       $ dismoen chat                          $ OPENAI_API_BASE=... aider --model openai/... │
└──────────────────────┬──────────────────────────────────────┬───────────────┘
                       │                                      │
                       │ (Direct STDIN/STDOUT)                │ HTTP POST /v1/chat/completions
                       ▼                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 CHAT ENGINE & SESSION ORCHESTRATOR (DISMOEN)                │
│                                                                             │
│  1. ChatML Template Engine (SUBSET TEXT-ONLY, §2.1):                        │
│     <|im_start|>system ... <|im_end|>                                       │
│     <|im_start|>user ... <|im_end|>                                         │
│     <|im_start|>assistant ...                                               │
│     Non-subset (tool/vision/reasoning terstruktur) → TOLAK eksplisit.       │
│                                                                             │
│  2. KMSS v1 State Continuity Manager:                                       │
│     - Mempertahankan KV Cache (Attention) & Recurrent State (GDN)           │
│     - Delta prefill: Hanya memproses token giliran baru                     │
│                                                                             │
│  3. Token-to-Text Streamer:                                                 │
│     - Live BPE Detokenizer dengan penanganan UTF-8 boundary                 │
│     - Format SSE chunk: data: {"choices": [{"delta": {"content": "..."}}]}  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼ (M11 Multi-Thread & Async I/O Engine)
┌─────────────────────────────────────────────────────────────────────────────┐
│             CORE DISK-STREAMING INFERENCE (M0–M11 ENGINE RUNTIME)           │
│         Worker Threads (c*) + Async Double-Buffer Ping-Pong (F18)           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 ChatML Formatter (subset text-only) & Prompt Invariant

> **M12-2 — M12 BUKAN template Qwen3.6 penuh.**
> Template aktual upstream (`tokenizer_config.json` → `chat_template`,
> revision ter-pin di lockfile) jauh lebih kaya dari tiga role teks:
> ia menangani role `system`/`user`/`assistant`/`tool`, array `tools`,
> `tool_calls`/`tool_response`, `reasoning_content`/blok `<think>`,
> serta blok konten multimodal (`vision_start`, `image_pad`, `video_pad`).
> Checkpoint-nya sendiri multimodal-capable (`config.json` memuat
> `vision_config` + `image/video_token_id`; processor `Qwen3VLProcessor`).
> Engine M0–M11 adalah engine LM text-only (tidak ada vision encoder),
> sehingga mengklaim "standar Qwen3.6" untuk renderer tiga-role adalah
> klaim berlebih. Milestone ini mendefinisikan dan meng-gate **subset
> text-only** di bawah; kompatibilitas template penuh eksplisit
> **di luar scope M12**.

**Subset yang didukung M12** (satu-satunya input yang boleh di-render):

| Dimensi | Subset M12 | Di luar subset → DITOLAK eksplisit |
| :--- | :--- | :--- |
| Role | `system`, `user`, `assistant` | Role lain (`tool`, `developer`, …) |
| Konten | String teks (termasuk blok `<think>` sebagai teks passthrough, tanpa parsing khusus) | Konten list terstruktur (image/video/audio), `image_url`, `reasoning_content` terstruktur |
| Tool use | — | `tools`, `tool_calls`, `tool_response` |
| Generation prompt | Suffix `<|im_start|>assistant` (level string) | Parameter template lain (`enable_thinking`, `preserve_thinking`, …) |

**Kontrak penolakan**: input di luar subset TIDAK BOLEH di-render
sebagian/salah. `dismoen serve` menjawab HTTP `400`/`422` dengan body
JSON ber-field `unsupported` yang menyebut fitur pemicunya
(mis. `"unsupported": "role=tool"`); `dismoen chat` menolak dengan error
CLI yang setara. Daftar fitur yang ditolak wajib terdokumentasi di
help/error output, bukan hanya di doc ini.

Format prompt subset (level string/template):

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Halo, siapa kamu?<|im_end|>
<|im_start|>assistant
```

> **M12-1 (P0) — DILARANG hardcode token ID ChatML.**
> Nilai era-Qwen2 (`151644`/`151645`) **salah** untuk checkpoint Qwen3.6
> saat ini; ID special-token bergantung pada revisi tokenizer yang
> dipublikasikan upstream dan dapat berubah antar revisi. Hardcode numerik
> — di spec, kode, maupun test — akan menghasilkan prompt/token contract
> yang salah secara diam-diam. Tidak ada angka ID token di milestone ini;
> satu-satunya sumber kebenaran adalah metadata tokenizer di `--model-dir`.

**Kontrak resolusi token (runtime, per `--model-dir`):**

1. **Sumber otoritatif**: `tokenizer.json` → `added_tokens`
   (`content` + `id` + `special: true`) untuk
   `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`.
2. **Cross-check wajib**: `tokenizer_config.json` → `added_tokens_decoder`
   (pasangan id ↔ content harus identik dengan butir 1) dan nama
   `eos_token` harus merujuk ke salah satu special-token di atas.
3. **Stop set**: `generation_config.json` → `eos_token_id` (list).
   Pembangkitan berhenti secara deterministik saat model memprediksi token
   yang termasuk himpunan `eos_token_ids` hasil resolve ini
   (mencakup `<|im_end|>` dan `<|endoftext|>` tanpa menyebut angka).
4. **Konsistensi atau tolak start**: ketiga sumber (butir 1–3) wajib sepakat
   untuk ketiga special-token; inkonsistensi apa pun → startup error
   (`CONFIG_MISMATCH`), bukan fallback diam-diam.
5. **Validasi range**: setiap resolved ID wajib `< vocab_size` dari config
   model (Qwen3.6: `248320`, M9-port §); pelanggaran → `TOKEN_INVALID`.
6. **Pin lockfile**: identitas tokenizer — SHA-256 `tokenizer.json` +
   `tokenizer_config.json` beserta revision HF — wajib tercatat di model
   lockfile (`models.lock.port.json`, sejajar pin shard SEC-1 /
   `ref-ground-truth`). Engine menegaskan pin saat startup; mismatch
   → tolak start. (`models.lock.port.json` saat ini hanya memuat shard;
   penambahan section tokenizer adalah pekerjaan wajib W1.)

**Oracle G-M12-1**: tokenizers HF (`Qwen2Tokenizer`) + `chat_template`
upstream pada revision ter-pin di lockfile. Formatter lulus bila dan hanya
bila: (a) untuk setiap input dalam subset §2.1, seluruh urutan ID —
termasuk special-token hasil resolve — match 100% vs oracle yang diberi
input subset yang sama; (b) uji negatif: metadata tokenizer yang digeser
ID-nya wajib terdeteksi (mismatch error), bukan lolos diam-diam;
(c) setiap input di luar subset wajib ditolak eksplisit sesuai kontrak
penolakan (tidak ter-render).

### 2.2 Kelanjutan Sesi Multi-Turn Tanpa Prefill Ulang (KMSS v1)

Pada obrolan multi-turn sekuensial, menghitung ulang _KV cache_ dan _GDN recurrent state_ dari awal giliran (_turn_) adalah pemborosan latensi ($O(L_{prompt})$).

> **M12-5 — identitas sesi wajib didefinisikan; pointer-mentah tidak cukup.**
> `POST /v1/chat/completions` adalah API stateless tanpa konsep session
> internal milik dismoen, sehingga "menyimpan pointer KMSS di memory dan
> memakai state yang sama di turn kedua" belum menjawab pertanyaan kunci:
> request A → session A dan request B → session B tanpa pernah tercampur.
> Dua skema identitas DILARANG:
>
> - **Koneksi socket = session** — klien boleh reconnect kapan saja;
>   sesi yang terikat pada soket akan hilang/salah pasca-reconnect.
> - **`model` + panjang `messages`** — dua user berbeda dapat punya prompt
>   dengan panjang sama; skema ini menjamin tabrakan.

**Rantai identitas kanonis** (satu-satunya cara me-resolve request → state):

```text
Canonical message history
        ↓ (render subset §2.1 → token ID, termasuk special-token)
tokenized prefix
        ↓ (hash kriptografis atas urutan ID + domain kunci)
prefix hash / session key
        ↓
KMSS state (KV cache + GDN state atas prefix itu)
```

- **Kanonikalisasi**: histori `messages` (role + konten string per pesan,
  berurutan) di-render oleh renderer subset §2.1 menjadi urutan token ID
  lengkap turn itu. Session key = hash atas urutan ID tersebut dalam domain
  kunci `(model_id, tokenizer_pin, template_subset_rev)`.
- **Aturan reuse/invalidasi**:

| Kondisi | Keputusan |
| :--- | :--- |
| Prefix eksak sama (hash cocok) | **Reusable** — sambung state, prefill hanya $$\Delta N_{tokens} = N_{new\_prompt}$$ |
| Prefix mismatch (pesan mana pun berbeda/berubah urutan) | **Invalidate** — recompute penuh dari awal |
| System prompt berubah | **Invalidate** (kasus khusus prefix mismatch, disebut eksplisit agar tak dioptimasi keliru) |
| Model berubah | **Invalidate by construction** (domain kunci memuat `model_id`) |
| Tokenizer berubah (pin/revisi) | **Invalidate by construction** (domain kunci memuat `tokenizer_pin`) |

- **Cache = optimasi opsional, bukan syarat semantik.** API wajib tetap
  benar bila cache mati/penuh/di-evict: setiap request yang recompute penuh
  menghasilkan output identik dengan yang memakai cache-hit. Eviksi
  (LRU berbatas — cache tidak boleh tumbuh tanpa batas) hanya mengubah
  latensi, tidak pernah mengubah kebenaran.
- Status KV cache 10 layer attention dan state matriks 30 layer GDN yang
  reusable langsung disambung (_appended_), memangkas latensi respon awal
  giliran ke-2 dan seterusnya hingga **< 0.5 detik**.

### 2.3 Protokol Streaming OpenAI SSE

Pada mode `dismoen serve`, response streaming mematuhi dua lapisan spesifikasi
yang berbeda dan jangan dicampur (M12-4):

- **Framing/encoding SSE** — WHATWG HTML Living Standard, § *Server-sent events*
  (normatif untuk `Content-Type: text/event-stream`, field `data:`, framing
  blank-line antar event, dan parsing stream UTF-8).
  RFC 8895 **bukan** spesifikasi SSE — ia hanya *memakai* SSE sebagai transport
  untuk ALTO Incremental Updates — sehingga dilarang disitir sebagai dasar SSE.
- **Skema payload chat** — OpenAI Chat Completions API reference
  (bentuk objek `chat.completion.chunk`, field `choices[].delta`,
  `finish_reason`, dan event terminator `data: [DONE]`):

```http
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive

data: {"id":"chatcmpl-dismoen-01","object":"chat.completion.chunk","created":1773800000,"model":"qwen3.6-35b-a3b","choices":[{"index":0,"delta":{"role":"assistant","content":"Halo"},"finish_reason":null}]}

data: {"id":"chatcmpl-dismoen-01","object":"chat.completion.chunk","created":1773800000,"model":"qwen3.6-35b-a3b","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-dismoen-01","object":"chat.completion.chunk","created":1773800000,"model":"qwen3.6-35b-a3b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### 2.4 Concurrency, Admission Control & Cancellation Ownership (M12-6)

> **Asumsi M11 tidak berlaku untuk server konkuren.**
> M11 mengkalibrasi **satu** engine: $c^*$ + worker pool + single pipeline +
> double-buffer I/O untuk workload `1 request × c*`. Dua request yang
> berbagi pool yang sama (`2 request × c*`) adalah workload yang berbeda:
> bandwidth DRAM, antrean NVMe, dan penalti $\beta(c-1)$ terbagi/berubah,
> sehingga model Amdahl fixed-workload M11 tidak langsung berlaku untuk
> concurrent inference. M12 **tidak** mengklaim konkurensi generasi.

**Kontrak M12 (serial-first):**

| Parameter | Default normatif | Perilaku |
| :--- | :---: | :--- |
| `max_concurrent_generations` | **1** | Tepat satu generasi aktif; generasi kedua tidak pernah berjalan paralel dengannya |
| `max_queue_depth` | **8** | Request berlebih antre FIFO; melampaui itu → HTTP `429` + header `Retry-After` |
| Budget memori per-request | konteks ≤ `model_max_length` + working set dalam `M_budget` (M11) | Melebihi → tolak sebelum admisi (`400`/`413`), tidak pernah OOM di tengah generasi |
| Kepemilikan pembatalan | server | Disconnect klien → server meng-abort generasi milik koneksi itu, melepas worker pool + transient KMSS, engine kembali idle; request antre milik koneksi yang putus dikeluarkan dari antrean |

**Batas resource request (M12-9).**
Satu request jahat pada model disk-streaming 35B bisa sangat mahal, dan
server tidak boleh mengizinkan konteks sebesar maksimum model
(`max_position_embeddings = 262144`, `text_config`) hanya karena model
mampu. Default konservatif untuk micro-daemon lokal (semua bisa dioverride
via flag CLI yang terdokumentasi):

| Limit | Default | Flag override | Pelanggaran |
| :--- | :---: | :--- | :--- |
| Max request body | 1 MiB | `--max-body-bytes` | `413` |
| Max messages | 128 | `--max-messages` | `413` |
| Max input tokens | 8192 | `--max-input-tokens` | `400` |
| Max output tokens | 2048 | `--max-output-tokens` | `400` |
| Max context length (input + output) | 16384 | `--max-context-tokens` (plafon keras: tidak boleh > 262144) | `400` |
| Max concurrent generations | 1 | `--max-concurrent-generations` | antre/429 (kontrak di atas) |
| Max queue depth | 8 | `--max-queue-depth` | `429` (kontrak di atas) |
| Request timeout (admisi → selesai) | 900 s | `--request-timeout-s` | abort + `500` (`type: "timeout"`); stream diputus dengan error event |
| Queue-wait timeout | 60 s | `--queue-wait-timeout-s` | `429` |
| SSE keepalive | komentar `: keepalive` per 15 s | `--sse-keepalive-s` | — (mencegah intermediary memutus stream idle) |

Semua penolakan limit memakai body error §2.5. Batas diverifikasi
per-request **sebelum** admisi ke executor — tidak ada alokasi besar yang
terjadi sebelum limit dicek.

> **Layering M10 → M11 → M12 → M13+ (M12-9).**
> Bentuk milestone yang benar saat ini:
>
> ```text
> M10  Model correctness
>        ↓
> M11  single-request performance (F16/F18 @ 1 request × c*)
>        ↓
> M12  single-request serving + protocol (kontrak §2.4–§2.5)
>        ↓
> M13+ multi-request scheduling (belum scope)
> ```
>
> M12 dilarang diam-diam berubah dari *single inference optimizer* menjadi
> *general-purpose concurrent server*. Daftar non-goal M12 yang eksplisit
> menjadi problem milestone scheduling kelak: scheduler, fairness,
> queueing di luar FIFO berbatas, admission control adaptif, isolasi cache
> antar-tenant di luar session key §2.2, akuntansi memori per-tenant, dan
> multi-request batching. Semuanya berpotensi menggeser kurva F16/F18,
> sehingga milestone tersebut wajib kalibrasi ulang — bukan reuse angka M11.

- Antrean bersifat FIFO dan berbatas; `429` adalah jawaban jujur kelebihan
  beban, bukan blocking tanpa batas.
- Abort (`Ctrl+C` di `chat`, disconnect di `serve`) tidak boleh meracuni
  engine: generasi berikutnya — dari sesi mana pun — wajib berjalan normal
  (di-gate G-M12-4).
- **Konkurensi > 1 eksplisit di luar scope M12** dan menjadi
  milestone/bench terpisah (dengan kalibrasi ulang $c^*$/$\beta$/$BW_{eff}$
  di bawah beban konkuren). Kode M12 dilarang mengandung jalur generasi
  paralel setengah-jadi; yang ada hanya serial + antrean + 429.

### 2.5 OpenAI Compatibility Contract (M12-8)

> **Keputusan sadar**: target kompatibilitas M12 adalah **Chat Completions API**
> (`/v1/chat/completions`) — endpoint ini masih disediakan OpenAI saat ini,
> sehingga memakainya untuk interoperabilitas tetap masuk akal walaupun
> dokumentasi baru OpenAI lebih mendorong Responses API.
> **Responses API eksplisit di luar scope M12.**
> Yang di-gate bukan "nama endpoint", melainkan kontrak skema di bawah.

**`GET /v1/models`** → `200` dengan body:

```json
{"object": "list", "data": [{"id": "qwen3.6-35b-a3b", "object": "model", "owned_by": "dismoen"}]}
```

Field `object`, `data[]`, `data[].id`, `data[].object`, `data[].owned_by`
wajib ada (field tambahan diizinkan).

**`POST /v1/chat/completions`** — field request yang didukung:

| Field | Status M12 |
| :--- | :--- |
| `model` | Wajib; id tak dikenal → `404 model not found` |
| `messages` | Wajib; array `{role, content}`; di luar subset §2.1 → `400`/`422` + field `unsupported` |
| `stream` | `true` (SSE §2.3) / `false` (default) |
| `max_tokens` / `max_completion_tokens` | Alias; didukung dan dibatasi plafon konteks; bila keduanya ada dan berbeda → `400` |
| `temperature`, `top_p` | Didukung; `0` → greedy deterministik |
| `stop` | Didukung (string atau array string) |
| `tools`, `tool_choice` | Dikenali tetapi **ditolak** → `400`/`422` + `unsupported` (subset text-only, §2.1) |
| Field tak dikenal lain | Diabaikan (forward-compat), tidak boleh menggagalkan request valid |

**Respons sukses:**

- Non-stream (`stream: false`): `object: "chat.completion"`,
  `choices[{index, message: {role, content}, finish_reason}]`,
  `usage: {prompt_tokens, completion_tokens, total_tokens}` wajib ada.
- Stream (`stream: true`): chunk `object: "chat.completion.chunk"`,
  `choices[{index, delta, finish_reason}]`, terminator `data: [DONE]`
  (framing §2.3); `usage` dikirim pada akhir stream bila klien meminta
  `stream_options.include_usage`.
- `finish_reason` dalam subset: `"stop"` | `"length"`.

**Matriks error** (body gaya OpenAI `{error: {message, type, code}}`;
`500` tidak boleh membocorkan path/stack internal):

| Kode | Makna | Pemicu |
| :---: | :--- | :--- |
| 400 | malformed request | JSON invalid, field wajib hilang, alias konflik, payload non-subset tanpa penanda |
| 401 | unauthorized | Header `Authorization` hilang/kosong (kunci non-kosong apa pun diterima — M12 tanpa auth ketat) |
| 404 | model not found | `model` tak dikenal (termasuk di `/v1/models`) |
| 405 | method not allowed | Method salah pada path yang ada |
| 413 | body too large | Body/konteks melebihi budget pra-admisi (§2.4) |
| 429 | overloaded | Antrean FIFO penuh (§2.4) |
| 500 | engine failure | Error internal engine |

Seluruh sel matriks di atas wajib tercakup verifier G-M12-3 (sukses +
setiap kode error dipicu dan diassert bentuknya).

**Rantai kanselasi deterministik (M12-7).**
"Tidak crash" belum cukup: worker yang masih menghitung puluhan token
setelah klien pergi berarti resource dibakar sia-sia. Saat disconnect,
seluruh execution graph wajib terputus berurutan:

```text
socket close
    ↓
HTTP request cancellation (hentikan SSE, bebaskan koneksi)
    ↓
generation cancellation (sinyal cancel ke loop decode; token-step baru dilarang dimulai)
    ↓
worker cancellation (worker pool join pada checkpoint aman antar token-step)
    ↓
pending I/O cancellation (chunk yang sudah di-submit boleh menyelesaikan
                          syscall-nya, tetapi hasilnya di-reclaim tanpa
                          pernah diumpankan ke worker; pread baru dilarang)
    ↓
buffer reclamation (buffer transient + slot ring kembali ke pool)
    ↓
KMSS policy (state prefix yang sudah komplit tetap cacheable per §2.2;
             turn parsial yang ter-abort dibuang, tidak pernah di-resume)
```

Di-gate G-M12-4 dengan tiga metrik kanselasi:

| Metrik | Definisi | Ambang |
| :--- | :--- | :---: |
| `cancel_propagation_ms` | $t_{\text{idle-bersih}} - t_{\text{cancel-signal}}$: waktu dari sinyal cancel hingga engine kembali idle + buffer ter-reclaim | p95 ≤ 500 ms (Project SLO) |
| `orphan_compute_tokens` | Token-step forward yang **dimulai** setelah sinyal cancel | == 0 |
| `pending_io_after_cancel` | Chunk I/O outstanding yang belum di-reclaim setelah kanselasi selesai (in-flight syscall yang selesai tepat saat cancel dihitung reclaim, bukan pending) | == 0 |

Aturan ukur:Verifier `test_m12_resilience.sh` memicu disconnect acak di
tengah generasi (streaming aktif), lalu menegaskan ketiga metrik di atas
plus kriteria G-M12-4 lama (0 segfault, 0 zombie threads, VmHWM stabil,
generasi berikutnya normal). "Tidak ada token yatim" berarti tidak ada
pekerjaan baru yang dimulai pasca-cancel — penyelesaian atomik operasi
SIMD yang sedang berjalan saat sinyal tiba bukan pelanggaran, tetapi
wajib tercatat di log kanselasi agar dapat diaudit.

---

## 3. Spesifikasi Penggunaan Antarmuka CLI

### 3.1 Mode Obrolan Interaktif Terminal

```bash
# Menjalankan sesi chat interaktif langsung
dismoen chat --model-dir /home/will/models/qwen3.6-35b-a3b --auto

# Dengan instruksi sistem kustom
dismoen chat --system-prompt "Kamu adalah asisten pemrograman ahli Linux kernel dan sistem terdistribusi."
```

Fitur Terminal:

- Streaming teks seketika dengan ANSI syntax highlighting.
- Pintasan `Ctrl+C`: Menghentikan generasi token saat ini tanpa mematikan sesi (_graceful abort_).
- Perintah internal: `/clear` (reset konteks KMSS), `/history` (lihat riwayat sesi), `/exit` (keluar).

### 3.2 Mode HTTP Server OpenAI-Compatible

```bash
# Menjalankan micro-server lokal di port 8000
dismoen serve --host 127.0.0.1 --port 8000 --model-dir /home/will/models/qwen3.6-35b-a3b --auto
```

#### Integrasi Langsung dengan Klien Populer:

1. **Aider (Terminal Coding Agent)** — pola resmi OpenAI-compatible
   ([docs](https://aider.chat/docs/llms/openai-compat.html)):

   ```bash
   OPENAI_API_BASE=http://127.0.0.1:8000/v1 \
   OPENAI_API_KEY=dismoen \
   aider --model openai/qwen3.6-35b-a3b
   ```

   > **M12-3 — pola lama SALAH dan dilarang dipakai.**
   > `aider --openai-api-base ...` bukan mekanisme yang didokumentasikan
   > (endpoint dikonfigurasi via env `OPENAI_API_BASE` / file `.aider.conf.yml`);
   > nama model wajib ber-prefix `openai/` agar aider me-route request ke
   > endpoint compatible (`--model qwen3.6-...` tanpa prefix tidak di-route
   > ke server ini); dan `--api-key` memakai format `provider=key`
   > (mis. `--api-key openai=dismoen`), bukan string kunci polos.
   > Server `dismoen serve` menerima API key apa pun (termasuk `dismoen`).
   > Contoh di atas adalah kontrak yang di-gate G-M12-5, bukan ilustrasi.

2. **Cline / Continue.dev (VSCode)**:
   - Base URL: `http://127.0.0.1:8000/v1`
   - Model ID: `qwen3.6-35b-a3b`
   - API Key: `dismoen` (atau string apa saja)

3. **Python `openai` SDK**:

   ```python
   from openai import OpenAI

   client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="dismoen")
   stream = client.chat.completions.create(
       model="qwen3.6-35b-a3b",
       messages=[{"role": "user", "content": "Jelaskan Hukum Amdahl"}],
       stream=True,
   )
   for chunk in stream:
       print(chunk.choices[0].delta.content or "", end="", flush=True)
   ```

```

---

## 4. Quality Gates (M12)

| Gate | Kriteria Penilaian | Ambang Batas | Verifier Tool |
| :--- | :--- | :---: | :--- |
| **G-M12-1** | **ChatML Prompt Fidelity (subset text-only) & Detokenization**: Resolver special-token (`<|im_start|>`, `<|im_end|>`, `eos_token_ids`) dari metadata tokenizer valid vs lockfile; input subset ter-encode tepat, input non-subset ditolak eksplisit, detokenisasi streaming UTF-8 multi-byte valid tanpa karakter corrupt | 100% token match vs oracle (HF tokenizer + chat_template @ pinned revision, input subset) + uji negatif tamper-ID & non-subset terdeteksi/ditolak, zero encoding error | `tests/unit/test_chatml_formatter.mojo` |
| **G-M12-2** | **KMSS v1 Session Identity & Delta Latency**: Request di-resolve ke state via session key (§2.2); prefix sama → reuse + TTFT turn-2 jauh lebih cepat vs recompute; prefix/system/model/tokenizer berbeda → invalidate; dua sesi beda-prefix tak pernah tercampur; output cache-hit identik dengan full-prefill | $TTFT_{turn2} \le 0{,}5 \times TTFT_{recompute}$ + reuse/invalidate/isolation/parity-cache-vs-recompute 100% benar | `tests/integration/test_m12_session_continuity.sh` |
| **G-M12-3** | **OpenAI Compatibility Contract (§2.5)**: `GET /v1/models` + `POST /v1/chat/completions` (subset) lulus via Python OpenAI SDK resmi (stream & non-stream: skema, `finish_reason`, `usage`); seluruh sel matriks error (400/401/404/405/413/429/500) terpicu dan berbentuk benar; non-subset → error terstruktur | Kontrak §2.5 100% tercakup (termasuk batas resource §2.4 → 400/413) + zero SDK exception pada jalur sukses | `tools/bench/verify_openai_conformance.py` |
| **G-M12-4** | **Serial Execution, Admission & Full-Graph Cancellation**: Tepat satu generasi aktif (`max_concurrent_generations = 1`); overflow antrean → 429 terstruktur; over-budget → tolak pre-admisi; disconnect/abort memutus seluruh execution graph (§2.4: HTTP → generasi → worker → I/O → buffer → KMSS) dengan metrik kanselasi terpenuhi; generasi berikutnya normal | 0 segfault, 0 zombie threads, VmHWM stabil, tidak pernah 2 generasi aktif, `orphan_compute_tokens == 0`, `pending_io_after_cancel == 0`, p95 `cancel_propagation_ms` ≤ 500 ms (Project SLO) | `tests/integration/test_m12_resilience.sh` |
| **G-M12-5** | **Harness Integration (Aider, pola resmi)**: Klien aider asli terhubung ke `dismoen serve` memakai kontrak §3.2 (env `OPENAI_API_BASE`/`OPENAI_API_KEY` + `--model openai/<id>`) dan menyelesaikan satu pertukaran pesan non-interaktif | Model ter-list via routing `openai/`, exit 0, respons assistant tak-kosong; skip eksplisit bila biner aider tak terinstal | `tests/integration/test_m12_harness_compat.sh` |

---

## 5. Rencana Gelombang Kerja (Execution Waves)

- **Gelombang 1 (M12-W1: ChatML Template & Streaming Detokenizer)**:
  - Implementasi resolver special-token runtime + pin section tokenizer di `models.lock.port.json` (kontrak §2.1 butir 1–6; tanpa konstanta ID numerik di kode/test).
  - Implementasi renderer subset text-only + jalur penolakan eksplisit untuk semua input non-subset (tabel §2.1).
  - Implementasi formatter prompt ChatML untuk format role system, user, dan assistant.
  - Streaming detokenizer yang menangani potongan byte UTF-8 multi-byte (Gate G-M12-1).
- **Gelombang 2 (M12-W2: Terminal Interactive REPL `dismoen chat`)**:
  - Implementasi derivasi session key + tabel invalidasi §2.2 (termasuk uji
    isolasi dua sesi sama-panjang dan uji paritas cache-hit vs full-prefill).
  - Implementasi antarmuka loop terminal interaktif dengan graceful abort `Ctrl+C`.
  - Integrasi session continuity KMSS v1 (Gate G-M12-2).
- **Gelombang 3 (M12-W3: Embedded HTTP Micro-Server `dismoen serve`)**:
  - Server HTTP ringan berbasis socket lokal berkinerja tinggi.
  - Penanganan router `/v1/models` dan `/v1/chat/completions`.
  - Serial executor + antrean FIFO berbatas + admission control + kepemilikan
    pembatalan sesuai kontrak §2.4 (Gate G-M12-4).
- **Gelombang 4 (M12-W4: OpenAI Protocol Conformance & Streaming SSE)**:
  - Implementasi chunk format SSE `data: {...}` dan event `[DONE]`.
  - Implementasi kontrak skema §2.5 (field request, `usage`, `finish_reason`,
    matriks error) + batas resource §2.4 (limit → 400/413/429, timeout,
    keepalive) + verifikasi SDK/`curl` dan uji negatif per sel error
    (Gate G-M12-3).
- **Gelombang 5 (M12-W5: Harness Integration, Client Abort & Closure)**:
  - Integration test harness: `tests/integration/test_m12_harness_compat.sh`
    menyalakan `dismoen serve`, mengekspor kontrak §3.2, lalu menjalankan
    aider asli satu pesan non-interaktif di repo scratch sekali-pakai
    (Gate G-M12-5; skip eksplisit + alasan bila biner aider tak terinstal —
    dilarang hijau-palsu). Uji Cline/Continue mengacu pada konformansi
    protokol G-M12-3 + field koneksi §3.2 butir 2.
  - Pengujian ketahanan pemutusan koneksi klien di tengah streaming plus
    audit metrik kanselasi (`cancel_propagation_ms`, `orphan_compute_tokens`,
    `pending_io_after_cancel`) (Gate G-M12-4).
  - Terbitkan scorecard kelulusan di `reports/YYYY-MM-DD/M12-gates-scorecard.md`.

---

## 6. Definisi Selesai (DoD M12)

- [ ] Subperintah `dismoen chat` berfungsi sebagai REPL terminal interaktif dengan live streaming dan formatting subset text-only yang akurat (Gate G-M12-1).
- [ ] Tidak ada hardcode ID ChatML di spec/kode/test; `models.lock.port.json` memuat pin tokenizer (SHA-256 + revision) dan engine menolak start saat mismatch (kontrak §2.1).
- [ ] Input non-subset (role `tool`, tool-calls, konten multimodal, dsb.) selalu ditolak eksplisit dengan penyebutan fitur pemicu — tidak pernah ter-render sebagian/salah.
- [ ] Multi-turn session state KMSS v1 berhasil menjaga konteks percakapan tanpa menghitung ulang prefill dari awal giliran (Gate G-M12-2).
- [ ] Identitas sesi mengikuti rantai kanonis §2.2 (skema socket=session dan model+length dilarang); cache murni optimasi — output tetap benar saat cache mati/di-evict.
- [ ] Subperintah `dismoen serve` mengimplementasikan kontrak kompatibilitas §2.5 (skema sukses + seluruh matriks error) dan lulus verifikasi SDK resmi (Gate G-M12-3).
- [ ] Seluruh batas resource §2.4 aktif by-default dengan nilai konservatif (konteks default 16384, jauh di bawah maksimum model 262144) dan dapat dioverride via flag; M12 tetap single-request serving — problem multi-request scheduling eksplisit non-goal untuk M13+.
- [ ] Terverifikasi sukses terhubung langsung dengan perkakas klien AI eksternal (Aider, Cline, Python `openai` SDK) pada alur chat teks subset tanpa lapisan perantara pihak ketiga (*zero-harness* dalam subset) — untuk aider via integration test kontrak resmi G-M12-5, bukan sekadar contoh doc.
- [ ] Server mengeksekusi tepat satu generasi dalam satu waktu (default `max_concurrent_generations = 1`), antrean FIFO berbatas dengan 429 jujur saat penuh, dan abort/disconnect memutus seluruh execution graph hingga idle bersih dengan `orphan_compute_tokens == 0` dan `pending_io_after_cancel == 0` (Gate G-M12-4); konkurensi > 1 di luar scope M12.
- [ ] Server tangguh terhadap *early disconnect / client abort* tanpa memicu thread deadlock atau kebocoran memori (Gate G-M12-4).
- [ ] Seluruh suite pengujian regresi (`validate-m11`, `validate-m10`, `validate-m9`) dan 13 hook `pre-commit` 100% hijau.
- [ ] Scorecard formal sertifikasi M12 ter-commit di `reports/YYYY-MM-DD/M12-gates-scorecard.md`.

```
