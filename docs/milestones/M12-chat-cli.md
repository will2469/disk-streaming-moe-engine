# M12 — Fast & Lightweight Chat CLI & Native OpenAI-Compatible API

> Proyek: `disk-streaming-moe-engine`. Fase: **Production Delivery & Interface**. Index: `../README.md`.
> Landasan Protokol: OpenAI Chat Completions API Specification, Server-Sent Events (SSE / RFC 8895), Qwen ChatML Prompt Format.
> Prasyarat: M10 hijau (Konsolidasi DISMOEN & Qwen 3.6 SSOT) + M11 hijau (Multi-Core Scaling F16, Async I/O F18, RAM Budgeting).
> Next: M13 (Multi-Token Prediction / MTP — Speculative Decoding).

| Field           | Nilai                                                                                                |
| :-------------- | :--------------------------------------------------------------------------------------------------- |
| **Deliverable** | CLI interaktif `dismoen chat` dan HTTP micro-daemon `dismoen serve` (OpenAI-compatible)              |
| **Komponen**    | Terminal REPL, ChatML Formatter, SSE Streaming Engine, KMSS v1 Multi-Turn Session State, HTTP Router |
| **Integrasi**   | Direct zero-harness connect: Aider, Cline, Continue.dev, Open WebUI, Python `openai` SDK, `curl`     |
| **Prasyarat**   | M11 hijau (Engine konkurensi $c^*$ stabil, latency hiding $\mathcal{E}_{overlap} \ge 80\%$)          |
| **Gate**        | G-M12-1..G-M12-4                                                                                     |

---

## 1. Latar Belakang & Sasaran Utama

Hingga milestone M11, engine `dismoen` beroperasi sebagai kernel komputasi tingkat rendah (_low-level engine_) yang menerima input via berkas/argumen dan memuntahkan output array token ID mentah.

Milestone M12 mentransformasi `dismoen` menjadi **produk inferensi mandiri (_self-contained engine_)** yang siap pakai untuk interaksi manusia maupun integrasi perkakas AI, tanpa memerlukan lapisan perantara (_no third-party harness needed_):

### Dua Mode Antarmuka Utama:

1. **`dismoen chat` (Fast & Lightweight Terminal REPL)**:
   - Antarmuka terminal interaktif langsung di shell bash.
   - Pembangkitan token di-stream secara instan ke _stdout_ karakter per karakter (_zero latency feeling_).
   - Penanganan otomatis template ChatML standar Qwen 3.6 (`<|im_start|>` / `<|im_end|>`).
   - Melanjutkan konteks multi-turn (_session continuation_) dengan memanfaatkan KMSS v1 (Kimo MoE Session State) tanpa perlu menghitung ulang (_prefill_) riwayat percakapan sebelumnya.

2. **`dismoen serve` (Native OpenAI-Compatible Micro HTTP Server)**:
   - Server HTTP ringan (_low footprint_) yang berjalan sebagai daemon lokal (default port `8000`).
   - Menyediakan endpoint REST standar OpenAI:
     - `GET /v1/models` $\to$ daftar model yang aktif (`qwen3.6-35b-a3b`).
     - `POST /v1/chat/completions` $\to$ melayani permintaan obrolan dengan dukungan _Server-Sent Events_ (SSE) streaming (`stream: true`) maupun respons JSON blok tunggal (`stream: false`).
   - **Zero-Harness Direct Compatibility**: Perkakas AI seperti **Aider** (terminal coding agent), **Cline** (VSCode extension agent), **Continue.dev**, dan **Open WebUI** dapat langsung diarahkan ke `http://localhost:8000/v1` tanpa adaptor atau wrapper tambahan!

---

## 2. Arsitektur Komponen M12

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                            PILIHAN ANTARMUKA USER                           │
│                                                                             │
│   [A] Terminal Interaktif                 [B] External Tools (Aider / Cline)│
│       $ dismoen chat                          $ aider --openai-api-base ... │
└──────────────────────┬──────────────────────────────────────┬───────────────┘
                       │                                      │
                       │ (Direct STDIN/STDOUT)                │ HTTP POST /v1/chat/completions
                       ▼                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 CHAT ENGINE & SESSION ORCHESTRATOR (DISMOEN)                │
│                                                                             │
│  1. ChatML Template Engine:                                                 │
│     <|im_start|>system ... <|im_end|>                                       │
│     <|im_start|>user ... <|im_end|>                                         │
│     <|im_start|>assistant ...                                               │
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

### 2.1 ChatML Formatter & Prompt Invariant

Format prompt mengikuti standar arsitektur Qwen 3.6:

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Halo, siapa kamu?<|im_end|>
<|im_start|>assistant
```

Engine menyuntikkan token khusus:

- `<|im_start|>`: ID 151644
- `<|im_end|>`: ID 151645
  Pembangkitan berhenti secara deterministik saat model memprediksi token `<|im_end|>` atau `<|endoftext|>`.

### 2.2 Kelanjutan Sesi Multi-Turn Tanpa Prefill Ulang (KMSS v1)

Pada obrolan multi-turn sekuensial, menghitung ulang _KV cache_ dan _GDN recurrent state_ dari awal giliran (_turn_) adalah pemborosan latensi ($O(L_{prompt})$).

- M12 menyimpan pointer status KMSS v1 di memori.
- Giliran baru hanya melakukan prefill pada token tambahan:
  $$\Delta N_{tokens} = N_{new\_prompt}$$
- Status KV cache 10 layer attention dan state matriks 30 layer GDN langsung disambung (_appended_), memangkas latensi respon awal giliran ke-2 dan seterusnya hingga **< 0.5 detik**.

### 2.3 Protokol Streaming OpenAI SSE

Pada mode `dismoen serve`, response streaming mematuhi standar RFC 8895 / OpenAI format:

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

1. **Aider (Terminal Coding Agent)**:

   ```bash
   aider --openai-api-base http://127.0.0.1:8000/v1 --model qwen3.6-35b-a3b --api-key none
   ```

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
| **G-M12-1** | **ChatML Prompt Fidelity & Detokenization**: Struktur prompt `<|im_start|>`/`<|im_end|>` ter-encode dengan tepat, detokenisasi streaming UTF-8 multi-byte valid tanpa karakter corrupt | 100% token match vs oracle, zero encoding error | `tests/unit/test_chatml_formatter.mojo` |
| **G-M12-2** | **KMSS v1 Multi-Turn Delta Latency**: Giliran ke-2 dan seterusnya memakai cache KMSS v1, latensi time-to-first-token (TTFT) giliran kedua berkurang drastis vs recompute dari awal | $TTFT_{turn2} \le 0{,}5 \times TTFT_{recompute}$ | `tests/integration/test_m12_session_continuity.sh` |
| **G-M12-3** | **OpenAI API Protocol Conformance**: Endpoint `/v1/models` dan `/v1/chat/completions` lulus pengujian kepatuhan menggunakan Python OpenAI SDK resmi (mode streaming & non-streaming) | HTTP 200, valid JSON & SSE format, zero SDK exception | `tools/bench/verify_openai_conformance.py` |
| **G-M12-4** | **Client Abort & Concurrency Resilience**: Server menangani pemutusan koneksi klien tiba-tiba (*broken pipe / early disconnect*) secara bersih tanpa memory leak dan tanpa engine crash | 0 segfault, 0 zombie threads, VmHWM stabil | `tests/integration/test_m12_resilience.sh` |

---

## 5. Rencana Gelombang Kerja (Execution Waves)

- **Gelombang 1 (M12-W1: ChatML Template & Streaming Detokenizer)**:
  - Implementasi formatter prompt ChatML untuk format role system, user, dan assistant.
  - Streaming detokenizer yang menangani potongan byte UTF-8 multi-byte (Gate G-M12-1).
- **Gelombang 2 (M12-W2: Terminal Interactive REPL `dismoen chat`)**:
  - Implementasi antarmuka loop terminal interaktif dengan graceful abort `Ctrl+C`.
  - Integrasi session continuity KMSS v1 (Gate G-M12-2).
- **Gelombang 3 (M12-W3: Embedded HTTP Micro-Server `dismoen serve`)**:
  - Server HTTP ringan berbasis socket lokal berkinerja tinggi.
  - Penanganan router `/v1/models` dan `/v1/chat/completions`.
- **Gelombang 4 (M12-W4: OpenAI Protocol Conformance & Streaming SSE)**:
  - Implementasi chunk format SSE `data: {...}` dan event `[DONE]`.
  - Verifikasi kompatibilitas langsung dengan Python OpenAI SDK dan `curl` (Gate G-M12-3).
- **Gelombang 5 (M12-W5: Harness Integration, Client Abort & Closure)**:
  - Uji end-to-end dengan Aider dan Cline.
  - Pengujian ketahanan pemutusan koneksi klien (Gate G-M12-4).
  - Terbitkan scorecard kelulusan di `reports/YYYY-MM-DD/M12-gates-scorecard.md`.

---

## 6. Definisi Selesai (DoD M12)

- [ ] Subperintah `dismoen chat` berfungsi sebagai REPL terminal interaktif dengan live streaming dan ChatML prompt formatting yang akurat (Gate G-M12-1).
- [ ] Multi-turn session state KMSS v1 berhasil menjaga konteks percakapan tanpa menghitung ulang prefill dari awal giliran (Gate G-M12-2).
- [ ] Subperintah `dismoen serve` berjalan stabil sebagai HTTP micro-daemon dan mematuhi spesifikasi OpenAI Chat Completions API (Gate G-M12-3).
- [ ] Terverifikasi sukses terhubung langsung dengan perkakas klien AI eksternal (Aider, Cline, Python `openai` SDK) tanpa lapisan perantara pihak ketiga (*zero-harness*).
- [ ] Server tangguh terhadap *early disconnect / client abort* tanpa memicu thread deadlock atau kebocoran memori (Gate G-M12-4).
- [ ] Seluruh suite pengujian regresi (`validate-m11`, `validate-m10`, `validate-m9`) dan 13 hook `pre-commit` 100% hijau.
- [ ] Scorecard formal sertifikasi M12 ter-commit di `reports/YYYY-MM-DD/M12-gates-scorecard.md`.

```
