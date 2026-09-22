# Real-Time Context-Aware Interview Co-Pilot (V3 Reliability Release)

Low-Latency, Reliability-Harden, Real-Time Interview Assistant designed for Google Data Analytics Apprenticeship (GCA + Googliness) and Technical Interviews on Windows 10/11.

---

## 🌟 Key Architecture & Capabilities

1. **Deterministic Audio & Voice Activity Detection (VAD)**
   - Windows WASAPI loopback capture for system audio (Interviewer channel) and microphone input (Candidate channel).
   - High-throughput float32 ring buffer downsampled via linear interpolation to 16kHz mono.
   - Dual-threshold energy VAD (0.012 speech, 0.008 silence) with 0.5s silence End-of-Speech (EOS) trigger.

2. **Automatic CPU / CUDA Hardware Whisper Selection**
   - Automatically detects NVIDIA CUDA, cuBLAS, and cuDNN DLLs on Windows.
   - Defaults to `cuda` with `float16` compute type; gracefully falls back to `cpu` with `int8` quantization if GPU is unavailable or throws an OOM/driver error.

3. **Speculative Partial-STT Preparation**
   - Runs early speculative Whisper passes on partial speech buffers before the interviewer completes their sentence.
   - Warms up retrieval cache; if the final transcript validates speculative intent, retrieval latency drops to 0ms.

4. **Interview Separation & Routing Engine (ISRE)**
   - Rule-based routing in <2ms with zero LLM overhead.
   - Classifies utterances into 7 distinct interview types:
     - `GCA` (General Cognitive Ability / business scenario breakdown)
     - `TECHNICAL` (Concept definitions, troubleshooting, SQL, BI tools)
     - `QUANTITATIVE` (Zero-latency deterministic math solver)
     - `PERSONAL` (Resume walkthrough, career pivot, motivation)
     - `PROJECT` (Candidate's authoritative Delinquency Dashboard deep-dive)
     - `BEHAVIORAL` (STAR framework: Teamwork, Conflict, Failure)
     - `FOLLOW_UP` (Contextual inheritance of parent question and topic)

5. **Entity Protection & Domain Normalizer**
   - Normalizes phonetic STT errors (e.g. `Power bee`, `pbi` -> `Power BI`, `dacs` -> `DAX`).
   - Strict preservation of `Power VA` (Volt-Ampere unit) to prevent accidental collapse into `Power BI`.
   - Contradiction guardrail flags biological, electrical, or financial homonyms (e.g., measuring Power BI in volts or confusing DAX with the German stock market).

6. **Authoritative Grounding & Hallucination Taxonomy (H1–H7)**
   - Single source of truth: `context.md` (39 numbered sections).
   - Candidate context is strictly isolated and never leaked into general technical queries.
   - Strict audit blocks ungrounded experience (H1), fabricated metrics (H2), entity confusion (H3), unsupported achievements (H5), and chronological distortion (H6).
   - Unverified queries reliably trigger safe fallback states (`NO_SAFE_ANSWER` / `NO_VERIFIED_STORY`).

7. **Multi-Tier LLM Provider Failover & Monotonic Cancellation**
   - Primary provider: Groq (`llama-3.3-70b-versatile`) for ultra-low Time-To-First-Token (TTFT < 350ms).
   - Automatic zero-restart failover: NVIDIA NIM (`meta/llama-3.3-70b-instruct`) on rate limits, connection timeouts, or API errors.
   - Monotonic generation IDs prevent stale answers from rendering when the interviewer interrupts or asks a follow-up.

8. **Cloaked Webcam Teleprompter Overlay**
   - Semi-transparent, top-aligned, borderless Tkinter window positioned directly beneath the webcam.
   - Windows API screen-share cloaking (`SetWindowDisplayAffinity`) keeps prompt invisible to screen-capture tools during video calls.

---

## 🚀 Quickstart & Installation

### Prerequisites
- Windows 10/11
- Python 3.10 or 3.11 (Python 3.11 recommended)
- Optional: NVIDIA GPU with CUDA 12.x for accelerated Whisper transcription

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/creative-kalfan/Interview_Copilot.git
cd Interview_Copilot

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys
Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```
Edit `.env` and configure your keys:
```ini
GROQ_API_KEY=gsk_...
NVIDIA_API_KEY=nvapi-...
INTERVIEWER_MIC_INDEX=0
CANDIDATE_MIC_INDEX=1
```

### 3. Run the Application
```bash
python copilot.py
```

---

## 🧪 Testing & Verification

The repository includes two complete test suites verifying pipeline latency, routing, grounding, and failover:

### 1. Standalone V3 Reliability Test Suite (18 Tests)
Validates all V3 domain normalization, contradiction detection, state transitions, hallucination taxonomy, and dataset structures:
```bash
python test_v3_reliability.py
```
*Expected: 18/18 PASSED (100%)*

### 2. Full Audio & Hardware Validation Suite (10 Tests)
Simulates end-to-end WASAPI audio, faster-whisper transcription, Groq/NVIDIA failovers, speculative STT, and GCA formatting:
```bash
python test_validation_suite.py
```
*Expected: 10/10 PASSED (100%) with median E2E first UI token < 500ms.*

---

## 📂 Repository Structure

```
Interview_Copilot/
├── copilot.py                 # Core unified application (VAD, STT, ISRE, LLM, Teleprompter)
├── context.md                 # Authoritative candidate context (39 structured sections)
├── test_v3_reliability.py     # Standalone V3 reliability test suite
├── test_validation_suite.py   # E2E Audio, Whisper, LLM, and Routing validation suite
├── .env.example               # Template environment configuration
├── requirements.txt           # Production Python dependencies
├── docs/                      # Supporting documentation & framework guidelines
│   └── gca_frameworks.md      # GCA structured breakdown frameworks
└── evaluation/                # Test datasets for benchmark evaluation
    ├── routing.jsonl          # Intent classification test cases
    ├── technical.jsonl        # Technical definition test cases
    ├── behavioral.jsonl       # STAR behavioral test cases
    ├── project.jsonl          # Project deep-dive test cases
    ├── followups.jsonl        # Follow-up inheritance test cases
    ├── grounding.jsonl        # Entity grounding & candidate privacy cases
    └── adversarial.jsonl      # Adversarial prompts & homonym checks
```

---

## 🔒 Security & Candidate Truthfulness Rules

- **Zero Invention Guarantee:** No candidate stories, employer names, project metrics, or certifications may be invented or accepted outside `context.md`.
- **Candidate Context Isolation:** General technical questions (SQL, Power BI, DAX, Python) strictly receive zero candidate context injection.
- **Privacy First:** `.env`, audio recordings, transcript logs, and API credentials are kept local and excluded from version control.
