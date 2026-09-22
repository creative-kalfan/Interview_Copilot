"""
Real-Time Context-Aware Interview Co-Pilot V2 (Windows 10/11)
Low-Latency GCA + Googliness Interview Practice Assistant.
"""

import os
import sys
import site
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. Dynamic CUDA / cuBLAS / cuDNN DLL Directory Registration (Windows)
# ---------------------------------------------------------------------------
def _register_cuda_dlls() -> None:
    """Register all installed NVIDIA / CUDA DLL directories for ctranslate2 / faster-whisper."""
    if sys.platform != "win32":
        return
    site_dirs = []
    try:
        site_dirs.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        u_site = site.getusersitepackages()
        if u_site:
            site_dirs.append(u_site)
    except Exception:
        pass

    registered = set()
    for s_dir in site_dirs:
        base = Path(s_dir)
        if not base.exists():
            continue
        for p in base.glob("*nvidia*"):
            if not p.is_dir():
                continue
            for sub in [p] + list(p.rglob("*")):
                if sub.is_dir() and sub.name.lower() in ("bin", "lib"):
                    s_str = str(sub.resolve())
                    if s_str not in registered:
                        registered.add(s_str)
                        try:
                            os.add_dll_directory(s_str)
                            os.environ["PATH"] = s_str + os.pathsep + os.environ.get("PATH", "")
                        except Exception:
                            pass

_register_cuda_dlls()

import time
import math
import queue
import logging
import threading
import re
import shutil
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np

# Configure Logging
LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("InterviewCopilot")

# ---------------------------------------------------------------------------
# 1. Workspace Bootstrapping & Path Management (Downloads/Interview_Copilot)
# ---------------------------------------------------------------------------
DOWNLOADS_DIR = Path.home() / "Downloads"
WORKSPACE_DIR = DOWNLOADS_DIR / "Interview_Copilot"
DOCS_DIR = WORKSPACE_DIR / "docs"
LOGS_DIR = WORKSPACE_DIR / "logs"
CONFIG_DIR = WORKSPACE_DIR / "config"
ENV_FILE = WORKSPACE_DIR / ".env"

DEFAULT_STAR_STORIES = """# Candidate STAR Interview Stories

## Story 1: Production Incident Resolution & Cache Storm Mitigation
- **Category**: Technical Leadership / Incident Management / Performance
- **Company / Context**: Cloud Infrastructure Platform
- **Situation**: A sudden thundering herd on user authentication service caused p99 latency to spike to 2.4s and cascaded failures across downstream API gateways during peak traffic.
- **Task**: Identify root cause under high pressure, restore service stability within 30 minutes, and implement permanent mitigation.
- **Action**:
  - Triaged logs and distributed traces; identified expired cache keys simultaneously triggering expensive database lookups.
  - Implemented probabilistic early expiration (XFetch algorithm) and distributed mutex locking on cache misses.
  - Coordinated with cross-functional teams to deploy canary fix with zero further downtime.
- **Result**: Latency returned to 85ms within 22 minutes; database query load dropped 78%; zero repeat incidents over following 12 months.
- **Emphasize**: Composure under pressure, root-cause focus, data-driven decisions.

## Story 2: Architectural Disagreement on API Design
- **Category**: Conflict Resolution / Collaboration / Googliness
- **Company / Context**: Distributed Services Team
- **Situation**: Tech lead favored a rigid gRPC-only mandate across all services, while frontend engineers needed REST/GraphQL for browser client performance and rapid iteration.
- **Task**: Align two engineering teams with opposing requirements without stalling delivery timeline.
- **Action**:
  - Organized a timeboxed evaluation matrix assessing payload size, browser client complexity, tooling maturity, and maintenance overhead.
  - Championed an Envoy-based gRPC-JSON transcoding layer as a pragmatic bridge: backend maintained pure gRPC schemas, while frontend consumed idiomatic REST endpoints.
  - Acknowledged valid concerns from both sides and documented trade-offs transparently in an RFC.
- **Result**: Shipped on schedule with full consensus; frontend team achieved 40% faster feature velocity while backend kept schema type safety.
- **Emphasize**: Intellectual humility, disagree and commit, empathy for partner team workflows.

## Story 3: Underestimating Scope & Project Delay
- **Category**: Failure / Retrospective / Learning
- **Company / Context**: Data Ingestion Platform
- **Situation**: Committed to a 4-week timeline for migrating historical telemetry data to a new columnar data store, but encountered undocumented edge-case schema drift in legacy partitions.
- **Task**: Re-align stakeholders, mitigate customer delivery impact, and successfully complete the migration.
- **Action**:
  - Immediately flagged the 2-week risk to engineering director instead of quietly working overtime and hiding the delay.
  - Built automated schema validation scripts with dead-letter queue isolation so clean partitions migrated ahead of schedule.
  - Conducted blameless retrospective and instituted pre-migration schema auditing checklist for future initiatives.
- **Result**: Migration finished 1.5 weeks behind original estimate, but with 100% data fidelity and zero data loss. The audit checklist became the org-wide migration standard.
- **Emphasize**: Transparency, accountability, turning setbacks into systemic process improvements.
"""

DEFAULT_GCA_FRAMEWORKS = """# GCA & System Design Problem Solving Frameworks

## 1. General Cognitive Ability (GCA) Framework
1. **Clarify Objective & Constraints**:
   - What metric are we optimizing (latency, retention, cost, throughput)?
   - What is the scope (global vs single region, all users vs cohorts)?
   - What are the operational constraints (budget, timeline, existing tech debt)?
2. **Decompose & Structure**:
   - Break problem into distinct MECE components (Mutually Exclusive, Collectively Exhaustive).
   - Trace flow end-to-end (Client -> CDN/Edge -> Gateway -> Services -> Storage).
3. **Hypothesis Generation & Root Cause**:
   - Brainstorm potential causes or bottlenecks.
   - Categorize by likelihood and operational impact.
4. **Prioritize & Validate**:
   - Propose rapid diagnostic steps (telemetry, log slicing, A/B isolation).
   - Formulate short-term containment vs long-term structural fix.
5. **Trade-offs & Edge Cases**:
   - Explicitly articulate trade-offs (e.g., consistency vs availability, memory vs CPU).

## 2. Behavioral & Googliness Cues
- Emphasize ownership and accountability without blaming peers.
- Demonstrate intellectual humility when proven wrong by data.
- Collaborative problem solving: disagree constructively, commit fully.
"""

ENV_TEMPLATE = """# Interview Co-Pilot V2 Configuration
GROQ_API_KEY=
NVIDIA_API_KEY=

# Audio & VAD Latency Tuning
PARTIAL_TRANSCRIBE_INTERVAL=0.35
END_OF_TURN_SILENCE=0.50
MIN_SPEECH_DURATION=0.35
SILENCE_RMS_THRESHOLD=0.012

# Whisper STT Configuration
WHISPER_MODEL=tiny.en
WHISPER_DEVICE=auto
WHISPER_COMPUTE_TYPE=float16

# LLM Providers & Models
GROQ_MODEL=qwen/qwen3.8-27b
NVIDIA_MODEL=meta/llama-3.2-11b-vision-instruct
LLM_MAX_TOKENS=200
LLM_TEMPERATURE=0.35
HISTORY_TURNS=6
RETRIEVAL_TOP_K=2
"""

def bootstrap_workspace() -> None:
    """Scaffold directory tree in Downloads, self-copy script, and initialize .env and docs."""
    for folder in [WORKSPACE_DIR, DOCS_DIR, LOGS_DIR, CONFIG_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    log_file = LOGS_DIR / f"session_{time.strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(fh)

    try:
        current_script = Path(__file__).resolve()
        dest_script = WORKSPACE_DIR / "copilot.py"
        if current_script.is_file() and current_script != dest_script:
            shutil.copy2(current_script, dest_script)
            logger.info(f"Self-copied executable script to: {dest_script}")
    except Exception as e:
        logger.debug(f"Self-copy skipped: {e}")

    star_file = DOCS_DIR / "star_stories.md"
    if not star_file.exists():
        star_file.write_text(DEFAULT_STAR_STORIES, encoding="utf-8")
        logger.info(f"Generated STAR stories template at: {star_file}")

    gca_file = DOCS_DIR / "gca_frameworks.md"
    if not gca_file.exists():
        gca_file.write_text(DEFAULT_GCA_FRAMEWORKS, encoding="utf-8")
        logger.info(f"Generated GCA frameworks at: {gca_file}")

    if not ENV_FILE.exists():
        old_env = Path("./.env")
        if old_env.exists():
            shutil.copy2(old_env, ENV_FILE)
            logger.info(f"Migrated existing .env to: {ENV_FILE}")
        else:
            ENV_FILE.write_text(ENV_TEMPLATE, encoding="utf-8")
            logger.warning(f"Created template .env at: {ENV_FILE}")

def load_env_variables() -> Dict[str, str]:
    """Parse .env cleanly and set environment variables."""
    env_vars = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env_vars[k.strip()] = v.strip().strip('"').strip("'")
                os.environ[k.strip()] = env_vars[k.strip()]
    return env_vars

# Load configuration values with sensible defaults
bootstrap_workspace()
load_env_variables()

# Performance & Latency Configurations
PARTIAL_TRANSCRIBE_INTERVAL = float(os.getenv("PARTIAL_TRANSCRIBE_INTERVAL", "0.35"))
END_OF_TURN_SILENCE = float(os.getenv("END_OF_TURN_SILENCE", "0.50"))
MIN_SPEECH_DURATION = float(os.getenv("MIN_SPEECH_DURATION", "0.35"))
SILENCE_RMS_THRESHOLD = float(os.getenv("SILENCE_RMS_THRESHOLD", "0.012"))

WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny.en")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")

GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.2-11b-vision-instruct")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "260"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.35"))
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "6"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "2"))

# Speculative partial-STT preparation (V2.1)
SPEC_PARTIAL_INTERVAL = float(os.getenv("SPEC_PARTIAL_INTERVAL", "1.0"))       # s between full-buffer speculative partials
SPEC_MIN_WORDS = int(os.getenv("SPEC_MIN_WORDS", "6"))                          # min words before speculative intent/retrieval
SPEC_SIMILARITY_THRESHOLD = float(os.getenv("SPEC_SIMILARITY_THRESHOLD", "0.6"))  # token Jaccard threshold for reuse
WARMUP_PROVIDERS = os.getenv("WARMUP_PROVIDERS", "1").strip().lower() not in ("0", "false", "no", "")
WARMUP_TIMEOUT = float(os.getenv("WARMUP_TIMEOUT", "6.0"))

# ---------------------------------------------------------------------------
# 2. Performance Telemetry Instrumentation
# ---------------------------------------------------------------------------
class PerformanceTelemetry:
    """Tracks latency breakdown across the entire lifecycle: audio -> VAD -> STT -> RAG -> LLM -> UI."""

    def __init__(self, question: str = ""):
        self.question = question
        self.t_audio_capture_start: float = 0.0
        self.t_speech_detected: float = 0.0
        self.t_partial_stt_queued: float = 0.0
        self.t_end_of_turn_detected: float = time.time()
        self.t_stt_started: float = 0.0
        self.t_stt_completed: float = 0.0
        self.t_intent_classification: float = 0.0
        self.t_retrieval: float = 0.0
        self.t_llm_request_sent: float = 0.0
        self.t_llm_ttft: float = 0.0
        self.t_first_useful_token_ui: float = 0.0
        self.t_final_response_rendered: float = 0.0
        self.intent: str = "UNKNOWN"
        self.provider: str = ""
        self.token_count: int = 0
        # V2.1 instrumentation
        self.t_prompt_enqueued: float = 0.0
        self.t_llm_stream_created: float = 0.0
        self.provider_events: List[Dict[str, Any]] = []
        self.spec_reused: bool = False
        self.spec_prep_ms: float = 0.0

    def mark_audio_start(self) -> None:
        self.t_audio_capture_start = time.time()

    def mark_speech_detected(self) -> None:
        self.t_speech_detected = time.time()

    def mark_partial_stt_queued(self) -> None:
        self.t_partial_stt_queued = time.time()

    def mark_turn_end(self) -> None:
        self.t_end_of_turn_detected = time.time()

    def mark_stt_start(self) -> None:
        self.t_stt_started = time.time()

    def mark_stt_done(self) -> None:
        self.t_stt_completed = time.time()

    def mark_intent_done(self, intent: str) -> None:
        self.intent = intent
        self.t_intent_classification = time.time()

    def mark_retrieval_done(self) -> None:
        self.t_retrieval = time.time()

    def mark_llm_request_sent(self) -> None:
        self.t_llm_request_sent = time.time()

    def mark_first_token(self, provider: str) -> None:
        self.provider = provider
        self.t_llm_ttft = time.time()
        self.t_first_useful_token_ui = self.t_llm_ttft

    def mark_generation_done(self, token_count: int = 0) -> None:
        self.token_count = token_count
        self.t_final_response_rendered = time.time()

    def mark_prompt_enqueued(self) -> None:
        self.t_prompt_enqueued = time.time()

    def mark_llm_stream_created(self) -> None:
        self.t_llm_stream_created = time.time()

    def add_provider_event(self, provider: str, event: str, detail: str = "") -> None:
        self.provider_events.append({
            "provider": provider,
            "event": event,
            "detail": detail,
            "t": time.time(),
        })

    def get_summary_dict(self) -> Dict[str, float]:
        eos = self.t_end_of_turn_detected

        stt_complete_ms = (self.t_stt_completed - eos) * 1000 if self.t_stt_completed else 0.0
        llm_request_ms = (self.t_llm_request_sent - eos) * 1000 if self.t_llm_request_sent else 0.0
        first_ui_token_ms = (self.t_first_useful_token_ui - eos) * 1000 if self.t_first_useful_token_ui else 0.0
        complete_response_ms = (self.t_final_response_rendered - eos) * 1000 if self.t_final_response_rendered else 0.0

        stt_duration_ms = (self.t_stt_completed - self.t_stt_started) * 1000 if self.t_stt_completed and self.t_stt_started else 0.0
        intent_ms = (self.t_intent_classification - self.t_stt_completed) * 1000 if self.t_intent_classification and self.t_stt_completed else 0.0
        rag_ms = (self.t_retrieval - self.t_intent_classification) * 1000 if self.t_retrieval and self.t_intent_classification else 0.0
        ttft_pure_ms = (self.t_llm_ttft - self.t_llm_request_sent) * 1000 if self.t_llm_ttft and self.t_llm_request_sent else 0.0

        return {
            "eos_to_stt_complete_ms": round(stt_complete_ms, 1),
            "eos_to_llm_request_ms": round(llm_request_ms, 1),
            "eos_to_first_token_ms": round(first_ui_token_ms, 1),
            "eos_to_complete_response_ms": round(complete_response_ms, 1),
            "stt_duration_ms": round(stt_duration_ms, 1),
            "intent_ms": round(intent_ms, 1),
            "rag_ms": round(rag_ms, 1),
            "ttft_pure_ms": round(ttft_pure_ms, 1),
            "spec_reused": self.spec_reused,
            "spec_prep_ms": round(self.spec_prep_ms, 1),
            "provider_events": [
                {"provider": e["provider"], "event": e["event"], "detail": e["detail"],
                 "offset_ms": round((e["t"] - eos) * 1000, 1) if eos else 0.0}
                for e in self.provider_events
            ],
        }

    def log_and_format(self) -> str:
        s = self.get_summary_dict()
        spec_str = f" | Spec: {'REUSED ' + str(s['spec_prep_ms']) + 'ms' if s['spec_reused'] else 'miss'}"
        ev_str = " | Events: " + ", ".join(f"{e['provider']}:{e['event']}@{e['offset_ms']:.0f}ms" for e in s["provider_events"]) if s["provider_events"] else ""
        msg = (
            f"[PERF] Intent={self.intent} | EOS->STT: {s['eos_to_stt_complete_ms']}ms | "
            f"EOS->LLMReq: {s['eos_to_llm_request_ms']}ms | EOS->FirstUI: {s['eos_to_first_token_ms']}ms | "
            f"EOS->Complete: {s['eos_to_complete_response_ms']}ms | TTFT: {s['ttft_pure_ms']}ms{spec_str}{ev_str}"
        )
        logger.info(msg)
        return f"STT: {int(s['eos_to_stt_complete_ms'])}ms | FirstUI: {int(s['eos_to_first_token_ms'])}ms | TTFT: {int(s['ttft_pure_ms'])}ms"

# ---------------------------------------------------------------------------
# 3. Intent Classification & Routing Layer (<2ms Local Heuristic)
# ---------------------------------------------------------------------------
class IntentClassifier:
    """Ultra-fast local intent classifier. Categorizes interviewer utterances without LLM latency."""

    GCA_INDICATORS = (
        "drop", "decrease", "increase", "spike", "metric", "engagement", "revenue",
        "investigate", "how would you solve", "how would you approach", "how would you investigate",
        "what if", "suppose", "imagine", "estimate", "how many", "troubleshoot", "root cause",
        "trade-off", "trade off", "problem", "solve", "situation where numbers", "why did"
    )

    BEHAVIORAL_INDICATORS = (
        "tell me about a time", "have you ever", "give an example", "describe a situation",
        "disagreement", "conflict", "disagreed", "mistake", "failed", "failure", "proud of",
        "challenging project", "difficult teammate", "leadership", "teamwork", "cross-functional",
        "pushback", "tight deadline", "under pressure", "learned from", "feedback",
        "difficult situation", "challenging situation", "learn something quickly"
    )

    TECHNICAL_INDICATORS = (
        "system design", "architecture", "scale", "concurrency", "distributed", "microservices",
        "database", "sharding", "cache", "redis", "kafka", "latency", "throughput", "p99",
        "api design", "rest", "grpc", "algorithm", "data structure", "deadlock", "thread"
    )

    PROJECT_INDICATORS = (
        "in your resume", "at your last company", "in your previous role", "walk me through your project",
        "what was your stack", "what did you build", "how did you implement"
    )

    CLARIFICATION_INDICATORS = (
        "what do you mean by", "can you clarify", "could you elaborate", "are you saying",
        "just to confirm", "so your point is"
    )

    SMALL_TALK_INDICATORS = (
        "can you hear me", "am i audible", "hello", "hi", "hey", "good morning", "good afternoon",
        "how are you", "thanks for joining", "let's get started", "nice to meet you",
        "yes", "no", "okay", "alright", "cool", "sounds good", "perfect"
    )

    FOLLOW_UP_INDICATORS = (
        "why that", "why that approach", "why did you", "what happened next",
        "what was your role", "what did you learn", "how did you measure", "and then",
        "what would you do differently", "why not", "how so", "instead of",
        "then what", "what changed", "and the trade", "how?", "what?"
    )

    @staticmethod
    def _hit(indicators, t: str, stem: bool = False) -> bool:
        """Indicator match. stem=True allows verb inflections (drop->dropped); exact otherwise
        (prevents 'rest' matching inside 'the rest of the team')."""
        for i in indicators:
            pat = rf"\b{re.escape(i)}\w*" if stem else rf"\b{re.escape(i)}\b"
            if re.search(pat, t):
                return True
        return False

    @classmethod
    def classify(cls, text: str) -> Tuple[str, bool]:
        """
        Classifies utterance into (intent, should_trigger_llm).
        Intents: GCA, BEHAVIORAL, TECHNICAL, PROJECT, CLARIFICATION, SMALL_TALK, FOLLOW_UP, OTHER.
        """
        t = text.strip().lower()
        words = t.split()
        if not words:
            return "OTHER", False

        # 1. Check small talk / greetings first (punctuation-tolerant word match)
        t_norm = re.sub(r"[^a-z0-9\s]", " ", t)
        for st in cls.SMALL_TALK_INDICATORS:
            if re.search(rf"\b{re.escape(st)}\b", t_norm):
                if len(words) <= 6 and not any(k in t_norm for k in ("why", "how", "what", "design", "time")):
                    return "SMALL_TALK", False

        # 2. Check direct follow-up questions
        for fu in cls.FOLLOW_UP_INDICATORS:
            if fu in t:
                return "FOLLOW_UP", True
        if t in ("why?", "why", "how so?", "what happened?", "and then?"):
            return "FOLLOW_UP", True
        # Short challenge/interrogative continuations ("What if the data is correct?", "How would you
        # validate that?", "And the trade-off?") inherit the previous question's context instead of
        # starting a fresh analytical answer. Cold start is handled by DialogueManager.
        if len(words) <= 8 and (t in ("how?", "what?") or t.startswith(
            ("what if ", "what about", "and ", "but ", "so ", "then ")
        )):
            return "FOLLOW_UP", True

        # 3. Check behavioral / STAR questions
        if cls._hit(cls.BEHAVIORAL_INDICATORS, t, stem=True):
            return "BEHAVIORAL", True

        # 4. Check GCA / Case study questions
        if cls._hit(cls.GCA_INDICATORS, t, stem=True):
            return "GCA", True

        # 5. Check Technical / System Design questions
        if cls._hit(cls.TECHNICAL_INDICATORS, t):
            return "TECHNICAL", True

        # 6. Check Project / Resume questions
        if cls._hit(cls.PROJECT_INDICATORS, t):
            return "PROJECT", True

        # 7. Check Clarifications
        if cls._hit(cls.CLARIFICATION_INDICATORS, t):
            return "CLARIFICATION", True

        # Fallback question detection
        is_question = t.endswith("?") or any(t.startswith(w) for w in ("what", "why", "how", "can you", "could you", "where"))
        # Short wh-questions ("How?", "Why?", "What would you check first?", "Then what?") inherit
        # the previous question's context instead of starting a fresh analytical answer.
        # Personal "your X" questions are NOT follow-ups ("What are your strengths?").
        if is_question and len(words) <= 6 and not t.startswith(("what are your", "what is your", "whats your")):
            return "FOLLOW_UP", True
        if is_question and len(words) >= 4:
            return "GCA", True  # default analytical question to GCA framing
        elif len(words) >= 12:
            return "GCA", True

        return "OTHER", False

# ---------------------------------------------------------------------------
# 3b. Interview Separation & Routing Engine (ISRE)
# Decides the interview TYPE before GCA-specific reasoning. Cheap, local, sub-ms.
# ---------------------------------------------------------------------------
class InterviewType(Enum):
    GCA = "GCA"
    TECHNICAL = "TECHNICAL"
    QUANTITATIVE = "QUANTITATIVE"
    PERSONAL = "PERSONAL"
    PROJECT = "PROJECT"
    BEHAVIORAL = "BEHAVIORAL"
    FOLLOW_UP = "FOLLOW_UP"
    OTHER = "OTHER"


@dataclass
class InterviewClassification:
    interview_type: InterviewType
    intent: str                 # pipeline-facing intent string; FOLLOW_UP resolves to parent intent
    confidence: float
    reason: str
    should_trigger: bool = True
    latency_ms: float = 0.0


_TECH_TOOLS = (
    "power bi", "tableau", "sql", "redis", "readies", "reedies", "readings", "postgres", "docker", "kafka", "api",
    "fastapi", "django", "react", "next.js", "python", "etl", "json", "nosql",
    "mongodb", "graphql", "aws", "azure", "kubernetes", "spark", "server",
)

_NUM = r"(\d[\d,]*(?:\.\d+)?)"

_QUANT_SHAPE = re.compile(
    r"(what\s+(?:percentage|percent)|%\s*of\b|percent\s+of\b|percentage\s+(?:increase|decrease)|"
    r"percent\s+(?:increase|decrease)|\btimes\s+" + _NUM + r")"
)

_TECH_DEFN = re.compile(
    r"^(what\s+(?:is|are)\s+(?!your\b|you\b|this\s+role)|explain\s+|define\s+|what.s\s+the\s+difference\s+between|"
    r"difference\s+between|compare\s+|how\s+does\s+\S+\s+work)"
)
_TECH_TROUBLE = re.compile(r"(can'?t|cannot|won'?t|fails?|failing|error|unable|not connect|broken|debug)")
_TECH_WHY_USE = re.compile(r"why\s+(?:would|did|do|should)\s+you\s+(?:use|choose|pick|select|go\s+with)")
_PROJECT_RE = re.compile(
    r"(careeros|career\s*os|care\s*arrows?|gear\s*arrows?|careyros|your project|the project|your (?:trading|job ingestion|blog|syntheseed)|"
    r"project architecture|walk me through your (?:project|architecture)|biggest challenge|job ingestion|"
    r"credit risk|delinquency|in (?:your|the) dashboard|your dashboard|the dashboard|built in power bi|portfolio project)"
)
_PERSONAL_RE = re.compile(
    r"(tell me about yourself|yourself\b|walk me through your resume|your background|"
    r"why are you interested|why (?:google|this role|our company|this company|apprenticeship)|your strengths|"
    r"your weaknesses|weakness|why did you (?:move|switch|transition)|civil engineering|about you\b|"
    r"why data|why analytics|where do you see yourself|future goals|transition from civil)"
)
_FOLLOW_CTX = re.compile(r"^(why|how|and|so|then|but|what did|what was|what would|what made)\b")
_FOLLOW_CTX_MAX_WORDS = 12

# ---------------------------------------------------------------------------
# 4. Dynamic Document Context Retrieval (Intent-Aware BM25)
# ---------------------------------------------------------------------------
def _looks_like_question(text: str) -> bool:
    """Mic-channel guard: question-shaped utterances (incl. short follow-ups) and interviewer-style
    imperative prompts trigger guidance; spoken candidate narration ("I built ...") does not."""
    t = text.strip().lower()
    return t.endswith("?") or t.startswith((
        "what", "why", "how", "which", "when", "where", "who", "can you", "could you", "then what", "and the",
        "estimate", "calculate", "suppose", "imagine", "design", "explain", "describe", "tell me",
        "walk me", "compare", "given ", "diagnose", "evaluate", "assess", "troubleshoot", "prioritize",
    ))


def _parse_num(s: str) -> float:
    return float(s.replace(",", ""))


def _fmt_num(x: float) -> str:
    return str(int(x)) if abs(x - round(x)) < 1e-9 else f"{x:.2f}"


def quant_answer(question: str) -> Optional[str]:
    """Deterministic arithmetic/percentage solver. Returns a formatted ANSWER string or None.
    Only fires on unambiguous arithmetic shapes; anything else falls to the LLM quant route."""
    t = question.lower().replace("what's", "what is")
    m = re.search(r"what\s+(?:percentage|percent)\s+is\s+" + _NUM + r"\s+of\s+" + _NUM, t)
    if m:
        x, y = _parse_num(m.group(1)), _parse_num(m.group(2))
        if y != 0:
            return f"{m.group(1)} is {x / y * 100:.2f}% of {m.group(2)}."
    m = re.search(r"what\s+(?:percentage|percent)\s+of\s+" + _NUM + r"\s+is\s+" + _NUM, t)
    if m:
        y, x = _parse_num(m.group(1)), _parse_num(m.group(2))
        if y != 0:
            return f"{m.group(2)} is {x / y * 100:.2f}% of {m.group(1)}."
    m = re.search(r"what\s+is\s+" + _NUM + r"\s*(?:%|percent)\s+of\s+" + _NUM, t)
    if m:
        p, n = _parse_num(m.group(1)), _parse_num(m.group(2))
        return f"{_fmt_num(p / 100 * n)} is {m.group(1)}% of {m.group(2)}."
    m = re.search(_NUM + r"\s*(?:\*|times|multiplied by)\s*" + _NUM, t)
    if m:
        a, b = _parse_num(m.group(1)), _parse_num(m.group(2))
        return f"{_fmt_num(a * b)}."
    m = re.search(_NUM + r"\s*(?:\+|plus)\s*" + _NUM, t)
    if m:
        a, b = _parse_num(m.group(1)), _parse_num(m.group(2))
        return f"{_fmt_num(a + b)}."
    m = re.search(_NUM + r"\s*(?:-|minus)\s*" + _NUM, t)
    if m:
        a, b = _parse_num(m.group(1)), _parse_num(m.group(2))
        return f"{_fmt_num(a - b)}."
    m = re.search(r"from\s+" + _NUM + r"\s+to\s+" + _NUM, t)
    if m and any(w in t for w in ("increase", "decrease", "drop", "growth", "change", "grew")):
        a, b = _parse_num(m.group(1)), _parse_num(m.group(2))
        if a != 0:
            pct = (b - a) / abs(a) * 100
            word = "increase" if pct >= 0 else "decrease"
            return f"That is a {abs(pct):.2f}% {word} (from {m.group(1)} to {m.group(2)})."
    m = re.search(_NUM + r"\s*(?:%|percent)\s+of\s+" + _NUM, t)
    if m and any(w in t for w in ("remain", "left", "churn", "survive")):
        p, n = _parse_num(m.group(1)), _parse_num(m.group(2))
        return f"{_fmt_num(n * (1 - p / 100))} remain out of {m.group(2)} (after {m.group(1)}% churn)."
    return None


class ISRE:
    """Interview Separation & Routing Engine.

    Classifies the interview TYPE (GCA / TECHNICAL / QUANTITATIVE / PERSONAL / PROJECT /
    BEHAVIORAL / FOLLOW_UP / OTHER) using question structure, semantic cues, technical
    vocabulary, and conversation context (previous question + previous type). Pure local
    heuristics, sub-millisecond, no LLM calls. Follow-ups inherit the parent interview type;
    unknown questions fall back to a generic direct answer, never to GCA by default."""

    _PARENT_TYPES = {
        InterviewType.GCA, InterviewType.TECHNICAL, InterviewType.QUANTITATIVE,
        InterviewType.PERSONAL, InterviewType.PROJECT, InterviewType.BEHAVIORAL,
    }

    _GCA_CUES = (
        "what would you do", "what factors would you consider", "how would you determine",
        "diagnose", "prioritize", "prioritization", "decomposition", "how to approach",
        "how to solve", "how would you approach", "how would you investigate",
        "how would you solve", "root cause", "investigation strategy",
    )

    @classmethod
    def classify(cls, text: str, prev_type: Optional[InterviewType] = None,
                 prev_question: str = "") -> InterviewClassification:
        t0 = time.perf_counter()
        t = text.strip().lower()
        words = t.split()

        def finish(itype: InterviewType, confidence: float, reason: str,
                   intent: Optional[str] = None, should_trigger: bool = True) -> InterviewClassification:
            return InterviewClassification(
                interview_type=itype,
                intent=intent if intent is not None else itype.value,
                confidence=confidence,
                reason=reason,
                should_trigger=should_trigger,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        if not words:
            return finish(InterviewType.OTHER, 0.9, "empty text", intent="OTHER", should_trigger=False)

        # 0. Check small talk / greetings first (punctuation-tolerant match)
        t_norm = re.sub(r"[^a-z0-9\s]", " ", t)
        for st in IntentClassifier.SMALL_TALK_INDICATORS:
            if re.search(rf"\b{re.escape(st)}\b", t_norm):
                if len(words) <= 6 and not any(k in t_norm for k in ("why", "how", "what", "design", "time", "drop", "solve")):
                    return finish(InterviewType.OTHER, 0.9, "greeting/small talk", intent="OTHER", should_trigger=False)

        # 1. QUANTITATIVE: deterministic arithmetic first (exact + zero LLM latency)
        det = quant_answer(text)
        if det is not None:
            return finish(InterviewType.QUANTITATIVE, 0.97, "deterministic arithmetic pattern")
        if _QUANT_SHAPE.search(t):
            return finish(InterviewType.QUANTITATIVE, 0.8, "quantitative reasoning shape")

        # 2. PERSONAL / background (evaluated early so "Why Google?" / "Why did you move..." route to PERSONAL)
        if _PERSONAL_RE.search(t):
            return finish(InterviewType.PERSONAL, 0.85, "personal/background question")

        # 3. PROJECT deep-dive (explicit project mentions)
        if _PROJECT_RE.search(t):
            return finish(InterviewType.PROJECT, 0.88, "project-specific deep-dive question")

        # 4. Direct technical / definitional knowledge (never GCA-scaffolded)
        if _TECH_DEFN.match(t):
            return finish(InterviewType.TECHNICAL, 0.9, "direct conceptual/definitional question")
        if _TECH_TROUBLE.search(t) and any(tool in t for tool in _TECH_TOOLS):
            return finish(InterviewType.TECHNICAL, 0.85, "technical troubleshooting question")

        # 5. BEHAVIORAL (STAR stories)
        if IntentClassifier._hit(IntentClassifier.BEHAVIORAL_INDICATORS, t, stem=True):
            return finish(InterviewType.BEHAVIORAL, 0.9, "behavioral/STAR experience question")

        # 6. FOLLOW_UP: contextual continuation or follow-up question shape
        if prev_type == InterviewType.PROJECT and (_TECH_WHY_USE.search(t) or any(tool in t for tool in _TECH_TOOLS)):
            return finish(InterviewType.FOLLOW_UP, 0.9, "follow-up inheriting parent PROJECT", intent="PROJECT")

        # Architecture and tech tool justifications without parent context route to PROJECT / TECHNICAL
        if prev_type is None:
            if "architecture" in t and any(w in t for w in ("why", "choose", "pick", "design")):
                return finish(InterviewType.PROJECT, 0.75, "project/architecture choice question")
            if _TECH_WHY_USE.search(t) and any(tool in t for tool in _TECH_TOOLS):
                return finish(InterviewType.TECHNICAL, 0.8, "tool/technology justification question")

        is_fu_word = any(fu in t for fu in IntentClassifier.FOLLOW_UP_INDICATORS)
        is_short_cont = (
            t in ("why?", "why", "how so?", "how?", "what?", "what happened?", "and then?")
            or (len(words) <= 8 and t.startswith(("what if ", "what about", "and ", "but ", "so ", "then ")))
            or (len(words) <= _FOLLOW_CTX_MAX_WORDS and _FOLLOW_CTX.match(t)
                and not any(cue in t for cue in cls._GCA_CUES))
        )
        if is_fu_word or is_short_cont:
            if prev_type in cls._PARENT_TYPES:
                return finish(InterviewType.FOLLOW_UP, 0.9,
                              f"follow-up inheriting parent {prev_type.value}", intent=prev_type.value)
            return finish(InterviewType.FOLLOW_UP, 0.75, "follow-up without parent context", intent="GCA")

        # 7. Project/Architecture choice without prior context
        if "architecture" in t and any(w in t for w in ("why", "choose", "pick", "design")):
            return finish(InterviewType.PROJECT, 0.75, "project/architecture choice question")

        # 8. TECHNICAL tool justification and domain questions
        if _TECH_WHY_USE.search(t) and any(tool in t for tool in _TECH_TOOLS):
            return finish(InterviewType.TECHNICAL, 0.8, "tool/technology justification question")
        if IntentClassifier._hit(IntentClassifier.TECHNICAL_INDICATORS, t):
            return finish(InterviewType.TECHNICAL, 0.8, "technical domain question")

        # 9. GCA: open-ended analytical problem solving
        if IntentClassifier._hit(IntentClassifier.GCA_INDICATORS, t, stem=True) or any(k in t for k in cls._GCA_CUES):
            return finish(InterviewType.GCA, 0.85, "open-ended analytical problem solving")

        # 10. Question vs Non-question check
        is_q = _looks_like_question(text) or t.endswith("?")

        # 11. Pure continuation without parent context (cold start)
        if is_q and t in ("why?", "why", "how so?", "what happened?", "and then?"):
            return finish(InterviewType.FOLLOW_UP, 0.7, "follow-up without parent context", intent="GCA")

        # 12. Safe fallback: generic direct answer (NEVER default to GCA)
        if is_q:
            return finish(InterviewType.OTHER, 0.5, "unclassified question -> generic direct answer", intent="OTHER")
        return finish(InterviewType.OTHER, 0.6, "non-question speech", intent="OTHER", should_trigger=False)


# ---------------------------------------------------------------------------
# 4. Controlled Candidate Context & Entity Grounding Engine
# ---------------------------------------------------------------------------
@dataclass
class AnswerTarget:
    name: str
    category: str
    forbidden_terms: List[str]
    expected_terms: List[str]
    safe_fallback: str


KNOWN_TARGETS: Dict[str, AnswerTarget] = {
    "Power BI": AnswerTarget(
        name="Power BI",
        category="BI_TOOL",
        forbidden_terms=[
            "power va", "volt-ampere", "volt ampere", "apparent power",
            "power factor", "ac circuit", "rms voltage", "electrical power", "electrical circuit"
        ],
        expected_terms=[
            "business intelligence", "data visualization", "microsoft",
            "report", "dashboard", "analytics", "power bi"
        ],
        safe_fallback=(
            "Power BI is a Microsoft business intelligence platform used to connect to diverse "
            "data sources, model relationships, and build interactive dashboards and reports."
        )
    ),
    "Power VA": AnswerTarget(
        name="Power VA",
        category="ELECTRICAL_UNIT",
        forbidden_terms=[
            "business intelligence", "data visualization", "power bi", "microsoft report", "dax", "dashboard"
        ],
        expected_terms=[
            "volt-ampere", "apparent power", "ac circuit", "rms", "electrical", "power va"
        ],
        safe_fallback=(
            "Power VA (Volt-Amperes) represents apparent power in an electrical alternating current (AC) "
            "circuit, calculated as the product of RMS voltage and RMS current."
        )
    ),
    "Power Query": AnswerTarget(
        name="Power Query",
        category="ETL_TOOL",
        forbidden_terms=["power va", "volt-ampere", "apparent power"],
        expected_terms=[
            "etl", "extract", "transform", "data preparation", "m language", "cleaning", "power query"
        ],
        safe_fallback=(
            "Power Query is an ETL and data transformation engine in Power BI and Excel used to extract, "
            "clean, reshape, and load data from heterogeneous sources."
        )
    ),
    "DAX": AnswerTarget(
        name="DAX",
        category="EXPRESSION_LANGUAGE",
        forbidden_terms=["power va", "volt-ampere", "apparent power"],
        expected_terms=[
            "data analysis expressions", "formula", "measure", "calculated column", "power bi", "tabular", "dax"
        ],
        safe_fallback=(
            "DAX (Data Analysis Expressions) is a formula expression language used in Power BI, Analysis Services, "
            "and Power Pivot to define custom measures and calculated columns."
        )
    ),
    "Tableau": AnswerTarget(
        name="Tableau",
        category="BI_TOOL",
        forbidden_terms=["power va", "volt-ampere", "apparent power"],
        expected_terms=[
            "data visualization", "business intelligence", "analytics", "dashboard", "tableau"
        ],
        safe_fallback=(
            "Tableau is a leading enterprise data visualization and visual analytics platform known for "
            "drag-and-drop report building and interactive dashboards."
        )
    ),
    "SQL": AnswerTarget(
        name="SQL",
        category="QUERY_LANGUAGE",
        forbidden_terms=["power va", "volt-ampere", "apparent power"],
        expected_terms=["relational", "database", "query", "select", "tables", "sql"],
        safe_fallback=(
            "SQL (Structured Query Language) is the domain-specific language used for managing, querying, "
            "and manipulating data in relational database systems."
        )
    )
}


def normalize_query_entities(q: str) -> str:
    """Normalize phonetics/speech misrecognitions while strictly preserving distinct entities."""
    t = re.sub(r"\bpower\s*(?:bee|b\s*i)\b", "Power BI", q, flags=re.I)
    t = re.sub(r"\bpowerbi\b", "Power BI", t, flags=re.I)
    t = re.sub(r"\bpower\s*(?:v\s*a|va)\b", "Power VA", t, flags=re.I)
    t = re.sub(r"\bpowerva\b", "Power VA", t, flags=re.I)
    t = re.sub(r"\bpower\s*query\b", "Power Query", t, flags=re.I)
    t = re.sub(r"\bd\.a\.x\.|\bd\s*a\s*x\b", "DAX", t, flags=re.I)
    t = re.sub(r"\b(?:tablo|tab\s*low)\b", "Tableau", t, flags=re.I)
    return t


def extract_target_entity(query: str) -> Optional[AnswerTarget]:
    """Extract authoritative technical target entity with alias normalization."""
    norm = normalize_query_entities(query)
    # Check Power VA before Power BI when 'power va' is in query
    if re.search(r"\bPower VA\b", norm, flags=re.I):
        return KNOWN_TARGETS["Power VA"]
    if re.search(r"\bPower Query\b", norm, flags=re.I):
        return KNOWN_TARGETS["Power Query"]
    if re.search(r"\bPower BI\b", norm, flags=re.I):
        return KNOWN_TARGETS["Power BI"]
    if re.search(r"\bDAX\b", norm, flags=re.I):
        return KNOWN_TARGETS["DAX"]
    if re.search(r"\bTableau\b", norm, flags=re.I):
        return KNOWN_TARGETS["Tableau"]
    if re.search(r"\bSQL\b", norm, flags=re.I):
        return KNOWN_TARGETS["SQL"]
    return None


def validate_technical_answer(target: Optional[AnswerTarget], answer_text: str) -> Tuple[bool, str]:
    """Validate that technical answer satisfies entity grounding and contains zero forbidden concepts."""
    if not target or not answer_text:
        return True, "OK"
    ans_low = answer_text.lower()
    for forb in target.forbidden_terms:
        if forb in ans_low:
            return False, f"FORBIDDEN_ENTITY: Answer contained forbidden term '{forb}' for {target.name}"
    found_expected = any(term in ans_low for term in target.expected_terms) or target.name.lower() in ans_low
    if not found_expected:
        return False, f"MISSING_EXPECTED: Answer lacks expected domain terminology for {target.name}"
    return True, "VALID"


class GoogleInterviewContext:
    """Manages authoritative candidate context from context.md for Google Data Analytics Apprenticeship.
    Enforces controlled section retrieval and zero candidate context injection for general technical questions."""

    def __init__(self, context_path: Optional[Path] = None):
        if context_path is None:
            context_path = WORKSPACE_DIR / "context.md"
            if not context_path.exists():
                context_path = Path(__file__).parent / "context.md"
        self.context_path = context_path
        self.sections: Dict[str, str] = {}
        self.load_context_file()

    def load_context_file(self, path: Optional[Path] = None) -> None:
        p = path or self.context_path
        if not p or not p.exists():
            return
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
            self.sections.clear()
            current_title = "Overview"
            current_body: List[str] = []
            for line in content.splitlines():
                if line.startswith("### "):
                    if current_body:
                        self.sections[current_title] = "\n".join(current_body).strip()
                    current_title = line[4:].strip()
                    current_body = []
                else:
                    current_body.append(line)
            if current_body:
                self.sections[current_title] = "\n".join(current_body).strip()
            logger.info(f"Loaded {len(self.sections)} controlled context sections from {p.name}")
        except Exception as e:
            logger.warning(f"Failed loading context file {p}: {e}")

    def get_relevant_context(self, intent: str, query: str, k: int = 2) -> List[str]:
        q = query.lower()
        # General TECHNICAL, QUANTITATIVE, and OTHER: NEVER inject candidate context
        if intent in ("TECHNICAL", "QUANTITATIVE", "OTHER"):
            return []

        # PROJECT questions: Retrieve Credit Risk / Delinquency Dashboard section
        if intent == "PROJECT":
            matched = []
            if "Credit Risk / Delinquency Dashboard" in self.sections:
                matched.append(self.sections["Credit Risk / Delinquency Dashboard"])
            if any(w in q for w in ("challenge", "limitation", "tradeoff", "next", "follow", "improve")) and "Follow-ups" in self.sections:
                matched.append(self.sections["Follow-ups"])
            return matched if matched else [self.sections.get("Credit Risk / Delinquency Dashboard", "")]

        # PERSONAL questions: Retrieve matched transition/motivation/goals or default opening
        if intent == "PERSONAL":
            matched = []
            if any(w in q for w in ("civil", "transition", "switch", "background", "degree", "move")):
                if "Civil -> Data Transition" in self.sections:
                    matched.append(self.sections["Civil -> Data Transition"])
            if any(w in q for w in ("why data", "why analytics", "interest", "passion")):
                if "Why Data Analytics" in self.sections:
                    matched.append(self.sections["Why Data Analytics"])
            if any(w in q for w in ("why google", "why apprenticeship", "role", "program", "company", "apply")):
                if "Why Apprenticeship" in self.sections:
                    matched.append(self.sections["Why Apprenticeship"])
                if "Candidate Value" in self.sections:
                    matched.append(self.sections["Candidate Value"])
            if any(w in q for w in ("future", "goal", "see yourself", "5 year", "five year")):
                if "Future Goals" in self.sections:
                    matched.append(self.sections["Future Goals"])
            if not matched:
                if "Opening / Tell Me About Yourself" in self.sections:
                    matched.append(self.sections["Opening / Tell Me About Yourself"])
                if "Candidate Profile" in self.sections:
                    matched.append(self.sections["Candidate Profile"])
            return matched

        # BEHAVIORAL questions: Retrieve matching STAR experience
        if intent == "BEHAVIORAL":
            matched = []
            if any(w in q for w in ("team", "teamwork", "conflict", "disagree", "dmx", "collaborat", "partner", "database", "performance", "high load", "license", "sql", "optimiz")):
                if "Teamwork" in self.sections:
                    matched.append(self.sections["Teamwork"])
            elif any(w in q for w in ("mistake", "failure", "failed", "wrong", "error", "flaw")):
                if "Mistake / Failure" in self.sections:
                    matched.append(self.sections["Mistake / Failure"])
            elif any(w in q for w in ("communicat", "stakeholder", "non-technical", "explain")):
                if "Communication" in self.sections:
                    matched.append(self.sections["Communication"])
            elif any(w in q for w in ("priorit", "deadline", "pressure", "multiple", "urgent")):
                if "Prioritization" in self.sections:
                    matched.append(self.sections["Prioritization"])

            if not matched:
                if "Behavioral" in self.sections:
                    matched.append(self.sections["Behavioral"])
                elif "Teamwork" in self.sections:
                    matched.append(self.sections["Teamwork"])
            return matched

        return []


class DocumentChunker(GoogleInterviewContext):
    """Subclass of GoogleInterviewContext providing backwards-compatibility.
    Guarantees controlled context.md retrieval and forbids CareerOS/Alphora ingestion."""

    def __init__(self, docs_path: Optional[Path] = None):
        super().__init__()
        self.docs_path = docs_path or DOCS_DIR
        self.gca_chunks: List[str] = []
        self._load_gca_frameworks()
        self.chunks: List[str] = list(self.sections.values()) + self.gca_chunks
        self.chunk_metadata: List[Dict[str, Any]] = [{"source": "context.md"} for _ in self.chunks]

    def _load_gca_frameworks(self) -> None:
        gca_file = self.docs_path / "gca_frameworks.md"
        if gca_file.exists():
            try:
                content = gca_file.read_text(encoding="utf-8", errors="ignore")
                self.gca_chunks = [p.strip() for p in content.split("\n\n") if len(p.strip().split()) > 20]
            except Exception:
                self.gca_chunks = []

    def load_and_index(self) -> None:
        self.load_context_file()
        self._load_gca_frameworks()
        self.chunks = list(self.sections.values()) + self.gca_chunks
        self.chunk_metadata = [{"source": "context.md"} for _ in self.chunks]

    def get_top_k(self, query: str, intent: str = "GCA", k: int = 2) -> List[str]:
        if intent == "GCA" and self.gca_chunks:
            return self.gca_chunks[:k]
        return self.get_relevant_context(intent, query, k=k)

# ---------------------------------------------------------------------------
# 5. Resilient Audio Capture & Windows WASAPI Streaming
# ---------------------------------------------------------------------------
def resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Downsample float32 audio chunk to 16kHz via NumPy linear interpolation."""
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    duration = len(audio) / orig_sr
    target_len = int(round(duration * target_sr))
    if target_len == 0:
        return np.empty(0, dtype=np.float32)
    orig_times = np.linspace(0, duration, len(audio), endpoint=False)
    target_times = np.linspace(0, duration, target_len, endpoint=False)
    return np.interp(target_times, orig_times, audio).astype(np.float32)

class AudioChannelListener:
    """Manages sounddevice stream, ring buffer, RMS VAD, and fast silence tracking."""

    def __init__(self, name: str, device_idx: int, is_loopback: bool, target_queue: queue.Queue):
        self.name = name
        self.device_idx = device_idx
        self.is_loopback = is_loopback
        self.target_queue = target_queue
        self.running = False
        self.paused = False

        self.silence_threshold = SILENCE_RMS_THRESHOLD
        self.silence_duration_trigger = END_OF_TURN_SILENCE
        self.partial_interval = PARTIAL_TRANSCRIBE_INTERVAL
        self.min_speech_duration_samples = int(16000 * MIN_SPEECH_DURATION)

        self.speech_buffer: List[np.ndarray] = []
        self.is_in_speech = False
        self.silence_start_time: Optional[float] = None
        self.last_partial_time: float = 0.0
        self.last_spec_partial_time: float = 0.0

        self.stream = None
        self.native_sr = 16000

    def start(self) -> None:
        import sounddevice as sd

        dev_info = sd.query_devices(self.device_idx)

        if self.is_loopback:
            extra = sd.WasapiSettings(exclusive=False, auto_convert=True)
            max_out = int(dev_info.get("max_output_channels", 2) or 2)
            def_sr = int(dev_info.get("default_samplerate", 48000) or 48000)
            attempts = [
                (self.device_idx, max_out, def_sr),
                (self.device_idx, 2, 48000),
                (self.device_idx, 1, def_sr),
            ]
            hp_name = dev_info.get("name", "").lower().split("(")[0].strip()
            for idx, d in enumerate(sd.query_devices()):
                d_name = d.get("name", "").lower()
                if "loopback" in d_name and (hp_name in d_name or not hp_name):
                    attempts.append((idx, 2, int(d.get("default_samplerate", 48000) or 48000)))
                    break
        else:
            extra = None
            max_in = int(dev_info.get("max_input_channels", 1) or 1)
            def_sr = int(dev_info.get("default_samplerate", 16000) or 16000)
            attempts = [
                (self.device_idx, 1, 16000),
                (self.device_idx, max_in, def_sr),
                (self.device_idx, 1, def_sr),
            ]

        stream_opened = False
        last_err = None

        for dev_id, ch, sr in attempts:
            try:
                self.native_sr = sr
                self.stream = sd.InputStream(
                    device=dev_id,
                    channels=ch,
                    samplerate=sr,
                    extra_settings=extra,
                    callback=self._audio_callback,
                    blocksize=int(sr * 0.03),  # 30ms blocks for low audio latency
                )
                self.stream.start()
                stream_opened = True
                d_name = sd.query_devices(dev_id).get("name", "Unknown")
                logger.info(f"Stream '{self.name}' started on [{dev_id}] '{d_name}' (channels={ch}, rate={sr}Hz)")
                break
            except Exception as err:
                last_err = err
                logger.warning(f"Audio attempt failed on dev {dev_id} (ch={ch}, sr={sr}): {err}. Falling back...")

        if not stream_opened:
            raise RuntimeError(f"Could not open audio stream for '{self.name}' on device [{self.device_idx}]: {last_err}")

        self.running = True

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if not self.running or self.paused:
            return

        mono = indata.mean(axis=1) if indata.ndim > 1 else indata.flatten()
        chunk_16k = resample_linear(mono, self.native_sr, 16000)
        if len(chunk_16k) == 0:
            return

        rms = float(np.sqrt(np.mean(chunk_16k**2)))
        now = time.time()

        if rms >= self.silence_threshold:
            self.speech_buffer.append(chunk_16k)
            self.is_in_speech = True
            self.silence_start_time = None

            # Periodic partial STT trigger while speaking
            if (now - self.last_partial_time) >= self.partial_interval:
                self.last_partial_time = now
                current_len = sum(len(c) for c in self.speech_buffer)
                if current_len >= self.min_speech_duration_samples:
                    # Send rolling partial audio without draining buffer
                    partial_pcm = np.concatenate(self.speech_buffer[-12:])  # latest ~400ms
                    try:
                        self.target_queue.put_nowait((self.name, partial_pcm, False, now))
                    except queue.Full:
                        pass
            # Full-buffer speculative partial at slower cadence (speculative intent+retrieval prep)
            if (now - self.last_spec_partial_time) >= SPEC_PARTIAL_INTERVAL:
                self.last_spec_partial_time = now
                if sum(len(c) for c in self.speech_buffer) >= self.min_speech_duration_samples:
                    try:
                        self.target_queue.put_nowait((self.name, np.concatenate(self.speech_buffer), False, now, True))
                    except queue.Full:
                        pass
        else:
            if self.is_in_speech:
                self.speech_buffer.append(chunk_16k)
                if self.silence_start_time is None:
                    self.silence_start_time = now
                elif (now - self.silence_start_time) >= self.silence_duration_trigger:
                    full_audio = np.concatenate(self.speech_buffer)
                    self.speech_buffer.clear()
                    self.is_in_speech = False
                    self.silence_start_time = None

                    if len(full_audio) >= self.min_speech_duration_samples:
                        # Final speech turn detected
                        logger.info(f"[QUESTION_DETECTION] ch={self.name} audio={len(full_audio)/16000:.2f}s -> FINAL turn emitted")
                        try:
                            self.target_queue.put_nowait((self.name, full_audio, True, now))
                        except queue.Full:
                            # Drop oldest if queue full to maintain low perceived latency
                            try:
                                self.target_queue.get_nowait()
                            except queue.Empty:
                                pass
                            self.target_queue.put((self.name, full_audio, True, now))
                    else:
                        logger.info(f"[QUESTION_DETECTION] ch={self.name} audio={len(full_audio)/16000:.2f}s -> DROPPED short speech (MIN_SPEECH_DURATION={MIN_SPEECH_DURATION}s)")

    def stop(self) -> None:
        self.running = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logger.debug(f"Stream close: {e}")


class SoundcardLoopbackListener:
    """WASAPI loopback via soundcard library with low-latency streaming."""

    def __init__(self, name: str, target_queue: queue.Queue, sample_rate: int = 48000):
        self.name = name
        self.target_queue = target_queue
        self.sample_rate = sample_rate
        self.running = False
        self.paused = False
        self._thread: Optional[threading.Thread] = None

        self.silence_threshold = SILENCE_RMS_THRESHOLD
        self.silence_duration_trigger = END_OF_TURN_SILENCE
        self.partial_interval = PARTIAL_TRANSCRIBE_INTERVAL
        self.min_speech_duration_samples = int(16000 * MIN_SPEECH_DURATION)

        self.speech_buffer: List[np.ndarray] = []
        self.is_in_speech = False
        self.silence_start_time: Optional[float] = None
        self.last_partial_time: float = 0.0
        self.last_spec_partial_time: float = 0.0

    def start(self) -> None:
        import soundcard as sc
        speaker = sc.default_speaker()
        mic = sc.get_microphone(id=str(speaker.id), include_loopback=True)
        logger.info(f"SoundcardLoopback: capturing from '{speaker.name}' (loopback)")
        self.running = True
        self._mic = mic
        self._thread = threading.Thread(target=self._capture_loop, name="SoundcardLoopback", daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        block_samples = int(self.sample_rate * 0.03)  # 30ms blocks
        try:
            with self._mic.recorder(samplerate=self.sample_rate, channels=1) as rec:
                while self.running:
                    if self.paused:
                        time.sleep(0.05)
                        continue
                    data = rec.record(numframes=block_samples)
                    mono = data.flatten().astype(np.float32)
                    chunk_16k = resample_linear(mono, self.sample_rate, 16000)
                    if len(chunk_16k) == 0:
                        continue

                    rms = float(np.sqrt(np.mean(chunk_16k ** 2)))
                    now = time.time()

                    if rms >= self.silence_threshold:
                        self.speech_buffer.append(chunk_16k)
                        self.is_in_speech = True
                        self.silence_start_time = None

                        if (now - self.last_partial_time) >= self.partial_interval:
                            self.last_partial_time = now
                            current_len = sum(len(c) for c in self.speech_buffer)
                            if current_len >= self.min_speech_duration_samples:
                                partial_pcm = np.concatenate(self.speech_buffer[-12:])
                                try:
                                    self.target_queue.put_nowait((self.name, partial_pcm, False, now))
                                except queue.Full:
                                    pass
                        # Full-buffer speculative partial at slower cadence (speculative intent+retrieval prep)
                        if (now - self.last_spec_partial_time) >= SPEC_PARTIAL_INTERVAL:
                            self.last_spec_partial_time = now
                            if sum(len(c) for c in self.speech_buffer) >= self.min_speech_duration_samples:
                                try:
                                    self.target_queue.put_nowait((self.name, np.concatenate(self.speech_buffer), False, now, True))
                                except queue.Full:
                                    pass
                    elif self.is_in_speech:
                        self.speech_buffer.append(chunk_16k)
                        if self.silence_start_time is None:
                            self.silence_start_time = now
                        elif (now - self.silence_start_time) >= self.silence_duration_trigger:
                            full_audio = np.concatenate(self.speech_buffer)
                            self.speech_buffer.clear()
                            self.is_in_speech = False
                            self.silence_start_time = None
                            if len(full_audio) >= self.min_speech_duration_samples:
                                logger.info(f"[QUESTION_DETECTION] ch={self.name} audio={len(full_audio)/16000:.2f}s -> FINAL turn emitted")
                                try:
                                    self.target_queue.put_nowait((self.name, full_audio, True, now))
                                except queue.Full:
                                    try:
                                        self.target_queue.get_nowait()
                                    except queue.Empty:
                                        pass
                                    self.target_queue.put((self.name, full_audio, True, now))
                            else:
                                logger.info(f"[QUESTION_DETECTION] ch={self.name} audio={len(full_audio)/16000:.2f}s -> DROPPED short speech (MIN_SPEECH_DURATION={MIN_SPEECH_DURATION}s)")
        except Exception as e:
            logger.error(f"SoundcardLoopback capture error: {e}", exc_info=True)

    def stop(self) -> None:
        self.running = False

# ---------------------------------------------------------------------------
# 6. Streaming Transcription Worker (faster-whisper) with Hardware Selection
# ---------------------------------------------------------------------------
def create_whisper_model(
    model_name: Optional[str] = None,
    device_pref: Optional[str] = None,
    compute_type_pref: Optional[str] = None
) -> Tuple[Any, Dict[str, Any]]:
    """Automatic hardware selection for faster-whisper.
    Attempts CUDA float16 when available/configured; automatically and safely
    falls back to CPU int8 if CUDA fails or is unavailable. Never crashes on missing CUDA."""
    from faster_whisper import WhisperModel

    model_to_load = model_name or WHISPER_MODEL_NAME or "tiny.en"
    pref = (device_pref or WHISPER_DEVICE or "auto").strip().lower()
    if pref not in ("auto", "cuda", "cpu"):
        logger.warning(f"Unknown WHISPER_DEVICE preference '{pref}'. Defaulting to 'auto'.")
        pref = "auto"

    cuda_attempted = False
    cuda_err = None

    if pref in ("auto", "cuda"):
        cuda_attempted = True
        try:
            import ctranslate2
            if hasattr(ctranslate2, "get_cuda_device_count") and ctranslate2.get_cuda_device_count() == 0:
                raise RuntimeError("No compatible NVIDIA/CUDA devices detected by ctranslate2")

            c_type = compute_type_pref or ("float16" if pref == "auto" else WHISPER_COMPUTE_TYPE) or "float16"
            logger.info(f"Attempting faster-whisper ('{model_to_load}') on CUDA ({c_type})...")
            model = WhisperModel(model_to_load, device="cuda", compute_type=c_type)
            # Warm up with dummy audio
            list(model.transcribe(np.zeros(8000, dtype=np.float32), beam_size=1))
            runtime_info = {
                "model": model_to_load,
                "device": "CUDA",
                "compute_type": c_type,
                "fallback_occurred": False,
                "cuda_error": None
            }
            logger.info(f"[WHISPER]\nmodel={model_to_load}\ndevice=CUDA\ncompute_type={c_type}")
            return model, runtime_info
        except Exception as e:
            cuda_err = str(e)
            logger.warning(f"[WHISPER] CUDA initialization failed: {cuda_err}\nFalling back to CPU int8...")

    # CPU mode (fallback or requested)
    c_type = "int8"
    logger.info(f"Initializing faster-whisper ('{model_to_load}') on CPU ({c_type})...")
    model = WhisperModel(model_to_load, device="cpu", compute_type=c_type)
    list(model.transcribe(np.zeros(8000, dtype=np.float32), beam_size=1))
    runtime_info = {
        "model": model_to_load,
        "device": "CPU",
        "compute_type": c_type,
        "fallback_occurred": cuda_attempted,
        "cuda_error": cuda_err
    }
    logger.info(f"[WHISPER]\nmodel={model_to_load}\ndevice=CPU\ncompute_type={c_type}")
    return model, runtime_info


class TranscriptionWorker(threading.Thread):
    """Processes partial audio cues and final speech turns via faster-whisper."""

    def __init__(self, audio_queue: queue.Queue, turn_queue: queue.Queue, ui_queue: queue.Queue, spec_queue: Optional[queue.Queue] = None):
        super().__init__(name="TranscriptionWorker", daemon=True)
        self.audio_queue = audio_queue
        self.turn_queue = turn_queue
        self.ui_queue = ui_queue
        self.spec_queue = spec_queue
        self.model = None
        self.runtime_info = {}
        self._init_whisper()

    def _init_whisper(self) -> None:
        self.model, self.runtime_info = create_whisper_model()
        logger.info(f"faster-whisper ready on {self.runtime_info['device']} ({self.runtime_info['compute_type']}).")

    def run(self) -> None:
        while True:
            try:
                task = self.audio_queue.get()
                if task is None or task[0] is None:
                    break

                if len(task) == 5:
                    channel_name, audio_pcm, is_final, t_turn_end, is_spec = task
                else:
                    channel_name, audio_pcm, is_final, t_turn_end = task
                    is_spec = False

                if not is_final:
                    # Partial interim transcript: fast preview in UI status
                    # Only do if audio_queue is not heavily backed up
                    if self.audio_queue.qsize() <= 2:
                        try:
                            segments, _ = self.model.transcribe(audio_pcm, language="en", vad_filter=False, beam_size=1)
                            partial_txt = " ".join(s.text.strip() for s in segments).strip()
                            if partial_txt:
                                self.ui_queue.put(("partial_stt", f"{channel_name}: {partial_txt}..."))
                                if is_spec and self.spec_queue is not None:
                                    self.spec_queue.put((channel_name, partial_txt, time.time()))
                        except Exception:
                            pass
                    continue

                # Final turn transcription
                t0 = time.time()
                segments, _ = self.model.transcribe(
                    audio_pcm,
                    language="en",
                    vad_filter=False,
                    beam_size=1
                )
                text = " ".join(s.text.strip() for s in segments).strip()
                stt_latency_ms = (time.time() - t0) * 1000

                audio_secs = len(audio_pcm) / 16000
                if text and len(text) >= 2:
                    logger.info(f"[TRANSCRIPT] ch={channel_name} audio={audio_secs:.2f}s stt={stt_latency_ms:.1f}ms text='{text}'")
                    self.turn_queue.put((channel_name, text, t_turn_end, time.time()))
                else:
                    logger.info(f"[TRANSCRIPT] ch={channel_name} audio={audio_secs:.2f}s stt={stt_latency_ms:.1f}ms -> DROPPED empty/short transcript text='{text}'")
            except Exception as e:
                logger.error(f"Transcription error: {e}", exc_info=True)
            finally:
                self.audio_queue.task_done()

# ---------------------------------------------------------------------------
# 7. Specialized GCA & Googliness Prompt Formats
# ---------------------------------------------------------------------------
GCA_SYSTEM_PROMPT = """You are an executive GCA interview reasoning coach.
Internally reason through: clarifying objective, decomposing, hypotheses, validation, and trade-offs. Express as natural spoken prose -- NEVER print labels like CLARIFY, FRAME, HYPOTHESES, OBJECTIVE, DECOMPOSITION.

MANDATORY OUTPUT FORMAT:
You MUST start your response immediately with ANSWER: and end with FOLLOW-UP CUE:.
Keep the answer concise (under 90 words).

ANSWER:
[Concise spoken answer, under 90 words: metric to clarify, decomposition, top hypothesis, validation, trade-off. No bullets.]

FOLLOW-UP CUE:
[One short clarification question]
"""

BEHAVIORAL_SYSTEM_PROMPT = """You are an executive behavioral & Googliness interview answer generator.
Match the question to an authentic story from [AUTHENTIC CANDIDATE STAR STORIES].
STRICT RULES:
1. NEVER fabricate companies, metrics, roles, projects, technologies, or achievements.
2. CONTEXT AUDIT:
- If a story in [AUTHENTIC CANDIDATE STAR STORIES] addresses the core challenge or scenario (e.g. database performance, Redis caching, license management, stepping up, schema migration), USE IT to construct your answer.
- If NO relevant story exists in the context (e.g. enterprise sales, labor unions, mobile games, hardware procurement, chip design), output EXACTLY:
NO VERIFIED STORY FOUND. Use a real example from your experience.
3. If a relevant story is found, output ONLY:

ANSWER:
[Natural first-person spoken answer, 80-130 words: situation and task, 2-3 concrete actions YOU took, quantified result and lesson learned. No S:/T:/A:/R: labels.]
"""

TECHNICAL_SYSTEM_PROMPT = """You are a Staff Systems Architect interview coach.
Format strictly as 4 scannable cues for natural spoken delivery:
1. REQUIREMENTS & SCALE: [p99 latency, throughput, consistency requirements]
2. ARCHITECTURE: [high-level components and data flow]
3. KEY BOTTLENECKS: [failover, cache invalidation, or partitioning]
4. TRADE-OFF: [latency vs consistency or compute vs cost]
"""

TECHNICAL_ANSWER_PROMPT = """You are a senior engineer answering a direct technical interview question.
Answer the question that was actually asked — do not restructure it into a design framework.
Output ONLY:

ANSWER:
[Concise natural spoken answer, 2-6 sentences. For a comparison question, briefly characterize BOTH
options and the key distinction; do not declare a winner unless the question explicitly asks for a
recommendation. For a design question, walk through the design in 4-6 sentences. Ground every claim;
if something is version- or context-dependent, say so briefly.]
"""

def build_technical_prompt(question: str, target: Optional[AnswerTarget] = None) -> str:
    """Constructs technical prompt with strict entity grounding and negative constraints."""
    base = TECHNICAL_ANSWER_PROMPT
    if not target:
        return base
    forb_list = ", ".join(f"'{f}'" for f in target.forbidden_terms[:4])
    exp_list = ", ".join(target.expected_terms[:4])
    return (
        f"{base}\n\n"
        f"[STRICT ENTITY GROUNDING]\n"
        f"TARGET ENTITY: {target.name} (Category: {target.category})\n"
        f"MANDATORY INSTRUCTIONS:\n"
        f"1. You MUST answer specifically regarding {target.name}. Never substitute another technology or electrical unit.\n"
        f"2. FORBIDDEN CONCEPTS: Never confuse with or mention {forb_list}.\n"
        f"3. Focus strictly on {exp_list}."
    )

PERSONAL_SYSTEM_PROMPT = """You generate first-person interview answers about the candidate, grounded
ONLY in [CANDIDATE PROFILE & RESUME]. NEVER invent employers, projects, metrics, education, or timelines.
If the provided context lacks the requested detail, answer with what IS verified and briefly note the
candidate would elaborate from direct experience — do not fabricate specifics.
Output ONLY:

ANSWER:
[Natural first-person spoken answer, 60-120 words]
"""

PROJECT_SYSTEM_PROMPT = """You answer project deep-dive questions about the candidate's work, grounded
ONLY in [PROJECT & CANDIDATE CONTEXT]. NEVER invent features, components, metrics, or decisions that are
not in the context. If a detail is missing, say what you would verify or how you'd find out rather than
inventing it.
Output ONLY:

ANSWER:
[Natural first-person spoken answer, 60-130 words]
"""

QUANTITATIVE_SYSTEM_PROMPT = """You are a precise quantitative reasoning coach. Work the calculation
step by step in natural language and double-check the arithmetic before answering.
Output ONLY:

ANSWER:
[The computed answer with a one-line explanation of the calculation]
"""

GENERIC_ANSWER_PROMPT = """You are a concise interview assistant. Answer the question directly, honestly,
and briefly in natural spoken language. If the question is ambiguous, state your assumption in one
clause, then answer.
Output ONLY:

ANSWER:
[Concise direct answer, 1-5 sentences]
"""

# ---------------------------------------------------------------------------
# 7b. Speculative Partial-STT Preparation (V2.1)
# ---------------------------------------------------------------------------
def _content_tokens(text: str) -> set:
    return {w.strip(".,?!\"'()").lower() for w in re.split(r"\s+", text) if len(w) > 2}


def spec_matches(final_text: str, spec_text: str) -> bool:
    """Token Jaccard similarity between final transcript and speculative partial."""
    a, b = _content_tokens(final_text), _content_tokens(spec_text)
    if not a or not b:
        return False
    return len(a & b) / max(1, len(a | b)) >= SPEC_SIMILARITY_THRESHOLD


class SpeculativeWorker(threading.Thread):
    """Consumes full-buffer partial transcripts while the interviewer is still speaking
    and pre-computes cheap local work: intent classification + BM25 retrieval.
    Final transcript stays authoritative; speculative state is reused only when
    intent matches and transcripts are similar. No LLM calls happen here."""

    def __init__(self, spec_queue: queue.Queue, chunker: DocumentChunker,
                 context_provider: Optional[Any] = None):
        super().__init__(name="SpeculativeWorker", daemon=True)
        self.spec_queue = spec_queue
        self.chunker = chunker
        self.context_provider = context_provider  # callable -> (prev_type, prev_question)
        self.lock = threading.Lock()
        self.state: Optional[Dict[str, Any]] = None

    def run(self) -> None:
        while True:
            try:
                channel_name, partial_txt, t_partial = self.spec_queue.get()
            except Exception:
                continue
            try:
                words = [w for w in re.split(r"\s+", partial_txt.strip()) if w]
                if len(words) < SPEC_MIN_WORDS:
                    continue
                t0 = time.time()
                prev_type, prev_question = (self.context_provider() or (None, "")) if self.context_provider else (None, "")
                cls = ISRE.classify(partial_txt, prev_type=prev_type, prev_question=prev_question)
                intent, should_trigger = cls.intent, cls.should_trigger
                if not should_trigger or cls.interview_type == InterviewType.FOLLOW_UP:
                    continue
                logger.info(f"[SEPARATION] type={cls.interview_type.value} confidence={cls.confidence:.2f} "
                            f"latency_ms={cls.latency_ms:.2f} reason='{cls.reason}' (speculative)")
                top_chunks = self.chunker.get_top_k(partial_txt, intent=intent, k=RETRIEVAL_TOP_K)
                prep_ms = (time.time() - t0) * 1000
                with self.lock:
                    self.state = {
                        "text": partial_txt,
                        "intent": intent,
                        "chunks": top_chunks,
                        "t_partial": t_partial,
                        "prep_ms": prep_ms,
                    }
                logger.info(f"[Spec] Prepped [{intent}] from partial ({len(words)} words, {prep_ms:.1f}ms): '{partial_txt[:60]}...'")
            except Exception as e:
                logger.warning(f"Speculative prep failed: {e}")
            finally:
                try:
                    self.spec_queue.task_done()
                except Exception:
                    pass

    def get_state(self) -> Optional[Dict[str, Any]]:
        with self.lock:
            return self.state

    def clear(self) -> None:
        with self.lock:
            self.state = None

# ---------------------------------------------------------------------------
# 8. Dual-Provider Streaming LLM Engine (Groq + NVIDIA NIM Fallback)
# ---------------------------------------------------------------------------
class LLMStreamingWorker(threading.Thread):
    """Low-latency streaming LLM worker with instant preemption and telemetry."""

    def __init__(self, prompt_queue: queue.Queue, ui_queue: queue.Queue, chunker: DocumentChunker, spec_worker: Optional["SpeculativeWorker"] = None):
        super().__init__(name="LLMStreamingWorker", daemon=True)
        self.prompt_queue = prompt_queue
        self.ui_queue = ui_queue
        self.chunker = chunker
        self.spec_worker = spec_worker
        self.active_generation_id = 0
        self.lock = threading.Lock()
        # Persistent OpenAI-compatible clients: keeps TLS/HTTP connection alive
        self.client_groq = None
        self.client_nvidia = None
        try:
            from openai import OpenAI
            groq_key = os.getenv("GROQ_API_KEY", "").strip()
            nvidia_key = os.getenv("NVIDIA_API_KEY", "").strip()
            if groq_key:
                self.client_groq = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key, max_retries=0)
            if nvidia_key:
                self.client_nvidia = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=nvidia_key, max_retries=1)
        except Exception as e:
            logger.warning(f"LLM client pre-init failed ({e}). Clients will be created lazily per request.")

    def cancel_current(self) -> None:
        """Immediately preempts the currently running generation."""
        with self.lock:
            self.active_generation_id += 1

    def warmup(self) -> Dict[str, Any]:
        """Optional one-time provider connection warmup: 1-token request, no answer generation.
        Safe: non-fatal on failure, consumes ~1 token per provider, runs once at startup."""
        results: Dict[str, Any] = {}
        if not WARMUP_PROVIDERS:
            results["skipped"] = "WARMUP_PROVIDERS disabled"
            return results
        for name, client, model in (
            ("groq", self.client_groq, GROQ_MODEL),
            ("nvidia", self.client_nvidia, NVIDIA_MODEL),
        ):
            if client is None:
                continue
            try:
                t0 = time.time()
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                    temperature=0.0,
                    stream=False,
                    timeout=WARMUP_TIMEOUT,
                )
                results[f"{name}_warmup_ms"] = round((time.time() - t0) * 1000, 1)
                logger.info(f"[Warmup] {name} connection warmed in {results[f'{name}_warmup_ms']}ms")
            except Exception as e:
                results[f"{name}_warmup_error"] = str(e)[:80]
                logger.warning(f"[Warmup] {name} warmup failed (non-fatal): {e}")
        return results

    def _stream_groq(self, messages: List[Dict[str, str]], api_key: str):
        client = self.client_groq
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key, max_retries=0)
            self.client_groq = client
        return client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            stream=True,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            timeout=8.0
        )

    def _stream_nvidia(self, messages: List[Dict[str, str]], api_key: str):
        client = self.client_nvidia
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key, max_retries=1)
            self.client_nvidia = client
        return client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=messages,
            stream=True,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            timeout=45.0
        )

    def run(self) -> None:
        while True:
            try:
                task = self.prompt_queue.get()
                if task is None:
                    break

                history, current_question, intent, telemetry = task
                with self.lock:
                    self.active_generation_id += 1
                    gen_id = self.active_generation_id

                self.ui_queue.put(("status", f"STREAMING [{intent}]..."))
                self.ui_queue.put(("clear_text", None))
                self.ui_queue.put(("intent_badge", intent))

                # QUANTITATIVE short-circuit: deterministic answer, zero LLM latency
                if intent == "QUANTITATIVE":
                    det = quant_answer(current_question)
                    if det is not None:
                        logger.info(f"[SEPARATION] deterministic quantitative answer (no LLM): {det}")
                        telemetry.mark_llm_request_sent()
                        telemetry.mark_first_token("Deterministic")
                        toks = det.split(" ")
                        for i in range(0, len(toks), 3):
                            self.ui_queue.put(("token", " ".join(toks[i:i + 3]) + " "))
                            time.sleep(0.01)
                        telemetry.mark_generation_done(len(toks))
                        telemetry_summary = telemetry.log_and_format()
                        self.ui_queue.put(("telemetry", telemetry_summary))
                        self.ui_queue.put(("status", "LISTENING"))
                        continue

                # Retrieval phase — reuse speculative prep when final transcript validates it
                top_chunks = None
                spec = self.spec_worker.get_state() if self.spec_worker else None
                if spec and spec.get("intent") == intent and spec_matches(current_question, spec.get("text", "")):
                    top_chunks = spec["chunks"]
                    telemetry.spec_reused = True
                    telemetry.spec_prep_ms = spec.get("prep_ms", 0.0)
                    logger.info(f"[Spec] Reused speculative retrieval (prep {telemetry.spec_prep_ms:.1f}ms) for: '{current_question[:50]}'")
                if top_chunks is None:
                    top_chunks = self.chunker.get_top_k(current_question, intent=intent, k=RETRIEVAL_TOP_K)
                logger.info(f"[RETRIEVAL] intent={intent} chunks={len(top_chunks)} spec_reused={telemetry.spec_reused} query='{current_question[:70]}'")
                if self.spec_worker:
                    self.spec_worker.clear()
                telemetry.mark_retrieval_done()

                context_str = "\n\n---\n\n".join(top_chunks) if top_chunks else "No specific documents provided."

                # Select specialized prompt based on interview type (ISRE routes; GCA is one mode)
                target_entity = None
                if intent == "BEHAVIORAL":
                    sys_prompt = f"{BEHAVIORAL_SYSTEM_PROMPT}\n\n[AUTHENTIC CANDIDATE STAR STORIES]\n{context_str}"
                elif intent == "TECHNICAL":
                    target_entity = extract_target_entity(current_question)
                    tech_prompt = build_technical_prompt(current_question, target_entity)
                    sys_prompt = f"{tech_prompt}\n\n[REFERENCE NOTES]\n{context_str}"
                elif intent == "PERSONAL":
                    sys_prompt = f"{PERSONAL_SYSTEM_PROMPT}\n\n[CANDIDATE PROFILE & RESUME]\n{context_str}"
                elif intent == "PROJECT":
                    sys_prompt = f"{PROJECT_SYSTEM_PROMPT}\n\n[PROJECT & CANDIDATE CONTEXT]\n{context_str}"
                elif intent == "QUANTITATIVE":
                    sys_prompt = f"{QUANTITATIVE_SYSTEM_PROMPT}\n\n[REFERENCE NOTES]\n{context_str}"
                elif intent == "OTHER":
                    sys_prompt = f"{GENERIC_ANSWER_PROMPT}\n\n[CONTEXT]\n{context_str}"
                else:
                    sys_prompt = f"{GCA_SYSTEM_PROMPT}\n\n[GCA FRAMEWORKS & CONTEXT]\n{context_str}"

                messages = [{"role": "system", "content": sys_prompt}]
                sliced_history = history[-HISTORY_TURNS:]
                for speaker, text in sliced_history:
                    role = "user" if speaker == "Interviewer" else "assistant"
                    messages.append({"role": role, "content": text})

                if not sliced_history or sliced_history[-1][1] != current_question:
                    messages.append({"role": "user", "content": current_question})

                groq_key = os.getenv("GROQ_API_KEY", "").strip()
                nvidia_key = os.getenv("NVIDIA_API_KEY", "").strip()

                stream = None
                provider_used = ""

                # 1. Primary provider: Groq
                telemetry.mark_llm_request_sent()
                logger.info(f"[LLM_REQUEST] intent={intent} provider_pool=groq->nvidia question='{current_question[:90]}'")
                if groq_key:
                    try:
                        self.ui_queue.put(("provider", f"Groq ({GROQ_MODEL})"))
                        t_groq0 = time.time()
                        stream = self._stream_groq(messages, groq_key)
                        telemetry.mark_llm_stream_created()
                        telemetry.add_provider_event("groq", "stream_created", f"{(time.time() - t_groq0) * 1000:.0f}ms")
                        provider_used = "Groq"
                    except Exception as err:
                        is_429 = "429" in str(err) or "rate" in str(err).lower()
                        retry_delay = 0.0
                        if is_429:
                            m_delay = re.search(r"try again in ([\d.]+)s", str(err))
                            retry_delay = float(m_delay.group(1)) if m_delay else 0.0
                        telemetry.add_provider_event("groq", "rate_limited" if is_429 else "failed",
                                                     f"retry_after={retry_delay:.2f}s" if is_429 else str(err)[:80])
                        # V2.1: on 429 with short provider-suggested retry delay, retry Groq once
                        # (much cheaper than slow NVIDIA fallback). Fallback stays intact otherwise.
                        if is_429 and 0.0 < retry_delay <= 3.0:
                            time.sleep(retry_delay + 0.1)
                            try:
                                telemetry.add_provider_event("groq", "retry_after_429", f"{retry_delay:.2f}s")
                                t_groq0 = time.time()
                                stream = self._stream_groq(messages, groq_key)
                                telemetry.mark_llm_stream_created()
                                telemetry.add_provider_event("groq", "stream_created", f"{(time.time() - t_groq0) * 1000:.0f}ms")
                                provider_used = "Groq"
                            except Exception as err2:
                                telemetry.add_provider_event("groq", "retry_failed", str(err2)[:80])
                                stream = None
                        if stream is None:
                            logger.warning(f"Groq stream error: {err}. Failing over to NVIDIA NIM...")
                            self.ui_queue.put(("status", "Groq failed. Trying NVIDIA NIM..."))

                # 2. Fallback provider: NVIDIA NIM
                if stream is None and nvidia_key:
                    try:
                        telemetry.add_provider_event("nvidia", "fallback_start")
                        self.ui_queue.put(("provider", f"NVIDIA NIM ({NVIDIA_MODEL})"))
                        t_nv0 = time.time()
                        stream = self._stream_nvidia(messages, nvidia_key)
                        telemetry.mark_llm_stream_created()
                        telemetry.add_provider_event("nvidia", "stream_created", f"{(time.time() - t_nv0) * 1000:.0f}ms")
                        provider_used = "NVIDIA NIM"
                    except Exception as err:
                        telemetry.add_provider_event("nvidia", "failed", str(err)[:80])
                        logger.error(f"NVIDIA failover failed: {err}")
                        self.ui_queue.put(("token", f"\n[Error: Providers failed: {err}]"))
                        self.ui_queue.put(("status", "STOPPED"))
                        continue

                if stream is None:
                    msg = "Set GROQ_API_KEY or NVIDIA_API_KEY in .env"
                    self.ui_queue.put(("token", f"\n[{msg}]"))
                    self.ui_queue.put(("status", "NO API KEYS"))
                    continue

                first_token_received = False
                token_count = 0
                accumulated_tokens = []
                stream_rejected = False

                for chunk in stream:
                    with self.lock:
                        if gen_id != self.active_generation_id or not self.prompt_queue.empty():
                            # Preempted by a newer question
                            self.active_generation_id += 1
                            break

                    delta = chunk.choices[0].delta.content if chunk.choices else ""
                    if delta:
                        accumulated_tokens.append(delta)
                        if target_entity and len(accumulated_tokens) % 3 == 0:
                            so_far = "".join(accumulated_tokens).lower()
                            if any(forb in so_far for forb in target_entity.forbidden_terms):
                                logger.warning(f"[GROUNDING_GUARD] Forbidden entity detected mid-stream for {target_entity.name}")
                                stream_rejected = True
                                self.ui_queue.put(("clear_text", None))
                                break

                        if not first_token_received:
                            first_token_received = True
                            telemetry.mark_first_token(provider_used)
                            logger.info(f"[LLM_RESPONSE] first token via {provider_used} ({chunk.choices[0].delta.content[:40]!r})")
                        token_count += 1
                        self.ui_queue.put(("token", delta))

                # Empty-stream guard: some models emit zero visible content (e.g. reasoning-only
                # output). Fall back to NVIDIA once rather than showing a blank answer.
                if token_count == 0 and not stream_rejected and provider_used.startswith("Groq") and nvidia_key \
                        and gen_id == self.active_generation_id and self.prompt_queue.empty():
                    logger.warning("[LLM_RESPONSE] Groq returned 0 content tokens -> retrying via NVIDIA NIM")
                    telemetry.add_provider_event("nvidia", "empty_retry")
                    try:
                        stream2 = self._stream_nvidia(messages, nvidia_key)
                        telemetry.mark_llm_stream_created()
                        provider_used = "NVIDIA NIM"
                        for chunk in stream2:
                            with self.lock:
                                if gen_id != self.active_generation_id or not self.prompt_queue.empty():
                                    self.active_generation_id += 1
                                    break
                            delta = chunk.choices[0].delta.content if chunk.choices else ""
                            if delta:
                                accumulated_tokens.append(delta)
                                if not first_token_received:
                                    first_token_received = True
                                    telemetry.mark_first_token(provider_used)
                                    logger.info(f"[LLM_RESPONSE] first token via {provider_used} ({delta[:40]!r})")
                                token_count += 1
                                self.ui_queue.put(("token", delta))
                    except Exception as err2:
                        telemetry.add_provider_event("nvidia", "empty_retry_failed", str(err2)[:80])
                        logger.error(f"NVIDIA empty-retry failed: {err2}")

                # Exact Entity Grounding Guard & Max 1 Regeneration for TECHNICAL questions
                if target_entity and gen_id == self.active_generation_id and self.prompt_queue.empty():
                    ans_text = "".join(accumulated_tokens)
                    is_valid = False
                    reason = ""
                    if not stream_rejected and ans_text:
                        is_valid, reason = validate_technical_answer(target_entity, ans_text)

                    if not is_valid or stream_rejected:
                        logger.warning(f"[GROUNDING_GUARD] Technical answer rejected for {target_entity.name}: {reason or 'Forbidden term'}. Clearing UI and retrying once.")
                        self.ui_queue.put(("clear_text", None))
                        retry_messages = list(messages)
                        retry_messages.append({"role": "assistant", "content": ans_text})
                        retry_messages.append({
                            "role": "user",
                            "content": (
                                f"CRITICAL CORRECTION: Previous response was REJECTED for entity mismatch ({reason or 'Forbidden term'}). "
                                f"You MUST answer strictly regarding {target_entity.name} ({target_entity.category}). "
                                f"Do NOT mention {', '.join(target_entity.forbidden_terms[:3])}."
                            )
                        })
                        retry_tokens = []
                        retry_stream = None
                        try:
                            if groq_key:
                                retry_stream = self._stream_groq(retry_messages, groq_key)
                            elif nvidia_key:
                                retry_stream = self._stream_nvidia(retry_messages, nvidia_key)
                        except Exception as e_ret:
                            logger.warning(f"[GROUNDING_GUARD] Retry stream creation failed: {e_ret}")

                        retry_valid = False
                        if retry_stream:
                            for chunk in retry_stream:
                                with self.lock:
                                    if gen_id != self.active_generation_id or not self.prompt_queue.empty():
                                        self.active_generation_id += 1
                                        break
                                delta = chunk.choices[0].delta.content if chunk.choices else ""
                                if delta:
                                    retry_tokens.append(delta)
                            retry_ans = "".join(retry_tokens)
                            retry_valid, _ = validate_technical_answer(target_entity, retry_ans)

                        if retry_valid and retry_tokens:
                            logger.info(f"[GROUNDING_GUARD] Retry succeeded for {target_entity.name}")
                            for delta in retry_tokens:
                                self.ui_queue.put(("token", delta))
                            token_count = len(retry_tokens)
                        else:
                            logger.warning(f"[GROUNDING_GUARD] Retry failed or invalid. Emitting safe fallback for {target_entity.name}")
                            self.ui_queue.put(("clear_text", None))
                            fb_tokens = target_entity.safe_fallback.split(" ")
                            for i in range(0, len(fb_tokens), 3):
                                self.ui_queue.put(("token", " ".join(fb_tokens[i:i+3]) + " "))
                                time.sleep(0.01)
                            token_count = len(fb_tokens)

                with self.lock:
                    if gen_id == self.active_generation_id and self.prompt_queue.empty():
                        telemetry.mark_generation_done(token_count)
                        telemetry_summary = telemetry.log_and_format()
                        logger.info(f"[LLM_RESPONSE] complete via {provider_used} tokens={token_count} intent={intent}")
                        self.ui_queue.put(("telemetry", telemetry_summary))
                        self.ui_queue.put(("status", "LISTENING"))
                    else:
                        logger.info(f"[LLM_RESPONSE] generation {gen_id} superseded by {self.active_generation_id} (tokens={token_count}, suppressed)")

            except Exception as e:
                logger.error(f"LLM Streaming Worker error: {e}", exc_info=True)
                self.ui_queue.put(("status", f"Error: {e}"))
            finally:
                self.prompt_queue.task_done()

# ---------------------------------------------------------------------------
# 9. Context Manager & Dialogue Orchestration
# ---------------------------------------------------------------------------
class DialogueManager(threading.Thread):
    """Tracks turns, classifies intent, handles follow-ups, and routes to LLM worker."""

    def __init__(self, turn_queue: queue.Queue, prompt_queue: queue.Queue, ui_queue: queue.Queue):
        super().__init__(name="DialogueManager", daemon=True)
        self.turn_queue = turn_queue
        self.prompt_queue = prompt_queue
        self.ui_queue = ui_queue
        self.history: List[Tuple[str, str]] = []
        self.listening_paused = False

        # Conversational memory state
        self.last_question = ""
        self.last_intent = "GCA"
        self.last_interview_type: Optional[InterviewType] = None
        self.last_story_retrieved = ""

    def clear(self) -> None:
        self.history.clear()
        self.last_question = ""
        self.last_intent = "GCA"
        self.last_interview_type = None
        self.last_story_retrieved = ""
        logger.info("Dialogue history and state cleared.")

    def run(self) -> None:
        while True:
            try:
                task = self.turn_queue.get()
                if task is None or task[0] is None:
                    break

                if len(task) == 3:
                    speaker, text, telemetry = task
                elif len(task) == 4:
                    speaker, text, t_turn_end, t_stt_done = task
                    telemetry = PerformanceTelemetry(text)
                    telemetry.t_end_of_turn_detected = t_turn_end
                    telemetry.t_stt_completed = t_stt_done
                else:
                    speaker, text = task[0], task[1]
                    telemetry = PerformanceTelemetry(text)
                    telemetry.t_end_of_turn_detected = time.time()
                    telemetry.t_stt_completed = time.time()

                if self.listening_paused:
                    continue

                self.history.append((speaker, text))
                self.ui_queue.put(("transcript_turn", (speaker, text)))

                if speaker == "Candidate" and not _looks_like_question(text):
                    logger.info(f"[QUESTION_DETECTION] ch=Candidate non-question turn ignored: '{text[:60]}'")
                    continue

                if speaker in ("Interviewer", "Candidate"):
                    cls = ISRE.classify(text, prev_type=self.last_interview_type, prev_question=self.last_question)
                    intent, should_trigger = cls.intent, cls.should_trigger
                    telemetry.mark_intent_done(intent)
                    logger.info(f"[SEPARATION] type={cls.interview_type.value} confidence={cls.confidence:.2f} "
                                f"latency_ms={cls.latency_ms:.2f} reason='{cls.reason}'")
                    logger.info(f"[INTENT] '{text}' -> {intent} (trigger={should_trigger})")

                    if not should_trigger:
                        logger.info(f"[QUESTION_DETECTION] DROPPED: no LLM trigger for '{text}' (classified '{intent}')")
                        self.ui_queue.put(("status", f"LISTENING [{intent}]"))
                        continue

                    # Handle follow-up context inheritance (parent interview type + intent)
                    if cls.interview_type == InterviewType.FOLLOW_UP and self.last_question:
                        effective_intent = intent  # already resolved to the parent intent by ISRE
                        routed_question = f"{self.last_question} -> Follow-up: {text}"
                        logger.info(f"[FOLLOW_UP] detected | [PREVIOUS_TYPE] {self.last_interview_type.value if self.last_interview_type else 'UNKNOWN'} "
                                    f"| [PREVIOUS_INTENT] {effective_intent} | [PREVIOUS_QUESTION] '{self.last_question}' | current='{text}'")
                    else:
                        # Cold start: a short follow-up-shaped question with no prior context is its own question
                        effective_intent = intent if cls.interview_type != InterviewType.FOLLOW_UP else "GCA"
                        routed_question = text
                        self.last_question = text
                        self.last_intent = effective_intent
                        self.last_interview_type = cls.interview_type
                        logger.info(f"[QUESTION_DETECTION] NEW question [{effective_intent}]: '{text}'")

                    logger.info(f"Interviewer question routed [{effective_intent}]: '{text}'")

                    # Backpressure: clear any older queued prompts
                    while not self.prompt_queue.empty():
                        try:
                            self.prompt_queue.get_nowait()
                            self.prompt_queue.task_done()
                        except (queue.Empty, ValueError):
                            break

                    telemetry.mark_prompt_enqueued()
                    self.prompt_queue.put((list(self.history), routed_question, effective_intent, telemetry))

            except Exception as e:
                logger.error(f"DialogueManager error: {e}", exc_info=True)
            finally:
                self.turn_queue.task_done()

# ---------------------------------------------------------------------------
# 10. Frameless Webcam Teleprompter Overlay & UI Controls
# ---------------------------------------------------------------------------
class TeleprompterOverlay:
    """Semi-transparent, top-aligned, screen-share cloaked teleprompter window."""

    def __init__(self, ui_queue: queue.Queue, chunker: DocumentChunker, on_clear_cb, on_toggle_pause_cb, on_test_cb=None, on_toggle_mic_role_cb=None):
        self.tk = tk
        self.ui_queue = ui_queue
        self.chunker = chunker
        self.on_clear_cb = on_clear_cb
        self.on_toggle_pause_cb = on_toggle_pause_cb
        self.on_test_cb = on_test_cb
        self.on_toggle_mic_role_cb = on_toggle_mic_role_cb
        self.mic_as_interviewer = False

        self.root = tk.Tk()
        self.root.title("Interview Copilot V2")

        self.bg_color = "#181818"
        self.fg_color = "#ffffff"
        self.accent_color = "#4caf50"
        self.subtext_color = "#9e9e9e"

        # Frameless, topmost, alpha
        self.root.overrideredirect(True)
        self.root.wm_attributes("-topmost", True)
        self.root.wm_attributes("-alpha", 0.92)
        self.root.configure(bg=self.bg_color)

        # Windows screen-share cloaking (WDA_EXCLUDEFROMCAPTURE = 0x00000011)
        try:
            import ctypes
            self.root.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id()) or self.root.winfo_id()
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, 0x00000011)
        except Exception as e:
            logger.debug(f"Screen capture affinity skipped: {e}")

        # Geometry: 820x220 top-aligned beneath webcam
        window_width = 820
        window_height = 220
        screen_width = self.root.winfo_screenwidth()
        pos_x = (screen_width - window_width) // 2
        pos_y = 8
        self.root.geometry(f"{window_width}x{window_height}+{pos_x}+{pos_y}")

        # Dragging handlers for repositioning
        self._drag_start_x = 0
        self._drag_start_y = 0
        self.root.bind("<Button-1>", self._start_drag)
        self.root.bind("<B1-Motion>", self._on_drag)

        # Header bar
        header_frame = tk.Frame(self.root, bg="#252525", height=28)
        header_frame.pack(fill=tk.X, side=tk.TOP)

        # Left status badges
        self.status_label = tk.Label(
            header_frame,
            text="● LISTENING",
            font=("Segoe UI", 8, "bold"),
            bg="#252525",
            fg=self.accent_color
        )
        self.status_label.pack(side=tk.LEFT, padx=(8, 4))

        self.intent_label = tk.Label(
            header_frame,
            text="[GCA]",
            font=("Segoe UI", 8, "bold"),
            bg="#252525",
            fg="#64b5f6"
        )
        self.intent_label.pack(side=tk.LEFT, padx=3)

        self.provider_label = tk.Label(
            header_frame,
            text=f"[{GROQ_MODEL}]",
            font=("Segoe UI", 8),
            bg="#252525",
            fg=self.subtext_color
        )
        self.provider_label.pack(side=tk.LEFT, padx=4)

        self.telemetry_label = tk.Label(
            header_frame,
            text="STT: -- | TTFT: --",
            font=("Segoe UI", 8),
            bg="#252525",
            fg="#ffb74d"
        )
        self.telemetry_label.pack(side=tk.LEFT, padx=6)

        doc_count = len(list(DOCS_DIR.glob("*")))
        self.doc_count_label = tk.Label(
            header_frame,
            text=f"Docs: {doc_count}",
            font=("Segoe UI", 8),
            bg="#252525",
            fg=self.subtext_color
        )
        self.doc_count_label.pack(side=tk.LEFT, padx=4)

        # Right action buttons
        self.clear_btn = tk.Button(
            header_frame,
            text="✕ Clear",
            command=self.clear_overlay,
            font=("Segoe UI", 8),
            bg="#37474f",
            fg="#eceff1",
            activebackground="#455a64",
            activeforeground="#ffffff",
            bd=0,
            padx=6,
            pady=1,
            cursor="hand2"
        )
        self.clear_btn.pack(side=tk.RIGHT, padx=3, pady=2)

        self.pause_btn = tk.Button(
            header_frame,
            text="■ Stop",
            command=self.toggle_pause,
            font=("Segoe UI", 8, "bold"),
            bg="#3e2723",
            fg="#e53935",
            activebackground="#4e342e",
            activeforeground="#ef5350",
            bd=0,
            padx=6,
            pady=1,
            cursor="hand2"
        )
        self.pause_btn.pack(side=tk.RIGHT, padx=3, pady=2)

        if self.on_test_cb:
            self.test_btn = tk.Button(
                header_frame,
                text="⚡ Test",
                command=self.on_test_cb,
                font=("Segoe UI", 8),
                bg="#37474f",
                fg="#ffd54f",
                activebackground="#455a64",
                activeforeground="#ffecb3",
                bd=0,
                padx=5,
                pady=1,
                cursor="hand2"
            )
            self.test_btn.pack(side=tk.RIGHT, padx=3, pady=2)

        self.upload_btn = tk.Button(
            header_frame,
            text="+ Upload Doc",
            command=self.upload_document,
            font=("Segoe UI", 8),
            bg="#1e3a5f",
            fg="#90caf9",
            activebackground="#2a4d7d",
            activeforeground="#bbdefb",
            bd=0,
            padx=6,
            pady=1,
            cursor="hand2"
        )
        self.upload_btn.pack(side=tk.RIGHT, padx=3, pady=2)

        self.mic_role_btn = tk.Button(
            header_frame,
            text="🎤 Mic→Cand",
            command=self.toggle_mic_role,
            font=("Segoe UI", 8),
            bg="#37474f",
            fg="#ce93d8",
            activebackground="#455a64",
            activeforeground="#e1bee7",
            bd=0,
            padx=5,
            pady=1,
            cursor="hand2"
        )
        self.mic_role_btn.pack(side=tk.RIGHT, padx=3, pady=2)

        # Main teleprompter text display
        self.text_widget = tk.Text(
            self.root,
            bg=self.bg_color,
            fg=self.fg_color,
            font=("Segoe UI", 11),
            wrap=tk.WORD,
            bd=0,
            padx=12,
            pady=8,
            highlightthickness=0
        )
        self.text_widget.pack(fill=tk.BOTH, expand=True)
        self.text_widget.insert(
            tk.END,
            "● Ready. Listening for interviewer questions...\n"
            "• GCA frameworks & authentic STAR stories stream here in real time.\n"
            "• Controls: [Ctrl+Shift+T: Test | +P: Pause | +C: Clear | +H: Hide]"
        )
        self.text_widget.configure(state=tk.DISABLED)

        self.is_hidden = False
        self.is_paused = False

        self.root.after(30, self._process_ui_queue)

    def _start_drag(self, event) -> None:
        self._drag_start_x = event.x
        self._drag_start_y = event.y

    def _on_drag(self, event) -> None:
        x = self.root.winfo_x() + (event.x - self._drag_start_x)
        y = self.root.winfo_y() + (event.y - self._drag_start_y)
        self.root.geometry(f"+{x}+{y}")

    def toggle_visibility(self) -> None:
        if self.is_hidden:
            self.root.deiconify()
            self.is_hidden = False
        else:
            self.root.withdraw()
            self.is_hidden = True

    def upload_document(self) -> None:
        try:
            file_path = filedialog.askopenfilename(
                initialdir=str(DOWNLOADS_DIR),
                title="Select Reference Document",
                filetypes=[
                    ("Supported Docs", "*.pdf;*.txt;*.md"),
                    ("PDFs", "*.pdf"),
                    ("Text & Markdown", "*.txt;*.md"),
                    ("All Files", "*.*")
                ]
            )
            if file_path:
                src = Path(file_path)
                dest = DOCS_DIR / src.name
                shutil.copy2(src, dest)
                self.chunker.load_and_index()
                doc_count = len(list(DOCS_DIR.glob("*")))
                self.doc_count_label.configure(text=f"Docs: {doc_count}")
                self.append_token(f"\n[✓ Indexed document: '{src.name}' ({len(self.chunker.chunks)} total chunks)]\n")
        except Exception as e:
            logger.error(f"Failed to upload document: {e}")
            messagebox.showerror("Upload Error", f"Could not index document: {e}")

    def toggle_pause(self) -> None:
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.status_label.configure(text="● STOPPED", fg="#e53935")
            self.pause_btn.configure(text="▶ Resume", bg="#1b5e20", fg="#4caf50")
            self.append_token("\n[■ Audio capture & LLM generation paused]\n")
        else:
            self.status_label.configure(text="● LISTENING", fg="#4caf50")
            self.pause_btn.configure(text="■ Stop", bg="#3e2723", fg="#e53935")
            self.append_token("\n[▶ Listening resumed]\n")
        self.on_toggle_pause_cb(self.is_paused)

    def toggle_mic_role(self) -> None:
        self.mic_as_interviewer = not self.mic_as_interviewer
        if self.mic_as_interviewer:
            self.mic_role_btn.configure(text="🎤 Mic→Int", bg="#4a148c", fg="#e1bee7")
            self.append_token("\n[🎤 Mic routed as INTERVIEWER — speak to trigger LLM]\n")
        else:
            self.mic_role_btn.configure(text="🎤 Mic→Cand", bg="#37474f", fg="#ce93d8")
            self.append_token("\n[🎤 Mic routed as CANDIDATE (default)]\n")
        if self.on_toggle_mic_role_cb:
            self.on_toggle_mic_role_cb(self.mic_as_interviewer)

    def clear_overlay(self) -> None:
        self.text_widget.configure(state=self.tk.NORMAL)
        self.text_widget.delete("1.0", self.tk.END)
        self.text_widget.configure(state=self.tk.DISABLED)
        self.on_clear_cb()

    def append_token(self, token: str) -> None:
        self.text_widget.configure(state=self.tk.NORMAL)
        self.text_widget.insert(self.tk.END, token)
        self.text_widget.see(self.tk.END)
        self.text_widget.configure(state=self.tk.DISABLED)
        try:
            self.root.update_idletasks()
        except Exception:
            pass

    def set_status(self, text: str) -> None:
        if not self.is_paused:
            color = self.accent_color if "LISTEN" in text.upper() else "#ffb74d"
            self.status_label.configure(text=f"● {text.upper()}", fg=color)

    def set_intent_badge(self, intent: str) -> None:
        badge_colors = {
            "GCA": "#64b5f6",
            "BEHAVIORAL": "#ba68c8",
            "TECHNICAL": "#4db6ac",
            "PROJECT": "#81c784",
            "FOLLOW_UP": "#ffb74d",
        }
        color = badge_colors.get(intent, "#90caf9")
        self.intent_label.configure(text=f"[{intent}]", fg=color)

    def set_telemetry(self, text: str) -> None:
        self.telemetry_label.configure(text=text)

    def set_provider(self, text: str) -> None:
        self.provider_label.configure(text=text)

    def _process_ui_queue(self) -> None:
        while not self.ui_queue.empty():
            try:
                msg_type, payload = self.ui_queue.get_nowait()
                if msg_type == "token":
                    self.append_token(payload)
                elif msg_type == "clear_text":
                    self.text_widget.configure(state=self.tk.NORMAL)
                    self.text_widget.delete("1.0", self.tk.END)
                    self.text_widget.configure(state=self.tk.DISABLED)
                elif msg_type == "status":
                    self.set_status(payload)
                elif msg_type == "intent_badge":
                    self.set_intent_badge(payload)
                elif msg_type == "telemetry":
                    self.set_telemetry(payload)
                elif msg_type == "provider":
                    self.set_provider(payload)
                elif msg_type == "partial_stt":
                    self.set_status(payload[:28])
                elif msg_type == "transcript_turn":
                    speaker, turn_text = payload
                    logger.info(f"[History] {speaker}: {turn_text}")
                self.ui_queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"UI Queue processing error: {e}")

        self.root.after(30, self._process_ui_queue)

    def start_loop(self) -> None:
        self.root.mainloop()

# ---------------------------------------------------------------------------
# 11. Device Verification & Audio Stream Resolution
# ---------------------------------------------------------------------------
def resolve_audio_devices() -> Tuple[Optional[int], Optional[int]]:
    """Resolve WASAPI loopback (playback) and candidate mic dynamically."""
    import sounddevice as sd

    devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    wasapi_index = None
    for idx, api in enumerate(hostapis):
        if "WASAPI" in api.get("name", "").upper():
            wasapi_index = idx
            break

    interviewer_idx = None
    candidate_idx = None

    if wasapi_index is not None:
        wasapi_api = hostapis[wasapi_index]
        def_out = wasapi_api.get("default_output_device", -1)
        def_in = wasapi_api.get("default_input_device", -1)

        if def_out >= 0:
            interviewer_idx = def_out
            def_out_name = devices[def_out].get("name", "")
            clean_name = def_out_name.lower().split("(")[0].strip()
            for idx, dev in enumerate(devices):
                if dev.get("hostapi") == wasapi_index and "loopback" in dev.get("name", "").lower():
                    if clean_name and clean_name in dev.get("name", "").lower():
                        logger.info(f"Found dedicated WASAPI loopback device: [{idx}] '{dev.get('name')}'")
                        interviewer_idx = idx
                        break

        if def_in >= 0:
            candidate_idx = def_in

    if interviewer_idx is None:
        if isinstance(sd.default.device, (list, tuple)) and len(sd.default.device) > 1:
            interviewer_idx = sd.default.device[1]
    if candidate_idx is None:
        if isinstance(sd.default.device, (list, tuple)) and len(sd.default.device) > 0:
            candidate_idx = sd.default.device[0]

    if interviewer_idx is not None and interviewer_idx < len(devices):
        idev = devices[interviewer_idx]
        logger.info(
            f"Resolved Interviewer Loopback -> Device [{interviewer_idx}] '{idev.get('name')}' "
            f"(channels={idev.get('max_output_channels')}, rate={idev.get('default_samplerate')}Hz)"
        )
    if candidate_idx is not None and candidate_idx < len(devices):
        cdev = devices[candidate_idx]
        logger.info(
            f"Resolved Candidate Mic -> Device [{candidate_idx}] '{cdev.get('name')}' "
            f"(channels={cdev.get('max_input_channels')}, rate={cdev.get('default_samplerate')}Hz)"
        )

    return interviewer_idx, candidate_idx

# ---------------------------------------------------------------------------
# 12. Automated Synthetic Benchmark & Test Runner (10 Realistic Prompts)
# ---------------------------------------------------------------------------
def run_benchmark() -> None:
    """Executes 10 realistic GCA, Googliness, Follow-up, and Technical questions through the full pipeline."""
    print("=" * 75)
    print(" INTERVIEW COPILOT V2 — LOW-LATENCY PIPELINE BENCHMARK (10 PROMPTS) ".center(75, "="))
    print("=" * 75)

    bootstrap_workspace()
    load_env_variables()
    chunker = DocumentChunker(DOCS_DIR)

    print(f"\n[1/3] Initializing faster-whisper ('{WHISPER_MODEL_NAME}')...")
    t_start = time.time()
    model, runtime_info = create_whisper_model()
    print(f"      [WHISPER] model={runtime_info['model']} device={runtime_info['device']} compute_type={runtime_info['compute_type']}")
    print(f"      Model loaded in {(time.time() - t_start):.2f}s")

    # Benchmark STT on 2.0s audio
    dummy_audio = (np.random.randn(32000) * 0.03).astype(np.float32)
    stt_latencies = []
    for _ in range(3):
        t0 = time.time()
        list(model.transcribe(dummy_audio, beam_size=1))
        stt_latencies.append((time.time() - t0) * 1000)
    median_stt = np.median(stt_latencies)
    print(f"      STT 2.0s audio inference (median): {median_stt:.1f}ms")

    test_cases = [
        ("GCA", "User engagement dropped 30% week-over-week after our latest release. How would you investigate?"),
        ("GCA", "Estimate the daily storage requirements for a global ride-sharing service."),
        ("GCA", "How would you prioritize feature development when two enterprise clients demand conflicting roadmaps?"),
        ("BEHAVIORAL", "Tell me about a time you had a strong disagreement with a technical decision made by your lead."),
        ("BEHAVIORAL", "Describe a situation where a critical production incident occurred under your watch."),
        ("BEHAVIORAL", "Tell me about a time you failed to meet a committed project deadline."),
        ("FOLLOW_UP", "Why did you choose that approach instead of escalating directly?"),
        ("FOLLOW_UP", "What was your specific individual contribution versus the rest of the team?"),
        ("TECHNICAL", "How would you design a distributed cache invalidation protocol under high write concurrency?"),
        ("TECHNICAL", "Explain the trade-offs between saga orchestration and choreography in distributed transactions.")
    ]

    print("\n[2/3] Running 10 Realistic Interview Test Cases...")
    from openai import OpenAI
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    nvidia_key = os.getenv("NVIDIA_API_KEY", "").strip()

    client_groq = None
    client_nvidia = None
    if groq_key:
        client_groq = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key, max_retries=0)
    if nvidia_key:
        client_nvidia = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=nvidia_key, max_retries=1)

    if not client_groq and not client_nvidia:
        print("      [ERROR] No valid API keys found in .env. Skipping LLM benchmark.")
        return

    provider_name = "Groq" if client_groq else "NVIDIA NIM"
    model_name = GROQ_MODEL if client_groq else NVIDIA_MODEL
    print(f"      Primary LLM Provider: {provider_name} ({model_name})")

    results = []
    history = []
    last_intent = "GCA"
    last_type: InterviewType = InterviewType.GCA
    last_q = ""
    groq_create_ms: List[float] = []
    nvidia_create_ms: List[float] = []
    fallback_count = 0
    provider_per_run: List[str] = []

    reps = int(os.getenv("BENCHMARK_REPS", "2"))
    all_cases = test_cases * reps

    for i, (expected_type, query) in enumerate(all_cases, 1):
        if i > 1:
            time.sleep(1.0)  # Avoid burst OTPM rate limiting
        print(f"\n--- Test {i}/{len(all_cases)}: [{expected_type}] \"{query}\" ---")

        telemetry = PerformanceTelemetry(query)

        # 1. Synthesize audio
        try:
            import win32com.client, tempfile, wave
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            stream = win32com.client.Dispatch("SAPI.SpFileStream")
            stream.Open(tmp_path, 3)
            voice.AudioOutputStream = stream
            voice.Speak(query)
            stream.Close()
            with wave.open(tmp_path, "rb") as w:
                pcm_data = w.readframes(w.getnframes())
                pcm_audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
                if w.getframerate() != 16000:
                    pcm_audio = resample_linear(pcm_audio, w.getframerate(), 16000)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            # Fallback to dummy 2s audio
            pcm_audio = (np.random.randn(32000) * 0.03).astype(np.float32)

        # Mark turn end (speech end declared)
        telemetry.mark_turn_end()

        # 2. STT
        telemetry.mark_stt_start()
        segments, _ = model.transcribe(pcm_audio, language="en", vad_filter=False, beam_size=1)
        stt_text = " ".join(s.text.strip() for s in segments).strip()
        telemetry.mark_stt_done()

        # Fallback to original text if SAPI speech was degraded
        actual_query = stt_text if len(stt_text) >= 5 else query

        # 3. ISRE: interview separation & routing
        cls = ISRE.classify(actual_query, prev_type=last_type, prev_question=last_q)
        intent, should_trigger = cls.intent, cls.should_trigger
        telemetry.mark_intent_done(intent)
        logger.info(f"[SEPARATION] type={cls.interview_type.value} confidence={cls.confidence:.2f} "
                    f"latency_ms={cls.latency_ms:.2f} reason='{cls.reason}'")

        if cls.interview_type == InterviewType.FOLLOW_UP:
            effective_intent = last_intent
            retrieval_query = f"{last_q} {actual_query}"
        else:
            effective_intent = intent
            last_intent = intent
            last_type = cls.interview_type
            last_q = actual_query
            retrieval_query = actual_query

        # 4. Retrieval
        top_chunks = chunker.get_top_k(retrieval_query, intent=effective_intent, k=RETRIEVAL_TOP_K)
        telemetry.mark_retrieval_done()

        # 5. LLM Streaming
        context_str = "\n\n---\n\n".join(top_chunks) if top_chunks else "No specific documents."
        if effective_intent == "BEHAVIORAL":
            prompt = f"{BEHAVIORAL_SYSTEM_PROMPT}\n\n[AUTHENTIC CANDIDATE STAR STORIES]\n{context_str}"
        elif effective_intent == "TECHNICAL":
            prompt = f"{TECHNICAL_SYSTEM_PROMPT}\n\n[ARCHITECTURE REFERENCE NOTES]\n{context_str}"
        else:
            prompt = f"{GCA_SYSTEM_PROMPT}\n\n[GCA FRAMEWORKS & CONTEXT]\n{context_str}"

        messages = [{"role": "system", "content": prompt}]
        for spk, txt in history[-4:]:
            messages.append({"role": "user" if spk == "Interviewer" else "assistant", "content": txt})
        messages.append({"role": "user", "content": actual_query})

        telemetry.mark_llm_request_sent()
        telemetry.mark_prompt_enqueued()

        resp_stream = None
        active_provider = provider_name

        if client_groq:
            try:
                t_g0 = time.time()
                resp_stream = client_groq.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=messages,
                    stream=True,
                    max_tokens=LLM_MAX_TOKENS,
                    temperature=LLM_TEMPERATURE
                )
                groq_create_ms.append((time.time() - t_g0) * 1000)
                telemetry.mark_llm_stream_created()
                telemetry.add_provider_event("groq", "stream_created", f"{groq_create_ms[-1]:.0f}ms")
                active_provider = f"Groq ({GROQ_MODEL})"
            except Exception as e:
                is_429 = "429" in str(e) or "rate" in str(e).lower()
                retry_delay = 0.0
                if is_429:
                    m_delay = re.search(r"try again in ([\d.]+)s", str(e))
                    retry_delay = float(m_delay.group(1)) if m_delay else 0.0
                telemetry.add_provider_event("groq", "rate_limited" if is_429 else "failed",
                                             f"retry_after={retry_delay:.2f}s" if is_429 else str(e)[:80])
                print(f"    [Groq Failed: 429 retry_after={retry_delay:.2f}s]" if is_429 else f"    [Groq Failed: {str(e)[:100]}]")
                # V2.1: short 429 retry before slow NVIDIA fallback (same policy as live worker)
                if is_429 and 0.0 < retry_delay <= 3.0:
                    print(f"    [Groq 429 retry-after {retry_delay:.2f}s -> retrying once]")
                    time.sleep(retry_delay + 0.1)
                    try:
                        t_g0 = time.time()
                        resp_stream = client_groq.chat.completions.create(
                            model=GROQ_MODEL,
                            messages=messages,
                            stream=True,
                            max_tokens=LLM_MAX_TOKENS,
                            temperature=LLM_TEMPERATURE
                        )
                        groq_create_ms.append((time.time() - t_g0) * 1000)
                        telemetry.mark_llm_stream_created()
                        telemetry.add_provider_event("groq", "retry_after_429", f"{retry_delay:.2f}s")
                        active_provider = f"Groq ({GROQ_MODEL})"
                    except Exception as e2:
                        telemetry.add_provider_event("groq", "retry_failed", str(e2)[:80])
                        print(f"    [Groq retry failed: {str(e2)[:80]} -> NVIDIA NIM fallback]")

        if resp_stream is None and client_nvidia:
            fallback_count += 1
            telemetry.add_provider_event("nvidia", "fallback_start")
            t_n0 = time.time()
            resp_stream = client_nvidia.chat.completions.create(
                model=NVIDIA_MODEL,
                messages=messages,
                stream=True,
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE
            )
            nvidia_create_ms.append((time.time() - t_n0) * 1000)
            telemetry.mark_llm_stream_created()
            telemetry.add_provider_event("nvidia", "stream_created", f"{nvidia_create_ms[-1]:.0f}ms")
            active_provider = f"NVIDIA NIM ({NVIDIA_MODEL})"
        provider_per_run.append(active_provider)

        first_token_time = None
        output_tokens = []
        token_count = 0
        for chunk in resp_stream:
            delta = chunk.choices[0].delta.content if chunk.choices else ""
            if delta:
                if first_token_time is None:
                    first_token_time = time.time()
                    telemetry.mark_first_token(active_provider)
                output_tokens.append(delta)
                token_count += 1

        telemetry.mark_generation_done(token_count)
        summary = telemetry.get_summary_dict()
        results.append(summary)

        first_line = "".join(output_tokens).strip().splitlines()[0] if output_tokens else ""
        print(f"    Intent: {intent} (effective: {effective_intent}) | Provider: {active_provider}")
        print(f"    EOS -> STT Complete:   {summary['eos_to_stt_complete_ms']}ms")
        print(f"    EOS -> LLM Request:    {summary['eos_to_llm_request_ms']}ms")
        print(f"    EOS -> First UI Token: {summary['eos_to_first_token_ms']}ms")
        print(f"    EOS -> Complete Resp:  {summary['eos_to_complete_response_ms']}ms")
        if summary["provider_events"]:
            print(f"    Provider Events: {summary['provider_events']}")
        print(f"    Spec Prep (intent+BM25, hidden behind speech in live mode): {summary['intent_ms'] + summary['rag_ms']:.1f}ms")
        print(f"    Output Preview: {first_line[:75]}...")

        history.append(("Interviewer", actual_query))
        history.append(("Assistant", "".join(output_tokens)))

    # Compute Summary Statistics
    eos_to_stt = [r["eos_to_stt_complete_ms"] for r in results]
    eos_to_req = [r["eos_to_llm_request_ms"] for r in results]
    eos_to_first = [r["eos_to_first_token_ms"] for r in results]
    eos_to_comp = [r["eos_to_complete_response_ms"] for r in results]
    spec_prep = [r["intent_ms"] + r["rag_ms"] for r in results]
    n_runs = len(results)

    cold_create = groq_create_ms[0] if groq_create_ms else float("nan")
    warm_creates = groq_create_ms[1:] if len(groq_create_ms) > 1 else groq_create_ms

    print("\n" + "=" * 75)
    print(f" [3/3] BENCHMARK SUMMARY — REAL PIPELINE ({n_runs} RUNS, {reps} REPS) ".center(75, "="))
    print("=" * 75)
    print(f"• Sample Count:                        {n_runs}")
    print(f"• End-of-Speech -> STT Complete:       Median={np.median(eos_to_stt):.1f}ms  |  P95={np.percentile(eos_to_stt, 95):.1f}ms")
    print(f"• End-of-Speech -> LLM Request:        Median={np.median(eos_to_req):.1f}ms  |  P95={np.percentile(eos_to_req, 95):.1f}ms")
    print(f"• End-of-Speech -> First Useful Token: Median={np.median(eos_to_first):.1f}ms  |  P95={np.percentile(eos_to_first, 95):.1f}ms")
    print(f"• End-of-Speech -> Complete Response:  Median={np.median(eos_to_comp):.1f}ms |  P95={np.percentile(eos_to_comp, 95):.1f}ms")
    print(f"• Speculative Prep (intent+BM25):      Median={np.median(spec_prep):.1f}ms  |  P95={np.percentile(spec_prep, 95):.1f}ms  (pre-computed behind speech in live mode)")
    if groq_create_ms:
        print(f"• Groq stream create — Cold (1st):     {cold_create:.1f}ms")
        print(f"• Groq stream create — Warm (median):  {np.median(warm_creates):.1f}ms  (n={len(warm_creates)})")
    print(f"• Groq fallback frequency:             {fallback_count}/{n_runs} runs")
    if nvidia_create_ms:
        print(f"• NVIDIA fallback stream create:       Median={np.median(nvidia_create_ms):.1f}ms  (n={len(nvidia_create_ms)})")
    print(f"• Providers used:                      {dict((p, provider_per_run.count(p)) for p in set(provider_per_run))}")
    print("=" * 75)

# ---------------------------------------------------------------------------
# 13. Main Orchestration & Global Hotkeys
# ---------------------------------------------------------------------------
def main():
    if "--benchmark" in sys.argv:
        run_benchmark()
        return

    bootstrap_workspace()
    load_env_variables()

    print("=" * 75)
    print(" REAL-TIME INTERVIEW CO-PILOT V2 (LOW-LATENCY GCA & GOOGLINESS) ".center(75, "="))
    print("=" * 75)
    print(f"• Workspace:        {WORKSPACE_DIR}")
    print(f"• Docs Folder:      {DOCS_DIR}")
    print(f"• STT Model:        {WHISPER_MODEL_NAME} ({WHISPER_DEVICE})")
    print(f"• Groq Model:       {GROQ_MODEL}")
    print(f"• NVIDIA Model:     {NVIDIA_MODEL}")
    print(f"• End-of-Turn Wait: {END_OF_TURN_SILENCE * 1000:.0f}ms")
    print("=" * 75)

    # 1. Document Indexing
    chunker = DocumentChunker(DOCS_DIR)

    # 2. Inter-thread Queues (Bounded for backpressure)
    audio_queue: queue.Queue = queue.Queue(maxsize=50)
    turn_queue: queue.Queue = queue.Queue(maxsize=50)
    prompt_queue: queue.Queue = queue.Queue(maxsize=10)
    ui_queue: queue.Queue = queue.Queue(maxsize=500)

    # 3. Background Workers
    spec_queue: queue.Queue = queue.Queue(maxsize=20)
    spec_worker = SpeculativeWorker(spec_queue, chunker)  # context wired after DialogueManager starts
    spec_worker.start()

    stt_worker = TranscriptionWorker(audio_queue, turn_queue, ui_queue, spec_queue=spec_queue)
    stt_worker.start()

    dialogue_manager = DialogueManager(turn_queue, prompt_queue, ui_queue)
    dialogue_manager.start()
    # Speculative classification reuses live conversation context (previous type/question)
    spec_worker.context_provider = lambda: (dialogue_manager.last_interview_type, dialogue_manager.last_question)

    llm_worker = LLMStreamingWorker(prompt_queue, ui_queue, chunker, spec_worker=spec_worker)
    llm_worker.start()

    # 3b. Optional one-time provider connection warmup (1 token per provider, non-fatal)
    if WARMUP_PROVIDERS:
        threading.Thread(target=llm_worker.warmup, name="ProviderWarmup", daemon=True).start()

    # 4. Audio Streams Setup
    interviewer_dev, candidate_dev = resolve_audio_devices()
    listeners = []
    candidate_listener = None

    # 4a. Interviewer loopback via soundcard (WASAPI playback)
    try:
        sc_loopback = SoundcardLoopbackListener(
            name="Interviewer",
            target_queue=audio_queue,
            sample_rate=48000
        )
        sc_loopback.start()
        listeners.append(sc_loopback)
    except Exception as e:
        logger.error(f"SoundcardLoopback failed: {e}. Interviewer loopback unavailable.")
        if interviewer_dev is not None:
            try:
                fb = AudioChannelListener("Interviewer", interviewer_dev, True, audio_queue)
                fb.start()
                listeners.append(fb)
            except Exception as e2:
                logger.error(f"PortAudio loopback fallback failed: {e2}.")

    # 4b. Candidate mic via sounddevice
    if candidate_dev is not None:
        try:
            candidate_listener = AudioChannelListener(
                name="Candidate",
                device_idx=candidate_dev,
                is_loopback=False,
                target_queue=audio_queue
            )
            candidate_listener.start()
            listeners.append(candidate_listener)
        except Exception as e:
            logger.error(f"Failed starting Candidate mic stream: {e}.")

    # 5. UI Callbacks
    def handle_clear():
        dialogue_manager.clear()
        ui_queue.put(("status", "Buffer cleared"))

    def handle_toggle_pause(is_paused: bool):
        dialogue_manager.listening_paused = is_paused
        for l in listeners:
            l.paused = is_paused
        logger.info(f"Listening paused state set to: {is_paused}")

    test_counter = 0
    test_questions = [
        "User engagement dropped 30% week-over-week. How would you investigate?",
        "Tell me about a time you had a significant disagreement with an engineering lead.",
        "Why did you choose that approach?",
        "How would you design a distributed telemetry aggregation pipeline?"
    ]

    def trigger_test_question():
        nonlocal test_counter
        sample_q = test_questions[test_counter % len(test_questions)]
        test_counter += 1
        logger.info(f"[Test Inject] Injecting question: '{sample_q}'")
        turn_queue.put(("Interviewer", sample_q, time.time(), time.time()))

    def handle_toggle_mic_role(mic_as_interviewer: bool):
        if candidate_listener:
            candidate_listener.name = "Interviewer" if mic_as_interviewer else "Candidate"
            logger.info(f"Mic role set to: {candidate_listener.name}")

    # 6. Initialize Teleprompter UI
    overlay = TeleprompterOverlay(
        ui_queue=ui_queue,
        chunker=chunker,
        on_clear_cb=handle_clear,
        on_toggle_pause_cb=handle_toggle_pause,
        on_test_cb=trigger_test_question,
        on_toggle_mic_role_cb=handle_toggle_mic_role
    )

    # 7. Global Hotkeys
    try:
        import keyboard
        keyboard.add_hotkey("ctrl+shift+h", overlay.toggle_visibility)
        keyboard.add_hotkey("ctrl+shift+p", overlay.toggle_pause)
        keyboard.add_hotkey("ctrl+shift+c", overlay.clear_overlay)
        keyboard.add_hotkey("ctrl+shift+t", trigger_test_question)
        logger.info("Global hotkeys: Ctrl+Shift+H (Hide), +P (Pause), +C (Clear), +T (Test)")
    except Exception as e:
        logger.warning(f"Failed registering global hotkeys via 'keyboard' ({e}).")

    # 8. Main GUI Loop
    try:
        overlay.start_loop()
    except KeyboardInterrupt:
        logger.info("Termination signal received.")
    finally:
        logger.info("Shutting down co-pilot audio streams and worker threads...")
        for l in listeners:
            l.stop()
        audio_queue.put((None, None, False, 0))
        turn_queue.put((None, None, 0, 0))
        prompt_queue.put(None)
        logger.info("Clean shutdown complete.")

if __name__ == "__main__":
    main()
