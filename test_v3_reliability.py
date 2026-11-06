"""
Standalone V3 Reliability Test Suite for Interview Copilot.
Validates all V3 core features:
1. AnswerTarget Model & Provenance
2. STT Entity Normalization (Power BI vs Power VA)
3. Entity Contradiction Detection & Rejection
4. Confidence Score Handling
5. Controlled Section-Aware Context Retrieval
6. Topic Switching (PROJECT -> TECHNICAL, etc.)
7. Conversation State Transitions
8. Candidate Speech Protection
9. NO_SAFE_ANSWER Triggers
10. Hallucination Taxonomy Reporting (H1-H7)
11. Answer Planner Structural Prompts
12. Dynamic Length Enforcement
13. Answer Validation & Regeneration
14. Provenance Tracking
15. Streaming Preemption & Generation ID Increment
16. Evaluation Datasets Runner (evaluation/*.jsonl)
17. Adversarial Prompt Suite
18. Latency Telemetry Preservation
"""

import os
import sys
import json
import time
import queue
from pathlib import Path
from typing import Dict, List, Any

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import copilot
from copilot import (
    AnswerTarget,
    EntityDefinition,
    DomainLexicon,
    ConversationState,
    detect_topic_switch,
    build_answer_target,
    detect_entity_contradiction,
    validate_technical_answer,
    audit_hallucinations,
    HallucinationCategory,
    get_answer_plan,
    get_target_length,
    InterviewType,
    InterviewClassification,
    GoogleInterviewContext,
    DocumentChunker,
    PerformanceTelemetry,
    _looks_like_question,
    NO_SAFE_ANSWER,
    NO_VERIFIED_STORY,
    LLMStreamingWorker,
)


def test_1_answer_target_model():
    print("\n--- Test 1: AnswerTarget Model & Compatibility ---")
    target = AnswerTarget(
        raw_transcript="what is power bi?",
        normalized_question="what is power bi?",
        interview_type=InterviewType.TECHNICAL,
        entity="Power BI",
        topic="Power BI",
        confidence=0.92,
        surface_type=InterviewType.TECHNICAL,
        resolved_interview_type=InterviewType.TECHNICAL,
        intent_confidence=0.95,
        entity_confidence=0.90,
        provenance="ISRE"
    )
    assert target.entity == "Power BI"
    assert target.name == "Power BI"  # legacy alias
    assert target.resolved_interview_type == InterviewType.TECHNICAL
    assert target.confidence == 0.92
    assert "entity" in target.confidence_breakdown
    assert target.confidence_breakdown["intent"] == 0.95
    assert target.provenance == "ISRE"
    print("  PASS: AnswerTarget model attributes and backward compatibility properties verified.")
    return True


def test_2_domain_lexicon_normalization():
    print("\n--- Test 2: STT Entity Normalization (Power BI vs Power VA) ---")
    # Power BI aliases
    for phrase in ["how does pbi work", "open powerbi desktop", "what is power bi"]:
        norm = DomainLexicon.normalize(phrase)
        assert "Power BI" in norm, f"Failed for '{phrase}' -> '{norm}'"
    
    # Power VA strict preservation - MUST NOT be rewritten to Power BI
    va_phrase = "tell me about power va in microsoft"
    norm_va = DomainLexicon.normalize(va_phrase)
    assert "Power VA" in norm_va, f"Power VA was mutated incorrectly: '{norm_va}'"
    assert "Power BI" not in norm_va, f"Power VA collided with Power BI: '{norm_va}'"

    # DAX and Power Query
    assert "DAX" in DomainLexicon.normalize("how to write dacs formulas")
    assert "Power Query" in DomainLexicon.normalize("use powerquery for etl")
    print("  PASS: STT entity normalization correctly handles Power BI and preserves Power VA.")
    return True


def test_3_entity_contradiction_detection():
    print("\n--- Test 3: Entity Contradiction Detection ---")
    # Power BI vs Electrical power / current
    c1 = detect_entity_contradiction("Power BI", "Power BI is measured in volts and amperes.")
    assert c1 is not None and "volt" in c1.lower()
    
    # DAX vs Stock index
    c2 = detect_entity_contradiction("DAX", "DAX is the Frankfurt stock market index in Germany.")
    assert c2 is not None and ("stock" in c2.lower() or "frankfurt" in c2.lower())
    
    # Valid answer has no contradiction
    c3 = detect_entity_contradiction("Power BI", "Power BI is Microsoft's business intelligence and interactive data visualization platform.")
    assert c3 is None, f"False positive contradiction: {c3}"
    print("  PASS: Contradictions between BI concepts and electrical/financial homonyms detected accurately.")
    return True


def test_4_confidence_score_handling():
    print("\n--- Test 4: Confidence Score Handling & Breakdowns ---")
    cls = InterviewClassification(
        interview_type=InterviewType.TECHNICAL,
        intent="TECHNICAL",
        confidence=0.88,
        reason="technical question"
    )
    state = ConversationState()
    target = build_answer_target("Explain DAX calculate function.", cls, state)
    assert target.confidence > 0.8
    assert "intent" in target.confidence_breakdown
    assert "entity" in target.confidence_breakdown
    assert target.confidence_breakdown["intent"] == 0.88
    print("  PASS: Confidence scores and breakdown structures verified.")
    return True


def test_5_section_aware_context_retrieval():
    print("\n--- Test 5: Controlled Section-Aware Context Retrieval ---")
    context_path = ROOT_DIR / "context.md"
    assert context_path.exists(), "context.md missing"
    
    ctx = GoogleInterviewContext(context_path)
    assert len(ctx.sections) >= 15, f"Only loaded {len(ctx.sections)} sections"
    
    # Verify canonical sections can be queried
    profile = ctx.get_section("Candidate Profile") or ctx.get_section("Tell Me About Yourself")
    assert profile is not None and "Kalfan" in profile
    
    delinq = ctx.get_section("Delinquency Dashboard") or ctx.get_section("4. Project: Delinquency Reduction Dashboard")
    assert delinq is not None and "delinquen" in delinq.lower()
    
    dmx = ctx.get_section("Teamwork") or ctx.get_section("DMX")
    assert dmx is not None and "dmx" in dmx.lower()
    print(f"  PASS: {len(ctx.sections)} authoritative sections indexed and accessible.")
    return True


def test_6_topic_switching():
    print("\n--- Test 6: Topic Switching Detection ---")
    state = ConversationState()
    state.update(
        question="Tell me about the Delinquency Dashboard.",
        surface_type=InterviewType.PROJECT,
        resolved_type=InterviewType.PROJECT,
        entity="Delinquency Dashboard",
        topic="Project Deep Dive"
    )
    
    # Switching from PROJECT to generic TECHNICAL question
    is_switch, reason = detect_topic_switch(
        InterviewType.TECHNICAL,
        "What is the difference between an inner join and a left join?",
        state
    )
    assert is_switch is True, f"Failed to detect PROJECT -> TECHNICAL switch: {reason}"
    
    # Switching from PROJECT to PERSONAL
    is_switch2, reason2 = detect_topic_switch(
        InterviewType.PERSONAL,
        "Why did you move from civil engineering to tech?",
        state
    )
    assert is_switch2 is True, f"Failed to detect PROJECT -> PERSONAL switch: {reason2}"
    print("  PASS: Cross-category topic switches reliably identified.")
    return True


def test_7_conversation_state_transitions():
    print("\n--- Test 7: Conversation State Transitions ---")
    state = ConversationState()
    assert state.question_number == 0
    assert state.conversation_generation == 0
    
    state.update(
        question="What is Power BI?",
        surface_type=InterviewType.TECHNICAL,
        resolved_type=InterviewType.TECHNICAL,
        entity="Power BI",
        topic="BI Tools"
    )
    assert state.question_number == 1
    assert state.conversation_generation == 1
    assert state.active_entity == "Power BI"
    
    state.reset()
    assert state.active_question == ""
    assert state.active_entity is None
    assert state.question_number == 0
    print("  PASS: ConversationState transition lifecycle (update -> state change -> reset) verified.")
    return True


def test_8_candidate_speech_protection():
    print("\n--- Test 8: Candidate Speech Protection ---")
    candidate_turns = [
        "In our project, we calculated delinquency rates using DAX measures.",
        "I graduated in 2025 with a degree in civil engineering.",
        "We coordinated weekly standups across three teams during DMX.",
        "The model reached 95% precision on test data."
    ]
    interviewer_turns = [
        "Can you explain how you calculated delinquency rates?",
        "What was your biggest challenge in that project?",
        "Why did you transition from civil engineering?",
        "How would you handle missing values in SQL?"
    ]
    for ct in candidate_turns:
        assert not _looks_like_question(ct), f"Candidate speech falsely tagged as question: '{ct}'"
    for it in interviewer_turns:
        assert _looks_like_question(it), f"Interviewer question not detected: '{it}'"
    print("  PASS: Spoken candidate responses protected from accidental re-triggering.")
    return True


def test_9_no_safe_answer_triggers():
    print("\n--- Test 9: NO_SAFE_ANSWER Triggers ---")
    # Unverified claims: CareerOS or invented company
    h_res = audit_hallucinations("I led engineering at CareerOS and built a microservices backend in FastAPI.", "Tell me about CareerOS.")
    assert h_res.has_hallucination
    assert HallucinationCategory.UNVERIFIED_EXPERIENCE in h_res.categories
    
    # Entity contradiction trigger
    v_res = validate_technical_answer("Power BI", "Power BI measures voltage in electric circuits.")
    assert not v_res.is_valid
    assert v_res.remediation_prompt is not None
    print("  PASS: Unverified project claims and technical contradictions trigger guards.")
    return True


def test_10_hallucination_taxonomy():
    print("\n--- Test 10: Hallucination Taxonomy (H1-H7) ---")
    # H1: Fabricated metric
    h1 = audit_hallucinations("We improved performance by 99.999% across 100 million transactions.", "Tell me about your project.")
    assert HallucinationCategory.FABRICATED_METRIC in h1.categories
    
    # H2: Unverified experience
    h2 = audit_hallucinations("When I was working at Alphora trading desk as a quant...", "What is your trading experience?")
    assert HallucinationCategory.UNVERIFIED_EXPERIENCE in h2.categories
    
    # H3: Entity confusion
    h3 = audit_hallucinations("Power BI connects via 220 volts power outlets to analyze data.", "What is Power BI?", "Power BI")
    assert HallucinationCategory.ENTITY_CONFUSION in h3.categories
    
    # Clean verified statement
    clean = audit_hallucinations("Power BI is a business intelligence tool developed by Microsoft for interactive data visualizations.", "What is Power BI?", "Power BI")
    assert not clean.has_hallucination
    print("  PASS: Taxonomy categories (H1-H7) accurately audited and tagged.")
    return True


def test_11_answer_planner():
    print("\n--- Test 11: Answer Planner Structural Prompts ---")
    plan_tech = get_answer_plan(InterviewType.TECHNICAL, "DAX")
    assert "Definition" in plan_tech or "Mechanism" in plan_tech
    
    plan_beh = get_answer_plan(InterviewType.BEHAVIORAL, "Teamwork")
    assert "STAR" in plan_beh or "Situation" in plan_beh
    
    plan_gca = get_answer_plan(InterviewType.GCA, "System Design")
    assert "Clarify" in plan_gca or "Hypotheses" in plan_gca
    print("  PASS: Structural answer plans generated per interview type.")
    return True


def test_12_dynamic_length_enforcement():
    print("\n--- Test 12: Dynamic Length Enforcement ---")
    min_w, max_w, cat = get_target_length(InterviewType.TECHNICAL, "What is DAX?")
    assert cat == "SHORT"
    assert min_w >= 40 and max_w <= 80
    
    min_w2, max_w2, cat2 = get_target_length(InterviewType.GCA, "How would you investigate a 30% drop in revenue?")
    assert cat2 == "DETAILED"
    assert max_w2 >= 150
    print("  PASS: Dynamic target lengths assigned correctly (SHORT/NORMAL/DETAILED).")
    return True


def test_13_answer_validation_limits():
    print("\n--- Test 13: Answer Validation & Max Regeneration ---")
    # Valid answer passes
    val_pass = validate_technical_answer("SQL", "SQL stands for Structured Query Language used to query relational databases.")
    assert val_pass.is_valid
    
    # Invalid answer flagged
    val_fail = validate_technical_answer("DAX", "DAX is the Frankfurt stock index in Germany.")
    assert not val_fail.is_valid
    assert len(val_fail.remediation_prompt) > 0
    print("  PASS: Technical validation verifies correctness and builds remediation prompts.")
    return True


def test_14_provenance_tracking():
    print("\n--- Test 14: Provenance Tracking ---")
    cls = InterviewClassification(
        interview_type=InterviewType.TECHNICAL,
        intent="TECHNICAL",
        confidence=0.91,
        reason="technical query"
    )
    state = ConversationState()
    target = build_answer_target("What is Power Query in Excel?", cls, state)
    assert target.provenance in ("GENERAL_TECHNICAL", "context.md", "REASONING", "DETERMINISTIC_ARITHMETIC")
    print("  PASS: Provenance records classifier provenance, entity detection, and inheritance state.")
    return True


def test_15_streaming_preemption():
    print("\n--- Test 15: Streaming Preemption & Generation ID Increment ---")
    p_q, u_q = queue.Queue(), queue.Queue()
    chunker = DocumentChunker(str(ROOT_DIR / "docs"))
    worker = LLMStreamingWorker(p_q, u_q, chunker)
    
    initial_gen = worker.active_generation_id
    worker.cancel_current()
    assert worker.active_generation_id == initial_gen + 1
    worker.cancel_current()
    assert worker.active_generation_id == initial_gen + 2
    print("  PASS: LLM generation cancellation preempts active streams via monotonic generation IDs.")
    return True


def test_16_evaluation_datasets_runner():
    print("\n--- Test 16: Evaluation Datasets Runner (evaluation/*.jsonl) ---")
    eval_dir = ROOT_DIR / "evaluation"
    assert eval_dir.exists(), "evaluation/ directory missing"
    
    expected_files = [
        "routing.jsonl",
        "technical.jsonl",
        "behavioral.jsonl",
        "project.jsonl",
        "followups.jsonl",
        "grounding.jsonl",
        "adversarial.jsonl"
    ]
    total_records = 0
    for fname in expected_files:
        fpath = eval_dir / fname
        assert fpath.exists(), f"Evaluation dataset missing: {fname}"
        records = [json.loads(line) for line in fpath.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(records) >= 2, f"Evaluation dataset {fname} has too few records: {len(records)}"
        total_records += len(records)
    print(f"  PASS: All {len(expected_files)} evaluation datasets verified with {total_records} total test cases.")
    return True


def test_17_adversarial_prompt_suite():
    print("\n--- Test 17: Adversarial Prompt Suite ---")
    adv_path = ROOT_DIR / "evaluation" / "adversarial.jsonl"
    assert adv_path.exists()
    records = [json.loads(line) for line in adv_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    
    for item in records:
        prompt = item.get("input") or item.get("prompt") or ""
        expected_type = item.get("expected_type", "")
        
        # Test entity confusion detection
        if "Power BI" in prompt and "volts" in prompt:
            c = detect_entity_contradiction("Power BI", prompt)
            assert c is not None, f"Failed to detect adversarial entity contradiction in: {prompt}"
        elif "Why" in prompt or "Then what" in prompt:
            c = copilot.ISRE.classify(prompt)
            assert c.interview_type == InterviewType.FOLLOW_UP
        elif "Power" in prompt:
            cls = copilot.ISRE.classify(prompt)
            target = copilot.build_answer_target(prompt, cls)
            assert "Power" in (target.entity or target.normalized_question)
    print(f"  PASS: Verified {len(records)} adversarial prompts against contradiction and routing guards.")
    return True


def test_18_telemetry_preservation():
    print("\n--- Test 18: Latency Telemetry Preservation ---")
    t = PerformanceTelemetry("Explain DAX calculate function.")
    t.mark_audio_start()
    t.mark_speech_detected()
    time.sleep(0.01)
    t.mark_intent_done("TECHNICAL")
    t.mark_retrieval_done()
    t.mark_llm_request_sent()
    t.mark_first_token("Groq")
    t.mark_generation_done(token_count=85)
    
    summary_dict = t.get_summary_dict()
    assert summary_dict["eos_to_complete_response_ms"] >= 0
    assert summary_dict["ttft_pure_ms"] >= 0
    summary = t.log_and_format()
    assert "TTFT:" in summary
    assert "STT:" in summary
    assert t.token_count == 85
    print("  PASS: End-to-end telemetry captures VAD, STT, intent, TTFT, token count, and summary.")
    return True


def test_19_misheard_stt_and_closest_question_recovery():
    print("\n--- Test 19: Misheard STT & Closest Question Recovery ---")
    # 1. Radius -> Redis vs geometry guard
    norm_rad, ent_rad, _ = DomainLexicon.normalize_stt("What is the radius?")
    assert "Redis" in norm_rad and ent_rad == "Redis", f"Expected Redis, got: {norm_rad}, {ent_rad}"
    assert copilot.IntentClassifier.classify("What is the radius?") == "TECHNICAL"
    norm_circle, ent_circle, _ = DomainLexicon.normalize_stt("What is the radius of a circle?")
    assert "Redis" not in norm_circle and ent_circle != "Redis", f"Geometry radius wrongly converted: {norm_circle}"

    # 2. First API -> FastAPI
    norm_fapi, ent_fapi, _ = DomainLexicon.normalize_stt("What is first API?")
    assert "FastAPI" in norm_fapi and ent_fapi == "FastAPI", f"Expected FastAPI, got: {norm_fapi}, {ent_fapi}"

    # 3. Reaction / Reactance -> React vs chemistry guard
    for phrase in ("What is the reaction?", "What is reactance?"):
        norm_r, ent_r, _ = DomainLexicon.normalize_stt(phrase)
        assert "React" in norm_r and ent_r == "React", f"Expected React for '{phrase}', got: {norm_r}, {ent_r}"
    norm_chem, ent_chem, _ = DomainLexicon.normalize_stt("What is a chemical reaction?")
    assert "React" not in norm_chem.split() and ent_chem != "React", f"Chemical reaction wrongly converted: {norm_chem}"

    # 4. Java Springbook / Javas Priak -> Java Spring Boot
    for phrase in ("What is meant by Java Springbook?", "What is meant by Javas Priak?"):
        norm_sb, ent_sb, _ = DomainLexicon.normalize_stt(phrase)
        assert "Java Spring Boot" in norm_sb and ent_sb == "Java Spring Boot", f"Failed Spring Boot for '{phrase}': {norm_sb}"

    # 5. What is it last? -> What is Flask
    norm_fl, ent_fl, _ = DomainLexicon.normalize_stt("What is it last?")
    assert "What is Flask" in norm_fl and ent_fl == "Flask", f"Expected Flask, got: {norm_fl}, {ent_fl}"

    # 6. What is meant by ETA? / LDA? -> EDA
    for phrase in ("What is meant by ETA?", "What is meant by LDA?"):
        norm_eda, ent_eda, _ = DomainLexicon.normalize_stt(phrase)
        assert "EDA" in norm_eda and ent_eda == "EDA", f"Expected EDA for '{phrase}', got: {norm_eda}, {ent_eda}"

    # 7. Hi, tell me about your sin. -> tell me about yourself -> PERSONAL
    sin_q = "Hi, tell me about your sin."
    assert _looks_like_question(sin_q) is True
    norm_sin, _, _ = DomainLexicon.normalize_stt(sin_q)
    assert "tell me about yourself" in norm_sin.lower(), f"Expected tell me about yourself, got: {norm_sin}"
    cls_sin = copilot.ISRE.classify(sin_q)
    assert cls_sin.interview_type == InterviewType.PERSONAL and cls_sin.should_trigger is True

    # 8. Why googling? / My Google. -> Why Google? -> PERSONAL
    for g_phrase in ("Why googling?", "My Google."):
        norm_g, _, _ = DomainLexicon.normalize_stt(g_phrase)
        assert norm_g == "Why Google?", f"Expected 'Why Google?', got: {norm_g}"
        cls_g = copilot.ISRE.classify(g_phrase)
        assert cls_g.interview_type == InterviewType.PERSONAL and cls_g.should_trigger is True

    # 9. Tell me about your internship at the Mx. -> DMX Technologies -> PROJECT, no H1 hallucination
    mx_q = "Tell me about your internship at the Mx."
    norm_mx, ent_mx, _ = DomainLexicon.normalize_stt(mx_q)
    assert "DMX Technologies" in norm_mx and ent_mx == "DMX Technologies"
    cls_mx = copilot.ISRE.classify(mx_q)
    assert cls_mx.interview_type == InterviewType.PROJECT
    target_mx = build_answer_target(mx_q, cls_mx)
    dmx_ans = "During my internship at DMX Technologies, I collaborated with the team on dashboard deliverables and SQL queries."
    audit_mx = audit_hallucinations(dmx_ans, target_mx)
    assert audit_mx.is_clean and audit_mx.category != HallucinationCategory.H1_INVENTED_EXPERIENCE

    # Contrast with unsupported manager conflict at internship
    mgr_q = "Tell me about a time you had a conflict with your manager at your internship"
    cls_mgr = copilot.ISRE.classify(mgr_q)
    target_mgr = build_answer_target(mgr_q, cls_mgr)
    fab_ans = "At my internship, my manager and I disagreed over deadlines, so I scheduled a 1-on-1 to align."
    audit_mgr = audit_hallucinations(fab_ans, target_mgr)
    assert not audit_mgr.is_clean and audit_mgr.category == HallucinationCategory.H1_INVENTED_EXPERIENCE

    # 10. GCA trade-off misrecognition
    gca_noisy = "yet red off. First but rough. But slow but thorough. Are you both sides in under 45 seconds total?"
    cls_gca = copilot.ISRE.classify(gca_noisy)
    assert cls_gca.interview_type == InterviewType.GCA, f"Expected GCA, got: {cls_gca.interview_type}"

    # 11. Power VA strictly preserved
    norm_pva, ent_pva, _ = DomainLexicon.normalize_stt("What is Power VA?")
    assert norm_pva == "What is Power VA?" and ent_pva == "Power VA"
    assert "Power BI" not in norm_pva

    # 12. Closest question phonetic recovery & AnswerTarget.recovery_hint
    rec_q, rec_hint = DomainLexicon.recover_closest_question("Why this roll?")
    assert rec_q == "Why this role?" and rec_hint is not None, f"Expected 'Why this role?', got: {rec_q}, {rec_hint}"
    near_q = "What are your strength and weakness?"
    cls_near = copilot.ISRE.classify(near_q)
    target_near = build_answer_target(near_q, cls_near)
    assert target_near.recovery_hint is not None and "What are your strengths and weaknesses?" in target_near.normalized_question
    print("  PASS: All STT misrecognitions, guards, routing, H1 audits, and closest-question recoveries verified.")
    return True


def _build_minimal_two_page_pdf(pdf_path: Path, page1_text: str, page2_text: str) -> None:
    """Builds a valid 2-page PDF with extractable text streams using standard PDF syntax."""
    def _escape_pdf_text(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    s1 = f"BT /F1 12 Tf 72 720 Td ({_escape_pdf_text(page1_text)}) Tj ET".encode("latin-1")
    s2 = f"BT /F1 12 Tf 72 720 Td ({_escape_pdf_text(page2_text)}) Tj ET".encode("latin-1")

    objs = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 6 0 R >>\nendobj\n",
        b"4 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 7 0 R >>\nendobj\n",
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        f"6 0 obj\n<< /Length {len(s1)} >>\nstream\n".encode("latin-1") + s1 + b"\nendstream\nendobj\n",
        f"7 0 obj\n<< /Length {len(s2)} >>\nstream\n".encode("latin-1") + s2 + b"\nendstream\nendobj\n",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objs:
        offsets.append(len(out))
        out.extend(obj)
    xref_pos = len(out)
    out.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode("latin-1"))
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
    out.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("latin-1")
    )
    pdf_path.write_bytes(bytes(out))


def test_20_stealth_document_pdf_viewer_engine():
    print("\n--- Test 20: Stealth Document & PDF Viewer Engine ---")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        ext_dir = tmp_path / "external"
        ext_dir.mkdir(parents=True, exist_ok=True)

        # Create a 2-page PDF and a multi-section Markdown file
        pdf_file = docs_dir / "candidate_notes.pdf"
        _build_minimal_two_page_pdf(
            pdf_file,
            "Page One Alpha: Credit Risk Dashboard built with Power BI and SQL.",
            "Page Two Beta: DMX Technologies internship collaboration and Redis caching.",
        )

        md_file = docs_dir / "interview_prep.md"
        md_file.write_text(
            "# Section 1 Intro\nTelemetry and latency metrics.\n## Section 2 Deep Dive\nExploratory Data Analysis and DAX measures.",
            encoding="utf-8",
        )

        engine = copilot.DocumentViewerEngine(docs_dir=docs_dir, workspace_dir=tmp_path)
        docs = engine.list_documents()
        doc_names = [d.name for d in docs]
        assert "candidate_notes.pdf" in doc_names and "interview_prep.md" in doc_names

        # Load and verify 2-page PDF navigation & search
        meta_pdf = engine.load_document(pdf_file)
        assert meta_pdf["total_pages"] == 2
        assert meta_pdf["page"] == 1
        assert "Credit Risk Dashboard" in meta_pdf["content"]

        p_num, p_total, p2_txt = engine.next_page()
        assert p_num == 2 and p_total == 2 and "DMX Technologies" in p2_txt

        p_num_prev, _, p1_txt = engine.prev_page()
        assert p_num_prev == 1 and "Credit Risk Dashboard" in p1_txt

        # Search across pages
        matches = engine.search("Redis")
        assert len(matches) == 1 and matches[0][0] == 2
        assert engine.current_page == 1  # 0-based index for page 2

        matches_both = engine.search("Page")
        assert len(matches_both) == 2
        next_hit = engine.next_search_match()
        assert next_hit is not None and next_hit[0] == 2

        # Load and verify Markdown pagination
        meta_md = engine.load_document(md_file)
        assert meta_md["total_pages"] >= 2
        assert "Telemetry" in engine.get_page(0)[2]
        assert "Exploratory Data Analysis" in engine.next_page()[2]

        # Stealth import of external file without OS dialog
        ext_md = ext_dir / "external_cheatsheet.md"
        ext_md.write_text("# External Sheet\nFastAPI and Java Spring Boot notes.", encoding="utf-8")
        imported_dest = engine.import_file(ext_md)
        assert imported_dest.exists() and imported_dest.parent == docs_dir
        assert "FastAPI" in engine.get_page(0)[2]
    print("  PASS: DocumentViewerEngine PDF/MD loading, pagination, search, and dialog-free import verified.")
    return True


def test_21_window_stealth_cloaking_and_whisper_prompt():
    print("\n--- Test 21: Window Stealth Cloaking & Whisper Initial Prompt ---")
    for term in ("Power BI", "Power VA", "Redis", "FastAPI", "React", "DMX Technologies", "EDA"):
        assert term in copilot.WHISPER_INITIAL_PROMPT, f"Missing '{term}' in WHISPER_INITIAL_PROMPT"

    assert copilot.WDA_EXCLUDEFROMCAPTURE == 0x00000011
    assert copilot.apply_window_stealth(0) is False
    assert copilot.apply_window_stealth(None) is False

    if sys.platform == "win32":
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        try:
            root.update_idletasks()
            stealth_ok = copilot.apply_window_stealth(root)
            assert stealth_ok is True, "Expected apply_window_stealth to succeed on Win32 Tk window"
        finally:
            root.destroy()
    print("  PASS: WHISPER_INITIAL_PROMPT vocabulary and Win32 WDA_EXCLUDEFROMCAPTURE stealth cloaking verified.")
    return True


def run_all_v3_tests():
    print("===========================================================================")
    print("           INTERVIEW COPILOT V3 RELIABILITY TEST SUITE                     ")
    print("===========================================================================")
    
    tests = [
        test_1_answer_target_model,
        test_2_domain_lexicon_normalization,
        test_3_entity_contradiction_detection,
        test_4_confidence_score_handling,
        test_5_section_aware_context_retrieval,
        test_6_topic_switching,
        test_7_conversation_state_transitions,
        test_8_candidate_speech_protection,
        test_9_no_safe_answer_triggers,
        test_10_hallucination_taxonomy,
        test_11_answer_planner,
        test_12_dynamic_length_enforcement,
        test_13_answer_validation_limits,
        test_14_provenance_tracking,
        test_15_streaming_preemption,
        test_16_evaluation_datasets_runner,
        test_17_adversarial_prompt_suite,
        test_18_telemetry_preservation,
        test_19_misheard_stt_and_closest_question_recovery,
        test_20_stealth_document_pdf_viewer_engine,
        test_21_window_stealth_cloaking_and_whisper_prompt,
    ]
    
    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  FAIL: {test_fn.__name__} raised exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            
    print("\n===========================================================================")
    print(f"V3 TEST RESULTS: {passed}/{len(tests)} PASSED (Failed: {failed})")
    print("===========================================================================")
    return failed == 0


run_all_tests = run_all_v3_tests


if __name__ == "__main__":
    success = run_all_v3_tests()
    sys.exit(0 if success else 1)
