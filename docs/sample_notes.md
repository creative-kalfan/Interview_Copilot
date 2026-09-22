# GCA & System Design Problem Solving Frameworks

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
