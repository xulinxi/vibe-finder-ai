# Vibe Finder AI — System Architecture

The diagram below shows the full agentic pipeline. To render and export
to PNG, paste the Mermaid block into <https://mermaid.live> and use
"Export → PNG" with a transparent background, then save the result to
`assets/architecture.png` (already referenced from the README).

```mermaid
flowchart TD
    User([User: natural language query]) --> S1

    subgraph INPUT["Stage 1: Input Guardrails"]
        S1["Sanitize<br/>(length / injection / empty)"]
    end

    subgraph PARSE["Stage 2: Parse · Claude API"]
        S2["parse_user_input<br/>specialized constrained-vocab prompt"]
        S2b{"Claude<br/>available?"}
        S2c["Keyword fallback parser"]
    end

    subgraph VALIDATE["Stage 3: Validate"]
        S3a["validate_parsed_input<br/>type / range checks"]
        S3b["detect_hallucination<br/>artist names not in user input"]
        S3c["apply_defaults"]
    end

    subgraph RETRIEVE["Stage 4: RAG · Last.fm"]
        S4a["artist.getSimilar<br/>for each preferred artist"]
        S4b["Cache + rate-limit"]
        S4c["Proxy artist mapping<br/>(fictional → real)"]
    end

    subgraph SCORE["Stage 5-6: Score + Boost"]
        S5["recommend_songs<br/>weighted scoring + diversity penalty<br/>(original Module 1-3 engine)"]
        S6["Last.fm similarity boost<br/>+ re-rank"]
    end

    subgraph CRITIQUE["Stage 7-8: Self-Critique · Claude API"]
        S7["critique_recommendations<br/>quality_score, issues, suggested_mode"]
        S7b{"quality < 0.4<br/>AND retry helpful?"}
        S8["Retry with new mode<br/>(max 1 retry)"]
    end

    subgraph OUTPUT["Stage 9: Explain + Confidence"]
        S9a["generate_explanation<br/>conversational summary"]
        S9b["compute_confidence<br/>magnitude + decisiveness + RAG agreement"]
        S9c["validate_recommendations<br/>hallucinated titles / duplicates"]
    end

    subgraph HUMAN["Where humans / testing review AI results"]
        H2["Test harness<br/>9 scenarios + RAG/specialization comparisons"]
        H3["Per-recommendation confidence label<br/>(high / medium / low)<br/>user reads label and decides to trust or refine query"]
    end

    LOG[("Structured logging<br/>at every stage<br/>logs/vibe_finder.log")]

    S1 --> S2
    S2 --> S2b
    S2b -- yes --> S3a
    S2b -- no --> S2c --> S3a
    S3a --> S3b --> S3c --> S4a
    S4a --> S4b --> S4c --> S5
    S5 --> S6 --> S7
    S7 --> S7b
    S7b -- no --> S9a
    S7b -- yes --> S8 --> S9a
    S9a --> S9b --> S9c --> Result([Final output])
    Result --> H3

    H2 -.validates.-> S5
    H2 -.validates.-> S6
    H2 -.validates.-> S2
    S7 -.observes.-> S6

    S1 -. logs .-> LOG
    S2 -. logs .-> LOG
    S4a -. logs .-> LOG
    S5 -. logs .-> LOG
    S7 -. logs .-> LOG
    S9a -. logs .-> LOG

    classDef llm fill:#fff4e1,stroke:#d97706,color:#92400e
    classDef rag fill:#e0f2fe,stroke:#0284c7,color:#075985
    classDef core fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef guard fill:#fee2e2,stroke:#dc2626,color:#991b1b
    classDef human fill:#f3e8ff,stroke:#9333ea,color:#581c87

    class S2,S7,S9a llm
    class S4a,S4b,S4c,S6 rag
    class S5 core
    class S1,S3a,S3b,S3c,S9c guard
    class H2,H3 human
```

## Component summary

| Component | Module | Role |
|---|---|---|
| **Retriever** | `src/lastfm.py` | RAG: queries Last.fm `artist.getSimilar` for context |
| **Agent / Orchestrator** | `src/agent.py` | Sequences all 9 stages; owns retry loop |
| **LLM (3 specialized prompts)** | `src/llm.py` | Parse, critique, explain — each with its own system prompt |
| **Evaluator** | `src/guardrails.py` | Input/output validation, hallucination detection, confidence scoring |
| **Tester** | `tests/test_harness.py` | Scenario suite + RAG before/after + Specialization comparison |
| **Core scoring** | `src/recommender.py` | Original Module 1-3 weighted scoring engine (untouched) |
| **Logging** | `src/logger_config.py` | Structured logs at every stage to `logs/vibe_finder.log` |
| **Configuration** | `src/config.py` | API keys, thresholds, valid genre/mood lists |

## Data flow at a glance

```
NL input
   ↓ sanitize (reject empty / oversized / injection)
   ↓ Claude parse  →  fallback to keyword parser if API unavailable
   ↓ validate + apply defaults + hallucination check
   ↓ Last.fm RAG (artist.getSimilar for each mentioned artist)
   ↓ score (rule-based engine + RAG match boost)
   ↓ Claude self-critique → maybe retry with different scoring mode
   ↓ Claude conversational explanation
   ↓ confidence label per pick + output validation
   ↓ formatted table + warnings + pipeline trace
```

Every external call (Claude, Last.fm) has try/except with a graceful
fallback path. The worst-case behavior — both APIs unreachable — is the
original Module 1-3 rule-based recommender plus structured logging and
input/output guardrails.
