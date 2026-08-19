# Agent Architecture — AlphaAgents

## Agent Contract

Every agent implements the following contract:
1. **Defined role** — one responsibility, one prompt file
2. **Structured input** — domain models (never raw strings)
3. **Structured output** — Pydantic schema (never free-form text)
4. **Evidence requirements** — FACT claims must reference evidence IDs
5. **Confidence score** — 0.0–1.0, reflects evidence quality
6. **Failure isolation** — exceptions return a `failed=True` AgentReport, never raise

---

## Fundamental Analyst (`app/agents/fundamental.py`)

**Input:**
- `FundamentalMetrics` — structured financial data from yfinance (P/E, margins, FCF, etc.)
- `list[Evidence]` — retrieved chunks from annual reports and filings

**Output schema:**
```python
class FundamentalAgentOutput(BaseModel):
    findings: list[AgentFinding]   # each with claim_type and evidence_ids
    risks: list[RiskFactor]
    summary: str
    confidence: float              # 0-1
```

**Prompt strategy:** Agent is explicitly told to distinguish FACT (evidence-backed), INFERENCE (derived), and UNCERTAINTY (evidence-insufficient). High evidence quality → higher confidence.

---

## Technical Analyst (`app/agents/technical.py`)

**Input:**
- `TechnicalIndicators` — deterministically computed (RSI, MACD, SMA, volatility, etc.)

**Critical design:** The LLM receives pre-computed numbers. It is never asked to calculate RSI. The prompt says: "You have been given pre-computed indicator values. Do NOT recalculate."

**Output schema:**
```python
class TechnicalAgentOutput(BaseModel):
    findings: list[AgentFinding]   # all INFERENCE or UNCERTAINTY (no document evidence)
    risks: list[RiskFactor]
    sentiment: Sentiment           # bullish/bearish/neutral/mixed
    summary: str
    confidence: float
```

---

## Sentiment Analyst (`app/agents/sentiment.py`)

**Input:**
- `list[NewsArticle]` — fetched in real-time from Yahoo Finance

**Graceful empty state:** If no articles are fetched, the agent returns a non-failed report with `confidence=0.0` and a summary noting the absence of data. This is not a failure — it is a data-unavailable state.

**Output schema:**
```python
class SentimentAgentOutput(BaseModel):
    findings: list[AgentFinding]
    risks: list[RiskFactor]
    sentiment: Sentiment
    summary: str
    confidence: float
    key_events: list[str]          # notable events (earnings, FDA approval, etc.)
```

---

## Risk / Critic Agent (`app/agents/critic.py`)

**Input:** The three specialist agent reports (as structured text).

**Role:** This agent does NOT analyse the underlying financial data. It audits the other agents' reasoning. It looks for:
- Claims typed as FACT with no evidence_ids
- Contradictions between agents (e.g., fundamental bullish, sentiment bearish with concrete events)
- Overconfident scores on thin evidence
- Missing data risks (stale documents, limited news coverage)

**Output schema:**
```python
class CriticAgentOutput(BaseModel):
    critic_findings: list[CriticFinding]   # each with affected_agent, severity, recommendation
    overall_risk_level: Severity
    summary: str
    confidence: float
```

---

## Synthesis Agent (`app/agents/synthesis.py`)

**Input:** All four specialist and critic reports + all evidence.

**Role:** Integration, not summarisation. The agent must:
1. Identify where agents agree on strong evidence
2. Surface disagreements without resolving them artificially
3. Incorporate critic findings — flagged claims are caveated or dropped
4. Produce a research report a professional investor could use

**Output schema:**
```python
class SynthesisOutput(BaseModel):
    executive_summary: str
    fundamental_view: str
    technical_view: str
    sentiment_view: str
    key_risks: list[RiskFactor]
    conflicting_signals: list[ConflictingSignal]
    overall_sentiment: Sentiment
    conclusion: str
    confidence_overall: float
    confidence_fundamental: float | None
    confidence_technical: float | None
    confidence_sentiment: float | None
```

---

## Execution Order and Dependencies

```
fetch data [concurrent]
    ↓
RAG retrieval
    ↓
Fundamental ─┐
Technical    ├─ [concurrent asyncio.gather]
Sentiment   ─┘
    ↓
Critic [sequential — needs all three]
    ↓
Synthesis [sequential — needs critic]
    ↓
ResearchReport assembly
```

See [`docs/CONCURRENCY.md`](docs/CONCURRENCY.md) for the full rationale.
