# Day 3 — GenAI: Prompt Engineering, RAG, Vector DBs, Agents, MCP & ACP (Tue, 17 Nov)

> This is **your home ground**. You've done this in production. Today's goal: **know the proper names** for things you already do, go deeper on design choices, and practice **whiteboarding a RAG + agent system** end to end.

**Time split:** Prompting (1 hr) · RAG + Vector DBs (3 hrs) · Agents (2 hrs) · MCP/ACP (1 hr) · Whiteboard practice (1 hr)

---

## 1. Prompt Engineering ⭐ (JD: "optimize prompts for accuracy, performance, and cost")

### 1.1 Techniques (know the names)
| Technique | What | Example |
|---|---|---|
| **Zero-shot** | Just ask | "Classify this email as spam or not." |
| **Few-shot** | Give 2–5 examples first | Show 3 labeled emails, then the new one |
| **Role / System prompt** | Set persona, rules, boundaries | "You are a restaurant assistant. Only answer about menu and orders." |
| **Chain-of-Thought (CoT)** | Ask it to reason step by step | "Think step by step before answering." Better at maths and logic |
| **ReAct** | Reason → Act (call tool) → Observe → repeat | The basis of tool-calling agents |
| **Self-consistency** | Generate several answers, take the majority | For reasoning accuracy |
| **Structured output** | Force JSON / a schema | "Return JSON: {name, email, job_id}". Use Pydantic or the provider's JSON mode / tool schemas |
| **Prompt chaining** | Break a task into steps, each prompt feeds the next | Extract → validate → summarize |
| **Delimiters** | Wrap inputs in tags/quotes | `<context>...</context>` reduces confusion and prompt injection |
| **Grounding instruction** | "Answer ONLY from the context. If not found, say you don't know." | Key anti-hallucination line for RAG |

### 1.2 Optimizing for accuracy
- Clear, specific instructions + output format + examples.
- Put the most important rules at the start (and repeat critical ones at the end for long prompts).
- Give the model an "out": *"If unsure, say 'I don't know'."*
- Lower temperature for factual tasks.
- Validate the output (Pydantic) and retry on failure.

### 1.3 Optimizing for cost & latency (← **your stories!**)
- **Shorter prompts**: move rules out of the system prompt into **tools/code**. *(Your 50% system-prompt reduction in the restaurant assistant: hours validation and order logic moved into modular tools.)*
- **Model routing**: a small/cheap model for easy tasks, a large one only when needed.
- **Prompt caching**: reuse a static prefix (system prompt) so providers bill it cheaper and respond faster.
- **Limit max_tokens**, ask for concise outputs.
- **Response caching**: same question → cached answer (semantic cache with embeddings).
- **Streaming**: the user sees the first tokens fast (better perceived latency).
- **Parallel calls** instead of sequential when the steps are independent.
- **Move orchestration server-side**: fewer round trips. *(Your 40% latency cut!)*

### 1.4 Prompt injection & security (bonus, ties to your PII/policy work)
- **Prompt injection** = user input (or a retrieved doc) contains "ignore previous instructions...".
- Defenses: separate system and user content, delimiters, input/output guardrails, least-privilege tools, human approval for risky actions, **PII detection (Presidio)**, **policy engine checks before and after the response** ← *exactly your Rust decision engine work.*

### 1.5 Evaluating prompts
- Build a small **golden test set** (questions + expected answers).
- Compare prompt versions on accuracy, format validity, tokens used and latency.
- **LLM-as-a-judge** for open-ended answers.
- Version prompts like code (Git, prompt registry, LangSmith/Langfuse).

---

## 2. RAG — Retrieval-Augmented Generation ⭐⭐⭐

### 2.1 Why RAG?
LLMs don't know your **private** or **recent** data and they hallucinate. RAG = **find the relevant documents first, then give them to the LLM as context** so it answers from facts.

### 2.2 The full pipeline (be able to draw this)

```
 ┌──────────── INGESTION (offline) ────────────┐
 │ Docs (PDF, web, DB) → Load → Clean → Chunk   │
 │      → Embed → Store in Vector DB (+metadata)│
 └──────────────────────────────────────────────┘
                         │
 ┌──────────── QUERY (online) ─────────────────────────────────────────┐
 │ User question → (rewrite query) → Embed → Retrieve top-k (hybrid)   │
 │   → Rerank → Build prompt (system + context + question)            │
 │   → LLM → Answer with citations → Guardrails → User                │
 └─────────────────────────────────────────────────────────────────────┘
```

### 2.3 Ingestion & Chunking
**Why chunk?** Documents are too long for the context window, and smaller focused chunks give **more precise retrieval**.

| Strategy | How | Pros / Cons |
|---|---|---|
| **Fixed-size** | Every N tokens (e.g. 500) | Simple. May cut sentences in half |
| **Recursive character** ⭐ | Split by paragraphs → sentences → words until small enough | LangChain default, respects structure |
| **Document-structure** | By headings, sections, Markdown/HTML tags, pages | Great for manuals, docs |
| **Semantic** | Split where the meaning changes (embedding similarity drops) | Best coherence, slower |
| **Parent-child / small-to-big** | Retrieve small chunks, send their bigger parent section to the LLM | Precise retrieval + full context |

- **Chunk overlap (e.g. 10–20%)**: neighboring chunks share some text, so **an idea split at a boundary isn't lost**.
- **Chunk size tradeoff:** too small → missing context. Too big → noisy, diluted embeddings, more tokens. Typical: 300–1000 tokens. **Tune it with evaluation.**
- **Metadata**: store source, page, date, department, access level → enables **filtering and citations**.

### 2.4 Embeddings & Similarity
- Embedding model turns text into a vector (e.g. 384, 768, 1536 dimensions).
- **Use the same embedding model for documents and queries.**
- **Cosine similarity** = the angle between vectors (ignores length), ranges −1 to 1. The most common choice.
- **Dot product** = angle × magnitude. Equals cosine **if the vectors are normalized**. Faster.
- **Euclidean (L2)** = straight-line distance.
- Popular models: OpenAI `text-embedding-3-small/large`, Azure OpenAI embeddings, `bge`, `e5`, `all-MiniLM-L6-v2` (local), Cohere embed.

### 2.5 Vector Databases & Indexing
**Why a vector DB?** Comparing a query with millions of vectors one by one is too slow. Vector DBs use **ANN (Approximate Nearest Neighbor)** indexes: slightly approximate, but massively faster.

| Index | Idea |
|---|---|
| **Flat / brute force** | Exact, compare with everything. Fine for small data |
| **HNSW** ⭐ | Multi-layer graph, "highway" layers on top for big jumps and detailed layers below. Very fast and accurate, uses more memory. The most popular |
| **IVF** | Cluster vectors into buckets (k-means) and search only the nearest buckets |
| **PQ (Product Quantization)** | Compress vectors to save memory |

| Vector DB | Notes |
|---|---|
| **FAISS** | Meta's library, in-memory, not a full DB |
| **Chroma** | Simple, local, great for prototypes |
| **Pinecone** | Fully managed cloud |
| **Qdrant / Weaviate / Milvus** | Open-source, production-grade, filtering, hybrid search |
| **pgvector** | Vectors inside **Postgres**, use SQL + vectors together |
| **Azure AI Search** ⭐ | Azure's managed search: **vector + keyword (hybrid) + semantic ranker**. Mention this for Azure! |
| **Elasticsearch / OpenSearch** | Keyword (BM25) + vector |
| **MongoDB Atlas Vector Search** | You know MongoDB, a good link |

> 📝 **Before Thursday, write down exactly which vector DB, embedding model and chunking you used at Quantal.** They WILL ask "which vector DB did you use and why?"

### 2.6 Improving Retrieval Quality ⭐
1. **Hybrid search** = **dense (vector, meaning)** + **sparse (BM25, exact keywords)**, merged using **Reciprocal Rank Fusion (RRF)**.
   - Why? Vectors miss exact terms like product codes, names and error IDs ("ERR-4021"). BM25 catches them.
2. **Reranking** with a **cross-encoder** (e.g. Cohere Rerank, `bge-reranker`):
   - Retrieve the top 20–50 quickly → the cross-encoder reads the (query, chunk) **together** and scores the true relevance → keep the top 3–5.
   - **Bi-encoder** (embeddings) = fast but rough. **Cross-encoder** = slow but accurate. Use both: retrieve, then rerank.
3. **Metadata filtering**: e.g. only HR docs, only 2026 docs, only docs this user may access.
4. **Query transformation**:
   - **Query rewriting**: make a vague or follow-up question standalone (use chat history).
   - **Multi-query**: generate 3 variations and retrieve for each.
   - **HyDE**: the LLM writes a hypothetical answer, and you embed *that* to search.
   - **Query decomposition**: split a complex question into sub-questions.
5. **Better chunking** + adding a section title or summary to each chunk (contextual chunking).

### 2.7 Generation & Hallucination Mitigation (your resume says "hallucination mitigation")
- Strong grounding prompt: *"Answer ONLY using the context. If the answer isn't there, say 'I don't have that information.'"*
- **Citations**: make the model cite chunk IDs/sources so users can verify.
- **Low temperature** (0–0.3).
- **Better retrieval** (most hallucinations in RAG are actually retrieval failures).
- **Output validation**: check the answer is supported by the context (an LLM judge or NLI check), structured outputs, guardrails.
- **Tools for facts** instead of guessing: e.g. your restaurant bot **validates hours through a tool** rather than the LLM guessing.
- **"I don't know" fallback / human handoff.**

### 2.8 Evaluating RAG ⭐⭐
Evaluate **retrieval** and **generation** separately:

| Part | Metric | Question it answers |
|---|---|---|
| Retrieval | **Context Precision** | Are the retrieved chunks relevant? (little noise) |
| Retrieval | **Context Recall** | Did we retrieve ALL the info needed? |
| Retrieval | Hit rate / Recall@k, MRR, nDCG | Is the right doc in the top-k, and how high? |
| Generation | **Faithfulness / Groundedness** | Is every claim in the answer supported by the context? (**= hallucination check**) |
| Generation | **Answer Relevance** | Does the answer actually address the question? |
| Generation | Answer correctness | Matches the ground truth? |
| System | Latency (p50/p95), cost per query, tokens | Production health |

- **Tools:** **RAGAS** ⭐, DeepEval, TruLens, LangSmith evals, Azure AI evaluation.
- **How:** build a golden dataset (50–200 Q&A pairs), run the pipeline, score it, change one thing (chunk size, top-k, reranker), re-score. Plus collect **user feedback** (👍/👎) in production.

### 2.9 Advanced RAG patterns (name-drop if relevant)
- **Agentic RAG**: an agent decides *whether* to retrieve, *which* source, and re-retrieves if the context is weak.
- **Corrective RAG (CRAG) / Self-RAG**: grade the retrieved docs and retry or web-search if they're bad.
- **GraphRAG**: build a knowledge graph of entities and relations, good for "connect the dots" questions.
- **Multimodal RAG**: tables, images and PDFs with charts.
- **Conversational RAG**: rewrite follow-up questions using chat history.

### 2.10 RAG minimal code (LangChain-style, conceptual)
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import Chroma

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
chunks = splitter.split_documents(docs)

emb = AzureOpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma.from_documents(chunks, emb)
retriever = db.as_retriever(search_kwargs={"k": 5})

llm = AzureChatOpenAI(model="gpt-4o-mini", temperature=0)

def answer(q):
    ctx = retriever.invoke(q)
    context = "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(ctx))
    prompt = f"""Answer ONLY from the context. Cite sources like [0].
If the answer is not in the context, say "I don't know".
<context>{context}</context>
Question: {q}"""
    return llm.invoke(prompt).content
```

---

## 3. Agentic AI ⭐⭐

### 3.1 What is an agent?
**An agent = an LLM + tools + a loop + memory, where the LLM decides what to do next.**

| Chain / Workflow | Agent |
|---|---|
| **Fixed** steps written by the developer | LLM **decides** the steps dynamically |
| Predictable, cheaper, easier to test | Flexible, handles open-ended tasks |
| e.g. RAG: retrieve → answer | e.g. "Book a table and email the confirmation" |

**Rule:** Use the simplest thing that works. A workflow if the steps are known, an agent only if they really need to be decided on the fly.

### 3.2 Core components
1. **LLM (brain)**: reasons and decides.
2. **Tools**: functions the LLM can call (APIs, DB queries, search, calculator). Defined with a **name + description + JSON schema** of the parameters.
3. **Memory**:
   - Short-term: the conversation and state in the current task.
   - Long-term: a vector store or DB of past facts and user preferences.
4. **Planning**: break a goal into steps (ReAct, plan-and-execute).
5. **Loop + stop condition**: max iterations, final answer detection.

### 3.3 How tool calling works (explain step by step)
1. You send the LLM the user message + a **list of tool schemas**.
2. The LLM replies *not with text* but with: `call create_order(item="pizza", qty=2)`.
3. **Your code** executes the function (the LLM never runs code itself).
4. You send the tool result back to the LLM.
5. The LLM either calls another tool or writes the final answer.

> 💡 Your restaurant assistant is exactly this: `create_order` and `check_restaurant_hours` tools. Moving logic into tools = **deterministic code for rules, LLM for language**. That's why the system prompt shrank 50% and reliability went up.

### 3.4 Agent patterns
| Pattern | Idea |
|---|---|
| **ReAct** | Thought → Action → Observation → repeat |
| **Plan-and-Execute** | Make a full plan first, then execute the steps (fewer LLM calls) |
| **Reflection / Self-critique** | Agent reviews its own output and fixes it |
| **Router** | LLM classifies intent → sends it to the right chain/agent/model |
| **Multi-agent: Supervisor** | One manager agent delegates to specialist agents (researcher, coder, writer) |
| **Multi-agent: Hierarchical / Swarm / Network** | Agents hand off to each other |
| **Human-in-the-loop** | Pause for approval before risky actions (payments, emails) |

### 3.5 LangChain vs LangGraph
- **LangChain**: building blocks (LLM wrappers, prompts, retrievers, tools) and simple chains/agents.
- **LangGraph** ⭐: builds agents as a **graph**: **nodes** (functions/LLM calls), **edges** (including **conditional edges**), and a shared **state**.
  - Supports **loops/cycles**, **checkpointing / persistence** (resume, memory), **human-in-the-loop interrupts**, streaming and multi-agent setups.
  - Why it's used in production: **more control and predictability** than a free-running agent.
```python
from langgraph.graph import StateGraph, END
g = StateGraph(State)
g.add_node("agent", call_llm)
g.add_node("tools", run_tools)
g.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
g.add_edge("tools", "agent")      # loop back after tool execution
g.set_entry_point("agent")
app = g.compile(checkpointer=memory)
```
- Other frameworks to know by name: **CrewAI**, **AutoGen / Microsoft Agent Framework** (Azure-side), **Semantic Kernel** (Microsoft), **LlamaIndex**, OpenAI Agents SDK.

### 3.6 Production challenges of agents (and fixes)
| Challenge | Fix |
|---|---|
| Infinite loops | Max iterations, timeouts |
| Wrong tool / bad arguments | Clear tool descriptions, schemas with Pydantic validation, fewer tools per agent |
| High latency & cost | Fewer LLM calls, parallel tool calls, smaller models for routing, caching |
| Hallucinated actions | Validation, human approval for risky tools, least-privilege permissions |
| Hard to debug | **Tracing/observability** (LangSmith, Langfuse, OpenTelemetry) ← *your trace ingestion work* |
| Security | Auth between components (*your API-key auth agent → decision engine*), PII masking, guardrails |

---

## 4. MCP — Model Context Protocol ⭐ (JD explicitly asks)

### 4.1 One-liner
> **MCP is an open standard (created by Anthropic in late 2024) that standardizes how AI applications connect to external tools and data. It's the "USB-C port for AI".**

### 4.2 The problem it solves
Without MCP: every app × every tool needs a custom integration (**M × N problem**). 5 AI apps × 10 tools = 50 integrations.
With MCP: each tool exposes **one MCP server**, each app implements **one MCP client** → **M + N**.

### 4.3 Architecture
```
 ┌───────────── HOST (AI app: Claude Desktop, IDE, your agent) ─────────────┐
 │   MCP Client  ──────►  MCP Server A (GitHub)                              │
 │   MCP Client  ──────►  MCP Server B (Postgres DB)                         │
 │   MCP Client  ──────►  MCP Server C (Slack)                               │
 └──────────────────────────────────────────────────────────────────────────┘
```
- **Host**: the AI application the user interacts with.
- **Client**: lives inside the host, one connection per server.
- **Server**: exposes capabilities of a tool or data source.
- **Protocol**: **JSON-RPC 2.0** messages.
- **Transports**: **stdio** (local process) and **Streamable HTTP** (remote servers, which replaced the older HTTP + SSE transport).

### 4.4 What a server exposes (3 core primitives)
| Primitive | Controlled by | Example |
|---|---|---|
| **Tools** | The **model** decides to call | `create_issue`, `run_sql_query`, `send_email` |
| **Resources** | The **application** provides read-only context | Files, DB schemas, documents |
| **Prompts** | The **user** picks reusable templates | "/summarize-pr" template |

The client side can also offer things like **sampling** (server asks the host's LLM to generate) and **elicitation** (server asks the user for input). There's also **discovery**: the client asks the server "what tools do you have?" (`tools/list`) and calls them with `tools/call`.

### 4.5 Tiny MCP server (Python SDK, FastMCP)
```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("restaurant")

@mcp.tool()
def check_hours(day: str) -> str:
    """Return opening hours for a given day."""
    return "10:00-22:00" if day != "Monday" else "Closed"

@mcp.resource("menu://today")
def menu() -> str:
    return "Pizza, Pasta, Salad"

if __name__ == "__main__":
    mcp.run()          # stdio by default
```

### 4.6 MCP vs normal function calling
- **Function calling** = the LLM's *ability* to request a function call (each app defines its own tools in its own code).
- **MCP** = a *standard protocol* for **packaging and sharing** tools/data so **any** MCP-compatible app can discover and use them. Function calling still happens under the hood.

### 4.7 Security points
Authentication (OAuth for remote servers), least privilege, **tool poisoning / prompt injection** via tool descriptions, user consent before tool calls, only use trusted servers.

### 4.8 Your link
> "At Quantal, I built modular tool-calling (restaurant assistant tools, agent ↔ decision-engine integration with API-key auth). MCP is the standardized version of that: instead of custom wrappers per app, I'd expose those tools as an MCP server so any agent or host could reuse them."

---

## 5. ACP — Agent Communication Protocol (and A2A) ⭐

### 5.1 One-liner
> **MCP connects an agent to *tools and data*. ACP/A2A connects *agents to other agents*.**

### 5.2 ACP
- An open protocol started by **IBM Research (BeeAI project)** under the **Linux Foundation**.
- **REST-based** ("HTTP for agents"): agents talk over plain HTTP endpoints, so you can even use curl/Postman. No special SDK required.
- Features: **agent discovery** (agent metadata/manifest describing its capabilities), **sync, async and streaming** communication, **multimodal messages** (text, images, files via MIME types), and **framework-agnostic** (a LangGraph agent can talk to a CrewAI agent).

### 5.3 A2A (Agent2Agent) — the important update
- **A2A** is Google's agent-to-agent protocol (2025), also under the **Linux Foundation**. Agents publish an **Agent Card** (JSON describing their skills and endpoint), and they exchange **tasks** and messages over HTTP (JSON-RPC, with streaming via SSE).
- **In 2025, the ACP team announced ACP would merge into A2A**, so the industry is consolidating around **A2A** for agent-to-agent communication.
- 💡 **Mentioning this shows you're up to date:** *"ACP and A2A solve the same problem. ACP has since been folded into A2A under the Linux Foundation, so for a new system I'd look at A2A, but the concepts are the same: discovery, task exchange, streaming."*

### 5.4 MCP vs ACP/A2A (draw this)
```
                 ┌────────── A2A / ACP ──────────┐
      Agent A (LangGraph)  ◄──────────────►  Agent B (CrewAI)
           │                                     │
          MCP                                   MCP
           │                                     │
    Tools / DBs / APIs                    Tools / DBs / APIs
```
| | MCP | ACP / A2A |
|---|---|---|
| Connects | Agent ↔ tools/data (**vertical**) | Agent ↔ agent (**horizontal**) |
| Analogy | USB-C for tools | HTTP / email between agents |
| Example | Agent queries a DB via an MCP server | Travel agent delegates "book hotel" to a hotel-booking agent |

### 5.5 "Equivalent agent orchestration approaches" (JD wording)
If they ask how you orchestrate agents *without* these protocols: **LangGraph supervisor / multi-agent graphs**, a message queue / event-driven design (*your event-driven trigger work!*), or REST microservices where each agent is a service.

---

## 6. 🧑‍🏫 Whiteboard Practice: "Design a RAG chatbot for our client's internal documents"

Practice saying this out loud in ~5 minutes:

1. **Clarify requirements:** Who are the users? Document types and volume? How often do docs change? Access control? Latency target? Languages? Cloud (Azure)?
2. **Ingestion:** Blob Storage → a parsing job (PDF/Docx; OCR if scanned) → recursive chunking (~800 tokens, 15% overlap) with metadata (source, page, department, ACL) → Azure OpenAI embeddings → **Azure AI Search** (hybrid index). Re-index on document change (event-driven trigger).
3. **Query path:** Flask API `/chat` → rewrite the query with chat history → hybrid search + metadata filter by user permissions → rerank to top 5 → grounded prompt with citations → Azure OpenAI GPT model → stream the response.
4. **Agentic layer (if needed):** LangGraph agent with tools: `search_docs`, `query_sql_db`, `create_ticket`. Tools exposed via **MCP** so other apps can reuse them.
5. **Safety:** PII masking (Presidio), prompt injection guardrails, content filters, auth (Entra ID), secrets in Key Vault.
6. **Evaluation:** a golden set + RAGAS (faithfulness, context recall), plus user feedback.
7. **Deployment:** Docker → AKS / Azure Container Apps, CI/CD (GitHub Actions + ArgoCD).
8. **Monitoring:** latency p95, tokens/cost per query, retrieval hit rate, error rates, traces (App Insights / Langfuse), drift in question topics.
9. **Optimizations:** semantic cache, smaller model for routing, prompt caching, streaming.

---

## ✅ Self-Test
1. Name 6 prompt engineering techniques.
2. How did you reduce prompt size by 50%? Why does that help cost AND accuracy?
3. Draw the RAG pipeline from memory.
4. Why chunk overlap? How do you choose the chunk size?
5. Cosine vs dot product?
6. What is HNSW in one sentence?
7. What is hybrid search and why is it better than vector-only?
8. Bi-encoder vs cross-encoder (reranker)?
9. Name the 4 core RAGAS metrics and what each measures.
10. 5 ways to reduce hallucination.
11. Agent vs chain? When NOT to use an agent?
12. Explain tool calling step by step.
13. Why LangGraph over a basic LangChain agent?
14. What is MCP? Host vs client vs server? 3 primitives?
15. MCP vs ACP/A2A in one line each. What happened to ACP in 2025?
16. Whiteboard the RAG chatbot design in 5 minutes.
