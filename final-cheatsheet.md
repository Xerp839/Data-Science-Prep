# 🏁 Final Cheat Sheet — Read on Thursday Morning (19 Nov)

> 20 minutes. No new learning. Just refresh, then walk in calm.

## Carry
3–4 printed resumes · ID · pen · notebook · phone charged · know your current CTC, expected CTC and notice period · reach Mahape by ~9:45 AM.

## Your numbers (say them confidently)
- **~40%** chatbot latency cut (orchestration moved frontend → backend + prompt/query optimization)
- **50%** system-prompt reduction (logic moved into modular tools)
- Presidio **16–18s → 8–12s** (model loaded at startup + prompt restructure)
- **26** Docker images secured · ArgoCD GitOps rollout
- **15+ hrs/week** saved (Gmail → Loxo ATS)
- Mind-Scope CNN: **91.3%** test accuracy, **0.915** F1, **11,137** params

## ML in one breath
- **Precision** = TP/(TP+FP) → "how many flagged were right" · **Recall** = TP/(TP+FN) → "how many real ones caught" · **F1** = their harmonic mean
- Tumor/fraud → **recall**. Spam → **precision**. Imbalanced → don't trust accuracy, use PR-AUC/F1.
- **Overfitting** = train ≫ test → more data, augmentation, dropout, L1/L2, early stopping, simpler model, CV
- **Bias** (too simple) vs **Variance** (too sensitive)
- **L1** → zeros, feature selection · **L2** → shrinks all
- **Bagging (RF)** parallel, ↓variance · **Boosting (XGB)** sequential, ↓bias
- Tabular → XGBoost · Images → CNN / transfer learning · Text → Transformers
- **Leakage** → fit the scaler/SMOTE/augmentation on train only, after splitting

## DL in one breath
- Forward → loss → **backprop** (chain rule gradients) → optimizer (**Adam**) updates weights
- ReLU hidden · sigmoid binary out · softmax multi-class
- CNN: conv (filters, weight sharing) → ReLU → pool → flatten → dense. Output size `(N−F+2P)/S+1`
- **Attention:** softmax(QKᵀ/√d)·V. Every token looks at every token
- BERT = encoder (understand/embeddings) · GPT = decoder (generate)
- **RAG = knowledge problem · Fine-tune = behavior problem** · **LoRA** = small trainable low-rank adapters on frozen weights · **QLoRA** = LoRA + 4-bit base

## SQL in one breath
- Order: FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT
- Top N per group: `DENSE_RANK() OVER (PARTITION BY dept ORDER BY salary DESC)` in a CTE, then `WHERE rnk <= N`
- ROW_NUMBER (1,2,3,4) · RANK (1,2,2,4) · DENSE_RANK (1,2,2,3)
- Never-ordered customers: LEFT JOIN … `WHERE o.id IS NULL`
- Rolling 7-day: `AVG(x) OVER (ORDER BY d ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)`
- Previous row: `LAG(x) OVER (ORDER BY d)`

## GenAI in one breath
- **RAG:** load → chunk (recursive, ~800 tokens, 10–20% overlap) → embed → vector DB (HNSW) → hybrid search (vector + BM25, RRF) → **rerank (cross-encoder)** → grounded prompt + citations → LLM → guardrails
- **Hallucination fixes:** better retrieval, "answer only from context / say I don't know", citations, temp 0, tools for facts, output validation
- **RAGAS:** Faithfulness · Answer Relevance · Context Precision · Context Recall
- **Agent** = LLM + tools + loop + memory, and the LLM picks the steps. Workflow = fixed steps. Use the simplest thing that works.
- **LangGraph** = nodes + conditional edges + state + checkpointing + human-in-the-loop
- **MCP** = USB-C for AI tools. Host → client → server. JSON-RPC. stdio / Streamable HTTP. **Tools, Resources, Prompts.** Turns M×N into M+N.
- **ACP / A2A** = agent ↔ agent (REST, discovery, sync/async/stream). ACP merged into **A2A** (Linux Foundation, 2025). **MCP = vertical (tools), A2A = horizontal (agents).**

## Backend & cloud in one breath
- Flask: **app factory + blueprints**, Pydantic validation, `gunicorn -w 4 "app:create_app()"` behind Nginx, load models at startup, stream with SSE
- Flask = WSGI, simple, mature · FastAPI = ASGI, async, Pydantic, auto docs
- Azure: **Azure OpenAI**, **AI Search** (hybrid vectors), **AI Foundry**, **AKS / Container Apps**, **ACR**, **Key Vault**, **Blob**, **App Insights / Monitor**, **Functions**, **Entra ID**
- Monitor: **p95 latency**, error/429 rate, tokens and cost, faithfulness, user feedback, drift, traces

## Mindset
- Definition → example → when/why. Tie it back to your own work.
- Unsure? *"I haven't used that directly, but here's how I'd approach it…"*
- Pause before answering. Asking a clarifying question is fine.
- You've **built this stuff in production**. Most candidates walking in have only done tutorials. 💪
