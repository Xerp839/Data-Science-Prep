# Day 4 — Flask, Python, Azure, Monitoring + Your Stories & Mock Interview (Wed, 18 Nov)

> **Morning:** Flask + Python (Parts 1–2) · **Afternoon:** Azure + Monitoring (Parts 3–4) · **Evening:** Stories + Mock (Parts 5–7). **Speak your stories OUT LOUD at least twice.** Sleep by 11.

---

## 1. Flask ⭐ (JD: "REST APIs using Flask")

Your bullets say FastAPI. Expect: *"The JD says Flask, you used FastAPI. Can you do Flask?"*
**Answer:** *"Yes. I've used Flask too, and the concepts are the same: routing, request validation, error handling, deployment behind a WSGI/ASGI server. I picked FastAPI for the LLM services because of native async and Pydantic validation, but I'm comfortable structuring the same API in Flask."*

### 1.1 Basic Flask API
```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok"), 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify(error="'text' is required"), 400
    result = model_predict(data["text"])
    return jsonify(result=result), 200

if __name__ == "__main__":
    app.run(debug=True)      # dev only, never in production
```

### 1.2 Production structure: App Factory + Blueprints ⭐
```
app/
 ├── __init__.py        # create_app()
 ├── config.py          # Dev/Prod configs (env vars)
 ├── routes/
 │    ├── chat.py       # Blueprint: /api/chat
 │    └── health.py
 ├── services/          # business logic (RAG, LLM calls)
 └── models/            # loaded ML models / schemas
wsgi.py                 # entrypoint for gunicorn
```
```python
# app/__init__.py
from flask import Flask
from .routes.chat import chat_bp

def create_app(config_name="prod"):
    app = Flask(__name__)
    app.config.from_object(f"app.config.{config_name.title()}Config")
    app.register_blueprint(chat_bp, url_prefix="/api")
    app.extensions["llm"] = load_llm_client()     # load ONCE at startup
    return app

# app/routes/chat.py
from flask import Blueprint, request, jsonify, current_app
chat_bp = Blueprint("chat", __name__)

@chat_bp.post("/chat")
def chat():
    q = request.json["question"]
    llm = current_app.extensions["llm"]
    return jsonify(answer=rag_answer(llm, q))
```
- **App factory (`create_app`)**: creates the app in a function, so you get different configs for dev/test/prod, easy testing and no circular imports.
- **Blueprints**: group related routes into modules (like FastAPI's `APIRouter`).
- **`current_app`, `request`, `g`**: Flask's **context locals**. `request` = current request data. `g` = per-request storage (e.g. a DB connection). `current_app` = the running app.
- **Error handling:** `@app.errorhandler(404)` / `@app.errorhandler(Exception)` returning JSON.
- **Validation:** Flask has none built in. Use **Pydantic** or **marshmallow** manually.
- **Extensions:** Flask-CORS, Flask-SQLAlchemy, Flask-JWT-Extended (auth), Flask-Limiter (rate limiting), flask-smorest / flasgger (Swagger docs).

### 1.3 Streaming LLM responses in Flask
```python
from flask import Response, stream_with_context

@app.post("/chat/stream")
def chat_stream():
    q = request.json["question"]
    def generate():
        for token in llm.stream(q):
            yield f"data: {token}\n\n"          # Server-Sent Events format
    return Response(stream_with_context(generate()), mimetype="text/event-stream")
```

### 1.4 Deploying Flask
- **Never** use `app.run()` in prod. Use a **WSGI server**: **Gunicorn**.
  `gunicorn -w 4 -b 0.0.0.0:8000 "app:create_app()"`
- **Workers:** a common rule is `2 × CPU cores + 1`. For I/O-heavy LLM calls, use **threads** (`--threads 8`) or **gevent** workers to handle many waiting requests.
- **Nginx** in front (reverse proxy, TLS, static files) → Gunicorn → Flask.
- Containerize with Docker → deploy to AKS / Container Apps / App Service.
- **Timeouts:** LLM calls are slow, so raise the gunicorn `--timeout` (default 30s).

### 1.5 Flask vs FastAPI ⭐
| | Flask | FastAPI |
|---|---|---|
| Protocol | **WSGI** (sync) | **ASGI** (async-native) |
| Async | Supported since 2.0 (`async def` views), but each request still uses a worker, so it isn't truly concurrent like ASGI | **Native async**, great for many concurrent I/O calls (LLM APIs) |
| Validation | Manual (Pydantic/marshmallow) | **Built-in via Pydantic** type hints |
| API docs | Needs extensions | **Auto Swagger/OpenAPI** at `/docs` |
| Maturity | Very mature, huge ecosystem, simple | Newer, very popular for ML/AI APIs |
| Server | Gunicorn | Uvicorn (or Gunicorn + Uvicorn workers) |

**Answer:** *"For LLM-heavy services with lots of waiting on external APIs, FastAPI's async is a natural fit. Flask is simpler and very mature, and with Gunicorn plus threaded/gevent workers it handles I/O-bound loads fine. Either way the important parts are the same: load models at startup, validate inputs, handle timeouts and retries, and stream responses."*

### 1.6 REST API basics (quick)
- Methods: **GET** (read), **POST** (create), **PUT** (replace), **PATCH** (partial update), **DELETE**.
- **Idempotent** = calling it twice has the same effect as once: GET, PUT, DELETE yes. POST no.
- Status codes: **200** OK, **201** Created, **204** No Content, **400** Bad Request, **401** Unauthorized (not logged in), **403** Forbidden (no permission), **404** Not Found, **422** Validation error, **429** Too Many Requests, **500** Server Error, **502/503/504** upstream/unavailable/timeout.
- **Stateless**: each request carries everything needed (auth token).
- Auth: **API keys** (*you implemented this*), **JWT**, **OAuth2**.
- **Rate limiting, pagination, versioning** (`/api/v1/...`).

---

## 2. Python ⭐

### 2.1 Core concepts they ask
| Topic | Quick answer |
|---|---|
| **List vs Tuple** | List is mutable `[]`, tuple is immutable `()` (hashable, can be a dict key, slightly faster) |
| **Set / Dict** | Hash-based, O(1) average lookup |
| **Mutable default arg trap** | `def f(x=[])` → the same list is shared across calls! Use `x=None` and create it inside |
| **`is` vs `==`** | Same object in memory vs equal value |
| **Shallow vs deep copy** | `copy.copy` copies the outer object only, `copy.deepcopy` copies everything nested |
| **`*args, **kwargs`** | Variable positional args (tuple) / keyword args (dict) |
| **List comprehension** | `[x*x for x in nums if x > 0]` |
| **Lambda** | Small anonymous function `lambda x: x * 2` |
| **GIL** | Global Interpreter Lock: only one thread runs Python bytecode at a time → **threads are good for I/O-bound** work (API calls), use **multiprocessing for CPU-bound** work |
| **Exceptions** | `try / except / else / finally`, raise custom exceptions |

### 2.2 Decorators
A function that wraps another function to add behavior (logging, timing, auth, retry). `@app.route` is a decorator!
```python
import time, functools

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.perf_counter() - start:.2f}s")
        return result
    return wrapper

@timer
def call_llm(prompt): ...
```
**Retry decorator** (very relevant to LLM APIs): use `tenacity` with `@retry(stop=stop_after_attempt(3), wait=wait_exponential())`.

### 2.3 Generators
Use `yield` to produce values **one at a time, lazily**, without holding everything in memory. Great for big files and for **streaming LLM tokens**.
```python
def read_large_file(path):
    with open(path) as f:
        for line in f:
            yield line.strip()
```

### 2.4 async / await
- For **I/O-bound concurrency**: while one request waits on the LLM API, others proceed.
```python
import asyncio

async def ask(llm, q):
    return await llm.ainvoke(q)

async def main():
    results = await asyncio.gather(ask(llm, "q1"), ask(llm, "q2"), ask(llm, "q3"))  # run in parallel
```
- **Common mistake:** calling blocking code (`requests.get`, `time.sleep`) inside `async def` blocks the whole event loop. Use `httpx.AsyncClient` / `asyncio.sleep`.

### 2.5 OOP
- **4 pillars:** Encapsulation, Abstraction, Inheritance, Polymorphism.
- `__init__`, `self`, `@classmethod`, `@staticmethod`, `@property`, dunder methods (`__str__`, `__len__`).
- **Context manager** (`with`): `__enter__` / `__exit__`. Auto-closes files and connections.

### 2.6 Pydantic
```python
from pydantic import BaseModel, Field, EmailStr

class Candidate(BaseModel):
    name: str
    email: EmailStr
    years_exp: float = Field(ge=0)
    job_id: str | None = None
```
Validates data and gives clear errors. Use it for **API input** and for **forcing LLM structured outputs** (*e.g. your Gmail → Loxo candidate parsing!*).

### 2.7 Pandas / NumPy quick hits (DS title)
```python
df.head(); df.info(); df.describe(); df.shape
df["col"].value_counts()
df.isna().sum(); df.dropna(); df.fillna(df["age"].median())
df.groupby("dept")["salary"].agg(["mean", "max", "count"])
df.merge(other, on="id", how="left")
df.sort_values("salary", ascending=False)
df.apply(lambda row: ..., axis=1)        # slow; prefer vectorized ops
df.pivot_table(index="month", columns="city", values="sales", aggfunc="sum")
pd.to_datetime(df["date"]); df["date"].dt.month
np.where(df["score"] > 0.5, 1, 0)        # vectorized if-else
```
**`loc` vs `iloc`:** label-based vs integer-position-based.
**Why vectorize?** NumPy runs in C, so it's 10–100× faster than Python loops.

### 2.8 Profiling latency & memory (JD: monitor latency)
- `time.perf_counter()`, `cProfile`, `line_profiler`, `memory_profiler`, `tracemalloc`, `py-spy` (live sampling).
- **Your story:** you *found* that Presidio was reloading the NLP model on every request. That's a profiling win.

### 2.9 Likely live-coding mini tasks
```python
# Reverse a string / check palindrome
s[::-1] == s

# Count word frequency
from collections import Counter
Counter(text.lower().split()).most_common(5)

# Two-sum
def two_sum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        if target - n in seen:
            return [seen[target - n], i]
        seen[n] = i

# Flatten a nested list
[x for sub in nested for x in sub]

# Remove duplicates, keep order
list(dict.fromkeys(items))

# Chunk text with overlap (RAG-flavored!)
def chunk(words, size=200, overlap=40):
    step = size - overlap
    return [" ".join(words[i:i+size]) for i in range(0, len(words), step)]
```

---

## 3. Azure / Cloud ⭐ (translate what you already know)

### 3.1 Your skills → Azure vocabulary
| You've used | Azure equivalent | AWS | GCP |
|---|---|---|---|
| Kubernetes | **AKS** (Azure Kubernetes Service) | EKS | GKE |
| Docker images | **ACR** (Azure Container Registry) | ECR | Artifact Registry |
| Simple container hosting | **Azure Container Apps** / **App Service** | ECS / App Runner | Cloud Run |
| Cron jobs / event triggers | **Azure Functions** (serverless), Logic Apps | Lambda | Cloud Functions |
| VM deployment | **Azure Virtual Machines** | EC2 | Compute Engine |
| HashiCorp Vault | **Azure Key Vault** | Secrets Manager | Secret Manager |
| Google Drive storage | **Azure Blob Storage** | S3 | Cloud Storage |
| MySQL / MongoDB | **Azure Database for MySQL/PostgreSQL**, **Cosmos DB** (has a MongoDB API + vector search) | RDS / DocumentDB | Cloud SQL / Firestore |
| Monitoring dashboard / logs | **Azure Monitor + Application Insights + Log Analytics** | CloudWatch | Cloud Monitoring |
| ArgoCD / GitOps | ArgoCD on AKS, **Azure DevOps Pipelines**, GitHub Actions | CodePipeline | Cloud Build |
| Event-driven triggers | **Event Grid**, **Service Bus**, Event Hubs | EventBridge / SQS / Kinesis | Pub/Sub |
| API-key auth | **Azure API Management (APIM)**, **Entra ID** (formerly Azure AD) | API Gateway / IAM | Apigee / IAM |

### 3.2 Azure AI services (most important for this JD)
- **Azure OpenAI Service**: GPT models and embeddings hosted in Azure, with **enterprise security** (data stays in your tenant, private networking, content filters). You call a **deployment name**, not just a model name.
- **Azure AI Foundry** (recently rebranded **Microsoft Foundry**): the platform for building, evaluating and deploying AI apps and agents. It has a model catalog (OpenAI, Llama, Mistral...), prompt flow, evaluations and an agent service.
- **Azure AI Search** ⭐: the managed vector + keyword **hybrid search** with a semantic ranker. **The go-to vector store for RAG on Azure.**
- **Azure Machine Learning**: train, track (MLflow) and deploy classic ML/DL models as endpoints.
- **Azure AI Document Intelligence**: OCR and extracting tables from PDFs (good for RAG ingestion).
- **Azure AI Content Safety**: harmful content filtering, prompt-injection detection (**Prompt Shields**).

### 3.3 Sample answer: "How would you deploy an LLM app on Azure?"
> "I'd containerize the Flask/FastAPI service with Docker and push it to Azure Container Registry. For hosting, I'd use Azure Container Apps for simplicity, or AKS if we need fine-grained control. I've run Kubernetes with ArgoCD, so I'd keep GitOps. The LLM would be Azure OpenAI, documents in Blob Storage, and vectors in Azure AI Search with hybrid search. Secrets go in Key Vault with managed identity, so no keys in code. Same idea as the HashiCorp Vault setup I standardized. Monitoring through Application Insights and Azure Monitor, tracking latency, errors, token usage and cost. CI/CD with GitHub Actions or Azure DevOps."

### 3.4 Cloud concepts
- **IaaS / PaaS / SaaS**: VM / App Service / Office 365.
- **Autoscaling** (horizontal = more instances, vertical = a bigger machine). K8s **HPA**.
- **Managed Identity**: Azure resources authenticate to each other without passwords.
- **Regions & quotas**: Azure OpenAI has **TPM (tokens per minute) quotas** per deployment → handle **429s** with retry + backoff, and spread load across deployments.

---

## 4. Monitoring AI in Production ⭐ (JD: "monitor model performance, latency, reliability")

| Category | What to track |
|---|---|
| **Latency** | p50 / **p95 / p99** response time, time-to-first-token (streaming), per step (retrieval, LLM, tools) |
| **Reliability** | Error rate, timeouts, 429 rate limits, retries, uptime, pod restarts |
| **Cost** | Tokens in/out per request, cost per conversation, cache hit rate |
| **Quality** | User feedback 👍/👎, faithfulness scores on sampled traffic, "I don't know" rate, escalation rate |
| **Retrieval** | Hit rate, empty-retrieval rate, top similarity scores |
| **Safety** | PII detections, blocked prompts, guardrail triggers |
| **Classic ML** | Data drift, prediction drift, accuracy once labels arrive |

- **Tools:** LangSmith, **Langfuse**, Arize Phoenix, **OpenTelemetry** traces, Prometheus + Grafana, Azure App Insights.
- **Why p95 not the average?** The average hides slow outliers. p95 = "95% of users get a response faster than this".
- **Your stories here:** trace discovery and ingestion microservices (observability!), the monitoring dashboard (emails, approvals, errors, workflow status → no more SSH), Slack failure alerts on the Loxo pipeline.

---

## 5. ⭐⭐⭐ YOUR RESUME STORIES (STAR format)

> **S**ituation → **T**ask → **A**ction → **R**esult. Keep each to 60–90 seconds.
> ⚠️ **Fill the `[ ]` blanks with your real details tonight.** They will dig: "What was the latency before and after? Which models? How did you measure?" Only state numbers you can stand behind.

### Story 1: 40% chatbot latency reduction 🚀
- **S:** A multi-LLM chatbot where the **frontend** orchestrated the calls → many round trips between browser and backend/LLMs, slow responses.
- **T:** Reduce response latency.
- **A:** Moved orchestration to the **backend** (one request from the client, the server coordinates the LLM calls). Optimized prompts (shorter, cleaner) and query execution `[e.g. parallel calls with asyncio.gather? caching? fewer sequential calls? — fill in]`.
- **R:** **~40% lower latency** `[from X s to Y s, measured how — logs / p95?]`. Also more secure (API keys no longer on the frontend), easier to maintain.
- **Likely follow-ups:** Why was frontend orchestration slow? (network hops, sequential calls, the browser can't parallelize well / exposed keys). How did you measure? What would you do next? (streaming, semantic caching, smaller routing model).

### Story 2: Restaurant assistant, 50% system-prompt reduction 🍕
- **S:** A LangChain + FastAPI restaurant assistant. The system prompt was bloated with business rules (hours, ordering logic).
- **T:** Make it reliable and cheaper.
- **A:** Moved complex logic into **modular tools**: an `order-creation` tool and a `restaurant-hours validation` tool. The LLM decides *when* to call them, and the code enforces the rules deterministically.
- **R:** **50% smaller system prompt** → fewer tokens (cost), faster responses, and fewer hallucinations (hours come from code, not the LLM's memory). Easier to test and update.
- **Lesson to say:** *"LLMs for language, code for rules."* This is exactly how MCP thinks about tools.

### Story 3: Presidio PII latency 16–18s → 8–12s 🔒
- **S:** A PII classification service (Microsoft Presidio) for an enterprise AI-security client was taking 16–18 seconds.
- **T:** Find and fix the bottleneck.
- **A:** Found the **NLP model was being loaded on every request**. Moved it to load **once at application startup** and reused it. Also **restructured the classification prompt**.
- **R:** **8–12s**, roughly 40–50% faster.
- **Follow-ups:** How did you find it? (timing logs / profiling per step). What does Presidio do? (detects PII entities like names, emails, phone numbers and card numbers using NER models + regex recognizers, and can anonymize them). Further improvements? (a smaller spaCy model, batching, caching, async, running on a GPU).
- **Connection:** Same principle as Mind-Scope loading the CNN once at startup.

### Story 4: Rust decision engine + API-key auth 🛡️
- Integrated a Rust-based policy engine with the AI agent layer: **pre-response checks** (input PII/policy) and **post-response checks** (output) → guardrails. Implemented **API-key authentication** for agent-to-engine calls.
- **Talking point:** a defense-in-depth guardrail pattern. Relevant to "agent orchestration" and production safety.

### Story 5: Trace discovery & ingestion (observability) 📡
- Built microservices that discover agents using credentials and **ingest traces on a schedule** into the platform backend.
- **Talking point:** you understand **LLM observability**, i.e. why traces (spans per LLM/tool call, tokens, latency) matter for monitoring in production. *Fill in: what trace source/API, what format, how scheduled.*

### Story 6: ArgoCD / GitOps + 26 Docker images ⚙️
- Replaced manual `kubectl` rollouts with **ArgoCD GitOps** (Git = the source of truth, auto-sync, easy rollback, no need for client-machine access). Containerized and secured **26 images**, fixed failing pods, reduced vulnerability scores, standardized tags, scripts and **Vault** config.
- **Talking point:** "I can take a model from notebook to production on Kubernetes." That covers the JD's "deploy and manage AI workloads".

### Story 7: Gmail → Loxo ATS pipeline (15+ hrs/week saved) 📧
- Parsed candidate details from emails (an LLM/NLP extraction step), mapped applications to jobs, pushed them to the Loxo ATS. Added **Slack summaries + failure alerts**.
- **R:** **15+ hours/week** of manual data entry eliminated.
- **Talking point:** business ROI + reliability (alerts). Likely question: "How did you ensure extraction accuracy?" → *structured output schema (Pydantic), validation, fallback/alert on failure `[fill in]`.*

### Story 8: AI video automation (event-driven) 🎬
- Replaced **two cron jobs with an event-driven trigger**, so approved clips flow automatically: review → editing → Instagram. Migrated from a client laptop to a **VM**, added auto-cleanup of old clips in Drive, and built a **monitoring dashboard** (emails, approvals, errors, status) so nobody needs to SSH in.
- **Talking point:** cron vs event-driven (lower latency, no wasted polling, no missed or duplicate runs).

### "Biggest challenge?" / "A failure?"
Pick one (e.g. the Presidio latency or failing pods during the ArgoCD migration). Structure: the problem → how you debugged it → what you learned. Be honest. Show the learning.

---

## 6. HR / Behavioral Questions

| Question | Your angle |
|---|---|
| **Tell me about yourself** | Use the pitch in `00-strategy`: current role → 2–3 achievements with numbers → ML base (Mind-Scope) → why this role |
| **Why leave Quantal after ~1.5 years?** | Positive: "I want a role that combines GenAI engineering with core data science and ML modeling, which this JD is. I want to grow on the modeling side, not only the engineering side." Never criticize your current employer |
| **Why Veracitiz?** | *Look up Veracitiz's website tonight (services, clients, domains) and write down 2 specifics.* Then: "The JD lines up closely with what I do: RAG, agents, Flask APIs, and Azure, which I want to deepen." |
| **Strength** | Taking AI from prototype to production: performance, deployment, reliability (backed by numbers) |
| **Weakness** | "My classic ML modeling experience is more academic than production. I've been actively strengthening it [give a specific]." Honest and shows growth |
| **Where in 5 years?** | Senior AI/Data Scientist leading GenAI solutions end to end, mentoring others |
| **Expected CTC / notice period** | Know your current CTC, your expectation (research the market range) and your exact notice period **before** you walk in |
| **Questions for them** | "What does the AI team look like? What are the main client use cases? Which LLM stack and cloud do you use (Azure OpenAI)? How do you evaluate GenAI solutions in production?" |

---

## 7. 🎤 Mock Interview (do this tonight, out loud, timed)

Answer each in **≤ 2 minutes**. Record yourself on your phone if you can.

1. Tell me about yourself.
2. Walk me through your most complex project at Quantal.
3. How did you get 40% lower latency? How did you measure it?
4. Explain your RAG pipeline. Which vector DB and embedding model? Why?
5. How do you prevent hallucinations?
6. How would you evaluate a RAG system?
7. What is an agent? Agent vs workflow?
8. Explain MCP. How is it different from ACP/A2A?
9. Prompting vs RAG vs fine-tuning?
10. Explain LoRA.
11. Overfitting: what is it and how do you fix it? (Use your Mind-Scope example!)
12. Precision vs recall with a real example.
13. Random Forest vs XGBoost.
14. SQL on paper: top 2 salaries per department.
15. Flask vs FastAPI. How would you deploy Flask in production?
16. Deploy a GenAI app on Azure: walk me through it.
17. How do you monitor an LLM application in production?
18. Explain a CNN using your project.
19. Design a chatbot over our company's 10,000 internal PDFs.
20. Why should we hire you?

**"Why should we hire you?"**
> "Because most of this JD is what I already do in production: RAG, agentic systems with LangChain and LangGraph, tool calling, prompt optimization, Python APIs and containerized deployment, with measurable results like 40% lower latency. And I have the ML foundation to train and evaluate models, which I've shown with my CNN project. I can contribute from day one on the GenAI side while growing on the classic data science side."
