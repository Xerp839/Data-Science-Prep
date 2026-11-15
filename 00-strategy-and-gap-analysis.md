# 00 — Strategy & Gap Analysis (read this first)

## 1. What is this job really?

The title says **Data Scientist**, but read the responsibilities. About 70% of it is **GenAI / AI Engineer** work: RAG, agents, prompts, Flask APIs, MCP/ACP and Azure. The other 30% is classic Data Scientist work: SQL, ML/DL training and evaluation.

**So:** your work experience already covers the big part. Your risk is the "classic DS" 30%. Interviewers *will* check it because the title says Data Scientist.

## 2. JD vs your resume

| JD asks for | Your resume | Status | What to do |
|---|---|---|---|
| Python | Python, FastAPI, Flask | ✅ Strong | Revise core Python + pandas (Day 4) |
| SQL | MySQL listed | ⚠️ Medium | **Practice queries by hand** (Day 2) |
| REST APIs with **Flask** | Flask listed, but bullets say FastAPI | ⚠️ Medium | Know Flask basics + "Flask vs FastAPI" (Day 4) |
| Prompt engineering | ✅ 50% prompt reduction story | ✅ Strong | Learn the names of the techniques (Day 3) |
| Azure / cloud | Docker, K8s, ArgoCD, Vault, VM | ⚠️ Medium | Translate your skills into Azure terms (Day 4) |
| **ML & DL fundamentals** | ❌ Nothing on resume | 🔴 **Gap** | **Day 1 + Day 2** + Mind-Scope project |
| RAG + vector DBs | RAG, vector search listed | ✅ Good | Go deeper: chunking, hybrid search, eval (Day 3) |
| Agentic AI | LangChain, LangGraph, tool calling | ✅ Strong | Know the patterns by name (Day 3) |
| MCP / ACP | Not on resume | ⚠️ Gap | Easy to learn. Day 3 |
| Monitoring latency and reliability in prod | Trace ingestion, monitoring dashboard, latency wins | ✅ Strong | Use these as stories (Day 4) |
| Train / fine-tune / evaluate models | ❌ | 🔴 Gap | Mind-Scope CNN + LoRA theory (Day 2) |

## 3. What a walk-in usually looks like

Walk-ins in Navi Mumbai and Thane usually go like this (it varies by company):
1. **Registration**, sometimes followed by a **short written or online test**: Python output questions, SQL queries, ML MCQs.
2. **Technical round 1**: resume deep-dive, then ML basics and SQL.
3. **Technical round 2 / managerial**: system design ("design a RAG chatbot for our client"), agents, cloud.
4. **HR**: salary, notice period, why this company.

**Carry:** 3–4 printed resumes, ID proof, a pen, a notebook (for whiteboard-style SQL), and your laptop if possible to show a GitHub demo.

## 4. Your positioning (memorize this pitch)

> "I'm an AI Engineer with a little over a year of production experience at Quantal. I build LLM agents, RAG chatbots and Python backends. I've cut chatbot latency by about 40% and halved system-prompt size by moving logic into modular tools. I've also handled the deployment side: Docker, Kubernetes and ArgoCD. On the ML side, I built a CNN for brain tumor detection from MRI scans in college with about 91% test accuracy, so I'm comfortable with model training and evaluation too. I'm looking for a role where I can combine GenAI engineering with core data science, and this JD is exactly that mix."

That last part matters. Bringing in the CNN project directly answers "you have no ML on your resume" before they ask.

## 5. Should you bring up Mind-Scope (the CNN project)?

**Yes, bring it up.** It's your only hands-on DL training experience, and the JD literally says "train, fine-tune, and evaluate ML and DL models."

How to use it:
- **Don't lead with it.** Your Quantal work is stronger, so open with that.
- **Use it when** they ask: "Any ML/DL experience?", "Have you trained a model?", "Explain CNNs", "Explain overfitting" (you have a real example!) or "Precision vs recall?" (medical context = recall matters).
- **Consider adding one line to your printed resume** under a "Projects" section, for example:
  > *Mind-Scope — Brain Tumor Detection (Academic):* CNN in TensorFlow/Keras on MRI scans with OpenCV preprocessing (contour-based cropping); 91% test accuracy, 0.915 F1; deployed as a Streamlit app with PDF report generation.
- **Be honest about the weaknesses** (they're in the Mind-Scope file). Telling them what you'd improve makes you look *more* senior.
- The Mind-Scope file is the full explainer. Read it on Day 2.

## 6. Top 15 questions you're most likely to get

1. Walk me through your resume / current project.
2. Explain your RAG pipeline end to end. How did you reduce hallucination?
3. How did you reduce latency by 40%? *(Day 4 stories)*
4. What is overfitting and how do you prevent it?
5. Precision vs recall. When is each more important?
6. Random Forest vs XGBoost / bagging vs boosting.
7. SQL: 2nd highest salary, or top N per group, with a window function.
8. What is an agent? How is it different from a chain?
9. What is MCP? What is ACP / A2A?
10. Prompting vs RAG vs fine-tuning: when to use which?
11. How do you evaluate a RAG system?
12. Flask vs FastAPI. How would you deploy a Flask app?
13. How would you deploy this on Azure?
14. Explain how a CNN works / explain your college project.
15. How do you monitor an LLM app in production?

All of these are answered across the day files.
