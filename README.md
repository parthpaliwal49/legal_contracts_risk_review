# **Graph-Augmented RAG for Legal Contract Risk Intelligence**

### 🚀 A Neo4j + Gemini powered system for clause-level reasoning in commercial contracts

---

## **📌 Overview**

This project implements an **enterprise-grade Retrieval-Augmented Generation (RAG) system** that understands legal contracts — not as raw text, but as a **knowledge graph with semantics, constraints, and risk signals**.

We are using dataset from www.atticusprojectai.org
Dataset URL : https://www.atticusprojectai.org/cuad

Unlike traditional RAG pipelines that hallucinate or ignore business logic, this architecture combines:

* **Neo4j** — contract ontology and graph relationships
* **Sentence Transformers** — clause embeddings for semantic search
* **Vector Indexing** — native HNSW inside Neo4j
* **Gemini 2.5 Flash** — deterministic legal reasoning
* **CUAD Dataset** — 510 annotated real-world commercial contracts

This system can answer questions like:

> *“Which Nevada contracts have uncapped liability for IP infringement?”*

**accurately and with evidence.**

---

## **🧠 Why This Matters**

Standard RAG retrieves “similar text”.
Contract analysis requires:

* parties
* governing law
* rights & obligations
* boolean risk signals
* clause scope and categories

These are **graph-structured**, not blob text.

This repo shows how to build RAG that respects **enterprise constraints**, not vibes.

---

## **🛠️ Architecture**

```
CUAD Dataset
   ↓
Clause Extraction + Embeddings (bge-large-en)
   ↓
Neo4j Knowledge Graph:
 Contract — Clause — Category — Party — Jurisdiction
   ↓
Hybrid Retrieval Engine (Vector + Cypher filters)
   ↓
Gemini 2.5 Flash for grounded answers
```

---

## **✨ Key Features**

| Capability                                               | Status |
| -------------------------------------------------------- | ------ |
| Contract ingestion from CUAD                             | ✅      |
| GPU-based clause embeddings                              | ✅      |
| Neo4j graph schema with categories/jurisdictions/parties | ✅      |
| Native HNSW vector index                                 | ✅      |
| Hybrid vector + graph retrieval                          | ✅      |
| Gemini answer generation with evidence                   | ✅      |

---

## **📂 Folder Structure**

```
legal_contracts_risk_review/
├── ingest_cuad.py        # ingestion, indexing, hybrid search, Gemini RAG
├── data/CUAD_v1/         # CUAD dataset (not included — download separately)
├── .env                  # Neo4j + Gemini keys
└── README.md
```

---

## **⚙️ Installation**

```bash
git clone <your-repo-url>
cd legal_contracts_risk_review
python -m venv .venv
source .venv/Scripts/activate   # Windows
pip install -r requirements.txt
```

---

## **🔑 Environment Variables**

Create a `.env` file:

```
NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=yourpassword
GEMINI_API_KEY=yourkey
```

---

## **🚀 Usage**

### **1) Ingest CUAD dataset into Neo4j**

```bash
python ingest_cuad.py ingest
```

This:

* embeds all clauses (GPU accelerated)
* writes Contract / Clause / Category nodes
* enriches graph with Parties + Jurisdiction + Boolean risk signals

---

### **2) Create Vector Index**

```bash
python ingest_cuad.py index
```

---

### **3) Hybrid Search Query**

```bash
python ingest_cuad.py search
```

Example output:

```
Contract: VIDEO-ON-DEMAND CONTENT LICENSE AGREEMENT
Jurisdiction: Ontario, Canada
Clause:
IN NO EVENT SHALL COMPANY BE LIABLE...
```

---

### **4) Ask Gemini a Legal Question**

```bash
python ingest_cuad.py ask
```

Gemini receives only **validated clauses**, ensuring grounded answers — not hallucinations.

---

## **🔍 Core Query Example**

This is the line that differentiates this project from generic RAG:

```cypher
CALL db.index.vector.queryNodes('clause_vec_idx', $top_k, $embedding)
YIELD node, score
MATCH (c:Contract)-[:HAS_CLAUSE]->(node)
WHERE j.name = "Nevada" AND node.uncapped_liability = true
```

That’s semantic retrieval **plus** business rules — impossible with pure vector stores.

---

## **🧪 Models & Libraries**

| Component  | Choice              | Why                                         |
| ---------- | ------------------- | ------------------------------------------- |
| Embeddings | `BAAI/bge-large-en` | Top-tier semantic ranking for legal text    |
| Graph DB   | Neo4j               | Native vector + Cypher filtering            |
| LLM        | Gemini 2.5 Flash    | Cheap, fast, large context, grounded output |
| Dataset    | CUAD                | Real contracts, real risk categories        |

---

## **🧩 Why This Project Stands Out**

Most RAG repos are **chunk + embed + pray**.

Here, retrieval is:

✔ semantic
✔ structural
✔ explainable
✔ legally aware

That’s the difference between a chatbot and a **contract intelligence engine**.

---

## **📈 Next Enhancements**

* Risk scoring per contract
* Clause similarity audit for M&A diligence
* Fine-grained category-level suggestions
* FastAPI wrapper for SaaS deployment

---

## **🤝 Contributions**

PRs welcome — especially around:

* performance tuning
* additional risk categories
* integrating non-CUAD corpora

---

## **📜 License**

CUAD dataset is under **CC-BY 4.0**.
This codebase is MIT licensed.

---

## **⭐ If this helped you**

Consider starring the repo — it helps professionals discover real RAG engineering, not marketing hype.

---

## **👤 Author**

**Parth**
Data Engineer / Graph-RAG Practitioner
Building AI systems that reason — not hallucinate.

---
