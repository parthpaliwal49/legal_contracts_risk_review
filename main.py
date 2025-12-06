# pip install sentence-transformers neo4j pandas tqdm
import pandas as pd
import sys
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import re
import dotenv
import os
from neo4j import GraphDatabase
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)

### CONFIG ###
load_status = dotenv.load_dotenv(".env")
if load_status is False:
    raise RuntimeError('Environment variables not loaded.')

CSV_PATH = "data/CUAD_v1/CUAD_v1/master_clauses.csv"
EMBED_MODEL = "BAAI/bge-large-en"
embedder = SentenceTransformer(EMBED_MODEL)

URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()
    print("Connection established.")
driver = GraphDatabase.driver(URI, auth=AUTH)


import google.generativeai as genai
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = genai.GenerativeModel("gemini-2.5-flash")

def build_context(hits, max_chars=8000):
    context = []
    used = 0
    for h in hits:
        chunk = (
            f"[CONTRACT: {h['contract_name']}]\n"
            f"[CLAUSE ID: {h['clause_id']}]\n"
            f"{h['clause_text']}\n\n"
        )
        if used + len(chunk) > max_chars:
            break
        context.append(chunk)
        used += len(chunk)
    return "".join(context)

def gemini_answer(question, hits):
    ctx = build_context(hits)
    prompt = f"""
        You are an expert legal contract analyst.
        Answer the question using ONLY the clauses below.
        If insufficient evidence exists, say so.
        
        QUESTION:
        {question}
        
        CLAUSE EVIDENCE:
        {ctx}
        
        Final answer (precise, legal tone):
    """
    resp = MODEL.generate_content(prompt)
    return resp.text




# Identify clause columns vs answer columns
def split_columns(columns):
    clause_cols = []
    answer_cols = []
    for col in columns:
        if col.endswith("-Answer"):
            answer_cols.append(col)
        else:
            clause_cols.append(col)
    return clause_cols, answer_cols

def normalize_id(name):
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


### NEO4J QUERIES ###
CREATE_CONTRACT = """
MERGE (c:Contract {id: $id})
ON CREATE SET c.name = $name
"""

CREATE_CLAUSE = """
MERGE (cl:Clause {id: $id})
SET cl.text = $text,
    cl.embedding = $embedding
"""

LINK_CLAUSE = """
MATCH (c:Contract {id: $cid}), (cl:Clause {id: $clid})
MERGE (c)-[:HAS_CLAUSE]->(cl)
"""

CREATE_CATEGORY = """
MERGE (cat:Category {name: $name})
"""

LINK_CATEGORY = """
MATCH (cl:Clause {id:$clid}), (cat:Category {name:$cat})
MERGE (cl)-[:BELONGS_TO]->(cat)
"""

SET_BOOLEAN = """
MATCH (cl:Clause {id:$clid})
SET cl.%s = $val
"""

CREATE_JURIS = """
MERGE (j:Jurisdiction {name: $name})
"""

LINK_JURIS = """
MATCH (cl:Clause {id:$clid}), (j:Jurisdiction {name:$jur})
MERGE (cl)-[:GOVERNED_BY]->(j)
"""

def ensure_vector_index():
    cypher = """
    CREATE VECTOR INDEX clause_vec_idx IF NOT EXISTS
    FOR (cl:Clause) ON (cl.embedding)
    OPTIONS {
      indexConfig: {
        `vector.dimensions`: 1024,
        `vector.similarity_function`: "cosine"
      }
    };
    """
    with driver.session() as sess:
        sess.run(cypher)
    print("Vector index ensured.")


def ingest():
    df = pd.read_csv(CSV_PATH)

    all_cols = df.columns.tolist()
    clause_cols = [c for c in all_cols if not c.endswith("-Answer") and c not in ["Filename", "Document Name"]]
    answer_cols = [c for c in all_cols if c.endswith("-Answer")]

    print("Embedding all clauses in batches...")
    clause_rows = []  # holds dicts for batch insert
    texts_to_embed = []

    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        contract = str(row["Document Name-Answer"]).strip()
        contract_id = normalize_id(contract)

        for col in clause_cols:
            clause_text = row[col]
            if isinstance(clause_text, str) and clause_text.strip() not in ["", "[]"]:
                clause_id = normalize_id(f"{contract_id}_{col}")
                texts_to_embed.append((clause_id, contract_id, col, clause_text))

    # --------- EMBEDDING BATCH ---------
    embeddings = []
    BATCH = 128  # GPU friendly

    for i in tqdm(range(0, len(texts_to_embed), BATCH), desc="Encoding"):
        batch = [t[3] for t in texts_to_embed[i:i + BATCH]]  # only texts
        vecs = embedder.encode(batch, convert_to_numpy=True)  # GPU accelerated
        embeddings.extend(vecs.tolist())

    print("Building rows for Neo4j...")
    for (cid, contract, cat, text), emb in zip(texts_to_embed, embeddings):
        clause_rows.append({
            "cid": contract,
            "clid": cid,
            "category": cat,
            "text": text,
            "embedding": emb
        })

    # --------- BULK WRITE ---------
    print("Writing to Neo4j...")

    query = """
    UNWIND $rows AS row
    MERGE (c:Contract {id: row.cid}) ON CREATE SET c.name = row.cid
    MERGE (cl:Clause {id: row.clid})
        SET cl.text = row.text,
            cl.embedding = row.embedding
    MERGE (cat:Category {name: row.category})
    MERGE (c)-[:HAS_CLAUSE]->(cl)
    MERGE (cl)-[:BELONGS_TO]->(cat)
    """

    with driver.session() as sess:
        for chunk in range(0, len(clause_rows), 500):
            sess.run(query, rows=clause_rows[chunk:chunk + 500])

    print("DONE - base graph imported.")

def enrich():
    df = pd.read_csv(CSV_PATH)
    all_cols = df.columns.tolist()

    clause_cols = [c for c in all_cols if not c.endswith("-Answer") and c not in ["Filename", "Document Name"]]
    answer_cols = [c for c in all_cols if c.endswith("-Answer")]

    bool_updates = []
    juris_links = []
    party_links = []

    for idx, row in df.iterrows():
        contract_name = str(row["Document Name-Answer"]).strip()
        cid = normalize_id(contract_name)

        for col in clause_cols:
            ans_col = f"{col}-Answer"
            if ans_col not in answer_cols:
                continue

            answer = row[ans_col]
            if not isinstance(answer, str) or answer.strip() == "" or answer.strip() == "[]":
                continue
            ans = answer.strip()

            clause_id = normalize_id(f"{cid}_{col}")

            # Boolean properties
            if ans.lower() in ["yes", "no"]:
                bool_updates.append({
                    "clid": clause_id,
                    "prop": normalize_id(col).lower(),
                    "val": True if ans.lower() == "yes" else False
                })

            # Jurisdiction enrichment
            if col == "Governing Law":
                juris_links.append({
                    "clid": clause_id,
                    "jur": ans
                })

            # Parties enrichment
            if col == "Parties":
                parties = [p.split("(")[0].strip() for p in ans.split(";") if p.strip()]
                for p in parties:
                    party_links.append({"cid": cid, "name": p})

    print("Writing booleans into graph...")
    query_bool = """
    UNWIND $rows AS row
    MATCH (cl:Clause {id: row.clid})
    SET cl[row.prop] = row.val
    """
    with driver.session() as sess:
        for chunk in range(0, len(bool_updates), 500):
            sess.run(query_bool, rows=bool_updates[chunk:chunk+500])

    print("Linking jurisdictions...")
    query_jur = """
    UNWIND $rows AS row
    MERGE (j:Jurisdiction {name: row.jur})
    WITH row, j
    MATCH (cl:Clause {id: row.clid})
    MERGE (cl)-[:GOVERNED_BY]->(j)
    """
    with driver.session() as sess:
        for chunk in range(0, len(juris_links), 500):
            sess.run(query_jur, rows=juris_links[chunk:chunk+500])

    print("Linking parties...")
    query_party = """
    UNWIND $rows AS row
    MERGE (p:Party {name: row.name})
    WITH row, p
    MATCH (c:Contract {id: row.cid})
    MERGE (c)-[:INVOLVES]->(p)
    """
    with driver.session() as sess:
        for chunk in range(0, len(party_links), 500):
            sess.run(query_party, rows=party_links[chunk:chunk+500])

    print("ENRICHMENT COMPLETE!")


def hybrid_search(
    query_text: str,
    top_k: int = 50,
    limit: int = 10,
    party: str | None = None,
    jurisdiction: str | None = None,
    category: str | None = None,
    flags: list[str] | None = None,
):
    """
    Hybrid vector + graph retrieval.
    flags -> list of boolean properties on Clause (e.g. ["uncapped_liability", "non_compete"])
    """

    # 1) Embed the query
    query_emb = embedder.encode(query_text).tolist()

    # 2) Build Cypher with optional filters
    cypher = """
    CALL db.index.vector.queryNodes('clause_vec_idx', $top_k, $embedding)
    YIELD node, score
    MATCH (c:Contract)-[:HAS_CLAUSE]->(node)
    OPTIONAL MATCH (node)-[:BELONGS_TO]->(cat:Category)
    OPTIONAL MATCH (node)-[:GOVERNED_BY]->(j:Jurisdiction)
    OPTIONAL MATCH (c)-[:INVOLVES]->(p:Party)
    WHERE ($category IS NULL OR cat.name = $category)
      AND ($jurisdiction IS NULL OR j.name = $jurisdiction)
      AND ($party IS NULL OR p.name = $party)
    """

    # flags like ["uncapped_liability", "non_compete"]
    if flags:
        for f in flags:
            cypher += f"\n      AND coalesce(node.{f}, false) = true"

    cypher += """
    RETURN
        node.id AS clause_id,
        node.text AS clause_text,
        c.id AS contract_id,
        c.name AS contract_name,
        collect(DISTINCT cat.name) AS categories,
        collect(DISTINCT j.name) AS jurisdictions,
        collect(DISTINCT p.name) AS parties,
        score
    ORDER BY score ASC
    LIMIT $limit
    """

    params = {
        "top_k": top_k,
        "embedding": query_emb,
        "category": category,
        "jurisdiction": jurisdiction,
        "party": party,
        "limit": limit,
    }

    with driver.session() as sess:
        res = sess.run(cypher, params)
        return [r.data() for r in res]



if __name__ == "__main__":

    mode = sys.argv[1] if len(sys.argv) > 1 else "search"

    if mode == "ingest":
        ingest()
        enrich()
    elif mode == "index":
        ensure_vector_index()
    elif mode == "search":
        ensure_vector_index()
        q = "Show me clauses where liability is uncapped and governed by Nevada law"
        results = hybrid_search(
            query_text=q,
            top_k=50,
            limit=5,
            jurisdiction="Nevada",
            flags=["uncapped_liability"]
        )
        for r in results:
            print("=" * 80)
            print("Contract:", r["contract_name"])
            print("Parties:", ", ".join([p for p in r["parties"] if p]))
            print("Jurisdictions:", ", ".join([j for j in r["jurisdictions"] if j]))
            print("Categories:", ", ".join([c for c in r["categories"] if c]))
            print("Score:", r["score"])
            print("Clause:\n", r["clause_text"])

    elif mode == "ask":
        ensure_vector_index()

        question = "Which contracts have uncapped liability for breach of confidentiality?"
        hits = hybrid_search(
            query_text=question,
            top_k=50,
            limit=5,
            flags=["uncapped_liability"]
        )

        print("Retrieved clauses:", len(hits))
        print("---- GEMINI ANSWER ----")
        print(gemini_answer(question, hits))

    else:
        print("Unknown mode. Use: ingest | index | search")
