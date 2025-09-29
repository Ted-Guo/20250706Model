import sqlite3;
import faiss;
import numpy as np;
from sentence_transformers import SentenceTransformer;

embedding_dim = 384;
db_path = "wiki_chunks.db";
faiss_index_path = "wiki.index";
model_name = "all-MiniLM-L6-v2";

#載入資料庫
conn = sqlite3.connect(db_path);
cur = conn.cursor();

#載入 FAISS
index = faiss.read_index(faiss_index_path);

#啟動模型
model = SentenceTransformer(model_name)

#測試查詢
query = "What is Retrieval-Augmented Generation";
q_emb = model.encode([query]).astype("float32");

# 搜尋相似 chunk
top_k = 5;
distances, ids = index.search(q_emb, top_k);

print("查詢結果")
for i, (score, row_id) in enumerate(zip(distances[0], ids[0])):
    cur.execute("SELECT content FROM chunks WHERE id = ?", (int(row_id),));
    row = cur.fetchone();
    print(f"[{i+1}] ID: {row_id}, Score: {score:.4f}");
    print(f"Chunk: {row[0][:80]}...\n");

