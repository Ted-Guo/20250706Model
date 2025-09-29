from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss, sqlite3, numpy as np, os

class RAGIndexer:
    def __init__(self, data_dir,
                 sqlite_path="wiki_chunks.db",
                 faiss_path="wiki.index",
                 batch_size=128,
                 chunk_size=300,
                 overlap=50,
                 model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        self.data_dir = Path(data_dir);
        self.sqlite_path = sqlite_path;
        self.faiss_path = faiss_path;
        self.batch_size = batch_size;
        self.chunk_size = chunk_size;
        self.overlap = overlap;

        # 載入模型
        self.model = SentenceTransformer(model_name);
        self.embedding_dim = self.model.get_sentence_embedding_dimension();

        # 建立資料庫
        self.conn, self.cur = self._create_DB();

        # 建立或載入 FAISS
        self._create_FAISS();

    # =========================
    # SQLite
    # =========================
    def _create_DB(self):
        conn = sqlite3.connect(self.sqlite_path);
        cur = conn.cursor();
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT
            )
        """);
        conn.commit();
        return conn, cur;

    # =========================
    # FAISS
    # =========================
    def _create_FAISS(self):
        base_index = faiss.IndexFlatL2(self.embedding_dim);
        if os.path.exists(self.faiss_path):
            self.index = faiss.read_index(self.faiss_path);
            print(f"已讀取 FAISS index, size: {self.index.ntotal}");
        else:
            self.index = faiss.IndexIDMap(base_index);

    # =========================
    # 文本切分
    # =========================
    def _chunk_words(self, text):
        words = text.split();
        for i in range(0, len(words), self.chunk_size - self.overlap):
            yield " ".join(words[i:i + self.chunk_size]);

    def _read_chunks(self, file_object, chunk_bytes=1024*1024):
        while True:
            data = file_object.read(chunk_bytes);
            if not data:
                break;
            yield data;

    # =========================
    # 建立向量索引
    # =========================
    def build_index(self, file_name):
        file_path = self.data_dir / file_name;
        self.cur.execute("SELECT MAX(id) FROM chunks");
        last_id = self.cur.fetchone()[0] or 0;

        pos = 0;
        with open(file_path, "r", encoding="utf-8") as f:
            for block in self._read_chunks(f):
                batch_chunks, batch_ids = [], [];
                for chunk in self._chunk_words(block):
                    # 跳過已處理 chunk
                    if pos + 1 <= last_id:
                        pos += 1;
                        continue;

                    # 存 SQLite
                    self.cur.execute("INSERT INTO chunks (content) VALUES (?)", (chunk,));
                    row_id = self.cur.lastrowid;
                    batch_chunks.append(chunk);
                    batch_ids.append(row_id);
                    self.conn.commit();  # commit 每批

                    # 批量加入 FAISS
                    if len(batch_chunks) >= self.batch_size:
                        self._add_to_index(batch_chunks, batch_ids);
                        batch_chunks, batch_ids = [], [];

                    pos += 1;

                # 處理最後不足 batch 的
                if batch_chunks:
                    self._add_to_index(batch_chunks, batch_ids);

        print(f"=== {file_name} 建檔完成 ===");

    def _add_to_index(self, chunks, ids):
        embs = self.model.encode(chunks, show_progress_bar=False);
        self.index.add_with_ids(np.array(embs, dtype="float32"),
                                np.array(ids, dtype=np.int64));

    # =========================
    # 儲存 FAISS
    # =========================
    def save_index(self):
        faiss.write_index(self.index, self.faiss_path);
        print(f"FAISS index 已儲存, size: {self.index.ntotal}");

    # =========================
    # 關閉資料庫
    # =========================
    def close(self):
        self.conn.close();
