from rag_indexer import RAGIndexer;



indexer = RAGIndexer("emmermarcell");

# 增量新增一個或多個 txt
indexer.build_index("wikipedia_processed_1.txt");

# 存 FAISS
indexer.save_index();

# 關閉資料庫
indexer.close();
