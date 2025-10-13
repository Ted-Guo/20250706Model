# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 15:18:06 2025

@author: user
"""

import re;
from sentence_transformers import SentenceTransformer, util;
import numpy as np;

class QADetector:
    def __init__(self, use_embeddings=False):
        # 啟用多語言 embedding (可選)
        self.use_embeddings = use_embeddings;
        if use_embeddings:
            self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2");
            # 兩個 prototype 問句
            self.prototype_embs = self.model.encode([
                "This is a question.",
                "這是一個問題。"
            ], convert_to_tensor=True);

        # 多語言疑問詞
        self.qa_words = [
            # 英文
            r"\b(what|why|how|who|where|when|which)\b",
            # 中文
            r"(請問|為什麼|怎麼|是否|哪裡|誰|幾點|多少)",
            # 日文
            r"(なぜ|どうやって|どこ|誰|いつ|何)",
            # 韓文
            r"(왜|어떻게|어디|누구|언제)",
            # 西班牙文
            r"(qué|por qué|cómo|dónde|quién|cuándo|cuál)",
            # 法文
            r"(quoi|pourquoi|comment|où|qui|quand|lequel)",
            # 德文
            r"(was|warum|wie|wo|wer|wann|welche)",
            # 俄文
            r"(что|почему|как|где|кто|когда)"
        ];
        self.qa_regex = re.compile("|".join(self.qa_words), re.IGNORECASE);

    def is_question(self, text, threshold=0.45):
        if not isinstance(text, str):
            return 0;

        text = text.strip();

        # 1. 檢查是否有問號
        if text.endswith("?") or text.endswith("？"):
            return 1;

        # 2. regex 疑問詞匹配
        if self.qa_regex.search(text):
            return 1;

        # 3. (可選) embedding classifier
        if self.use_embeddings:
            emb = self.model.encode(text, convert_to_tensor=True);
            sim = util.cos_sim(emb, self.prototype_embs).max().item();
            return 1 if sim > threshold else 0;

        return 0
