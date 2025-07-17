import pandas as pd;
import numpy as np;

from sentence_transformers import SentenceTransformer;


class BertPreprocessor:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name);

    def clean_text(self, text):
        if pd.isnull(text) or not isinstance(text, str) or text.strip() == "":
            return "";
        text = text.replace('\n', ' ').replace('\r', ' ').strip();
        return text;


    def combine_prompt_response(self, prompt, response):
        return f"Question: {prompt} Answer: {response}";


    def get_label(self,row):
        if row['winner_model_a'] == 1:
            return 'A';
        elif row['winner_model_b'] == 1:
            return 'B';
        else:
            return 'tie';

    def transform(self, df):
        # 清理文字欄位
        for col in ['prompt', 'response_a', 'response_b']:
            df[col] = df[col].apply(self.clean_text);

        # 合併成模型輸入格式
        df['input_a'] = df.apply(lambda row: self.combine_prompt_response(row['prompt'], row['response_a']), axis=1);
        df['input_b'] = df.apply(lambda row: self.combine_prompt_response(row['prompt'], row['response_b']), axis=1);

        # 轉換成向量
        X_a = self.model.encode(df['input_a'].tolist(), show_progress_bar=True);
        X_b = self.model.encode(df['input_b'].tolist(), show_progress_bar=True);
        X_diff = X_a - X_b;

        # 轉換標籤
        y = df['label'] = df.apply(self.get_label, axis=1).values;

        return X_a, X_b, X_diff, y;


