import pandas as pd;
import numpy as np;
import torch;

from sentence_transformers import SentenceTransformer;
from sklearn.decomposition import PCA;

class Pre_sentence_trans:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name);
    
    def clean_text(self, text):
        if pd.isnull(text) or not isinstance(text, str) or text.strip() == "":
            return "";
        text = text.replace('\n', ' ').replace('\r', ' ').strip();
        return text;

    def combine_prompt_response(self, prompt, response):
      return f"Question: {prompt} Answer: {response}";
  
    def get_label(self, row):
       if row['winner_model_a'] == 1:
           return 'A';
       elif row['winner_model_b'] == 1:
           return 'B';
       else:
           return 'tie';
  
    def pca(self,n_com,X_a,X_b,X_q):

        all_vecs_np = np.vstack([X_q,X_a, X_b]);
        pca = PCA(n_components = n_com);
        all_pca_np = pca.fit_transform(all_vecs_np);

        N = len(X_a);
        q_pca = all_pca_np[:N];
        a_pca = all_pca_np[N:2*N];
        b_pca = all_pca_np[2*N:];
        
        return a_pca,b_pca,q_pca;
        
    def cosine_batch(self,u, v):
        u_norm = torch.nn.functional.normalize(u, dim=1);
        v_norm = torch.nn.functional.normalize(v, dim=1);

        return (u_norm * v_norm).sum(dim=1).cpu().numpy();

    def fit(self, df):

       # 清理文字欄位
       for col in ['prompt', 'response_a', 'response_b']:
           df[col] = df[col].apply(self.clean_text);


       #get semantic similarity
       prompt_encode = torch.tensor(
           self.model.encode(df['prompt'].tolist(), show_progress_bar=True),
           dtype=torch.float32
       );

       response_a_encode = torch.tensor(
           self.model.encode(df['response_a'].tolist(), show_progress_bar=True),
           dtype=torch.float32
       );

       response_b_encode = torch.tensor(
           self.model.encode(df['response_b'].tolist(), show_progress_bar=True),
           dtype=torch.float32
       );
       cos_a = self.cosine_batch(prompt_encode,response_a_encode);
       cos_b = self.cosine_batch(prompt_encode,response_b_encode);

       #PCA for response_a_encode,response_b_encode
       a_pca_npy,b_pca_npy,q_pca_npy = self.pca(128,response_a_encode,response_b_encode,prompt_encode);
    

       # 建 input_a, input_b
       df['input_a'] = df.apply(lambda row: self.combine_prompt_response(row['prompt'], row['response_a']), axis=1);
       df['input_b'] = df.apply(lambda row: self.combine_prompt_response(row['prompt'], row['response_b']), axis=1);

       # prompt+response Embeddings
       X_a = self.model.encode(df['input_a'].tolist(), show_progress_bar=True);
       X_b = self.model.encode(df['input_b'].tolist(), show_progress_bar=True);
       X_diff = X_a - X_b;

       # 對稱特徵增強
       X_abs_diff = np.abs(X_a - X_b);
       X_prod = X_a * X_b;  # 元素相乘

       df_embed = pd.concat([
           df['id'],
           pd.DataFrame(X_a, columns=[f'emb_prompt_a_{i}' for i in range(X_a.shape[1])]),
           pd.DataFrame(X_b, columns=[f'emb_prompt_b_{i}' for i in range(X_b.shape[1])]),
           pd.DataFrame(X_diff, columns=[f'emb_diff_{i}' for i in range(X_diff.shape[1])]),
           pd.DataFrame(X_abs_diff, columns=[f'emb_abs_diff_{i}' for i in range(X_abs_diff.shape[1])]),
           pd.DataFrame(X_prod, columns=[f'emb_prod_{i}' for i in range(X_prod.shape[1])]),
           pd.DataFrame({'cos_prompt_a': cos_a, 'cos_prompt_b': cos_b}),
           pd.DataFrame(q_pca_npy, columns=[f'emb_prompt_pca_{i}' for i in range(q_pca_npy.shape[1])]),
           pd.DataFrame(a_pca_npy, columns=[f'emb_a_pca_{i}' for i in range(a_pca_npy.shape[1])]),
           pd.DataFrame(b_pca_npy, columns=[f'emb_b_pca_{i}' for i in range(b_pca_npy.shape[1])])
       ], axis=1);


       print("df_embed before merge:", df_embed.shape)
       


       # Label
       df['label'] = df.apply(self.get_label, axis=1);
       df_label = df[['id', 'label']];

       # Merge
       df_final = df[['id', 'prompt', 'response_a', 'response_b']].copy();
       print("df_final:", df_final.shape)
       
       df_final = df_final.merge(df_embed, on='id');
       df_final = df_final.merge(df_label, on='id');


       print("df_final after merge:", df_final.shape)

       # 分開特徵與標籤
       X = df_final.drop(columns=['id']);
       y = df_final['label'];

       return df_final, y;
  
