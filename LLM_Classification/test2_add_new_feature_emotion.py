import pandas as pd;
import numpy as np;
import seaborn as sns;
import matplotlib.pyplot as plt;


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer;


# =============================================================================
# check emotion 
# =============================================================================
df_train = pd.read_csv("train.csv");
analyzer = SentimentIntensityAnalyzer();

def analyze_sentiments(text):
    return analyzer.polarity_scores(text);



sentiment_a_scores = df_train['response_a'].apply(analyze_sentiments).apply(pd.Series);
sentiment_a_scores = sentiment_a_scores.add_prefix("a_")

df_train = pd.concat([df_train, sentiment_a_scores], axis=1);


sentiment_b_scores = df_train['response_b'].apply(analyze_sentiments).apply(pd.Series);
sentiment_b_scores = sentiment_b_scores.add_prefix("b_")
df_train = pd.concat([df_train, sentiment_b_scores], axis=1);


# =============================================================================
# check positive emotion is a good feature?
# =============================================================================
df_train['a_pos_bin'] = pd.cut(df_train['a_pos'], bins=10);
# 建立資料表，每一區間下，A/B/TIE 各出現幾次
pivot = df_train.groupby('a_pos_bin')[['winner_model_a', 'winner_model_b', 'winner_tie']].sum();

# 轉為百分比（每行加總為1）
pivot_percent = pivot.div(pivot.sum(axis=1), axis=0);

sns.heatmap(pivot_percent, annot=True, cmap="YlGnBu", fmt=".2f");
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'];
plt.rcParams['axes.unicode_minus'] = False;
plt.title("a_pos vs Winner 類別（按比例）");
plt.ylabel("a_pos 分數區間");
plt.xlabel("Winner 類別");
plt.show();


df_train['a_neu_bin'] = pd.cut(df_train['a_neu'], bins=10);
pivot = df_train.groupby('a_neu_bin')[['winner_model_a', 'winner_model_b', 'winner_tie']].sum();
pivot_percent = pivot.div(pivot.sum(axis=1), axis=0);
sns.heatmap(pivot_percent, annot=True, cmap="YlGnBu", fmt=".2f");
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'];
plt.rcParams['axes.unicode_minus'] = False;
plt.title("a_neu vs Winner 類別（按比例）");
plt.ylabel("a_neu 分數區間");
plt.xlabel("Winner 類別");
plt.show();


df_train['a_compound_bin'] = pd.cut(df_train['a_compound'], bins=10);
pivot = df_train.groupby('a_compound_bin')[['winner_model_a', 'winner_model_b', 'winner_tie']].sum();
pivot_percent = pivot.div(pivot.sum(axis=1), axis=0);
sns.heatmap(pivot_percent, annot=True, cmap="YlGnBu", fmt=".2f");
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'];
plt.rcParams['axes.unicode_minus'] = False;
plt.title("a_compound vs Winner 類別（按比例）");
plt.ylabel("a_compound 分數區間");
plt.xlabel("Winner 類別");
plt.show();