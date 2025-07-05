import xgboost as xgb
import pandas as pd

class MLRanker:
    def __init__(self):
        self.model = xgb.XGBRanker(
            objective='rank:pairwise',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4
        )

    def train(self, feature_df, relevance_labels, group_sizes):
        self.model.fit(feature_df, relevance_labels, group=group_sizes)

    def predict(self, feature_df):
        return self.model.predict(feature_df)

    def save(self, path="models/ml_ranker.json"):
        self.model.save_model(path)

    def load(self, path="models/ml_ranker.json"):
        self.model.load_model(path)
