import pandas as pd
import numpy as np
from itertools import product
from surprise import AlgoBase, Dataset as SurpriseDataset
from surprise import Reader
from surprise.model_selection import KFold
from surprise import accuracy

class HybridModel(AlgoBase):
    def __init__(self, user_data, item_data):
        AlgoBase.__init__(self)
        self.user_data = user_data.cleaned_data.set_index('user_id')
        self.item_data = item_data.cleaned_data.set_index('movie_id')
        self.trainset = None

    def fit(self, trainset):
        self.trainset = trainset
        return self

    def estimate(self, u, i):
        try:
            uid = self.trainset.to_raw_uid(u)
            iid = self.trainset.to_raw_iid(i)

            user_features = self.user_data.loc[uid] if uid in self.user_data.index else pd.Series()
            item_features = self.item_data.loc[iid] if iid in self.item_data.index else pd.Series()

            user_score = user_features.mean() if not user_features.empty else 0
            item_score = item_features.mean() if not item_features.empty else 0

            baseline = self.trainset.global_mean
            prediction = baseline + 0.5 * user_score + 0.5 * item_score

            return prediction
        except Exception:
            return self.trainset.global_mean

class ModelOptimizer:
    def __init__(self, rating_data, user_data, item_data):
        self.train_data = rating_data.get_train_data()
        self.test_data = rating_data.get_test_data()
        self.reader = Reader(rating_scale=(1, 5))
        self.user_data = user_data
        self.item_data = item_data
        self.model = None

    def train_model(self):
        self.model = HybridModel(self.user_data, self.item_data)
        trainset = SurpriseDataset.load_from_df(self.train_data[['user_id', 'item_id', 'rating']], self.reader).build_full_trainset()
        self.model.fit(trainset)

    def evaluate_model(self):
        testset = list(self.test_data[['user_id', 'item_id', 'rating']].itertuples(index=False, name=None))
        predictions = self.model.test(testset)
        rmse = accuracy.rmse(predictions)
        mae = accuracy.mae(predictions)
        return rmse, mae

    def cross_validate_model(self, k=5):
        data = SurpriseDataset.load_from_df(self.train_data[['user_id', 'item_id', 'rating']], self.reader)
        kf = KFold(n_splits=k, random_state=42, shuffle=True)

        rmses, maes = [], []

        for trainset, valset in kf.split(data):
            model = HybridModel(self.user_data, self.item_data)
            model.fit(trainset)
            predictions = model.test(valset)
            rmses.append(accuracy.rmse(predictions, verbose=False))
            maes.append(accuracy.mae(predictions, verbose=False))

        print(f"\nCross-Validation RMSE: {np.mean(rmses):.4f}")
        print(f"Cross-Validation MAE: {np.mean(maes):.4f}")
        return np.mean(rmses), np.mean(maes)

    def evaluate_on_test(self):
        testset = list(self.test_data[['user_id', 'item_id', 'rating']].itertuples(index=False, name=None))
        predictions = self.model.test(testset)
        rmse = accuracy.rmse(predictions)
        mae = accuracy.mae(predictions)
        print(f"\nFinal Evaluation on Test Set:")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test MAE: {mae:.4f}")
        return rmse, mae
