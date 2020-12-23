from math import cos
from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np
import recmetrics


class Analyser():
    def __init__(self, listens_df, cosine_results, pearson_results, pearson_baseline_results):
        self.index_limit = 25
        self.model_names = ['Cosine based CF', 'Pearson based CF', 'Pearson Baseline based CF']

        self.listens_df = listens_df

        self.cosine_results = self.convert_results_to_df(results=cosine_results)
        self.pearson_results = self.convert_results_to_df(results=pearson_results)
        self.pearson_baseline_results = self.convert_results_to_df(results=pearson_baseline_results)

        self.cosine_model = self.get_model(results=self.cosine_results)
        self.pearson_model = self.get_model(results=self.pearson_results)
        self.pearson_baseline_model = self.get_model(results=self.pearson_baseline_results)

    def convert_results_to_df(self, results):
        df = pd.DataFrame(results)
        df.drop("details", inplace=True, axis=1)
        df.columns = ['user_id', 'recording_id', 'scores', 'cf_predictions']
        return df

    def compare_mse(self):
        print("Cosine MSE: ", recmetrics.mse(self.cosine_results.scores, self.cosine_results.cf_predictions))
        print("Pearson MSE: ", recmetrics.mse(self.pearson_results.scores, self.pearson_results.cf_predictions))
        print("Pearson Baseline MSE: ", recmetrics.mse(
            self.pearson_baseline_results.scores, self.pearson_baseline_results.cf_predictions))

    def compare_rmse(self):
        print("Cosine RMSE: ", recmetrics.rmse(self.cosine_results.scores, self.cosine_results.cf_predictions))
        print("Pearson RMSE: ", recmetrics.rmse(self.pearson_results.scores, self.pearson_results.cf_predictions))
        print("Pearson Baseline RMSE: ", recmetrics.rmse(
            self.pearson_baseline_results.scores, self.pearson_baseline_results.cf_predictions))

    def get_model(self, results):
        model = results.pivot_table(index='user_id', columns='recording_id', values='cf_predictions').fillna(0)
        return model

    def get_users_predictions(self, user_id, n: int, model):
        recommended_items = pd.DataFrame(model.loc[user_id])
        recommended_items.columns = ["predicted_rating"]
        recommended_items = recommended_items.sort_values('predicted_rating', ascending=False)
        recommended_items = recommended_items.head(n)
        return recommended_items.index.tolist()

    def generate_predictions(self, cf_results, cf_model):
        cf_recs = [] = []
        # print(cf_results.index)
        for user in cf_results.index:
            cf_predictions = self. get_users_predictions(user, self.index_limit, cf_model)
            cf_recs.append(cf_predictions)
        return cf_recs

    def calculate_mark_scores(self, actual, predictions):
        mark = []
        for K in np.arange(1, self.index_limit+1):
            mark.extend([recmetrics.mark(actual, predictions, k=K)])
        return mark

    def compare_mark_scores(self):
        self.cosine_results = self.cosine_results.copy().groupby('user_id', as_index=False)[
            'recording_id'].agg({'scores': (lambda x: list(set(x)))})

        self.cosine_results = self.cosine_results.set_index("user_id")

        self.cosine_results['cf_predictions'] = self.generate_predictions(
            cf_results=self.cosine_results, cf_model=self.cosine_model)
        cosine_predictions = self.cosine_results.cf_predictions.values.tolist()
        cosine_actual = self.cosine_results.scores.values.tolist()

        cosine_mark_scores = self.calculate_mark_scores(actual=cosine_actual, predictions=cosine_predictions)

        self.pearson_results = self.pearson_results.copy().groupby('user_id', as_index=False)[
            'recording_id'].agg({'scores': (lambda x: list(set(x)))})
        self.pearson_results = self.pearson_results.set_index("user_id")

        self.pearson_results['cf_predictions'] = self.generate_predictions(
            cf_results=self.pearson_results, cf_model=self.pearson_model)
        pearson_predictions = self.pearson_results.cf_predictions.values.tolist()
        pearson_actual = self.pearson_results.scores.values.tolist()

        pearson_mark_scores = self.calculate_mark_scores(actual=pearson_actual, predictions=pearson_predictions)

        self.pearson_baseline_results = self.pearson_baseline_results.copy().groupby('user_id', as_index=False)[
            'recording_id'].agg({'scores': (lambda x: list(set(x)))})
        self.pearson_baseline_results = self.pearson_baseline_results.set_index("user_id")

        self.pearson_baseline_results['cf_predictions'] = self.generate_predictions(
            cf_results=self.pearson_baseline_results, cf_model=self.pearson_baseline_model)
        pearson_baseline_predictions = self.pearson_baseline_results.cf_predictions.values.tolist()
        pearson_baseline_actual = self.pearson_baseline_results.scores.values.tolist()

        pearson_baseline_mark_scores = self.calculate_mark_scores(
            actual=pearson_baseline_actual, predictions=pearson_baseline_predictions)

        mark_scores = [cosine_mark_scores, pearson_mark_scores, pearson_baseline_mark_scores]
        index = range(1, self.index_limit+1)
        fig = plt.figure(figsize=(15, 7))
        recmetrics.mark_plot(mark_scores, model_names=self.model_names, k_range=index)

    def compare_prediction_coverage(self, cosine_recs, pearson_recs, pearson_baseline_recs):
        catalog = self.listens_df.recording_id.unique().tolist()

        cosine_coverage = recmetrics.prediction_coverage(cosine_recs, catalog)
        pearson_coverage = recmetrics.prediction_coverage(pearson_recs, catalog)
        pearson_baseline_coverage = recmetrics.prediction_coverage(pearson_baseline_recs, catalog)

        coverage_scores = [cosine_coverage, pearson_coverage, pearson_baseline_coverage]
        fig = plt.figure(figsize=(7, 5))
        recmetrics.coverage_plot(coverage_scores, self.model_names)

    def compare_novelity(self, cosine_recs, pearson_recs, pearson_baseline_recs):
        nov = dict(self.listens_df.recording_id.value_counts())
        users = self.listens_df["user_id"].value_counts()
        cosine_novelity, _ = recmetrics.novelty(cosine_recs, nov, len(users), self.index_limit)
        pearson_novelity, _ = recmetrics.novelty(pearson_recs, nov, len(users), self.index_limit)
        pearson_baseline_novelity, _ = recmetrics.novelty(pearson_baseline_recs, nov, len(users), self.index_limit)

        print("Cosine Novelty:", cosine_novelity)
        print("Pearson Novelty:", pearson_novelity)
        print("Pearson Baseline Novelty:", pearson_baseline_novelity)

    def analyse(self):
        self.compare_mse()
        self.compare_rmse()
        self.compare_mark_scores()

        cosine_recs = self.generate_predictions(self.cosine_results, self.cosine_model)
        pearson_recs = self.generate_predictions(self.pearson_results, self.pearson_model)
        pearson_baseline_recs = self.generate_predictions(self.pearson_baseline_results, self.pearson_baseline_model)

        self.compare_prediction_coverage(cosine_recs=cosine_recs, pearson_recs=pearson_recs,
                                         pearson_baseline_recs=pearson_baseline_recs)
        self.compare_novelity(cosine_recs=cosine_recs, pearson_recs=pearson_recs,
                              pearson_baseline_recs=pearson_baseline_recs)
