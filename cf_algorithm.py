from surprise import Dataset, KNNBasic
from surprise.reader import Reader
from surprise.model_selection import train_test_split


class CFAlgorithm():
    def __init__(self, similarity_type: str, listens_df):
        self.similarity_options = {
            "name": similarity_type,
            "user_based": False,  # Compute  similarities between items
        }

        self.cf_algo = KNNBasic(sim_options=self.similarity_options)
        self.trainset, self.testset = self.read_dataframe(listens_df=listens_df)

    def read_dataframe(self, listens_df):
        reader = Reader(rating_scale=(1, 5))
        listens = Dataset.load_from_df(listens_df[['user_id', 'recording_id', 'listen_count']], reader)
        trainset, testset = train_test_split(listens, test_size=.25)

        return trainset, testset

    def train_algorithm(self):
        self.cf_algo.fit(self.trainset)

    def generate_predictions(self):
        self.train_algorithm()
        predictions = self.cf_algo.test(self.testset)
        return predictions
