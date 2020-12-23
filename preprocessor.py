import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Preporocessor():
    def __init__(self):
        self.LISTENS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'listens')

        self.listens_json = self.import_listens_form_json()
        self.listens_df = self.create_data_frame()

    def import_listens_form_json(self):
        listens_json = []
        for root, dirs, files in os.walk(self.LISTENS_DIR):
            for name in files:
                if name.endswith('json'):
                    listens_json.append(os.path.join(root, name))
        return listens_json

    def create_data_frame(self):
        dfs = [pd.read_json(f, lines=True).filter(['user_name', 'recording_msid', 'listened_at']) for f in self.listens_json]
        df = pd.concat(dfs, ignore_index=True)

        df = df.sample(frac=0.001)
        return df

    def calculate_listen_counts(self):
        self.listens_df['listen_count'] = self.listens_df.groupby(['user_name', 'recording_msid'])[
            'listened_at'].transform('count')

        # droo duplicate rows
        self.listens_df = self.listens_df.drop_duplicates(subset=['user_name', 'recording_msid', 'listen_count'])

    def normalize_listen_counts_to_scores(self):
        users = self.listens_df['user_name'].unique()

        for user in users:
            mms = MinMaxScaler(feature_range=(1, 5))
            self.listens_df.loc[self.listens_df['user_name'] == user, ['listen_count']] = mms.fit_transform(
                self.listens_df.loc[self.listens_df['user_name'] == user][['listen_count']])

        self.listens_df.rename(columns={"listen_count": "scores"})

    def convert_keys_to_ids(self):
        self.listens_df['user_id'] = pd.factorize(self.listens_df['user_name'])[0]
        self.listens_df['recording_id'] = pd.factorize(self.listens_df['recording_msid'])[0]

        self.listens_df.drop(columns=['user_name', 'recording_msid', 'listened_at'])

    def get_dataframe(self):
        self.calculate_listen_counts()
        self.normalize_listen_counts_to_scores()
        self.convert_keys_to_ids()

        return self.listens_df
