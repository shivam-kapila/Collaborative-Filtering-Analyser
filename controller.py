from preprocessor import Preporocessor
from cf_algorithm import CFAlgorithm
from analyser import Analyser


class Controller():
    def __init__(self):
        self.listens_df = self. get_dataframe()
        self.cosine_results = self.run_cf_algorithm(sim_type="cosine")
        self.pearson_results = self.run_cf_algorithm(sim_type="pearson")
        self.pearson_baseline_results = self.run_cf_algorithm(sim_type="pearson_baseline")

    def get_dataframe(self):
        pre_processor = Preporocessor()
        df = pre_processor.get_dataframe()

        return df

    def run_cf_algorithm(self, sim_type: str):
        cf = CFAlgorithm(similarity_type=sim_type, listens_df=self.listens_df)
        results = cf.generate_predictions()
        return results

    def run_analyser(self):
        analyser = Analyser(listens_df=self.listens_df, cosine_results=self.cosine_results,
                            pearson_results=self.pearson_results, pearson_baseline_results=self.pearson_baseline_results)
        analyser.analyse()


if __name__ == "__main__":
    controller = Controller()
    controller.run_analyser()
