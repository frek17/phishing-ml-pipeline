import logging
import pandas as pd
from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer
from sklearn.model_selection import train_test_split

from config import training_config


class TrainingPipeline:
    def __init__(self, config):
        self.config = config
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(filename='training.log', level=logging.INFO)
        logging.info('Training started')

    def _load_data(self):
        df = pd.read_csv(self.config.data_path)

        column_mapping = {
            self.config.label_column: 'label',
            self.config.text_column: 'text',
        }
        df.rename(columns=column_mapping, inplace=True)
        
        logging.info('Data loaded')
        return df

    def _load_model(self):
        model = SetFitModel.from_pretrained(self.config.base_model_id)
        model.to(self.config.device)
        logging.info(f'Pretrained model is loaded and on device {self.config.device}')
        return model

    def _prepare_datasets(self, df):
        df_train, df_test = train_test_split(df, test_size=self.config.test_size, random_state=42)
        df_train = Dataset.from_pandas(df_train)
        df_test = Dataset.from_pandas(df_test)
        return df_train, df_test

    def train_model(self):
        df = self._load_data()
        model = self._load_model()
        df_train, df_test = self._prepare_datasets(df)

        trainer = SetFitTrainer(
            model=model,
            train_dataset=df_train,
            eval_dataset=df_test,
            loss_class=CosineSimilarityLoss,
            **self.config.model_parameters
        )

        logging.info('Trainer is ready')
        trainer.train()
        final_evaluation = trainer.evaluate()
        logging.info(f'Final evaluation metrics: {final_evaluation}')

        model.save_pretrained(self.config.model_path)
        logging.info(f'Model Saved - {self.config.model_path}. Training completed')

if __name__ == "__main__":
    training_pipeline = TrainingPipeline(training_config)
    training_pipeline.train_model()
