from dogCatClassification.config.configuration import EvaluationConfig
import tensorflow as tf
import mlflow
import mlflow.keras
from dogCatClassification.utils.common import save_json
from pathlib import Path
from urllib.parse import urlparse
import os


class Evaluation:

    def __init__(self, config:EvaluationConfig) -> None:
        self.config = config


    def get_trained_model(self):
        self.model = tf.keras.models.load_model(
                self.config.path_of_model
            )


    def _valid_generator(self):


        datagenerator_kwargs = dict(
            rescale = 1./255
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.test_data,
            shuffle=False,
            **dataflow_kwargs
        )

    def evaluate(self):
        self.get_trained_model()
        self._valid_generator()
        print(os.listdir(self.config.test_data))
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):

        #recupérer toutes les mesures (à faire)
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("metrics.json"), data=scores)


    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]} #recupérer toutes les mesures (à faire)
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")
