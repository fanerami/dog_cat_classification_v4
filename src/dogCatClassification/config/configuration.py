from dogCatClassification.utils.common import read_yaml, create_directories
from dogCatClassification.constants import (CONFIG_FILE_PATH,
                                     PARAMS_FILE_PATH)
from dogCatClassification.entity.config_entity import (DataIngestionConfig,
                                                       PrepareBaseModelConfig,
                                                       TrainingConfig,
                                                       EvaluationConfig)

import os
class ConfigurationManager:

    def __init__(self):
        self.config = read_yaml(CONFIG_FILE_PATH)
        self.params = read_yaml(PARAMS_FILE_PATH)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self):

        create_directories([self.config.data_ingestion.root_dir])

        data_ingestion = DataIngestionConfig(root_dir=self.config.data_ingestion.root_dir,
            source_URL=self.config.data_ingestion.source_URL,
            local_data_file=self.config.data_ingestion.local_data_file,
            unzip_dir=self.config.data_ingestion.unzip_dir)

        return data_ingestion


    def get_prepare_base_model_config(self):
        create_directories([self.config.prepare_base_model.root_dir])
        return PrepareBaseModelConfig(
            root_dir= self.config.prepare_base_model.root_dir,
            base_model_path = self.config.prepare_base_model.base_model_path,
            updated_base_model_path = self.config.prepare_base_model.updated_base_model_path,
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
            params_outptut_activation_function=self.params.OUTPTUT_ACTIVATION_FUNCTION,
            params_metrics=self.params.METRICS
        )


    def get_training_config(self):

        create_directories([self.config.training.root_dir])


        return TrainingConfig(
            root_dir=self.config.training.root_dir,
            trained_model_path=self.config.training.trained_model_path,
            updated_base_model_path=self.config.prepare_base_model.updated_base_model_path,
            training_data=self.config.data_ingestion.training_data,
            params_epochs=self.params.EPOCHS,
            params_batch_size=self.params.BATCH_SIZE,
            params_is_augmentation=self.params.AUGMENTATION,
            params_image_size=self.params.IMAGE_SIZE

        )



    def get_evaluation_config(self):


        return EvaluationConfig(
            path_of_model=self.config.training.trained_model_path,
            test_data=self.config.data_ingestion.test_data,
            mlflow_uri=self.config.evaluation.mlfow_tracking_uri,
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )

