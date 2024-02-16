from dogCatClassification.config.configuration import ConfigurationManager
from dogCatClassification.components.training import Training

from dogCatClassification import logger

STAGE_NAME = "Training Model"


class TrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(training_config)
        training.get_update_base_model()
        training.train_valid_generator()
        training.training_model()


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = TrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
