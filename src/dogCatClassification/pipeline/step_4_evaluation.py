from dogCatClassification.config.configuration import ConfigurationManager
from dogCatClassification.components.evaluation import Evaluation
from dogCatClassification import logger

STAGE_NAME =  "Model Evaluation"

class EvaluationPipeline:

    def __init__(self) -> None:
        pass


    def main(self):
        config = ConfigurationManager()
        evaluation_config = config.get_evaluation_config()
        evaluation = Evaluation(evaluation_config)
        evaluation.evaluate()
        evaluation.log_into_mlflow()


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
