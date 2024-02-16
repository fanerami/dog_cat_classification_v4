from dogCatClassification.entity.config_entity import DataIngestionConfig
import gdown
from dogCatClassification import logger
import zipfile

class DataIngestion:

    def __init__(self, config:DataIngestionConfig):
        self.config = config


    def download_data(self):

        try:

            key = self.config.source_URL.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='

            logger.info(f"Downloading data from {self.config.source_URL} into file {self.config.local_data_file}")

            gdown.download(prefix+key, self.config.local_data_file)

            logger.info(f"Downloaded data from {self.config.source_URL} into file {self.config.local_data_file}")

        except Exception as e:
            raise e

    def unzip(self) :
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
