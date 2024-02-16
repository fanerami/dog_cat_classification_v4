from dogCatClassification.utils.common import read_yaml, create_directories
from dogCatClassification.constants import (CONFIG_FILE_PATH,
                                     PARAMS_FILE_PATH)
from dogCatClassification.entity.config_entity import (DataIngestionConfig)

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
