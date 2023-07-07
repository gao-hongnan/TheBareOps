from common_utils.core.common import generate_uuid
from pydantic import BaseModel
from rich.pretty import pprint

from conf.clean.base import Cleaner
from conf.directory.base import ROOT_DIR, Directories
from conf.environment.base import Environment
from conf.extract.base import Extract
from conf.general.base import General
from conf.load.base import Load
from conf.resample.base import Resample

RUN_ID = generate_uuid()


# TODO: explore how to use model_post_init to make run_id inside Config
class Config(BaseModel):
    """Main configuration class that compose all the other configuration classes."""

    dirs: Directories = Directories.create_instance(ROOT_DIR, RUN_ID)
    env: Environment = Environment.create_instance()
    extract: Extract = Extract()
    load: Load = Load()
    clean: Cleaner = Cleaner()
    resample: Resample = Resample()
    general: General = General(pipeline_name="pipeline-training")


if __name__ == "__main__":
    config = Config()
    pprint(config)
    pprint(config.extract.from_api.model_dump())
