from common_utils.core.common import generate_uuid
from pydantic import BaseModel
from rich.pretty import pprint

from conf.directory.base import ROOT_DIR, Directories
from conf.environment.base import Environment
from conf.extract.base import Extract
from conf.general.base import General
from conf.transform.base import Transform

RUN_ID = generate_uuid()


# TODO: explore how to use model_post_init to make run_id inside Config
class Config(BaseModel):
    dirs: Directories = Directories.create_instance(ROOT_DIR, RUN_ID)
    env: Environment = Environment.create_instance()
    extract: Extract = Extract()
    transform: Transform = Transform()
    general: General = General()


if __name__ == "__main__":
    config = Config()
    pprint(config)
    pprint(config.extract.from_api.model_dump())
