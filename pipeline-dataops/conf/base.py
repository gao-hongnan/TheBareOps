from conf.directory.base import ROOT_DIR, Directories
from conf.environment.base import Environment
from rich.pretty import pprint
from pydantic import BaseModel


class Config(BaseModel):
    directories: Directories
    environment: Environment


if __name__ == "__main__":
    config = Config(
        directories=Directories.create_instance(root_dir=ROOT_DIR),
        environment=Environment.create_instance(),
    )
    pprint(config)
