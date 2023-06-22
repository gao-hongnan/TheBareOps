# TODO: Do you think we can instantiate Logger here and compose it within Config?
# Doing the below does not work well because log_root_dir depends on
# cfg.dirs.stores.logs and pydantic does not offer interpolation.

# from typing import Optional
# from pydantic import BaseModel, Field
# import logging


# class Logger(BaseModel):
#     log_file: Optional[str] = Field(
#         default="pipeline.log",
#         description="The name of the log file",
#     )
#     module_name: Optional[str] = Field(
#         default=__name__,
#         description="The name of the module where the logger is used",
#     )
#     level: int = Field(
#         default=logging.INFO,
#         description="The level of the logger",
#     )
#     propagate: bool = Field(
#         default=False,
#         description="Whether the logger should propagate the logs to its parent",
#     )
#     log_root_dir: Optional[str] = Field(
#         default=None,
#         description="The root directory where the log file should be saved",
#     )
