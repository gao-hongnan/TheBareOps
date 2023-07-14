import os
from typing import Any, Literal, Union

from common_utils.versioning.git.core import get_git_commit_hash
from pydantic import BaseModel, Field


class General(BaseModel):
    pipeline_name: str = "thebareops-pipeline-training"
    seed: int = 1992
    device: str = "cpu"  # TODO: add dynamic device generation

    git_commit_hash: Union[str, Literal["N/A"]] = Field(
        default=None, model_post_init=True
    )  # FIXME: How to use it properly like dataclass? Now I need set default=None

    def model_post_init(self, __context: Any) -> None:
        git_commit_hash = get_git_commit_hash()
        if git_commit_hash == "N/A":
            git_commit_hash = os.getenv("GIT_COMMIT_HASH", None)
            if git_commit_hash is None:
                raise ValueError(
                    """Git commit hash is 'N/A' and environment variable 'GIT_COMMIT_HASH'
                    is not set. Please set the 'GIT_COMMIT_HASH' environment variable
                    when running inside a Docker container."""
                )
        self.git_commit_hash = git_commit_hash
