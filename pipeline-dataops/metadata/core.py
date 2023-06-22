from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict

import pandas as pd
import pytz

# TODO: Compose Metadata in a similar way as Config instead of scattering.


# pylint: disable=unnecessary-dunder-call
# mutable
@dataclass(frozen=False)
class Metadata:
    # tracks the inner state of the pipeline, update as it traverses the pipeline.
    # e.g. if the pipeline fails at a certain stage, we can use this to restart
    # from that stage.

    # general
    pipeline_name: str = None
    git_commit_hash: str = None

    # inside extract
    symbol: str = None
    interval: str = None
    limit: int = None
    start_time: int = None
    end_time: int = None
    base_url: str = None
    endpoint: str = None
    exported_at_utc: int = None

    raw_df: pd.DataFrame = None

    updated_at: datetime = field(
        default_factory=lambda: datetime.now(pytz.timezone("Asia/Singapore"))
    )

    transformed_df: pd.DataFrame = None

    def release(self, attribute: str) -> Any:
        self.__setattr__(attribute, None)

    def set_attrs(self, attr_dict: Dict[str, Any]):
        """Sets attributes on the Metadata instance.

        Parameters
        ----------
        attr_dict: Dict[str, Any]
            A dictionary where keys are attribute names and values are the
            corresponding values to be set.
        """
        for attr_name, value in attr_dict.items():
            self.__setattr__(attr_name, value)
