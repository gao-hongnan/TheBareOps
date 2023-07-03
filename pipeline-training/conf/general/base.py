from pydantic import BaseModel


class General(BaseModel):
    pipeline_name: str = None
    seed: int = 1992
