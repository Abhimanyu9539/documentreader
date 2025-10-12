from pydantic import BaseModel, Field, RootModel
from typing import List, Union


class Metadata(BaseModel):
    Summary : List[str] = Field(default_factory=list, description="Summary of the document")
    Title : str
    Author : str
    DateCreated : str
    LastModified : str
    Publisher : str
    PageCount : Union[int, str]
    Language : str
    SentimentTone : str


class ChangeFormat(BaseModel):
    Page: str
    change: str

class SummaryResponse(RootModel[list[ChangeFormat]]):
    pass