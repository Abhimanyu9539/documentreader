from pydantic import BaseModel, Field
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