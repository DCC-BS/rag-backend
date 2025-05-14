from pydantic import BaseModel


class User(BaseModel):
    id: str  # 'sub' claim
    username: str
    email: str | None = None
    picture: str | None = None
    organization: str
