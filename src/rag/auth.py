import os
from datetime import UTC, datetime, timedelta
from typing import Annotated

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt.exceptions import InvalidTokenError
from pydantic import BaseModel
from sqlmodel import Session, select

from rag.models.user import User, get_session

SECRET_KEY = os.environ.get("JWT_SECRET_KEY") or "81e7ad176e1d9e40d3c467791f4b7b11b76700e9ad9f5e3d119f3537d11e4b46"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


def verify_password(plain_password: str, hashed_password: bytes) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password)


def get_password_hash(password: str) -> bytes:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt)


def get_user(username: str, session: Session) -> User | None:
    user = session.exec(select(User).where(User.username == username)).first()
    return user


def authenticate_user(username: str, password: str, session: Session):
    user = get_user(username, session)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict[str, str | datetime], expires_delta: timedelta | None = None) -> str:
    to_encode: dict[str, str | datetime] = data.copy()
    if expires_delta:
        expire_time = datetime.now(tz=UTC) + expires_delta
    else:
        expire_time = datetime.now(tz=UTC) + timedelta(minutes=15)
    to_encode.update({"exp": expire_time})
    encoded_jwt = jwt.encode(payload=to_encode, key=SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    session: Annotated[Session, Depends(get_session)],
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload: dict[str, str | datetime] = jwt.decode(jwt=token, key=SECRET_KEY, algorithms=[ALGORITHM])
        username: str | None | datetime = payload.get("username")
        if username is None or not isinstance(username, str):
            raise credentials_exception
        token_data = TokenData(username=username)
    except InvalidTokenError as err:
        raise credentials_exception from err
    if token_data.username is None:
        raise credentials_exception
    user = get_user(token_data.username, session)
    if user is None:
        raise credentials_exception
    return user
