import base64
import os
from typing import Annotated, Any

import httpx
from cachetools import TTLCache
from dotenv import load_dotenv
from fastapi import Depends, Header, HTTPException, status
from fastapi.security import OpenIdConnect
from jose import (
    exceptions as jose_exceptions,
    jwt,
)

from rag.models.user import User

_ = load_dotenv()

AZURE_CLIENT_ID = os.environ.get("AZURE_CLIENT_ID")
AZURE_TENANT_ID = os.environ.get("AZURE_TENANT_ID")
AZURE_DISCOVERY_URL = f"https://login.microsoftonline.com/{AZURE_TENANT_ID}/v2.0/.well-known/openid-configuration"

key_cache: TTLCache[str, dict[str, list[dict[str, Any]]]] = TTLCache(maxsize=1, ttl=3600)


def _raise_no_jwks_uri_exception() -> None:
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="No JWKS URI found in discovery document"
    )


async def get_azure_public_keys() -> dict[str, list[dict[str, Any]]]:
    cached_keys = key_cache.get("keys")
    if cached_keys:
        return cached_keys

    async with httpx.AsyncClient() as client:
        try:
            # 1. Get discovery document
            disovery_response = await client.get(AZURE_DISCOVERY_URL)
            _ = disovery_response.raise_for_status()
            discovery_data = disovery_response.json()
            jwks_uri = discovery_data.get("jwks_uri")
            if not jwks_uri:
                _raise_no_jwks_uri_exception()
            # 2. Get JWKS (public keys)
            jwks_response = await client.get(jwks_uri)
            _ = jwks_response.raise_for_status()
            jwks = jwks_response.json()
            key_cache["keys"] = jwks
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to fetch Azure keys: {e.response.text}",
            ) from e
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error fetching Azure keys: {e!s}"
            ) from e
        else:
            return jwks
    raise AssertionError("Should not be reached: Failed to get Azure public keys")


oauth2_scheme = OpenIdConnect(
    openIdConnectUrl=AZURE_DISCOVERY_URL, auto_error=False
)  # auto_error=False to customize error


def _validate_and_decode_id_token(
    token: str,
    jwks: dict[str, list[dict[str, Any]]],
    access_token_from_header: str | None,
    audience: str | None,
    credentials_exception: HTTPException,
) -> dict[str, Any]:
    try:
        unverified_header = jwt.get_unverified_header(token)
    except jose_exceptions.JWTError as e:
        raise credentials_exception from e

    rsa_key: dict[str, str] = {}
    for key in jwks.get("keys", []):
        if key.get("kid") == unverified_header.get("kid"):
            rsa_key = {
                "kty": key["kty"],
                "kid": key["kid"],
                "use": key["use"],
                "n": key["n"],
                "e": key["e"],
            }
            break
    if not rsa_key:
        raise credentials_exception

    if "at_hash" in jwt.get_unverified_claims(token) and not access_token_from_header:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-Access-Token header is missing for at_hash validation when at_hash claim is present in ID Token.",
        )

    # jwt.decode can raise ExpiredSignatureError, JWTClaimsError, JWTError
    payload = jwt.decode(
        token,
        rsa_key,
        algorithms=["RS256"],
        audience=audience,
        options={"verify_iss": False},  # Disable default issuer validation for common endpoint
        access_token=access_token_from_header,
    )

    # Manual issuer validation based on tid
    token_issuer = payload.get("iss")
    token_tid = payload.get("tid")

    if not token_issuer or not token_tid:
        raise credentials_exception  # Missing iss or tid claim

    expected_issuer = f"https://login.microsoftonline.com/{token_tid}/v2.0"
    if token_issuer != expected_issuer:
        raise credentials_exception  # Issuer does not match tenant ID

    return payload


async def _get_user_photo(token: str) -> str | None:
    """Fetches user's photo from MS Graph and returns it as a data URI."""
    graph_photo_url = "https://graph.microsoft.com/v1.0/me/photo/$value"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                graph_photo_url,
                headers={"Authorization": f"Bearer {token}"},
            )
            if response.status_code == 200:
                photo_data = await response.aread()
                content_type = response.headers.get("Content-Type", "image/jpeg")
                encoded_photo = base64.b64encode(photo_data).decode("utf-8")
                return f"data:{content_type};base64,{encoded_photo}"
        except httpx.RequestError:
            # Log this maybe, but for now just return None
            return None
        else:
            # If user has no photo, Graph API returns 404. We can ignore this.
            # We also ignore other errors to not fail the login process.
            return None


def _create_user_from_payload(payload: dict[str, Any], credentials_exception: HTTPException) -> User:
    user_id_val: str | None = payload.get("sub")
    email: str | None = payload.get("email")
    picture: str | None = payload.get("picture")
    username_val: str | None = payload.get("name")
    roles_val: list[str] = payload.get("roles", [])

    if user_id_val is None:
        raise credentials_exception
    user_id: str = user_id_val

    if username_val is None:
        raise credentials_exception
    username: str = username_val

    if roles_val == []:
        raise credentials_exception
    roles: list[str] = roles_val

    return User(
        id=user_id,
        email=email,
        picture=picture,
        username=username,
        organizations=roles,
    )


async def get_current_user(
    token: Annotated[str | None, Depends(oauth2_scheme)],
    access_token_from_header: Annotated[str | None, Header(alias="X-Access-Token")] = None,
) -> User:
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        jwks = await get_azure_public_keys()
        payload = _validate_and_decode_id_token(
            token,
            jwks,
            access_token_from_header,
            AZURE_CLIENT_ID,
            credentials_exception,
        )
        user = _create_user_from_payload(payload, credentials_exception)
        if not user.picture and access_token_from_header:
            user.picture = await _get_user_photo(access_token_from_header)
    except jose_exceptions.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None
    except jose_exceptions.JWTClaimsError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token claims: {e!s}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
    except jose_exceptions.JWTError as e:
        print(f"JWTError: {e!s}")
        raise credentials_exception from e
    except Exception as e:
        print(f"Unexpected error validating token: {e!s}")
        raise credentials_exception from e
    else:
        return user
