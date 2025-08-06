import hashlib
import hmac

from fastapi_azure_auth.user import User


def get_pseudonymized_user_id(user: User, secret_key: str) -> str:
    """
    Generates a consistent, one-way pseudonym for a given user ID.
    """
    user_id = user.oid
    if user_id is None:
        user_id = user.sub
    message = user_id.encode("utf-8")
    signature = hmac.new(secret_key.encode("utf-8"), message, hashlib.sha256).hexdigest()
    return signature
