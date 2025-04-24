import argparse
from argparse import ArgumentParser, Namespace

from sqlmodel import Session, select

from rag.auth import get_password_hash
from rag.models.user import User, create_db_and_tables, engine


def create_user(username: str, password: str, organization: str, disabled: bool) -> None:
    hashed_password = get_password_hash(password)
    with Session(engine) as session:
        existing_user = session.exec(select(User).where(User.username == username)).first()
        if existing_user:
            print(f"User '{username}' already exists.")
            return
        user = User(
            username=username,
            hashed_password=hashed_password,
            organization=organization,
            disabled=disabled,
        )
        session.add(user)
        session.commit()
        print(f"User '{username}' created successfully.")


if __name__ == "__main__":
    parser: ArgumentParser = argparse.ArgumentParser(description="Create a new user in the SQLite user database.")
    _ = parser.add_argument("username", help="Username for the new user")
    _ = parser.add_argument("password", help="Password for the new user")
    _ = parser.add_argument(
        "--organization",
        default="",
        help="Leave empty for no organization.",
    )
    _ = parser.add_argument("--disabled", action="store_true", help="Mark the user as disabled")
    args: Namespace = parser.parse_args()

    create_db_and_tables()
    if args.username is None:
        raise ValueError("Username is required")
    if args.password is None:
        raise ValueError("Password is required")
    if args.organization is None:
        raise ValueError("Organization is required")
    if not isinstance(args.username, str):
        raise ValueError("Username must be a string")
    if not isinstance(args.password, str):
        raise ValueError("Password must be a string")
    if not isinstance(args.organization, str):
        raise ValueError("Organization must be a string")
    if not isinstance(args.disabled, bool):
        raise ValueError("Disabled must be a boolean")
    create_user(username=args.username, password=args.password, organization=args.organization, disabled=args.disabled)
