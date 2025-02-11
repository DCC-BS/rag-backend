import argparse
from argparse import Namespace

from auth import get_password_hash
from models import User, create_db_and_tables, engine
from sqlmodel import Session, select


def create_user(
    username: str, password: str, organization: str, disabled: bool
) -> None:
    hashed_password = get_password_hash(password)
    with Session(engine) as session:
        existing_user = session.exec(
            select(User).where(User.username == username)
        ).first()
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
    parser = argparse.ArgumentParser(
        description="Create a new user in the SQLite user database."
    )
    parser.add_argument("username", help="Username for the new user")
    parser.add_argument("password", help="Password for the new user")
    parser.add_argument(
        "--organization",
        default="",
        help="Leave empty for no organization.",
    )
    parser.add_argument(
        "--disabled", action="store_true", help="Mark the user as disabled"
    )
    args: Namespace = parser.parse_args()

    create_db_and_tables()  # Ensure the tables exist before inserting
    create_user(args.username, args.password, args.organization, args.disabled)
