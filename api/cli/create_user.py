import argparse
import sqlite3
from argparse import Namespace
from sqlite3 import Connection

import bcrypt


def get_password_hash(password: str) -> bytes:
    pwd_bytes: bytes = password.encode(encoding="utf-8")
    salt: bytes = bcrypt.gensalt()
    hashed_password: bytes = bcrypt.hashpw(password=pwd_bytes, salt=salt)
    return hashed_password


def init_db(conn: sqlite3.Connection):
    with conn:
        _ = conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                hashed_password BLOB NOT NULL,
                organization TEXT NOT NULL,
                disabled INTEGER NOT NULL
            )

            """
        )


def create_user(
    conn: sqlite3.Connection,
    username: str,
    password: str,
    organization: str,
    disabled: bool,
) -> None:
    hashed_password: bytes = get_password_hash(password)
    try:
        with conn:
            _ = conn.execute(
                "INSERT INTO user(username, hashed_password, organization, disabled) VALUES (?, ?, ?, ?)",
                (username, hashed_password, organization, int(disabled)),
            )
        print(f"User '{username}' created successfully.")
    except sqlite3.IntegrityError:
        print(f"User '{username}' already exists.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a new user in the SQLite user database."
    )
    _ = parser.add_argument("username", help="Username for the new user")
    _ = parser.add_argument("password", help="Password for the new user")
    _ = parser.add_argument(
        "--organization",
        default="",
        help="Leave empty for no organization.",
    )
    _ = parser.add_argument(
        "--disabled", action="store_true", help="Mark the user as disabled"
    )
    args: Namespace = parser.parse_args()

    conn: Connection = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row

    init_db(conn)

    # Create the user.
    create_user(conn, args.username, args.password, args.organization, args.disabled)
