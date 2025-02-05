import argparse
import json
import sqlite3

from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def init_db(conn: sqlite3.Connection):
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                hashed_password TEXT NOT NULL,
                role TEXT NOT NULL,
                disabled INTEGER NOT NULL
            )

            """
        )


def create_user(
    conn: sqlite3.Connection,
    username: str,
    password: str,
    role: str,
    disabled: bool,
):
    hashed_password = get_password_hash(password)
    try:
        with conn:
            conn.execute(
                "INSERT INTO user(username, hashed_password, role, disabled) VALUES (?, ?, ?, ?)",
                (username, hashed_password, role, int(disabled)),
            )
        print(f"User '{username}' created successfully.")
    except sqlite3.IntegrityError:
        print(f"User '{username}' already exists.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a new user in the SQLite user database."
    )
    parser.add_argument("username", help="Username for the new user")
    parser.add_argument("password", help="Password for the new user")
    parser.add_argument(
        "--role",
        default="",
        help="Leave empty for no role.",
    )
    parser.add_argument(
        "--disabled", action="store_true", help="Mark the user as disabled"
    )
    args = parser.parse_args()

    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row

    init_db(conn)

    # Create the user.
    create_user(conn, args.username, args.password, args.role, args.disabled)
