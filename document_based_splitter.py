# Used when texts are not a general english, hindi sentence. It can be code snippet, It can be a API documentation, it can be something like .md files, it can be python language code
#basically it uses recursive structure based but with different separater such as '\nclasss', '\ndef

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text="""
import asyncio
from dataclasses import dataclass
from typing import Optional


@dataclass
class User:
    id: int
    name: str
    email: Optional[str] = None


class UserService:

    def __init__(self):
        self.users = {}

    def add_user(self, user: User):
        

        if user.id in self.users:
            raise ValueError(
                f"User with id {user.id} already exists"
            )

        self.users[user.id] = user
        print(f"Added user: {user.name}")

    def get_user(self, user_id: int):
        return self.users.get(user_id)

    def delete_user(self, user_id: int):
        if user_id not in self.users:
            return False

        del self.users[user_id]
        return True


async def fetch_user_data(user_id: int):
    print(f"Fetching data for user {user_id}...")

    await asyncio.sleep(1)

    return {
        "id": user_id,
        "orders": [
            {"id": 101, "amount": 450},
            {"id": 102, "amount": 900},
            {"id": 103, "amount": 1200},
        ]
    }


def calculate_total(orders):

    total = 0

    for order in orders:
        if "amount" not in order:
            continue

        amount = order["amount"]

        if amount < 0:
            raise ValueError("Amount cannot be negative")

        total += amount

    return total


async def main():

    service = UserService()

    users = [
        User(1, "Alice", "alice@example.com"),
        User(2, "Bob"),
        User(3, "Charlie", "charlie@example.com"),
    ]

    for user in users:
        service.add_user(user)

    data = await fetch_user_data(1)

    total = calculate_total(data["orders"])

    print("User:", service.get_user(1))
    print("Orders:", data["orders"])
    print("Total:", total)


if __name__ == "__main__":
    asyncio.run(main())
    """
    
splitter= RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0   
)    

result= splitter.split_text(text)
print(result[0])