import asyncio
import threading
import time

import bentoml
from bento_embeddings import BentoEmbeddings

emb = BentoEmbeddings(api_url="http://localhost:50001")


def embedding():
    test_data = ["aaaa" * 100] * 10_000
    start_time = time.time()
    emb.embed_documents(test_data)
    end_time = time.time()
    print(f"Embedding took: {end_time - start_time:.4f} seconds")


async def ranking_task(client, test_docs, test_query):
    start_time = time.time()
    await client.rerank(documents=test_docs, query=test_query)
    end_time = time.time()
    return end_time - start_time  # Return the execution time of each task


async def ranking():
    test_query = "How much is the fish?"
    test_docs = [
        "The fish is 10 dollars",
        "I dont know",
        "Der Fisch ist Gratis",
        "Ich esse Gemüse",
    ]

    async with bentoml.AsyncHTTPClient("http://localhost:50001") as client:
        if await client.is_ready():
            tasks = []
            for _ in range(100):  # Create 100 asynchronous tasks
                tasks.append(ranking_task(client, test_docs, test_query))

            start_time = time.time()
            execution_times = await asyncio.gather(
                *tasks
            )  # Run tasks and get individual times
            end_time = time.time()

            total_execution_time = sum(execution_times)
            print(
                f"Total execution time for all ranking calls: {total_execution_time:.4f} seconds"
            )
            print(
                f"Time from start to finish of all ranking calls: {end_time-start_time:.4f} seconds"
            )
        else:
            print("Client is not ready yet.")


async def main():
    embedding_thread = threading.Thread(target=embedding)
    embedding_thread.start()

    await ranking()  # Run the asynchronous ranking tasks

    embedding_thread.join()
    print("Both embedding and ranking tasks completed.")


if __name__ == "__main__":
    asyncio.run(main())
