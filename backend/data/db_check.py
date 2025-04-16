# minimal_chroma_test.py
import chromadb
import logging
import os
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_DB_PATH = './chroma_db_test'
COLLECTION_NAME = "test_collection"

try:
    logger.info(f"Ensuring directory exists: {CHROMA_DB_PATH}")
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)

    logger.info("Initializing Chroma client...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    logger.info(f"Getting or creating collection: {COLLECTION_NAME}")
    collection = client.get_or_create_collection(COLLECTION_NAME)

    test_id = "TEST001"
    test_embedding = list(np.random.rand(1536)) # Generate a random embedding
    test_metadata = {"name": "Test Item", "price": 10.99, "valid": True}
    test_document = "This is the document for the test item."

    logger.info(f"Attempting to add item with ID: {test_id}")
    logger.info(f"Metadata: {test_metadata}")

    collection.add(
        ids=[test_id],
        embeddings=[test_embedding],
        metadatas=[test_metadata],
        documents=[test_document]
    )

    logger.info(f"Successfully added item with ID: {test_id}")

    # Verify
    count = collection.count()
    logger.info(f"Collection count: {count}")
    results = collection.get(ids=[test_id])
    logger.info(f"Retrieved item: {results}")

except Exception as e:
    logger.exception("An error occurred during the minimal test:")

finally:
    # Optional: Clean up the test directory
    # import shutil
    # if os.path.exists(CHROMA_DB_PATH):
    #     logger.info(f"Cleaning up test directory: {CHROMA_DB_PATH}")
    #     shutil.rmtree(CHROMA_DB_PATH)
    pass
