import os
import json
import logging
import asyncio
import argparse
from typing import List, Dict, Any
from openai import AsyncOpenAI
from dotenv import load_dotenv
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.data.vectordb_qdrant import VectorDBManager

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
DEFAULT_DATA_DIR = './scraped_data'
DEFAULT_BATCH_SIZE = 50
DEFAULT_QDRANT_HOST = "localhost"
DEFAULT_QDRANT_PORT = 6333

def construct_product_description(product_data: dict) -> str:
    """Constructs a description if missing, using title and manufacturer part number."""
    title = product_data.get('title', '')
    manufacturer_part_number = product_data.get('manufacturerPartNumber', '')
    description = f"{title} - Manufacturer Part Number: {manufacturer_part_number}"
    description = description.replace('\n', ' ').strip()
    if description == f" - Manufacturer Part Number: {manufacturer_part_number}":
        description = f"Part Number: {manufacturer_part_number}"
    elif description == " - Manufacturer Part Number: ":
         return "No description available"
    return description if description else "No description available"


async def load_directory_batch(
    data_dir: str,
    vector_db_manager: VectorDBManager,
    batch_size: int
):
    """
    Loads product data from JSON files in a directory in batches.
    (This function is compatible with the Qdrant manager's batch method)
    """
    if not os.path.isdir(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        return 0, 0

    logger.info(f"Starting data loading from directory: {data_dir}")
    logger.info(f"Using batch size: {batch_size}")

    product_batch: List[Dict[str, Any]] = []
    files_processed = 0
    products_submitted_count = 0

    for filename in os.listdir(data_dir):
        if filename.lower().endswith('.json'):
            file_path = os.path.join(data_dir, filename)
            files_processed += 1
            logger.debug(f"Processing file: {filename}")

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    product_data = json.load(f)

                part_number = product_data.get("partSelectNumber")
                if not part_number:
                    logger.warning(f"Skipping file {filename}: Missing 'partSelectNumber'.")
                    continue

                description = product_data.get("description")
                if not description or not description.strip():
                    logger.warning(f"File {filename} (Part: {part_number}): Missing or empty 'description'. Attempting to construct.")
                    constructed_description = construct_product_description(product_data)
                    if constructed_description == "No description available":
                        logger.error(f"Skipping file {filename} (Part: {part_number}): Could not construct a meaningful description.")
                        continue
                    else:
                        description = constructed_description
                        product_data["description"] = description
                        logger.info(f"File {filename} (Part: {part_number}): Using constructed description: '{description[:100]}...'")

                if not description or not description.strip():
                     logger.error(f"Skipping file {filename} (Part: {part_number}): Final description is still invalid after checks.")
                     continue

                product_batch.append(product_data)

                # Process batch if full
                if len(product_batch) >= batch_size:
                    logger.info(f"Processing batch of {len(product_batch)} products...")
                    try:
                        await vector_db_manager.add_products_batch(product_batch)
                        products_submitted_count += len(product_batch)
                    except Exception as batch_err:
                        logger.error(f"Error processing batch ending with file {filename}: {batch_err}")
                    product_batch = []

            except json.JSONDecodeError:
                logger.error(f"Skipping file {filename}: Invalid JSON.")
            except IOError as e:
                logger.error(f"Skipping file {filename}: Error reading file - {e}")
            except Exception as e:
                logger.exception(f"Skipping file {filename}: Unexpected error - {e}")

    # Process final batch
    if product_batch:
        logger.info(f"Processing final batch of {len(product_batch)} products...")
        try:
            await vector_db_manager.add_products_batch(product_batch)
            products_submitted_count += len(product_batch)
        except Exception as batch_err:
            logger.error(f"Error processing final batch: {batch_err}")

    logger.info(f"Finished loading data. Processed {files_processed} files.")
    logger.info(f"Total products submitted in batches: {products_submitted_count}. Check logs for embedding success/failure details.")
    return files_processed, products_submitted_count


async def main():
    """Main function to set up clients and run the data loading process for Qdrant."""
    parser = argparse.ArgumentParser(description="Load PartSelect product data into Qdrant.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing product JSON files (default: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--qdrant-host",
        type=str,
        default=DEFAULT_QDRANT_HOST,
        help=f"Qdrant server host (default: {DEFAULT_QDRANT_HOST})"
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=DEFAULT_QDRANT_PORT,
        help=f"Qdrant server port (default: {DEFAULT_QDRANT_PORT})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of products to add per batch (default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--clear-collection",
        action="store_true",
        help="Clear the existing Qdrant collection before loading data (USE WITH CAUTION!)."
    )

    args = parser.parse_args()

    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable not set.")
        return

    vector_db_manager = None
    try:
        logger.info("Initializing OpenAI client...")
        openai_client = AsyncOpenAI(api_key=openai_api_key)

        logger.info(f"Initializing VectorDBManager for Qdrant (connecting to {args.qdrant_host}:{args.qdrant_port})...")
        vector_db_manager = VectorDBManager(
            openai_client=openai_client,
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port
        )

        # --- Clear Collection if Flagged ---
        if args.clear_collection:
            collection_name = vector_db_manager.collection_name
            logger.warning(f"Attempting to clear Qdrant collection: {collection_name}")
            confirm = input(f"Are you sure you want to delete all data in Qdrant collection '{collection_name}'? (yes/no): ")
            if confirm.lower() == 'yes':
                try:
                    await vector_db_manager.clear_database()
                    logger.info(f"Qdrant collection '{collection_name}' cleared successfully.")
                    await vector_db_manager._ensure_collection()
                except Exception as clear_err:
                     logger.error(f"Failed to clear Qdrant collection: {clear_err}")
                     return 
            else:
                logger.info("Collection clearing cancelled.")
                return 
        else:
             await vector_db_manager._ensure_collection()


        # --- Run Loading Process ---
        files_processed, products_submitted = await load_directory_batch(
            data_dir=args.data_dir,
            vector_db_manager=vector_db_manager,
            batch_size=args.batch_size
        )

        logger.info(f"Data loading complete. Total files processed: {files_processed}.")
        logger.info(f"Total products submitted for addition: {products_submitted}. Check logs for details.")

    except Exception as e:
        logger.exception(f"An error occurred during the setup or loading process: {e}")
    finally:
        # --- Close Client Connection ---
        if vector_db_manager:
            logger.info("Closing Qdrant client connection...")
            await vector_db_manager.close()
            logger.info("Qdrant client connection closed.")


if __name__ == "__main__":
    asyncio.run(main())
