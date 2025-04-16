import logging
import json
from typing import List, Dict, Any, Optional, Tuple
import qdrant_client
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from openai import AsyncOpenAI
import asyncio

logger = logging.getLogger(__name__)

# --- Constants ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "product_information_qdrant"
EMBEDDING_DIM = 1536 # Assuming text-embedding-ada-002
EMBEDDING_MODEL = "text-embedding-ada-002"

class VectorDBManager:
    """Manages interactions with the Qdrant vector database"""

    def __init__(self, openai_client: AsyncOpenAI, qdrant_host: str = QDRANT_HOST, qdrant_port: int = QDRANT_PORT):
        self.client = AsyncQdrantClient(host=qdrant_host, port=qdrant_port)
        self.openai_client = openai_client
        self.collection_name = COLLECTION_NAME


    async def _ensure_collection(self):
        """Ensure the collection exists in Qdrant"""
        try:
            collections = await self.client.get_collections()
            if self.collection_name not in [c.name for c in collections.collections]:
                logger.info(f"Collection '{self.collection_name}' not found. Creating...")
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
                )
                logger.info(f"Created new collection: {self.collection_name}")
            else:
                logger.info(f"Using existing collection: {self.collection_name}")
        except Exception as e:
            logger.exception(f"Error ensuring Qdrant collection '{self.collection_name}': {e}")
            raise

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text using OpenAI"""
        if not text:
            logger.warning("Attempted to generate embedding for empty text.")
            return [0.0] * EMBEDDING_DIM

        try:
            response = await self.openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.exception(f"Error generating OpenAI embedding for text snippet '{text[:100]}...': {str(e)}")
            raise

    def _extract_payload(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts filterable fields and the full document for Qdrant payload.
        Ensures values are compatible types (str, int, float, bool, list[str|int|float|bool]).
        """
        payload = {"type": "product"}
        product_id_for_log = product_data.get('partSelectNumber', product_data.get('manufacturerPartNumber', 'N/A'))

        key_mapping = {
            "partSelectNumber": "part_number",
            "manufacturerPartNumber": "manufacturer_part_number",
            "stockStatus": "availability"
        }
        numeric_key_mapping = { "price": "price" }

        for json_key, meta_key in key_mapping.items():
            if json_key in product_data:
                value = product_data[json_key]
                if isinstance(value, (str, bool, int, float)):
                    if meta_key in ["part_number", "manufacturer_part_number"] and isinstance(value, str):
                        payload[meta_key] = value.upper()
                    else:
                        payload[meta_key] = value
                else:
                    logger.warning(f"Product ID {product_id_for_log}: Skipping payload key '{meta_key}' (from JSON key '{json_key}') due to incompatible type ({type(value)}). Value: {str(value)[:50]}...")

        for json_key, meta_key in numeric_key_mapping.items():
            if json_key in product_data:
                value_str = product_data[json_key]
                try:
                    payload[meta_key] = float(value_str)
                except (ValueError, TypeError):
                    logger.warning(f"Product ID {product_id_for_log}: Payload key '{meta_key}' (from JSON key '{json_key}') failed to convert to float from ({type(value_str)}). Value: {value_str}")

        # Derive 'appliance_type'
        if "title" in product_data and isinstance(product_data["title"], str):
            title_lower = product_data["title"].lower()
            if "refrigerator" in title_lower or "fridge" in title_lower: payload["appliance_type"] = "Refrigerator"
            elif "dishwasher" in title_lower: payload["appliance_type"] = "Dishwasher"
            elif "oven" in title_lower or "range" in title_lower: payload["appliance_type"] = "Oven/Range"
            elif "washer" in title_lower or "washing machine" in title_lower: payload["appliance_type"] = "Washer"
            elif "dryer" in title_lower: payload["appliance_type"] = "Dryer"
            elif "microwave" in title_lower: payload["appliance_type"] = "Microwave"
            else: logger.warning(f"Product ID {product_id_for_log}: Could not derive 'appliance_type' from title: '{product_data['title']}'.")

        # --- Qdrant Specific: Store the full document in the payload ---
        payload["full_document"] = json.dumps(product_data)
        # -------------------------------------------------------------

        # --- Final Checks ---
        if "part_number" not in payload: logger.error(...)
        if "appliance_type" not in payload: logger.warning(...)

        return payload

    async def add_products_batch(self, products: List[Dict[str, Any]]) -> None:
        """Adds a batch of products to Qdrant using upsert."""
        await self._ensure_collection()

        points_to_upsert: List[PointStruct] = []
        embeddings_to_generate: List[Tuple[str, str]] = []
        payloads_prepared: Dict[str, Dict] = {}
        original_indices: Dict[str, int] = {} 

        logger.info(f"Preparing batch of {len(products)} products for Qdrant upsert...")

        for i, product in enumerate(products):
            product_id_val = product.get("partSelectNumber")
            if not product_id_val:
                logger.warning(f"Skipping product in batch (index {i}) due to missing 'partSelectNumber'.")
                continue
            qdrant_id = str(product_id_val).upper() # Use normalized part number as Qdrant ID

            product_description = product.get("description", "")
            if not product_description:
                logger.warning(f"Skipping product in batch with ID '{qdrant_id}' due to missing 'description'.")
                continue

            # Prepare payload (includes full document)
            payload = self._extract_payload(product)
            if not payload.get("part_number"): # Critical check from extraction
                 logger.error(f"Skipping product ID {qdrant_id} because critical 'part_number' missing from payload after extraction.")
                 continue

            # Store data needed for embedding generation and final point creation
            embeddings_to_generate.append((qdrant_id, product_description))
            payloads_prepared[qdrant_id] = payload
            original_indices[qdrant_id] = i

        if not embeddings_to_generate:
            logger.info("No valid products found in the batch to prepare for upsert.")
            return

        # --- Generate Embeddings (Can be done concurrently) ---
        logger.info(f"Generating {len(embeddings_to_generate)} embeddings...")
        embedding_results: Dict[str, Optional[List[float]]] = {}
        tasks = [self.generate_embedding(desc) for _, desc in embeddings_to_generate]
        try:
            # Use asyncio.gather to run them concurrently
            generated_embeddings = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results, handling potential errors from gather
            for idx, result in enumerate(generated_embeddings):
                qdrant_id, _ = embeddings_to_generate[idx]
                if isinstance(result, Exception):
                    logger.error(f"Failed to generate embedding for product ID {qdrant_id} in batch: {result}. Skipping this product.")
                    embedding_results[qdrant_id] = None
                elif isinstance(result, list):
                    embedding_results[qdrant_id] = result
                else:
                    logger.error(f"Unexpected result type from embedding generation for ID {qdrant_id}: {type(result)}. Skipping.")
                    embedding_results[qdrant_id] = None

        except Exception as gather_err:
             logger.exception(f"Error during concurrent embedding generation: {gather_err}")
             pass

        # --- Create PointStructs for successful embeddings ---
        successful_count = 0
        for qdrant_id, embedding in embedding_results.items():
            if embedding:
                point = PointStruct(
                    id=qdrant_id,
                    vector=embedding,
                    payload=payloads_prepared[qdrant_id]
                )
                points_to_upsert.append(point)
                successful_count += 1

        if not points_to_upsert:
            logger.warning("No products had successful embedding generation in the batch. Nothing to upsert.")
            return

        # --- Perform Qdrant Upsert ---
        logger.info(f"Attempting to upsert {len(points_to_upsert)} points to Qdrant collection '{self.collection_name}'...")
        try:
            response = await self.client.upsert(
                collection_name=self.collection_name,
                points=points_to_upsert,
                wait=True 
            )
            logger.info(f"Qdrant upsert completed. Status: {response.status}. Attempted: {len(products)}, Successfully prepared & upserted: {len(points_to_upsert)}")

        except Exception as e:
            logger.exception(f"CRITICAL ERROR during Qdrant client.upsert:")
            logger.error(f"Number of points intended for upsert: {len(points_to_upsert)}")
            try:
                logger.error(f"First 3 point IDs: {[p.id for p in points_to_upsert[:3]]}")
                logger.error("First 3 Payloads (partial):")
                for i in range(min(3, len(points_to_upsert))):
                     payload_preview = {k: v for k, v in points_to_upsert[i].payload.items() if k != 'full_document'}
                     logger.error(f"  Point {i} (ID: {points_to_upsert[i].id}): {payload_preview}")
            except Exception as log_e:
                 logger.error(f"Additional error occurred during error logging: {log_e}")


    def _build_qdrant_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[models.Filter]:
        """Converts a simple key-value filter dict to a Qdrant Filter object."""
        if not filters:
            return None

        must_conditions = []
        for key, value in filters.items():
            if isinstance(value, (str, int, float, bool)):
                must_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            else:
                logger.warning(f"Unsupported filter type for key '{key}': {type(value)}. Skipping this filter condition.")

        if not must_conditions:
            return None

        return Filter(must=must_conditions)


    async def search_products(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for products in Qdrant based on query text and filters."""
        await self._ensure_collection()

        if not query_text:
            logger.warning("Search query is empty.")
            return []

        try:
            query_embedding = await self.generate_embedding(query_text)
            qdrant_filter = self._build_qdrant_filter(filters)

            logger.debug(f"Qdrant search: top_k={top_k}, filter={qdrant_filter.json() if qdrant_filter else 'None'}")

            search_result = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True, 
                with_vectors=False
            )

            # --- Parse Qdrant results (ScoredPoint objects) ---
            products = []
            for hit in search_result:
                payload = hit.payload if hit.payload else {}
                # Extract the full document JSON string and parse it
                product_doc = {}
                document_str = payload.get("full_document")
                if document_str and isinstance(document_str, str):
                    try:
                        product_doc = json.loads(document_str)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse product document JSON from payload for ID {hit.id}: {document_str[:100]}...")
                else:
                     logger.warning(f"Missing or invalid 'full_document' in payload for ID {hit.id}.")

                # Remove full_document from the main payload dict to avoid redundancy
                metadata_only_payload = {k: v for k, v in payload.items() if k != 'full_document'}

                products.append({
                    "id": hit.id, # Qdrant point ID
                    "metadata": metadata_only_payload, # Filterable fields
                    "document": product_doc, # Parsed full original data
                    "score": hit.score # Qdrant score
                })

            logger.info(f"Found {len(products)} products via Qdrant for query: '{query_text[:50]}...' with filters: {filters}")
            return products

        except Exception as e:
            logger.exception(f"Error searching Qdrant products for query '{query_text[:50]}...': {str(e)}")
            return []

    async def delete_product(self, product_id: str) -> None:
        """Delete a product from Qdrant using its normalized ID"""
        await self._ensure_collection()
        try:
            qdrant_id = str(product_id).upper()
            logger.info(f"Attempting to delete point with ID '{qdrant_id}' from Qdrant collection '{self.collection_name}'...")
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[qdrant_id]),
                wait=True
            )
            logger.info(f"Successfully deleted point ID '{qdrant_id}'.")
        except Exception as e:
            logger.exception(f"Error deleting Qdrant point ID '{product_id}' (normalized: '{qdrant_id}'): {str(e)}")

    async def clear_database(self) -> None:
        """Delete and recreate the Qdrant collection"""
        logger.warning(f"Attempting to delete Qdrant collection: {self.collection_name}")
        try:
            await self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Deleted Qdrant collection: {self.collection_name}")
            await self._ensure_collection()
        except Exception as e:
            logger.exception(f"Error clearing Qdrant database (deleting collection '{self.collection_name}'): {str(e)}")
            try:
                 await self._ensure_collection()
            except Exception as e2:
                 logger.error(f"Failed to ensure collection exists after clear attempt: {e2}")

    async def close(self):
        """Close the Qdrant client connection."""
        await self.client.close()
        logger.info("Qdrant client closed.")

