import logging
import json
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class VectorDBManager:
    """Manages interactions with the vector database"""

    def __init__(self, client: chromadb.Client, openai_client: AsyncOpenAI):
        self.client = client
        self.openai_client = openai_client
        self.collection_name = "product_information"
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure the collection exists"""
        try:
            # Check if collection exists before getting it
            collections = self.client.list_collections()
            if self.collection_name in [c.name for c in collections]:
                 self.collection = self.client.get_collection(self.collection_name)
                 logger.info(f"Retrieved existing collection: {self.collection_name}")
            else:
                 self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"} # Example: Specify distance metric if needed
                 )
                 logger.info(f"Created new collection: {self.collection_name}")
        except Exception as e:
             logger.error(f"Error ensuring collection '{self.collection_name}': {e}")
             raise # Re-raise for now to indicate critical failure


    async def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text using OpenAI"""
        if not text: # Handle empty strings
            logger.warning("Attempted to generate embedding for empty text.")
            return [0.0] * 1536 # Assuming ada-002 dimension

        try:
            response = await self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.exception(f"Error generating embedding for text snippet '{text[:100]}...': {str(e)}") # Log exception for stack trace
            raise

    def _extract_metadata(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts filterable fields for metadata storage based on PartSelect requirements.
        Ensures values are compatible types (str, int, float, bool).
        ADJUSTED based on sample JSON structure (e.g., partSelectNumber, stockStatus, title).
        """
        metadata = {"type": "product"} 
        product_id_for_log = product_data.get('partSelectNumber', product_data.get('manufacturerPartNumber', 'N/A'))

        # --- Define keys expected in JSON and their target metadata key name ---
        # Format: { "json_key": "metadata_key" }
        key_mapping = {
            "partSelectNumber": "part_number",            
            "manufacturerPartNumber": "manufacturer_part_number", 
            "stockStatus": "availability"                

        }
        numeric_key_mapping = {
            "price": "price"                      
        }

        # --- Process standard filterable keys using mapping ---
        for json_key, meta_key in key_mapping.items():
            if json_key in product_data:
                value = product_data[json_key]
                if isinstance(value, (str, bool, int, float)):
                    if meta_key in ["part_number", "manufacturer_part_number"] and isinstance(value, str):
                         metadata[meta_key] = value.upper()
                    elif meta_key == "availability" and isinstance(value, str):
                         metadata[meta_key] = value # Store as is for now
                    else:
                         metadata[meta_key] = value
                else:
                    logger.warning(f"Product ID {product_id_for_log}: Skipping metadata key '{meta_key}' (from JSON key '{json_key}') due to incompatible type ({type(value)}). Value: {str(value)[:50]}...")
            else:
                 logger.debug(f"Product ID {product_id_for_log}: Source data missing potential metadata source key '{json_key}'.")

        # --- Process specific numeric keys using mapping ---
        for json_key, meta_key in numeric_key_mapping.items():
            if json_key in product_data:
                 value_str = product_data[json_key] # Price is a string in the JSON
                 try:
                     # Attempt to convert price string to float
                     value = float(value_str)
                     metadata[meta_key] = value
                 except (ValueError, TypeError):
                      logger.warning(f"Product ID {product_id_for_log}: Metadata key '{meta_key}' (from JSON key '{json_key}') expected numeric type but failed to convert from ({type(value_str)}). Value: {value_str}")
            else:
                 logger.debug(f"Product ID {product_id_for_log}: Source data missing potential metadata source key '{json_key}'.")

        # --- Derive 'appliance_type' from 'title' --- NEW LOGIC
        if "title" in product_data:
            title_value = product_data["title"]
            if isinstance(title_value, str):
                title_lower = title_value.lower()
                # Add more keywords and appliance types as needed
                if "refrigerator" in title_lower or "fridge" in title_lower:
                    metadata["appliance_type"] = "Refrigerator"
                elif "dishwasher" in title_lower:
                    metadata["appliance_type"] = "Dishwasher"
                elif "oven" in title_lower or "range" in title_lower:
                     metadata["appliance_type"] = "Oven/Range"
                elif "washer" in title_lower or "washing machine" in title_lower:
                     metadata["appliance_type"] = "Washer"
                elif "dryer" in title_lower:
                     metadata["appliance_type"] = "Dryer"
                elif "microwave" in title_lower:
                     metadata["appliance_type"] = "Microwave"
                # Add more appliance types as needed
                else:
                     # Log if no known appliance keyword is found in the title
                     logger.warning(f"Product ID {product_id_for_log}: Could not derive 'appliance_type' from title: '{title_value}'.")
            else:
                 logger.warning(f"Product ID {product_id_for_log}: Cannot derive 'appliance_type' as title value is not a string: {title_value}")
        else:
             logger.warning(f"Product ID {product_id_for_log}: Cannot derive 'appliance_type' as 'title' is missing from product data.")


        # --- Crucial Check: Ensure required filter fields are present ---
        # part_number is derived from partSelectNumber which is present in the sample
        if "part_number" not in metadata:
             # This is critical, make it an error
             logger.error(f"Product ID {product_id_for_log}: Critical filter field 'part_number' is missing from final metadata! Check source JSON for 'partSelectNumber'.")
        # appliance_type derivation might fail, so warn if missing
        if "appliance_type" not in metadata:
             # This might be acceptable if title doesn't map, but good to warn
             logger.warning(f"Product ID {product_id_for_log}: Filter field 'appliance_type' could not be derived/added to final metadata.")
        # manufacturer_part_number is present in sample, but check just in case
        if "manufacturer_part_number" not in metadata:
             logger.warning(f"Product ID {product_id_for_log}: Filter field 'manufacturer_part_number' is missing from final metadata.")
        # Brand is no longer expected based on sample JSON

        return metadata


    async def add_product(self, product_id: str, product_data: Dict[str, Any], product_description: Optional[str] = None) -> None:
        # Use product_id from input args OR product_data['partNumber'] as the canonical ID
        # Ensure consistency in how the ID is determined and normalized
        chroma_id_source = product_data.get("partSelectNumber", product_id) # Prefer partNumber from data if available
        if not chroma_id_source:
             logger.error(f"Cannot determine a valid ID (from input arg or 'partNumber' in data) for product: {str(product_data)[:100]}...")
             return
        chroma_id = str(chroma_id_source).upper() # Normalize ID for consistency

        if not product_description:
            product_description = product_data.get("description")
        if not product_description:
             logger.error(f"Cannot add product with ID '{chroma_id}': Missing description for embedding.")
             return

        try:
            # Ensure the product_data itself contains 'partNumber' matching the one used for Chroma ID
            if 'partSelectNumber' not in product_data or str(product_data['partSelectNumber']).upper() != chroma_id:
                 logger.warning(f"Product data for Chroma ID '{chroma_id}' has inconsistent or missing internal 'partSelectNumber' field.")
                 # Optionally add/overwrite it for consistency within the stored document
                 # product_data['partNumber'] = chroma_id

            embedding = await self.generate_embedding(product_description)
            metadata = self._extract_metadata(product_data) # Use the updated extraction logic

            self.collection.add(
                ids=[chroma_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[json.dumps(product_data)]
            )
            logger.info(f"Added product with ID '{chroma_id}' to vector database.")
        except Exception as e:
            logger.exception(f"Failed to add product with ID '{chroma_id}': {str(e)}")


    async def add_products_batch(self, products: List[Dict[str, Any]]) -> None:
        ids = []
        embeddings_to_generate = []
        metadatas = []
        documents = []
        valid_product_indices = []

        for i, product in enumerate(products):
            product_id_val = product.get("partSelectNumber")
            if not product_id_val:
                 logger.warning(f"Skipping product in batch (index {i}) due to missing 'partSelectNumber'. Name: {product.get('name', 'N/A')}")
                 continue
            chroma_id = str(product_id_val).upper() # Normalize

            product_description = product.get("description", "")
            if not product_description:
                logger.warning(f"Skipping product in batch with ID '{chroma_id}' due to missing 'description'. Name: {product.get('name', 'N/A')}")
                continue

            # Prepare data for batch add
            embeddings_to_generate.append(product_description)
            metadatas.append(self._extract_metadata(product)) # Use updated extraction
            documents.append(json.dumps(product))
            ids.append(chroma_id)
            valid_product_indices.append(i)

        # ... (rest of batch add logic remains the same, using the prepared lists) ...
        if not ids:
            logger.info("No valid products found in the batch to add.")
            return

        try:
            logger.info(f"Generating {len(embeddings_to_generate)} embeddings sequentially...")
            generated_embeddings = []
            successful_indices = [] # Track which ones succeeded
            for idx, desc in enumerate(embeddings_to_generate):
                 try:
                     embedding = await self.generate_embedding(desc)
                     generated_embeddings.append(embedding)
                     successful_indices.append(idx) # Mark as successful
                 except Exception as embed_err:
                      failed_id = ids[idx] # Get the ID that failed
                      logger.error(f"Failed to generate embedding for product ID {failed_id} in batch: {embed_err}. Skipping this product.")
                      # Don't raise, just skip this one product

            # Filter down the lists to only include successfully embedded products
            final_ids = [ids[i] for i in successful_indices]
            final_metadatas = [metadatas[i] for i in successful_indices]
            final_documents = [documents[i] for i in successful_indices]
            final_embeddings = generated_embeddings # Already filtered by successful appends

            if not final_ids:
                 logger.warning("No products had successful embedding generation in the batch. Nothing added.")
                 return
            
            logger.info(f"DEBUG: Preparing to call collection.add with {len(final_ids)} items.")
            logger.info(f"DEBUG: final_ids count: {len(final_ids)}")
            logger.info(f"DEBUG: final_embeddings count: {len(final_embeddings)}")
            logger.info(f"DEBUG: final_metadatas count: {len(final_metadatas)}")
            logger.info(f"DEBUG: final_documents count: {len(final_documents)}")
            logger.info(f"DEBUG: Sample embedding shape: {len(final_embeddings[0]) if final_embeddings else 'None'}")

            # --- Add batch to collection ---
            """
            try:
                self.collection.add(
                    ids=final_ids,
                    embeddings=final_embeddings,
                    metadatas=final_metadatas,
                    documents=final_documents
                )
                logger.info(f"DEBUG: collection.add call completed.")
                logger.info(f"Attempted to add {len(products)} products. Successfully generated embeddings and added {len(final_ids)} products to vector database.")

            except Exception as e:
                # Ensure the exception block logs clearly
                logger.exception(f"ERROR: Exception occurred *during* collection.add or subsequent logging: {str(e)}")
                # Re-raise or handle as appropriate
                raise
            """
            try:
                logger.info(f"DEBUG: First ID in batch: {final_ids[0] if final_ids else 'None'}")
                logger.info(f"DEBUG: Preparing to call collection.add with {len(final_ids)} items.")

                # --- Check for duplicate IDs within the batch ---
                if len(final_ids) != len(set(final_ids)):
                    logger.error("Duplicate IDs found within the batch being sent to collection.add!")
                    # Find and log duplicates
                    seen = set()
                    dupes = [x for x in final_ids if x in seen or seen.add(x)]
                    logger.error(f"Duplicate IDs: {list(set(dupes))}")
                    # Decide how to handle: raise error, skip batch, or filter duplicates?
                    # For now, let's raise to stop execution and investigate
                    raise ValueError("Duplicate IDs detected within a single batch add operation.")
                # --- End Duplicate ID Check ---

                self.collection.add(
                    ids=final_ids,
                    embeddings=final_embeddings,
                    metadatas=final_metadatas,
                    documents=final_documents
                )
                logger.info(f"DEBUG: collection.add call completed successfully for {len(final_ids)} items.")
                # Log success *after* the call completes
                logger.info(f"Attempted to add {len(products)} products. Successfully generated embeddings and added {len(final_ids)} products to vector database.")
            except Exception as e:
                # Use logger.exception to get the full traceback
                logger.exception(f"CRITICAL ERROR during collection.add:")

                # Log details about the data that likely caused the error
                logger.error(f"Data details at time of error:")
                logger.error(f"Number of items intended for add: {len(final_ids)}")
                try:
                    logger.error(f"First 3 IDs: {final_ids[:3]}")
                    # CRITICAL: Print the metadata that likely caused the issue
                    logger.error("First 3 Metadata objects:")
                    for i in range(min(3, len(final_metadatas))):
                        logger.error(f"  Item {i} (ID: {final_ids[i]}): {final_metadatas[i]}")
                        # Check types within the metadata
                        for k, v in final_metadatas[i].items():
                            if type(v) not in [str, int, float, bool]:
                                logger.error(f"    --> POTENTIAL ISSUE: Key '{k}' has invalid type: {type(v)} (Value: {str(v)[:50]}...)")

                except Exception as log_e:
                    logger.error(f"Additional error occurred during error logging: {log_e}")

                # Re-raise the original exception after logging
                raise
        except Exception as e:
            logger.exception(f"Failed during batch add process (post-embedding): {str(e)}")



    async def search_products(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for products based on a query string, with optional metadata filtering.
        Filters should target keys extracted by _extract_metadata (e.g., part_number, brand).

        Args:
            query_text: The text query for semantic search.
            top_k: The maximum number of results to return.
            filters: A dictionary specifying metadata filters (compatible with ChromaDB's 'where' clause).
                     Example: {"part_number": "WP8544771", "brand": "WHIRLPOOL"}

        Returns:
            A list of product dictionaries, each containing 'id', 'metadata',
            'document' (the full product JSON), and 'distance'.
        """
        if not query_text:
            logger.warning("Search query is empty.")
            return []

        try:
            query_embedding = await self.generate_embedding(query_text)

            # Construct the where clause for filtering if filters are provided
            where_clause = filters if filters else None

            # Log the actual query being sent to ChromaDB
            logger.debug(f"ChromaDB query: top_k={top_k}, where={json.dumps(where_clause)}")

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause, # Pass the filters directly
                include=['metadatas', 'documents', 'distances'] # Ensure all needed fields are included
            )

            # --- Parse and structure the results ---
            products = []
            if results and results.get('ids') and len(results['ids']) > 0:
                # Chroma returns lists of lists, even for single query embedding
                result_ids = results['ids'][0]
                result_metadatas = results['metadatas'][0] if results.get('metadatas') else [None] * len(result_ids)
                result_documents = results['documents'][0] if results.get('documents') else [None] * len(result_ids)
                result_distances = results['distances'][0] if results.get('distances') else [None] * len(result_ids)

                for i, product_id in enumerate(result_ids):
                    try:
                        metadata = result_metadatas[i]
                        document_str = result_documents[i]
                        distance = float(result_distances[i]) if result_distances[i] is not None else None

                        # Parse the document JSON string back into a dict
                        product_doc = {}
                        if document_str:
                            try:
                                product_doc = json.loads(document_str)
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse product document JSON for ID {product_id}: {document_str[:100]}...")
                                # Continue with empty doc or skip? Let's continue with empty.
                        else:
                             logger.warning(f"Missing document string for ID {product_id} in search results.")


                        products.append({
                            "id": product_id, # The ID used in Chroma (normalized part number)
                            "metadata": metadata, # Contains filterable fields like brand, price, part_number
                            "document": product_doc, # Contains the full original product data (incl. compatibleModels, videoTutorials)
                            "distance": distance # Lower is more similar
                        })
                    except IndexError:
                         logger.warning(f"Index error processing results for ID {product_id}. Results structure might be inconsistent. Index: {i}, Total IDs: {len(result_ids)}")
                    except Exception as parse_err:
                         logger.error(f"Unexpected error processing result for ID {product_id}: {parse_err}")


            logger.info(f"Found {len(products)} products for query: '{query_text[:50]}...' with filters: {filters}")
            return products

        except Exception as e:
            logger.exception(f"Error searching products for query '{query_text[:50]}...': {str(e)}")
            return [] # Return empty list on error


    def delete_product(self, product_id: str) -> None:
        """Delete a product from the vector database using its normalized ID"""
        try:
            chroma_id = str(product_id).upper() # Normalize ID for deletion
            self.collection.delete(ids=[chroma_id])
            logger.info(f"Deleted product with ID '{chroma_id}' from vector database")
        except Exception as e:
            logger.error(f"Error deleting product with ID '{product_id}' (normalized: '{chroma_id}'): {str(e)}")


    def clear_database(self) -> None:
        """Delete and recreate the collection"""
        try:
            logger.warning(f"Attempting to delete collection: {self.collection_name}")
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            self._ensure_collection() # Recreate it
        except Exception as e:
            logger.exception(f"Error clearing database (deleting collection '{self.collection_name}'): {str(e)}")
            try:
                self._ensure_collection()
            except Exception as e2:
                 logger.error(f"Failed to recreate collection after deletion error: {e2}")

