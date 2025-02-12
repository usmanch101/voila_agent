import asyncio
import requests
from utils.logger import logger

class ExternalSearchService:
    def __init__(self, google_api_key: str, google_cse_id: str):
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id

    async def search(self, query: str) -> str:
        logger.info(f"Executing external search with query: {query}")
        url = (
            f"https://www.googleapis.com/customsearch/v1"
            f"?key={self.google_api_key}&cx={self.google_cse_id}&q={query}"
        )
        try:
            response = await asyncio.to_thread(requests.get, url)
            response.raise_for_status()
            data = response.json()
            
            if "items" in data:
                results = [item["snippet"] for item in data["items"][:3]]
                return "\n".join(results)
            else:
                logger.warning("No search results found.")
                return "No relevant information found."
        except Exception as e:
            logger.error(f"Error during external search: {e}", exc_info=True)
            return "Error occurred while fetching search results."
