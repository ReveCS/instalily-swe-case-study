"""
import asyncio
from crawl4ai import *

async def main():

    run_config = CrawlerRunConfig(
        # Anti-bot
        magic=True,
        simulate_user=True,
        override_navigator=True
    )

    browser_conf = BrowserConfig(
        browser_type="firefox",
        headless=False,
        text_mode=True,
        user_agent_mode="random"
    )


    async with AsyncWebCrawler(config=browser_conf) as crawler:
        result = await crawler.arun(
            url="https://www.partselect.com/PS11739119-Whirlpool-WP2188656-Refrigerator-Crisper-Drawer-with-Humidity-Control.htm?SourceCode=10",
            config=run_config
        )
        print(result.markdown)

if __name__ == "__main__":
    asyncio.run(main())
"""

import os
import json
import time
import random
import re
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException

# --- Constants ---
BASE_URL = 'https://www.partselect.com'

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1'
]

# --- Helper Functions ---

def get_random_user_agent():
    """Returns a random user agent string."""
    return random.choice(USER_AGENTS)

def sleep(ms):
    """Pauses execution for a specified number of milliseconds."""
    time.sleep(ms / 1000.0)

def clean_text(text):
    """Clean text to remove excessive whitespace and newlines."""
    if not text:
        return ''
    text = re.sub(r'\s+', ' ', str(text))
    text = re.sub(r'\n+', ' ', text)
    return text.strip()

def clean_price(price_text):
    """Clean price data to extract just the first price number."""
    if not price_text:
        return ''
    # Remove currency symbols and whitespace
    cleaned_text = re.sub(r'[$\s]', '', str(price_text))
    # Look for repeating price patterns (match the first occurrence)
    price_match = re.match(r'^(\d+\.\d{2})', cleaned_text)
    if price_match and price_match.group(1):
        return price_match.group(1)
    # Fallback: return the cleaned text if pattern doesn't match
    # (handles cases where it might just be digits without decimal)
    numeric_match = re.match(r'^(\d+(\.\d+)?)', cleaned_text)
    return numeric_match.group(1) if numeric_match else cleaned_text


async def fetch_html_with_user_agent(url, retries=10, delay_ms=2000, timeout_sec=10):
    """
    Fetch HTML content from a URL with custom user agent and retry logic.
    Uses requests library.
    """
    attempt = 0
    last_exception = None

    while attempt < retries:
        user_agent = get_random_user_agent()
        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
            'Cache-Control': 'no-cache'
        }
        print(f"Attempt {attempt + 1}/{retries}: Fetching {url} with user agent: {user_agent}")

        try:
            response = requests.get(url, headers=headers, timeout=timeout_sec)
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

            # Check if status code is 200 explicitly (though raise_for_status covers it)
            if response.status_code == 200:
                # It's good practice to check content type if possible,
                # but for scraping we often assume HTML
                # if 'text/html' in response.headers.get('Content-Type', ''):
                return response.text
                # else:
                #     print(f"Warning: Content-Type is not text/html for {url}")
                #     return response.text # Or handle non-HTML content differently

        except RequestException as e:
            print(f"Attempt {attempt + 1}/{retries} failed for {url}: {e}")
            last_exception = e

        attempt += 1
        if attempt < retries:
            print(f"Waiting {delay_ms}ms before retry...")
            sleep(delay_ms)

    raise Exception(f"Failed to fetch {url} after {retries} attempts. Last error: {last_exception}") from last_exception


# --- Selectors ---
SELECTORS = {
    'brandPage': {
        'brands': '.semi-bold a, .brand_links a, .nf__links a',
        'parts': '.nf__part, .part-item, .product-item',
        'partTitle': '.nf__part__title, .part-title, .product-title',
        'partImage': '.nf__part__img img, .part-image img, .product-image img',
        'partPrice': '.nf__part__price, .part-price, .price',
        'readMoreLink': '.nf__part__detail a[href*="PS"], a.part-detail-link, a[href*=".htm"]',
        'pagination': '.pagination a, .next a, a:contains("Next")', # :contains might need lambda
    },
    'partDetailPage': {
        # Using lists for fallbacks
        'title': ['h1', '.product-title h1', '.product-title', '.page-title', 'h1.product-heading'],
        'partSelectNumber_container': [
            '.product-specs:has(span:contains("PartSelect Number"))', # :has might need adjustment
            'span:contains("PartSelect Number")', # :contains needs lambda
            '.pd__part-number',
            '.nf__part__detail__part-number'
         ],
        'manufacturerPartNumber_container': [
            '.product-specs:has(span:contains("Manufacturer Part Number"))', # :has might need adjustment
            'span:contains("Manufacturer Part Number")', # :contains needs lambda
            '.nf__part__detail__part-number:contains("Manufacturer")' # :contains needs lambda
        ],
        'price': ['.price:not(.original-price)', '.your-price', '.price', '.pd__price'], # :not might work
        'originalPrice': ['.price.original-price', '.original-price', '.was-price'],
        'stockStatus': ['span:contains("In Stock")', '.stock-status', '.availability'], # :contains needs lambda
        'description': ['.product-description', '.part-description', '.pd__description', 'p.description', '.nf__part__detail > p'],
        'installationInstructions': ['.installation-instructions', '.install-instructions', '.customer-instruction', '.pd__cust-review__submitted-review'],
        'reviews_container': ['.reviews', '.customer-reviews', '.ratings-container', '.pd__cust-review'],
        'review_item': ['.review-item', '.customer-review', '.pd__cust-review__submitted-review'],
        'review_title': ['.review-title', '.review-heading', '.bold:not(.pd__cust-review__submitted-review__header)'], # :not might work
        'review_text': ['.review-text', '.review-content', '.js-searchKeys'],
        'review_author': ['.review-author', '.reviewer-name', '.pd__cust-review__submitted-review__header'],
        'review_rating_text': ['.rating', '.star-rating', '.rating__stars'],
        'review_count_text': ['.review-count', '.ratings-count', '.rating__count'],
        'compatibleModels_container': ['.compatible-models', '.models-list', '.fits-models', '.nf__part__detail__compatibility'],
        'compatibleModels_item': ['li', '.model-item'],
        'symptoms_container': ['.fixes-symptoms', '.symptoms-list', '.symptoms', '.nf__part__detail__symptoms'],
        'symptoms_item': ['li', '.symptom-item'],
        'main_image': [
            '.product-image img', '.nf__part__left-col__img img', '.main-image img',
            '.pd__img img', 'img.js-imgTagHelper', 'img.b-lazy', '.product-page-image img'
        ],
        'additional_images_container': ['.product-images', '.thumbnail-images', '.additional-images', '.product-thumbnails'],
        'additional_image_item': 'img',
        'video_container': ['.yt-video', '[data-yt-init]', '.youtube-video', '.video-container'],
        'installation_guide_container': ['.installation-guide', '.installation-help', '.install-steps', '.how-to-install'],
    },
}

# --- Extraction Functions ---

def extract_part_detail_page(html, source_url=""):
    """Extracts detailed information from a part detail page HTML."""
    soup = BeautifulSoup(html, 'html.parser')
    details = {'sourceUrl': source_url}
    sel = SELECTORS['partDetailPage']

    print("--- Starting Detail Page Extraction ---")

    # Title
    for selector in sel['title']:
        el = soup.select_one(selector)
        if el:
            details['title'] = clean_text(el.get_text())
            print(f"Found title using selector '{selector}': {details['title']}")
            break
    if 'title' not in details:
         details['title'] = "Title Not Found"
         print("Warning: Title not found.")


    # PartSelect Number
    ps_number_found = False
    for selector in sel['partSelectNumber_container']:
        # Handle :contains and :has manually if needed
        elements = soup.select(selector)
        for el in elements:
            text_content = el.get_text()
            if "PartSelect Number" in text_content or selector in ['.pd__part-number', '.nf__part__detail__part-number']:
                 ps_match = re.search(r'PS\d+', text_content)
                 if ps_match:
                     details['partSelectNumber'] = ps_match.group(0)
                     print(f"Found PartSelect Number: {details['partSelectNumber']} using container '{selector}'")
                     ps_number_found = True
                     break
        if ps_number_found: break

    if not ps_number_found and source_url:
        url_match = re.search(r'PS\d+', source_url)
        if url_match:
            details['partSelectNumber'] = url_match.group(0)
            print(f"Extracted PartSelect Number from URL: {details['partSelectNumber']}")
            ps_number_found = True

    if not ps_number_found:
        print("Warning: PartSelect Number not found.")


    # Manufacturer Part Number
    mfr_num_found = False
    for selector in sel['manufacturerPartNumber_container']:
        elements = soup.select(selector)
        for el in elements:
            text_content = el.get_text()
            if "Manufacturer Part Number" in text_content or "Manufacturer:" in text_content:
                print(f"Found text for manufacturer part: {text_content[:100]}...")
                # Try various patterns to extract the number
                patterns = [
                    r'Number\s+([A-Za-z0-9-]+)',
                    r'Manufacturer\s+Part\s+Number\s+([A-Za-z0-9-]+)',
                    r'Manufacturer:\s+([A-Za-z0-9-]+)'
                ]
                for pattern in patterns:
                    match = re.search(pattern, text_content, re.IGNORECASE)
                    if match and match.group(1):
                        details['manufacturerPartNumber'] = match.group(1).strip()
                        print(f"Found Manufacturer Part Number: {details['manufacturerPartNumber']} using container '{selector}'")
                        mfr_num_found = True
                        break
            if mfr_num_found: break
        if mfr_num_found: break

    # Fallback from URL if possible (less reliable)
    if not mfr_num_found and source_url and 'partSelectNumber' in details:
        try:
            url_path = urlparse(source_url).path
            filename = os.path.basename(url_path)
            base, _ = os.path.splitext(filename)
            url_parts = base.split('-')
            # Example URL format: /PS12345-Whirlpool-WP67890-Part-Name.htm
            if len(url_parts) >= 3 and url_parts[0] == details.get('partSelectNumber'):
                 # Check if the potential Mfr number looks like one
                 potential_mfr = url_parts[2]
                 if re.match(r'^[A-Za-z0-9-]+$', potential_mfr) and len(potential_mfr) > 2:
                     details['manufacturerPartNumber'] = potential_mfr
                     print(f"Extracted Manufacturer Part Number from URL: {details['manufacturerPartNumber']}")
                     mfr_num_found = True
        except Exception as e:
            print(f"Could not extract Mfr number from URL: {e}")


    if not mfr_num_found:
        print("Warning: Manufacturer Part Number not found.")

    # Price
    price_found = False
    for selector in sel['price']:
        # Handle :not(.original-price) - select all, then filter
        elements = soup.select(selector)
        for el in elements:
            # Check if it has the 'original-price' class
            if 'original-price' not in el.get('class', []):
                raw_price = el.get_text()
                cleaned = clean_price(raw_price)
                if cleaned: # Ensure we got a valid price
                    details['price'] = cleaned
                    print(f"Found price using selector '{selector}': {details['price']} (raw: {raw_price.strip()})")
                    price_found = True
                    break
        if price_found: break

    if not price_found:
        print("Warning: Price not found.")

    # Stock Status
    stock_found = False
    for selector in sel['stockStatus']:
         elements = soup.select(selector)
         for el in elements:
             # Handle :contains manually
             text_content = el.get_text()
             if "In Stock" in text_content or selector in ['.stock-status', '.availability']:
                 details['stockStatus'] = clean_text(text_content)
                 print(f"Found stock status: {details['stockStatus']} using selector '{selector}'")
                 stock_found = True
                 break
         if stock_found: break

    if not stock_found:
        details['stockStatus'] = 'Unknown'
        print("Warning: Stock status not found, set to Unknown.")


    # Description
    desc_found = False
    for selector in sel['description']:
        el = soup.select_one(selector)
        if el:
            desc_text = clean_text(el.get_text())
            if len(desc_text) > 20: # Basic check for meaningful description
                details['description'] = desc_text
                print(f"Found description using selector '{selector}': {details['description'][:50]}...")
                desc_found = True
                break

    if not desc_found:
        # Fallback: Find the longest paragraph as a potential description
        longest_p = ""
        for p in soup.find_all('p'):
            text = clean_text(p.get_text())
            if len(text) > len(longest_p) and len(text) > 100: # Heuristic
                longest_p = text
        if longest_p:
            details['description'] = longest_p
            print(f"Found description in generic paragraph: {details['description'][:50]}...")
        else:
            print("Warning: Description not found.")


    # Installation Instructions (Simplified: gets first found block)
    inst_found = False
    for selector in sel['installationInstructions']:
        el = soup.select_one(selector)
        if el:
            inst_text = clean_text(el.get_text())
            if len(inst_text) > 20:
                details['installationInstructions'] = inst_text
                print(f"Found installation instructions using selector '{selector}'")
                inst_found = True
                break
    if not inst_found:
        print("Info: Installation instructions text block not found.")


    # Reviews
    details['reviews'] = {'averageRating': 0, 'count': 0, 'items': []}
    review_section_found = False
    for container_selector in sel['reviews_container']:
        section = soup.select_one(container_selector)
        if section:
            print(f"Found reviews section using selector '{container_selector}'")
            review_section_found = True

            # Extract average rating and count
            rating_text = ""
            for rating_sel in sel['review_rating_text']:
                rating_el = section.select_one(rating_sel)
                if rating_el:
                    rating_text = rating_el.get_text()
                    break
            rating_match = re.search(r'[\d.]+', rating_text)
            if rating_match:
                try:
                    details['reviews']['averageRating'] = float(rating_match.group(0))
                except ValueError:
                    pass # Keep default 0

            count_text = ""
            for count_sel in sel['review_count_text']:
                count_el = section.select_one(count_sel)
                if count_el:
                    count_text = count_el.get_text()
                    break
            count_match = re.search(r'\d+', count_text)
            if count_match:
                 try:
                    details['reviews']['count'] = int(count_match.group(0))
                 except ValueError:
                    pass # Keep default 0

            print(f"Found review summary: Rating={details['reviews']['averageRating']}, Count={details['reviews']['count']}")

            # Extract individual review items
            items_found = False
            for item_selector in sel['review_item']:
                review_elements = section.select(item_selector)
                if review_elements:
                    print(f"Found {len(review_elements)} potential review items using '{item_selector}'")
                    for el in review_elements:
                        review = {}
                        # Extract title, text, author, date
                        for title_sel in sel['review_title']:
                            title_el = el.select_one(title_sel)
                            if title_el:
                                # Exclude author header if it matches this selector
                                if 'pd__cust-review__submitted-review__header' not in title_el.get('class', []):
                                    review['title'] = clean_text(title_el.get_text())
                                    break
                        for text_sel in sel['review_text']:
                            text_el = el.select_one(text_sel)
                            if text_el:
                                review['text'] = clean_text(text_el.get_text())
                                break
                        for author_sel in sel['review_author']:
                            author_el = el.select_one(author_sel)
                            if author_el:
                                author_text = author_el.get_text().strip()
                                # Try to split author and date (common pattern: "Author Name - Date")
                                author_match = re.match(r'([^-]+)-\s*(.*)', author_text)
                                if author_match:
                                    review['author'] = author_match.group(1).strip()
                                    review['date'] = author_match.group(2).strip()
                                else:
                                    review['author'] = author_text # Assume whole text is author if no pattern
                                break

                        # Add review if it has text content
                        if review.get('text'):
                            review.setdefault('title', '')
                            review.setdefault('author', 'Anonymous')
                            review.setdefault('date', '')
                            details['reviews']['items'].append(review)
                            print(f"  Found review: \"{review['title'][:30]}...\" by {review['author']}")
                            items_found = True # Mark that we found items with this selector

                    if items_found: break # Stop trying item selectors if one worked

            if items_found: break # Stop trying container selectors if we found reviews

    if not review_section_found:
        print("Info: Reviews section not found.")


    # Compatible Models
    details['compatibleModels'] = []
    models_found = False
    for container_selector in sel['compatibleModels_container']:
        section = soup.select_one(container_selector)
        if section:
            print(f"Found compatible models section using selector '{container_selector}'")
            for item_selector in sel['compatibleModels_item']:
                model_elements = section.select(item_selector)
                if model_elements:
                    for el in model_elements:
                        model = clean_text(el.get_text())
                        # Avoid "See more" links or empty items
                        if model and 'see more' not in model.lower():
                            details['compatibleModels'].append(model)
                    if details['compatibleModels']:
                        print(f"Found {len(details['compatibleModels'])} compatible models using item selector '{item_selector}'")
                        models_found = True
                        break # Stop trying item selectors
            if models_found: break # Stop trying container selectors

    if not models_found:
        print("Info: Compatible models section not found.")


    # Symptoms
    details['symptoms'] = []
    symptoms_found = False
    for container_selector in sel['symptoms_container']:
        section = soup.select_one(container_selector)
        if section:
            print(f"Found symptoms section using selector '{container_selector}'")
            for item_selector in sel['symptoms_item']:
                symptom_elements = section.select(item_selector)
                if symptom_elements:
                    for el in symptom_elements:
                        symptom = clean_text(el.get_text())
                        if symptom and 'see more' not in symptom.lower():
                            details['symptoms'].append(symptom)
                    if details['symptoms']:
                        print(f"Found {len(details['symptoms'])} symptoms using item selector '{item_selector}'")
                        symptoms_found = True
                        break # Stop trying item selectors
            if symptoms_found: break # Stop trying container selectors

    if not symptoms_found:
        print("Info: Symptoms section not found.")


    # Images
    details['images'] = []
    details['imageUrl'] = None # Main image URL
    main_image_found = False

    # Find Main Image
    for selector in sel['main_image']:
        el = soup.select_one(selector)
        if el:
            img_src = el.get('data-src') or el.get('src') or el.get('data-original')
            if img_src:
                full_image_url = urljoin(BASE_URL, img_src)
                details['imageUrl'] = full_image_url
                details['images'].append({
                    'url': full_image_url,
                    'isPrimary': True,
                    'caption': el.get('alt', 'Main Product Image'),
                    'type': 'product'
                })
                print(f"Found main image URL: {full_image_url} using selector '{selector}'")
                main_image_found = True
                break

    # Fallback: Check background images (less reliable)
    if not main_image_found:
        for el in soup.select('[style*="background"]'):
            style = el.get('style', '')
            match = re.search(r'url\([\'"]?([^\'"\)]+)[\'"]?\)', style)
            if match and match.group(1):
                img_src = match.group(1)
                full_image_url = urljoin(BASE_URL, img_src)
                # Basic check if it looks like a product image URL
                if any(kw in full_image_url.lower() for kw in ['.jpg', '.png', '.jpeg', 'image', 'part']):
                    details['imageUrl'] = full_image_url
                    details['images'].append({
                        'url': full_image_url,
                        'isPrimary': True,
                        'caption': 'Product Image (Background)',
                        'type': 'product'
                    })
                    print(f"Found background image URL: {full_image_url}")
                    main_image_found = True
                    break # Take the first likely one

    # Find Additional Images
    additional_image_urls = set(img['url'] for img in details['images']) # Track existing URLs
    for container_selector in sel['additional_images_container']:
        container = soup.select_one(container_selector)
        if container:
            print(f"Checking for additional images in '{container_selector}'")
            img_elements = container.select(sel['additional_image_item'])
            for el in img_elements:
                img_src = el.get('data-src') or el.get('src') or el.get('data-original')
                if img_src:
                    full_image_url = urljoin(BASE_URL, img_src)
                    if full_image_url not in additional_image_urls:
                        details['images'].append({
                            'url': full_image_url,
                            'isPrimary': False,
                            'caption': el.get('alt', 'Additional Product Image'),
                            'type': 'product'
                        })
                        additional_image_urls.add(full_image_url)
                        print(f"  Found additional image: {full_image_url}")
            # If we found images in one container type, maybe stop? Or continue to be thorough?
            # Let's continue for now to catch all possibilities.

    print(f"Total product images found: {len([img for img in details['images'] if img['type'] == 'product'])}")


    # Video Tutorials
    details['videoTutorials'] = []
    video_ids_found = set()
    for selector in sel['video_container']:
        video_elements = soup.select(selector)
        for el in video_elements:
            video_id = el.get('data-yt-init') or el.get('data-video-id')

            # Try finding iframe src if no data attribute
            if not video_id:
                iframe = el.select_one('iframe')
                if iframe:
                    src = iframe.get('src', '')
                    match = re.search(r'/embed/([^\/\?]+)', src)
                    if match and match.group(1):
                        video_id = match.group(1)

            if video_id and video_id not in video_ids_found:
                video_img = el.select_one('img')
                video_title = "Installation Video"
                thumbnail_url = None
                if video_img:
                    video_title = video_img.get('title') or video_img.get('alt') or video_title
                    thumbnail_url = video_img.get('src')

                if not thumbnail_url:
                    thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg" # Default HQ thumb

                # Try to get a description from nearby heading
                description = "Installation Tutorial"
                heading = el.find_previous(['h3', 'h4']) or el.find_next(['h3', 'h4']) # Simple proximity check
                if heading:
                    description = clean_text(heading.get_text())

                video_data = {
                    'videoId': video_id,
                    'videoUrl': f"https://www.youtube.com/watch?v={video_id}",
                    'title': clean_text(video_title),
                    'thumbnailUrl': urljoin(BASE_URL, thumbnail_url),
                    'description': description
                }
                details['videoTutorials'].append(video_data)
                video_ids_found.add(video_id)

                # Add video thumbnail to images list
                if video_data['thumbnailUrl'] not in additional_image_urls:
                     details['images'].append({
                        'url': video_data['thumbnailUrl'],
                        'isPrimary': False,
                        'caption': video_data['title'],
                        'type': 'video',
                        'videoId': video_id
                    })
                     additional_image_urls.add(video_data['thumbnailUrl'])

                print(f"Found video tutorial: {video_data['title']} ({video_id})")

    print(f"Total video tutorials found: {len(details['videoTutorials'])}")


    # Installation Guides (Text/Image based)
    details['installationGuides'] = []
    for selector in sel['installation_guide_container']:
        guide_elements = soup.select(selector)
        for el in guide_elements:
            guide_title = "Installation Guide"
            title_el = el.select_one('h3, h4, .title') # Common title elements
            if title_el:
                guide_title = clean_text(title_el.get_text())

            guide_text = ""
            text_el = el.select_one('p, .text, .steps') # Common text elements
            if text_el:
                guide_text = clean_text(text_el.get_text())

            if guide_text: # Only add if there's text content
                guide = {
                    'title': guide_title,
                    'text': guide_text,
                    'imageUrl': None
                }
                guide_img = el.select_one('img')
                if guide_img:
                    img_src = guide_img.get('data-src') or guide_img.get('src') or guide_img.get('data-original')
                    if img_src:
                        guide['imageUrl'] = urljoin(BASE_URL, img_src)
                        # Add guide image to main images list if not already present
                        if guide['imageUrl'] not in additional_image_urls:
                            details['images'].append({
                                'url': guide['imageUrl'],
                                'isPrimary': False,
                                'caption': guide_title,
                                'type': 'guide'
                            })
                            additional_image_urls.add(guide['imageUrl'])

                details['installationGuides'].append(guide)
                print(f"Found installation guide: {guide_title}")

    print(f"Total installation guides found: {len(details['installationGuides'])}")
    print(f"--- Finished Detail Page Extraction for {details.get('title', 'unknown part')} ---")

    return details


def extract_data_from_html(html, selectors=None, source_url=""):
    """
    Extracts data from HTML content based on CSS selectors.
    Handles both list pages (brands, parts list) and delegates to
    extract_part_detail_page if selectors indicate a detail page.
    """
    if selectors is None:
        selectors = {}

    # Check if the provided selectors match the detail page structure
    # This check is a bit heuristic - might need refinement
    is_detail_page = selectors == SELECTORS['partDetailPage'] or selectors.get('title') or selectors.get('partSelectNumber_container')

    if is_detail_page:
        print("Detected or specified Detail Page selectors. Running detail extraction.")
        return extract_part_detail_page(html, source_url=source_url)
    else:
        # Assume it's a list page (like brand page or search results)
        print("Running List Page extraction (Brands/Parts List).")
        soup = BeautifulSoup(html, 'html.parser')
        result = {
            'brands': [],
            'parts': []
        }
        list_sel = SELECTORS['brandPage'] # Use brandPage selectors as default for lists

        # Extract brands
        brand_selector = selectors.get('brands', list_sel['brands'])
        for element in soup.select(brand_selector):
            brand_name = clean_text(element.get_text())
            brand_url = element.get('href')
            if brand_name and brand_url and 'see all' not in brand_name.lower():
                result['brands'].append({
                    'name': brand_name,
                    'url': urljoin(BASE_URL, brand_url) # Ensure absolute URL
                })
        print(f"Found {len(result['brands'])} brands.")

        # Extract parts list
        parts_selector = selectors.get('parts', list_sel['parts'])
        part_title_selector = selectors.get('partTitle', list_sel['partTitle'])
        part_image_selector = selectors.get('partImage', list_sel['partImage'])
        part_price_selector = selectors.get('partPrice', list_sel['partPrice'])
        read_more_selector = selectors.get('readMoreLink', list_sel['readMoreLink'])

        for element in soup.select(parts_selector):
            part = {}

            # Extract part title
            title_element = element.select_one(part_title_selector)
            if title_element:
                part['title'] = clean_text(title_element.get_text())

            # Extract part image
            img_element = element.select_one(part_image_selector)
            if img_element:
                img_src = img_element.get('data-src') or img_element.get('src')
                if img_src:
                    part['imageUrl'] = urljoin(BASE_URL, img_src)

            # Extract part price
            price_element = element.select_one(part_price_selector)
            if price_element:
                 # Avoid crossed-out prices if possible (check class)
                 if 'original-price' not in price_element.get('class', []):
                    part['price'] = clean_price(price_element.get_text())

            # Extract detail URL (Read More link)
            detail_url = None
            read_more_element = element.select_one(read_more_selector)
            if read_more_element:
                detail_url = read_more_element.get('href')
            else:
                # Fallback: check all links within the part container
                found_detail_link = False
                for link in element.select('a'):
                    href = link.get('href')
                    # Heuristic: look for PS number or .htm in the link
                    if href and ('PS' in href or ('-' in href and href.endswith('.htm'))):
                        detail_url = href
                        found_detail_link = True
                        break
                # Fallback 2: Construct URL from PS number in title if available
                if not found_detail_link and part.get('title'):
                    ps_match = re.search(r'PS\d+', part['title'])
                    if ps_match:
                        detail_url = f"/{ps_match.group(0)}.htm" # Relative path

            if detail_url:
                part['detailUrl'] = urljoin(BASE_URL, detail_url)
                # Add SourceCode parameter if missing (as in JS)
                # Note: This might be specific tracking, keep if needed
                # parsed_url = urlparse(part['detailUrl'])
                # query = parsed_url.query
                # if 'SourceCode=' not in query:
                #     separator = '&' if query else '?'
                #     part['detailUrl'] += f"{separator}SourceCode=18"


            # Add part if essential info is present
            if part.get('title') and part.get('detailUrl'):
                result['parts'].append(part)
            elif part.get('title'):
                 print(f"Warning: Part '{part['title']}' found without a detail URL.")


        print(f"Found {len(result['parts'])} parts on the list page.")
        return result


# --- Processing Functions ---

async def process_url_to_json(url, output_json_path, selectors=None):
    """Fetches a URL, extracts data, and saves it to a JSON file."""
    if selectors is None: selectors = {}
    try:
        print(f"Processing URL: {url}")
        # Fetch HTML
        html = await fetch_html_with_user_agent(url)

        # Extract data
        # Pass URL to extractor for context (e.g., fallback extraction)
        output_data = extract_data_from_html(html, selectors, source_url=url)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

        # Write JSON file
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"Successfully processed {url} to {output_json_path}")
        return output_data
    except Exception as error:
        print(f"Error processing {url}: {error}")
        # Optionally, save error information
        error_path = output_json_path.replace('.json', '.error.txt')
        os.makedirs(os.path.dirname(error_path), exist_ok=True)
        with open(error_path, 'w', encoding='utf-8') as f:
            f.write(f"URL: {url}\nError: {error}\n")
            import traceback
            traceback.print_exc(file=f)
        return None


def ensure_directories(dir_paths):
  """Creates directories if they don't exist."""
  for dir_path in dir_paths:
    if not os.path.exists(dir_path):
      os.makedirs(dir_path, exist_ok=True)
      print(f"Created directory: {dir_path}")


# --- Main Execution Example ---
if __name__ == "__main__":
    import asyncio

    async def main():
        # --- Example Usage ---
        print("--- HTML Parser Python Script ---")

        # Define output directories
        output_dir = "scraped_data"

        ensure_directories([output_dir])

        # == Example 1: Process a single Part Detail Page URL ==
        # Use a known part detail page URL from partselect.com
        detail_page_url = "https://www.partselect.com/PS11752778-Whirlpool-WPW10321304-Refrigerator-Door-Shelf-Bin.htm?SourceCode=3"
        print(f"\n--- Example 1: Processing Single Detail URL: {detail_page_url} ---")
        await process_url_to_json(
            detail_page_url,
            os.path.join(output_dir, "WPW10321304_detail.json"),
            selectors=SELECTORS['partDetailPage'] # Explicitly use detail selectors
        )
        sleep(2000) # Pause between examples

        print("\n--- Script Finished ---")

    # Run the async main function
    asyncio.run(main())


