# document_searcher.py

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models
# from qdrant_client.http.models import Distance, VectorParams

from openai import OpenAI
# from anthropic import Anthropic
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from transformers import CLIPProcessor, CLIPModel
import torch
import re
from pathlib import Path
import logging
import json
import os
# import shutil
# import atexit
import string


@dataclass
class ImageSearchFilters:
    """
    Dataclass for image search filter parameters.
    Defines the three main filters used for image searching:
    - categories: List of image categories to filter by
    - text_in_image_query: Text content within the image
    - image_caption: Caption/description of the image
    """
    categories: Optional[List[str]] = None      # Filter by image_category field
    text_in_image_query: Optional[str] = None         # Filter by text_in_image_query field
    image_caption: Optional[str] = None         # Filter by image_caption field


@dataclass
class ScoreConfig:
    """Configuration for score calculations"""
    BASE_THRESHOLD: float = 0.2
    MAX_BOOST: float = 1.65  # Maximum total boost possible (65% boost)
    # Individual boost weights as percentages of MAX_BOOST
    WEIGHT_DISTRIBUTION = {
        'category': 0.40,    # 40% - Image category importance
        'text': 0.35,        # 35% - Text in image importance
        'caption': 0.25      # 25% - Image caption importance
    }
    FILTER_WEIGHT: float = 0.05


class ScoreCalculator:
    def __init__(self, config: ScoreConfig):
        self.config = config
        # Calculate individual boosts based on weight distribution
        total_boost_range = self.config.MAX_BOOST - 1.0
        self.boosts = {
            'category': 1.0 + (total_boost_range * self.config.WEIGHT_DISTRIBUTION['category']),
            'text': 1.0 + (total_boost_range * self.config.WEIGHT_DISTRIBUTION['text']),
            'caption': 1.0 + (total_boost_range * self.config.WEIGHT_DISTRIBUTION['caption'])
        }

    def calculate_adjusted_threshold(self, filters: ImageSearchFilters) -> float:
        """Calculate adjusted threshold based on available filters"""
        if not filters:
            return self.config.BASE_THRESHOLD

        available_weights = sum(
            self.config.WEIGHT_DISTRIBUTION[filter_type]
            for filter_type, has_filter in {
                'category': bool(filters.categories),
                'text': bool(filters.text_in_image_query),
                'caption': bool(filters.image_caption)
            }.items()
            if has_filter
        )

        if available_weights > 0:
            return self.config.BASE_THRESHOLD * (1.0 + available_weights)
        return self.config.BASE_THRESHOLD

    def preprocess_text(self, input_text: str) -> str:
        """Preprocess text by removing punctuation and standardizing whitespace"""

        # Remove punctuation
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        cleaned = input_text.translate(translator)

        # Convert to lowercase and standardize whitespace
        cleaned = ' '.join(cleaned.lower().split())
        return cleaned


    def _text_relevance_check(self, text: str, query: str) -> float:
        """Check text relevance and return a score between 0 and 1"""
        if isinstance(text, list):
            text = ' '.join(text)

        if not text or not query:
            return 0.0

        # Preprocess both text
        processed_text = self.preprocess_text(text)

        query_tokens = set(query.lower().split())
        text_tokens = set(processed_text.split())

        print("*" * 60)
        print("\ntext relevance check results")
        print("\nDEBUG: User Query : ", query)
        print("\nDEBUG: Embedded Text : ", text)
        print("\nDEBUG: Processed Embedded Text : ", processed_text)
        print("query_tokens : ", query_tokens)
        print("text_tokens : ", text_tokens)

        if not query_tokens:
            return 0.0

        overlap = query_tokens.intersection(text_tokens)
        print("overlap : ", overlap)
        print("*" * 60)

        return len(overlap) / len(query_tokens)

    def calculate_final_score(self, base_score: float, result_payload: Dict,
                          filters: ImageSearchFilters) -> float:
        """Calculate final score with redistributed weights"""
        if not filters:
            return base_score

        # Initialize score components
        score_components = {
            'base': base_score,
            'category': 0.0,
            'text': 0.0,
            'caption': 0.0
        }

        # Calculate individual component scores
        if filters.categories:
            category_match = any(cat in result_payload.get('image_category', [])
                             for cat in filters.categories)
            score_components['category'] = 1.0 if category_match else 0.0

        if filters.text_in_image_query:
            print("DEBUG: calculate_final_score: text_in_image_query : ", filters.text_in_image_query)
            score_components['text'] = self._text_relevance_check(
                result_payload.get('text_in_image', ''), filters.text_in_image_query)

        if filters.image_caption:
            score_components['caption'] = self._text_relevance_check(
                result_payload.get('image_caption', ''), filters.image_caption)

        # Calculate available filters and total weight
        available_filters = {
            'category': bool(filters.categories),
            'text': bool(filters.text_in_image_query),
            'caption': bool(filters.image_caption)
        }

        total_available_weight = sum(
            self.config.WEIGHT_DISTRIBUTION[filter_type]
            for filter_type, is_available in available_filters.items()
            if is_available
        )

        # Calculate final score with weight redistribution
        if total_available_weight > 0:
            weighted_sum = sum(
                score_components[filter_type] * (
                    self.config.WEIGHT_DISTRIBUTION[filter_type] / total_available_weight
                )
                for filter_type, is_available in available_filters.items()
                if is_available
            )
            boost_factor = 1.0 + (weighted_sum * (self.config.MAX_BOOST - 1.0))
            final_score = base_score * boost_factor
        else:
            final_score = base_score

        return min(final_score, 1.0)

    def get_score_breakdown(self, base_score: float, result_payload: Dict,
                        filters: ImageSearchFilters) -> Dict[str, float]:
        """Get detailed breakdown of score components"""
        if not filters:
            return {'final_score': base_score, 'components': {}}

        score_components = {
            'base_score': base_score,
            'category_contribution': 0.0,
            'text_contribution': 0.0,
            'caption_contribution': 0.0
        }

        final_score = self.calculate_final_score(base_score, result_payload, filters)

        # Calculate individual contributions
        if filters.categories:
            category_match = any(cat in result_payload.get('image_category', [])
                             for cat in filters.categories)
            score_components['category_contribution'] = (
                self.boosts['category'] - 1.0) if category_match else 0.0

        if filters.text_in_image_query:
            text_relevance = self._text_relevance_check(
                result_payload.get('text_in_image', ''), filters.text_in_image_query)
            score_components['text_contribution'] = (
                self.boosts['text'] - 1.0) * text_relevance

        if filters.image_caption:
            caption_relevance = self._text_relevance_check(
                result_payload.get('image_caption', ''), filters.image_caption)
            score_components['caption_contribution'] = (
                self.boosts['caption'] - 1.0) * caption_relevance

        score_components['final_score'] = final_score
        return score_components


class CustomCLIPEmbeddings:
    """Custom CLIP embeddings class that uses existing get_clip_text_embedding method"""

    def __init__(self, parent_instance):
        self.parent = parent_instance

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get CLIP embeddings for a list of texts using existing method."""
        try:
            return [self.parent.get_clip_text_embedding(text) for text in texts]
        except Exception as e:
            self.parent.logger.error(f"Error in embed_documents: {str(e)}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Get CLIP embedding for a single query text using existing method."""
        try:
            return self.parent.get_clip_text_embedding(text)
        except Exception as e:
            self.parent.logger.error(f"Error in embed_query: {str(e)}")
            raise


class DocumentSearcher:
    def __init__(self, config):
        """Initialize searcher with OpenAI client and Qdrant client"""

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY, timeout=60.0)
        # self.anthropic_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)

        # Initialize Qdrant Cloud client
        try:
            if not config.QDRANT_URL or not config.QDRANT_API_KEY:
                raise ValueError("Qdrant Cloud credentials not found in environment variables")

            self.qdrant_client = QdrantClient(
                url=config.QDRANT_URL,
                api_key=config.QDRANT_API_KEY
            )

            # Initialize collections
            self._init_collections()

        except Exception as e:
            error_msg = f"Qdrant Cloud initialization failed: {str(e)}"
            self.logger.error(error_msg)
            raise

        # Test connection immediately
        collections = self.qdrant_client.get_collections()
        self.logger.info(
            f"Successfully connected to Qdrant. Found collections: {[c.name for c in collections.collections]}")

        # Initialize score calculator
        self.score_calculator = ScoreCalculator(ScoreConfig())

        # Initialize CLIP for image search
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clip_model = CLIPModel.from_pretrained(config.CLIP_MODEL).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(config.CLIP_PROCESSOR)

    def _init_collections(self):
        """Verify Qdrant collections exist"""
        try:
            # Text collection
            try:
                text_collection = self.qdrant_client.get_collection(self.config.COLLECTION_NAME)
                self.logger.info(f"Found text collection: {self.config.COLLECTION_NAME}")

            except Exception as e:
                error_msg = f"Text collection '{self.config.COLLECTION_NAME}' not found: {str(e)}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Image collection
            image_collection = f"{self.config.COLLECTION_NAME}_images"
            try:
                image_coll = self.qdrant_client.get_collection(image_collection)
                self.logger.info(f"Found image collection: {image_collection}")
            except Exception as e:
                error_msg = f"Image collection '{image_collection}' not found: {str(e)}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Verify collection configurations
            if text_collection.config.params.vectors.size != 1536:
                raise ValueError(
                    f"Text collection has incorrect vector size: {text_collection.config.params.vectors.size}")

            if image_coll.config.params.vectors.size != 512:
                raise ValueError(f"Image collection has incorrect vector size: {image_coll.config.params.vectors.size}")

        except Exception as e:
            error_msg = f"Collection verification failed: {str(e)}"
            self.logger.error(error_msg)
            raise

    def _cleanup_qdrant(self):
        """Clean up Qdrant connection"""
        try:
            if hasattr(self, 'qdrant_client'):
                self.qdrant_client.close()
                self.logger.info("Qdrant client closed successfully")
        except Exception as e:
            self.logger.error(f"Error during Qdrant cleanup: {str(e)}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self._cleanup_qdrant()
        if hasattr(self, 'task'):
            self.task.close()

    def get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for text search query"""
        response = self.openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=[text]
        )
        return response.data[0].embedding

    @torch.no_grad()  # Add decorator for better performance
    def get_clip_text_embedding(self, text: str) -> List[float]:
        """Get CLIP embedding for image search query"""
        try:
            inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True).to(self.device)

            text_features = self.clip_model.get_text_features(**inputs)

            # Normalize the embeddings
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            return text_features.squeeze().cpu().numpy().tolist()
        except Exception as e:
            self.logger.error(f"Error generating CLIP embedding: {str(e)}")
            raise

    def clean_text(self, text: str) -> str:
        """Clean text by removing trailing numbers and extra whitespace"""
        # Remove trailing numbers and clean whitespace
        text = text.strip()
        text = re.sub(r'\s*\d+\s*$', '', text)
        # Remove multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text

    def is_valid_content(self, text: str, query: str) -> bool:
        """Filter out metadata, headers, and navigational content"""

        if len(text.strip()) < 100:  # Minimum 100 characters
            return False
            
        metadata_patterns = [
            r"^Page \d+$",
            r"^Chapter \d+$",
            r"^\d{1,2}/\d{1,2}/\d{4}$",
            r"^Table of Contents$",
            r"^Questions and Answers$",
            r"^[\d\s\-—]*$",  # Just numbers, spaces, dashes
            r"^\s*\(.*\)\s*$"  # Just parenthetical content
        ]

        for pattern in metadata_patterns:
            if re.match(pattern, text.strip()):
                return False
                
        return True

    def create_filter_prompt(self, query: str) -> str:
        """
        Create prompt for extracting key words from query
        """
        prompt = f"""
        Extract the most important keywords from this image search query that would directly influence the search results. 
        Return only the keywords as a JSON array.

        Query: {query}

        Example:
        Query: "Show me a circuit diagram with resistor and capacitor"
        Response: ["resistor", "capacitor"]

        Query: "Find an image of a red sports car on a racetrack"
        Response: ["red", "sports car", "racetrack"]
        """
        return prompt

    def detect_image_filters(self, query: str) -> str:
        """
        Extract critical keywords using Claude with improved prompt
        """
        try:
            system_prompt = """You are an expert at extracting critical keywords from search queries.
    Your task is to identify only the most meaningful words that capture the core intent of the query.

    Guidelines:
    - Focus on content-carrying words that are essential for understanding the query intent
    - Automatically remove stop words, articles, prepositions, and generic verbs
    - Extract nouns, important adjectives, and domain-specific terms
    - Return ONLY a list of words, like: ["word1", "word2", "word3"]
    
    INCLUDE ONLY:
    - Domain-specific technical terms
    - Meaningful adjectives that modify core concepts
    - Specific nouns representing entities, systems, or concepts
    - Action words that represent specific technical operations

    EXCLUDE from response:
    - Articles (a, an, the)
    - Prepositions (of, in, at, etc.)
    - Generic verbs (show, get, find, display)
    - Common generic words: data, information, details, dataset,list, content, value, values, score, scores,and similar words

    Example:
    Query: "analyze machine learning performance in production environment"
    Response: ["analyze", "machine", "learning", "performance", "production", "environment"]
    
    Remember: If a word is generic, descriptive, or metadata-related (like data, details, information, 
    values, etc.), it must NOT be included.
    """

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract critical keywords from: {query}"}
            ]

            chat = ChatAnthropic(
                model="claude-3-5-haiku-20241022",
                temperature=0,
                max_tokens=4096
            )

            # Get analysis from Claude
            response = chat.invoke(messages)

            content = response.content
            cleaned_content = content.replace('```json\n', '').replace('\n```', '')
            keywords = json.loads(cleaned_content)

            if isinstance(keywords, list) and keywords:
                combined_keywords = " ".join(keywords)
                print(f"\nDEBUG: Query: {query}")
                print(f"DEBUG: Extracted Keywords: {combined_keywords}")
                return combined_keywords

            return ""

        except json.JSONDecodeError as e:
            print(f"Failed to parse response: {str(e)}")
            return ""
        except Exception as e:
            print(f"Error extracting keywords: {str(e)}")
            return ""


    def detect_filters(self, query: str) -> ImageSearchFilters:
        """
        Automatically detect and create filters based on query content and context
        """
        filters = ImageSearchFilters()
        query_lower = query.lower()

        # 1. Detect categories
        category_keywords = {
            'diagram': ['diagram', 'schematic', 'blueprint', 'flowchart', 'layout'],
            'photo': ['photo', 'picture', 'image', 'photograph'],
            'graph': ['graph', 'chart', 'plot', 'visualization'],
            'table': ['table', 'grid', 'matrix'],
            'technical': ['technical', 'specification', 'spec', 'engineering'],
            'document': ['document', 'pdf', 'doc', 'report']
        }

        detected_categories = []
        for category, keywords in category_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_categories.append(category)

        if detected_categories:
            filters.categories = detected_categories

        filters.text_in_image_query = self.detect_image_filters(query)
        print("\n\nDEBUG:detect_filters: filter.text_in_image_query : ", filters.text_in_image_query)

        # 3. Extract potential caption content
        caption_patterns = [
            r'captioned "([^"]+)"',
            r'labeled "([^"]+)"',
            r'titled "([^"]+)"',
            r'described as "([^"]+)"'
        ]

        for pattern in caption_patterns:
            matches = re.findall(pattern, query)
            if matches:
                filters.caption = matches[0]
                break

        # If no specific text or caption patterns found, use remaining text
        if not any([filters.text_in_image_query, filters.image_caption]):
            # Clean query by removing category keywords
            cleaned_query = query_lower
            for keywords in category_keywords.values():
                for keyword in keywords:
                    cleaned_query = cleaned_query.replace(keyword, '')

            # Remove special characters and extra spaces
            cleaned_query = re.sub(r'[^\w\s]', ' ', cleaned_query)
            cleaned_query = ' '.join(cleaned_query.split())

            if len(cleaned_query) > 3:  # More than 3 characters
                filters.text_in_image_query = cleaned_query  # Use as text_in_image search

        return filters

    def search_text(self, query: str, limit: int = None, score_threshold: float = None) -> List[Dict]:
        """Search for similar text chunks"""
        query_vector = self.get_text_embedding(query)
        
        results = self.qdrant_client.search(
            collection_name=self.config.COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="type",
                        match=models.MatchValue(value="text")
                    )
                ]
            ),
            limit=(limit or self.config.SEARCH_LIMIT) * 2,
            score_threshold=score_threshold or self.config.SIMILARITY_THRESHOLD
        )

        filtered_results = []
        for result in results:
            try:
                if self.is_valid_content(result.payload['text'], query):
                    cleaned_text = self.clean_text(result.payload['text'])
                    cleaned_text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', cleaned_text)
                
                    filtered_result = {
                        'type': 'text',
                        'filename': result.payload['filename'],
                        'page_number': result.payload['page_number'],
                        'page_header': result.payload.get('page_header', '').strip(),
                        'text': cleaned_text,
                        'score': result.score
                    }
                    filtered_results.append(filtered_result)
                
            except Exception as e:
                print(f"Error processing text result: {str(e)}")
                continue
            
        return filtered_results[:limit or self.config.SEARCH_LIMIT]

    def search_images(self,
                      query: str,
                      filters: Optional[ImageSearchFilters] = None,
                      limit: int = None,
                      score_threshold: float = 0.3) -> List[Dict]:
        """Enhanced image search with two-stage filtering"""

        # Check if collection exists and has points
        try:
            # Get CLIP text embedding for search
            query_for_vector_search = filters.text_in_image_query
            print("\nDEBUG: search_images: query: ", query_for_vector_search)

            # query_vector = self.get_clip_text_embedding(query_for_vector_search)
            query_vector = self.get_clip_text_embedding(query)

            # Stage 1: Vector Search - with relaxed threshold
            results = self.qdrant_client.search(
                collection_name=f"{self.config.COLLECTION_NAME}_images",
                query_vector=query_vector,
                limit=(limit or self.config.SEARCH_LIMIT) * 2,
                score_threshold=0.22  # Relaxed threshold for initial retrieval
            )

            print("\nDebug - Raw search results with scores:")
            for res in results:
                print(f"Score: {res.score:.4f} - Text: {res.payload.get('text_in_image', 'No text')}")

            print("\nDebug - Raw image search results:")

            processed_results = []
            query_vector_text = self.get_text_embedding(query_for_vector_search)

            for result in results:
                try:
                    # Get the storage path from result payload
                    stored_path = result.payload['source_path']

                    # Convert to absolute path if needed
                    # image_path = Path(self.config.IMAGE_STORAGE_PATH) / Path(stored_path).name
                    # image_path = Path(self.config.IMAGE_STORAGE_PATH) / os.path.basename(stored_path)
                    stored_path = stored_path.replace('\\', '/')
                    filename = stored_path.split('/')[-1]
                    image_path = Path(self.config.IMAGE_STORAGE_PATH) / filename

                    print(f"\nChecking image at: {image_path}")

                    if not image_path.exists():
                        print(f"Warning: Image not found at {image_path}")
                        continue

                    # Get text from image
                    text_in_image = result.payload.get('text_in_image', '')

                    if not text_in_image or text_in_image.strip() == '':
                        print(f"Skipping result due to empty text_in_image")
                        continue

                    print("\n\nDEBUG: search_images : text_in_image : ", text_in_image)
                    cleaned_text_in_image = self.score_calculator.preprocess_text(text_in_image)
                    print("\n\nDEBUG: search_images : cleaned text_in_image : ", cleaned_text_in_image)

                    # Stage 2: Keyword Similarity Check
                    text_similarity = self.calculate_cosine_similarity(
                            query_vector_text, self.get_text_embedding(cleaned_text_in_image)
                    )

                    # Only process results that pass keyword similarity threshold
                    KEYWORD_SIMILARITY_THRESHOLD = 0.8
                    if text_similarity >= KEYWORD_SIMILARITY_THRESHOLD:
                        # Calculate final score combining vector and keyword similarity
                        combined_score = (result.score * 0.6) + (text_similarity * 0.4)

                        if combined_score >= 0.33:  # Final quality threshold
                            print(f"\nScore details for {image_path.name}:")
                            print(f"Original score: {result.score:.4f}")
                            print(f"text similarity score: {text_similarity:.4f}")
                            print(f"Final adjusted score: {combined_score:.4f}")
                            print(f"Text in image: {result.payload.get('text_in_image', 'No text')}")

                            processed_result = {
                                'type': 'image',
                                'source_path': str(image_path),
                                'filename': image_path.name,
                                'storage_path': str(self.config.IMAGE_STORAGE_PATH),
                                'page_number': result.payload['page_number'],
                                'image_number': result.payload['image_number'],
                                'text_in_image': result.payload.get('text_in_image', ''),
                                'image_caption': result.payload.get('image_caption', ''),
                                'image_category': result.payload.get('image_category', []),
                                'score': combined_score,
                                'score_breakdown': {
                                    'vector_similarity': result.score,
                                    'keyword_similarity': text_similarity,
                                    'combined_score': combined_score
                                }
                            }
                            processed_results.append(processed_result)
                        else:
                            print(f"Dropped result due to low keyword similarity: {text_similarity:.4f}")

                except Exception as e:
                    print(f"Error processing result: {str(e)}")
                    continue

            # Sort results by similarity score in descending order
            processed_results.sort(key=lambda x: x['score'], reverse=True)

            if processed_results:
                self.logger.info(f"""
                    Image Search Summary:
                    - Total results found: {len(processed_results)}
                    - Score range: {min([r['score'] for r in processed_results])} - {max([r['score'] for r in processed_results])}
                    - Categories found: {set([cat for r in processed_results for cat in r['image_category']])}
                    """)
            else:
                self.logger.info("No image results found matching the criteria")

            # return processed_results[:limit or self.config.SEARCH_LIMIT]
            return processed_results

        except Exception as e:
            print(f"Error in image search: {str(e)}")
            return []

    def search_excel(self, query: str, limit: int = None, score_threshold: float = None) -> List[Dict]:
        """Search for Excel sheets and their contents"""
        query_vector = self.get_text_embedding(query)

        try:
            results = self.qdrant_client.search(
                collection_name=self.config.COLLECTION_NAME,
                query_vector=query_vector,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="type",
                            match=models.MatchValue(value="excel")
                        )
                    ]
                ),
                limit=(limit or self.config.SEARCH_LIMIT) * 2,
                score_threshold=score_threshold or self.config.SIMILARITY_THRESHOLD,
            )

            print("\n\nDebug:search_excel: - qdrant result:", results)
            filtered_results = []
            for result in results:
                try:
                    # Split text into lines and parse each line
                    text_lines = result.payload['text'].split('\n')
                    filename = None
                    filepath = None
                    headers = []
                    sheet_name = None

                    # Parse the text content
                    for line in text_lines:
                        if line.startswith('File: '):
                            filename = line.replace('File: ', '').strip()
                        elif line.startswith('Path: '):
                            filepath = line.replace('Path: ', '').strip()
                        elif line.startswith('Headers: '):
                            headers = [h.strip() for h in line.replace('Headers: ', '').split(',')]
                        elif line.startswith('Sheet: '):
                            sheet_name = line.replace('Sheet: ', '').strip()

                    # Use values from text content or fall back to payload fields
                    filename = filename or result.payload.get('filename')
                    sheet_name = sheet_name or result.payload.get('page_header', '')

                    if not filename or not filename.endswith('.xlsx'):
                        continue

                    filtered_result = {
                        'type': 'excel',
                        'filename': filename,
                        'filepath': filepath,
                        'sheet_name': sheet_name,
                        'headers': headers,
                        'score': result.score
                    }
                    print("\n\nDebug:search_excel: - Excel search result:", filtered_result)
                    if filepath:  # Only add if we found a valid filepath
                        filtered_results.append(filtered_result)

                except Exception as e:
                    print(f"Error processing excel result: {str(e)}")
                    continue

            print("\n\nDebug:search_excel: - Excel search results:", filtered_results)
            return filtered_results[:limit or self.config.SEARCH_LIMIT]

        except Exception as e:
            print(f"Error in excel search: {str(e)}")
            return []

    def create_excel_prompt(self, query: str, excel_content: str) -> str:
        """Create prompt for LLM to analyze Excel structure and query requirements"""

        prompt = f"""You are an expert at analyzing Excel data structures and requirements. Analyze this Excel file content and query.

    Excel content (first few rows):
    {excel_content}

    Analysis steps:
    1. Analyze the Excel structure:
       - Identify title/header rows
       - Find actual column headers where data exists below them
       - Determine where data rows begin

    2. For the user query: "{query}"
       - Determine which columns need to be extracted
       - Identify any filtering conditions

    Verification requirements:
    - Only include columns that have actual data beneath them
    - Verify data patterns match column structures
    - Ensure column names exactly match where data appears

    Return your response in this exact JSON structure:
    {{
        "excel_structure": {{
            "title": "title if present",
            "columns": ["list of columns that have actual data beneath them"],
            "data_starts_at_row": row_number
        }},
        "query_requirements": {{
            "columns_to_extract": ["list of needed columns confirmed to have data"],
            "filters": "filter conditions"
        }}
    }}

    Important: Your response must be valid JSON. Replace the placeholder values with actual analysis results while maintaining this exact structure.

    User Query: {query}"""

        return prompt

    def process_excel_data(self, query: str, file_path: str) -> dict:
        """Process Excel file based on GPT response"""
        try:
            # Read first few rows to analyze structure including raw data
            df = pd.read_excel(file_path, header=None, nrows=10)  # Increased rows to get better context

            # Convert first few rows to string format for LLM analysis
            excel_content = df.to_string(index=False, na_rep='')
            # print("\n\nDEBUG:process_excel_data: excel_content for LLM:", excel_content)

            # Create prompt with raw content
            prompt = self.create_excel_prompt(query, excel_content)
            print("\n\nDEBUG:process_excel_data: created prompt:", prompt)

            # Initialize ChatAnthropic
            chat = ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                temperature=0.3,
                max_tokens=4096
            )

            # Get LLM response using ChatAnthropic
            llm_response = chat.invoke(prompt)

            print("\n\nDEBUG:process_excel_data: gpt_reponse :", llm_response)

            # Extract and clean JSON from response
            content = llm_response.content

            # Find JSON block in the response
            try:
                # First try to find JSON in code block
                json_start = content.find('```json') + 7
                json_end = content.find('```', json_start)
                if json_start > 6 and json_end > json_start:  # Valid JSON block found
                    json_str = content[json_start:json_end].strip()
                else:
                    # If no code block, try to find JSON object directly
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = content[json_start:json_end].strip()
                    else:
                        raise ValueError("No valid JSON found in response")
            except Exception as e:
                print(f"JSON extraction error: {str(e)}")
                return {"error": "Failed to extract JSON from LLM response"}

            # Clean the JSON string
            json_str = json_str.replace('\n', '')  # Remove newlines
            json_str = ' '.join(json_str.split())  # Normalize spaces

            print("DEBUG: Cleaned JSON string:", json_str)

            requirements = json.loads(json_str)
            print("DEBUG: Parsed requirements:", requirements)

            # Use LLM-identified structure to read Excel properly
            header_row = requirements['excel_structure']['data_starts_at_row'] - 2
            df = pd.read_excel(file_path, header=header_row)

            print("DEBUG: DataFrame columns:", df.columns.tolist())

            # Get columns to extract directly from requirements
            columns_to_fetch = requirements['query_requirements']['columns_to_extract']

            # Create initial DataFrame with required columns
            filtered_df = df[columns_to_fetch].copy()

            # Handle filtering
            filters = requirements['query_requirements']['filters']
            if filters and filters != "None" and filters.strip():
                try:
                    # Split on OR
                    or_conditions = filters.split(' OR ')
                    print("\n\nDEBUG: OR conditions : ", or_conditions)
                    result_df = None  # Start with None instead of empty DataFrame

                    for condition in or_conditions:
                        # Clean up the condition
                        condition = condition.strip()
                        print("\n\nDEBUG: individual condition : ", condition)

                        # Parse single condition
                        parts = condition.split(' ', 2)  # Split into max 3 parts
                        if len(parts) >= 3:
                            column = parts[0].strip()
                            operator = parts[1].strip()
                            value = parts[2].strip().strip('"').strip("'")

                            # Check exact values in DataFrame for this condition
                            print(f"DEBUG: Values found in DataFrame that start with '{value}':")
                            print(filtered_df[filtered_df[column].str.startswith(value, na=False)][column].unique())
                            print(f"\nDEBUG: Processing condition - Column: {column}, Operator: {operator}, Value: {value}")

                            # Apply the condition
                            if operator == '=':
                                temp_df = filtered_df[filtered_df[column].astype(str) == value]
                            elif operator == '>':
                                temp_df = filtered_df[filtered_df[column].astype(float) > float(value)]
                            elif operator == '<':
                                temp_df = filtered_df[filtered_df[column].astype(float) < float(value)]
                            elif operator == 'contains':
                                temp_df = filtered_df[filtered_df[column].astype(str).str.contains(value, case=False)]
                            else:
                                print(f"Unsupported operator: {operator}")
                                temp_df = filtered_df

                            print(f"DEBUG: Rows found for this condition: {len(temp_df)}")

                            # Combine results with OR
                            if result_df is None:
                                result_df = temp_df
                            else:
                                result_df = pd.concat([result_df, temp_df])
                                result_df = result_df.drop_duplicates()  # Remove any duplicates

                            print(f"DEBUG: Current total rows after combining: {len(result_df)}")

                        print(f"\nDEBUG: Applied filter: {filters}")
                        print(f"DEBUG: Original rows: {len(filtered_df)}, Filtered rows: {len(result_df)}")

                        if result_df.empty:
                            return {
                                "file_path": file_path,
                                "total_rows": 0,
                                "columns": columns_to_fetch,
                                "data": [],
                                "message": "No data matches the filter criteria",
                                "excel_structure": requirements['excel_structure']
                            }

                    output_df = result_df

                except Exception as e:
                    print(f"Filter error: {str(e)}")
                    return {"error": f"Filter error: {str(e)}"}

            # Get results
            result = {
                "file_path": file_path,
                # "total_rows": len(filtered_df),
                "total_rows": len(output_df),
                "columns": columns_to_fetch,
                # "data": filtered_df.to_dict('records'),
                "data": output_df.to_dict('records'),
                "excel_structure": requirements['excel_structure']
            }
            print("\n\nDEBUG:process_excel_data: result : ", result)

            return result

        except Exception as e:
            return {"error": f"Error processing Excel: {str(e)}"}

    def search(self,
               query: str,
               image_filters: Optional[ImageSearchFilters] = None,
               limit: int = None,
               score_threshold: float = None) -> Dict[str, List[Dict]]:
        """
        Combined search function with automatic filter detection
        """
        if not query:
            raise ValueError("Query text is required")
        print("\n\nDEBUG: search : user query : ", query)

        # Auto-detect filters if none provided
        if image_filters is None:
            image_filters = self.detect_filters(query)
            print("\nAutomatically detected filters:")
            print(f"Categories: {image_filters.categories}")
            print(f"Text in Image: {image_filters.text_in_image_query}")
            print(f"Image Caption: {image_filters.image_caption}")
            print("-" * 50)

        print("\n\nDEBUG: search : user query after detect_filter : ", query)
        text_results = self.search_text(
            query=query,
            limit=limit,
            score_threshold=score_threshold
        )
        print("\nDebug - Text search results:", len(text_results))

        image_results = self.search_images(
            query=query,
            filters=image_filters,
            limit=limit,
            score_threshold=score_threshold
        )
        print("\nDebug - Image search results:", len(image_results))

        results = {
            'text': text_results,
            'images': image_results
        }

        print("\n\nDEBUG: search : user query before search excel: ", query)
        # Excel search
        excel_search_out = self.search_excel(
            query=query,
            limit=limit,
            score_threshold=self.config.SCORE_THRESHOLD_EXCEL
        )
        print("\n\nDebug:search: - excel_search_out :", excel_search_out)

        excel_results = []
        # Process Excel data if results found
        if excel_search_out:
            for excel_res in excel_search_out:
                try:
                    print("\n\nDEBUG:Search: query :", query)
                    print("\n\nDEBUG:Search: filepath :", excel_res['filepath'])

                    f_path = excel_res['filepath']
                    converted_path = f_path.replace("\\", "/")

                    processed_data = self.process_excel_data(
                        query,
                        converted_path
                    )
                    excel_results.append(processed_data)
                    print("\n\nDebug - Excel processed results:", processed_data)

                except Exception as e:
                    print(f"Error processing Excel file {excel_res['filepath']}: {str(e)}")

        results = {
            'text': text_results,
            'images': image_results,
            'excel': excel_results  # Excel results
        }

        # Analyze and group results
        analyzed_results = self.analyze_and_group_results(query, text_results, image_results, excel_results)

        return analyzed_results

    def get_collection_info(self) -> Dict[str, Dict]:
        """Get information about both collections"""
        collections_info = {}

        try:
            # Get text collection info
            text_info = self.qdrant_client.get_collection(
                collection_name=self.config.COLLECTION_NAME
            )
            print("\n\nDebugging: text_info : ", text_info)

            # Create a dictionary with text collection information
            collections_info['text'] = {
                'name': self.config.COLLECTION_NAME,            # Store collection name (e.g., "knowledge_base")
                'status': text_info.status,                     # Store collection status (e.g., "green" if healthy)
                'vectors_count': text_info.vectors_count,       # Number of vectors in the collection
                'indexed_vectors_count': text_info.vectors_count,  # Number of indexed vectors (same as vectors_count in this case)
                'points_count': text_info.points_count if hasattr(text_info, 'points_count') else None,
                'config': {
                    'vector_size': text_info.config.params.vectors.size,
                    'distance': text_info.config.params.vectors.distance
                }
            }
        except Exception as e:
            # If any error occurs while getting text collection info
            print(f"Error getting text collection info: {str(e)}")
            collections_info['text'] = {'error': str(e)}
            # Store error message instead of collection info
        
        try:
            # Start of error handling block for image collection
            # Get image collection info
            image_collection_name = f"{self.config.COLLECTION_NAME}_images"
            image_info = self.qdrant_client.get_collection(
                collection_name=image_collection_name
            )

            # Create a dictionary with image collection information
            collections_info['images'] = {
                'name': image_collection_name,                                  # Store collection name (e.g., "knowledge_base_images")
                'status': image_info.status,
                'vectors_count': image_info.vectors_count,                      # Changed from points_count
                'indexed_vectors_count': image_info.vectors_count,              # Changed from indexed_vectors_count
                'points_count': image_info.points_count if hasattr(image_info, 'points_count') else None,
                'config': {
                    'vector_size': image_info.config.params.vectors.size,
                    'distance': image_info.config.params.vectors.distance
                }
            }
        except Exception as e:
            # If any error occurs while getting image collection info
            print(f"Error getting image collection info: {str(e)}")
            collections_info['images'] = {'error': str(e)}
            # Store error message instead of collection info
        
        return collections_info
        # Return the dictionary containing information about both collections

    def analyze_and_group_results(self,
                                  query: str,
                                  text_results: List[Dict],
                                  image_results: List[Dict],
                                  excel_results: List[Dict]) -> Dict:
        """
        Analyze and group search results by source document
        """
        grouped_results = {}

        # First, group text results by filename
        for text_result in text_results:
            filename = text_result.get('filename')
            if not filename:
                continue

            if filename not in grouped_results:
                grouped_results[filename] = {
                    'texts': [],
                    'images': [],
                    'excel_data': [],
                    'summary': '',
                    'insights': '',
                    'data_analysis': ''
                }

            grouped_results[filename]['texts'].append({
                'text': str(text_result.get('text', '')),
                'page': int(text_result.get('page_number', 0)),
                'score': float(text_result.get('score', 0.0))
            })

        # Group Excel results
        for excel_result in excel_results:
            print("\n\nDEBUG: Control coming inside the excel_results for loop")
            filename = excel_result.get('file_path')
            if not filename:
                continue

            if filename not in grouped_results:
                grouped_results[filename] = {
                    'texts': [],
                    'images': [],
                    'excel_data': [],
                    'summary': '',
                    'insights': '',
                    'data_analysis': ''
                }

            # Add Excel specific information
            grouped_results[filename]['excel_data'].append({
                'data': excel_result.get('data', []),
                'columns': excel_result.get('columns', []),
                'total_rows': excel_result.get('total_rows', 0),
                'structure': excel_result.get('excel_structure', {})
            })

        # Then, group image results by matching base filename
        for image_result in image_results:
            image_filename = image_result.get('filename', '')
            if not image_filename:
                continue

            # Determine base filename based on the type of file
            if image_filename.startswith('confluence_'):
                # For confluence files
                base_filename = image_filename.split('_page')[0]
            elif image_filename.endswith('.png'):
                # For PDF and PPTX files
                base_name = image_filename.split('_page')[0]
                # Check if this is from a PPTX file first
                if any(base_name + '.pptx' == text_result.get('filename') for text_result in text_results):
                    base_filename = base_name + '.pptx'
                # Then check for PDF
                elif any(base_name + '.pdf' == text_result.get('filename') for text_result in text_results):
                    base_filename = base_name + '.pdf'
                else:
                    # Default to PDF if no match found
                    base_filename = base_name + '.pdf'
            else:
                continue

            if base_filename not in grouped_results:
                grouped_results[base_filename] = {
                    'texts': [],
                    'images': [],
                    'excel_data': [],
                    'summary': '',
                    'insights': '',
                    'data_analysis': ''
                }

            grouped_results[base_filename]['images'].append({
                'path': str(image_result.get('source_path', '')),
                'page': int(image_result.get('page_number', 0)),
                'caption': str(image_result.get('image_caption', '')),
                'score': float(image_result.get('score', 0.0))
            })

        # Process each document's results
        for doc_name, doc_results in grouped_results.items():
            # Sort by page number
            doc_results['texts'].sort(key=lambda x: x.get('page', 0))
            doc_results['images'].sort(key=lambda x: x.get('page', 0))

            # Generate insights from text content
            if doc_results['texts']:
                try:
                    text_contents = []
                    for t in doc_results['texts']:
                        if isinstance(t, dict) and 'text' in t:
                            text_contents.append(str(t['text']))

                    combined_text = ' '.join(text_contents)

                    if combined_text.strip():
                        insights_prompt = f"""
                        Based on the following text content and the query "{query}", 
                        provide key insights and relevant information:
                        {combined_text[:4000]}
                        """

                        insights_prompt = f"""
                        Analyze all the following content holistically based on the query "{query}".
                        Present a unified analysis that compares and synthesizes information across all documents.

                        Requirements:
                        1. Compare information and find relationships across all documents
                        2. When organizations or entities are mentioned, analyze their connections and comparisons
                        3. Format numerical comparisons as: [Value1] vs [Value2] (↑/↓ XX.XX%)
                        4. Identify contradictions or complementary information across sources
                        5. Structure your response with clear headings and bullet points
                                
                        Content from documents:
                        {combined_text[:4000]}
                        """

                        # Initialize ChatAnthropic
                        chat = ChatAnthropic(
                            model="claude-3-5-haiku-20241022",
                            temperature=0.3,
                            max_tokens=5000
                        )

                        # Get insights using ChatAnthropic
                        insights_response = chat.invoke(insights_prompt)
                        doc_results['insights'] = insights_response.content

                except Exception as e:
                    print(f"Error generating insights for {doc_name}: {str(e)}")
                    doc_results['insights'] = "Error generating insights"

            # Generate analysis for Excel data
            if doc_results['excel_data']:
                print("\n\nDEBUG: doc_results['excel_data'] ", doc_results['excel_data'])
                try:
                    excel_info = []
                    for excel_data in doc_results['excel_data']:
                        data = excel_data.get('data', [])
                        columns = excel_data.get('columns', [])
                        total_rows = excel_data.get('total_rows', 0)

                        excel_info.append(f"""
                        Total Rows: {total_rows}
                        Columns: {', '.join(columns)}
                        Sample Data (up to 5 rows): {str(data[:5])}
                        """)

                    if excel_info:
                        analysis_prompt = f"""
                        Based on the following Excel data and the query "{query}", 
                        provide a concise analysis of the data. Format your response using markdown syntax:
                        - Use '##' for the main 'Data Analysis' heading
                        - Use '###' for section headings
                        - Use regular text for data points and comparisons
                        
                        {' '.join(excel_info)}
                        """

                        analysis_response = self.openai_client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": """You are an expert data analyst providing clear insights from Excel data.
                                            When comparing two data points:
                                            1. Always calculate and show the percentage change
                                            2. Use ↑ for increases and ↓ for decreases
                                            3. Format as: [Value1] vs [value2] (↑/↓ XX.XX%)
                                            
                                            Always round percentages to 2 decimal places and use appropriate formatting 
                                            for numbers (e.g., commas for thousands)."""},
                                {"role": "user", "content": analysis_prompt}
                            ],
                            temperature=0.3,
                            max_tokens=5000
                        )
                        doc_results['data_analysis'] = analysis_response.choices[0].message.content
                except Exception as e:
                    print(f"Error generating Excel analysis for {doc_name}: {str(e)}")
                    doc_results['data_analysis'] = "Error analyzing Excel data"

            # Generate image summaries
            if doc_results['images']:
                try:
                    import base64
                    from PIL import Image
                    from io import BytesIO

                    messages = []
                    # System message
                    messages.append({
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": """You are an expert at analyzing visual content and providing detailed insights.
                                Important guidelines:
                                - Only include information that is directly visible or can be accurately inferred from the content
                                - Do not include any speculative or uncertain information
                                - If certain details are not clearly visible or cannot be confidently inferred, exclude them from the response
                                - For inferred information, base it strictly on clear visual evidence present in the content
                                - Focus on providing factual, observable details rather than assumptions"""
                            }
                        ]
                    })

                    # User message with query context and images
                    user_content = []
                    user_content.append({
                        "type": "text",
                        "text": f"Query: {query}\n\nAnalyze these images and provide relevant insights:"
                    })

                    # Process each image
                    for img in doc_results['images']:
                        if isinstance(img, dict):
                            image_path = img.get('path', '')
                            if image_path and os.path.exists(image_path):
                                # Read and encode image - using same path as Streamlit
                                with Image.open(image_path) as image:
                                    if image.mode not in ['RGB', 'L']:
                                        image = image.convert('RGB')

                                    # Resize if needed
                                    max_size = (1024, 1024)
                                    image.thumbnail(max_size, Image.Resampling.LANCZOS)

                                    # Convert to base64
                                    buffered = BytesIO()
                                    image.save(buffered, format="JPEG")
                                    img_base64 = base64.b64encode(buffered.getvalue()).decode()

                                    # Add image to content
                                    user_content.append({
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/jpeg",
                                            "data": img_base64
                                        }
                                    })

                                    # Add context matching the display caption
                                    user_content.append({
                                        "type": "text",
                                        "text": f"\nPage {img.get('page', '')}\nCaption: {img.get('caption', '')}\nScore: {img.get('score', 0.0):.4f}\n"
                                    })

                    if user_content:
                        messages.append({
                            "role": "user",
                            "content": user_content
                        })

                        chat = ChatAnthropic(
                            model="claude-3-5-haiku-20241022",
                            temperature=0,
                            max_tokens=4096
                        )

                        # Get analysis from Claude
                        summary_response = chat.invoke(messages)
                        doc_results['summary'] = summary_response.content

                except Exception as e:
                    print(f"Error generating image summary for {doc_name}: {str(e)}")
                    doc_results['summary'] = f"Error analyzing images: {str(e)}"

        print("\n\nDEBUG:analyze_and_group_results: grouped_results : ", grouped_results)

        return grouped_results

    def calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            if not vec1 or not vec2:
                return 0.0

            # Convert to numpy arrays
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)

            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)

            # Ensure the result is between 0 and 1
            return float(max(0.0, min(1.0, similarity)))

        except Exception as e:
            print(f"Error calculating cosine similarity: {str(e)}")
            return 0.0