# app.py

import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Dict, Tuple
from openai import OpenAI
from document_searcher import DocumentSearcher
from langchain_anthropic import ChatAnthropic
from config import Config
import os
import re
import logging

# Initialize logging
logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class QASystem:
    def __init__(self):
    
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.searcher = DocumentSearcher(Config)

    def get_chat_response(self, query: str, search_results: Dict[str, Dict]) -> Tuple[str, Dict]:
        """
        Get response from GPT-3.5 Turbo using context
        Args:
            query: User's question
            search_results: Dictionary containing results by filename
        Returns:
            Tuple of (response text, context used)
        """
        # Prepare context by combining all document data
        context_parts = []
        for filename, doc_data in search_results.items():
            doc_context = f"Document: {filename}\n"
            if doc_data.get('summary'):
                doc_context += f"Summary: {doc_data['summary']}\n"
            if doc_data.get('insights'):
                doc_context += f"Insights: {doc_data['insights']}\n"
            if doc_data.get('data_analysis'):
                doc_context += f"Analysis: {doc_data['data_analysis']}\n"

            # Add references
            if doc_data['texts']:
                doc_context += "Text References:\n" + "\n".join(
                    f"- Page {text['page']}: {text['text']}"
                    for text in doc_data['texts']
                ) + "\n"

            context_parts.append(doc_context)

        context_str = "\n\n".join(context_parts)

        system_prompt = """You are an executive dashboard generator specialized in synthesizing information 
        from multiple artifacts.

Generate only 4 sections as mentioned below. Each section should provide the details by considering all input
documents.

REQUIRED SECTIONS AND FORMATS:

# 📊 Executive Summary
This section summarizes the all the findings. Synthesize the data and relevant information into a 
cohesive and detailed narrative, pertinent to the user query. Present your findings in paragraph format. 
Highlight the key points in bold.

# 💡 Key Insights
This section provides a detailed analytics. Create NEW detailed insights by Analyzing patterns 
across all documents, finding relationships between different data points, and identifying trends and 
correlations among the data points.
CRITICAL: YOU MUST STRICTLY FOLLOW THE FORMAT SHOWN IN THE EXAMPLES BELOW - INCLUDING LINE BREAKS AND INDENTATION.

MANDATORY FORMAT FOR KEY INSIGHTS:
----------------------------------------
🔍 [Topic Name in bold]

• [Main Point 1]
    - [Supporting detail 1]
    - [Supporting detail 2]
    - [Supporting detail 3]

• [Main Point 2]
    - [Supporting detail 1]
    - [Supporting detail 2]
    - [Supporting detail 3]
----------------------------------------

For example:
🔍 **Market Leadership & Scale**

• Elsevier demonstrates dominant market position
- Controls 17% of global research output 
- Generates 29% of global citations 
- Hosts 22 million content pieces on ScienceDirect

• Springer Nature shows strong growth momentum 
- Published 420,000+ articles and 14,000+ books 
- Achieved 2.9% underlying revenue growth 
- Maintains 1.8 million registered platform users

# 📈 Data Analysis
Include only if numerical/comparative data exists. Present a comparative view if so asked. 
Format numerical comparisons as: [Value1] vs [Value2] (↑/↓ XX.XX%), ↑ in Green and ↓ in Red.
Always format the data professionally, the way the CXOs would like to see.

# 📚 References
[Required: Just write the text "The references are mentioned below")]

STRICT FORMATTING RULES FOR ALL SECTIONS:
1. Every section header must be formatted as "# [emoji] [Title]"
2. Every sub-header must be formatted with 🔍 and in bold
3. Main points must use • bullet points
4. Each sub-point MUST:
   - Start on a new line with -
5. NO horizontal lists with hyphens
6. NEVER combine sub-points with hyphens in one line
7. Always include blank lines between:
   - Section headers and content
   - Between main bullet points
   - Before and after sub-headers
8. Always provide data points when drawing insights
9. Skip sections (except References) if no meaningful content exists
10. Maintain professional, executive-level language
11. Ensure response is well-spaced and not cluttered"""

        user_prompt = f"Analyze this content:\n{context_str}\n\nQuery: {query}"

        try:
            # Initialize ChatAnthropic
            chat = ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                temperature=0.7,
                max_tokens=6000
            )

            # Get LLM response using ChatAnthropic
            response = chat.invoke(
                system_prompt + "\n\n" + user_prompt
            )
            print("\n\nDEBUG: app.py: get_chat_response: final response from LLM: ", response.content)
            # cleaned_response = clean_response(response.content)
            return response.content, search_results

        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            raise


def initialize_session_state():
    """Initialize session state variables"""
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = QASystem()


def display_results(grouped_results: Dict):
    """Display search results in an executive dashboard format"""
    if not grouped_results:
        st.info("No relevant results found. Try adjusting your query.")
        return

    # Custom CSS for better styling
    st.markdown("""
        <style>
        .section-header {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1.5rem 0 1rem 0;
            border-left: 4px solid #0066cc;
        }
        .insight-card {
            background-color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #e0e0e0;
            margin: 0.5rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .reference-item {
            border-left: 2px solid #0066cc;
            padding-left: 1rem;
            margin: 0.5rem 0;
        }
        .executive-summary {
            background-color: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border: 1px solid #e0e0e0;
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    # Reset button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🔄 Reset", key="reset_button"):
            st.rerun()

    # Process LLM response first
    if 'llm_response' in grouped_results:
        # st.markdown(grouped_results['llm_response'])
        content = grouped_results['llm_response']

        # Split into sections
        sections = content.split("# ")

        for section in sections:
            if not section:  # Skip empty sections
                continue

            if "💡 Key Insights" in section:
                # Format Key Insights section
                formatted_section = section.replace(" - ", "\n      - ")
                st.markdown("# " + formatted_section)
            else:
                # Display other sections as is
                st.markdown("# " + section)

    # Display document references
    for doc_name, doc_data in grouped_results.items():
        if doc_name == 'llm_response' or not isinstance(doc_data, dict):
            continue

        with st.expander(f"📄 {doc_name}"):
            # Text references
            if isinstance(doc_data.get('texts'), list) and doc_data['texts']:
                valid_texts = [text for text in doc_data['texts']
                               if isinstance(text, dict) and text.get('text')]
                if valid_texts:
                    st.markdown("**Text References:**")
                    for text in valid_texts:
                        st.markdown(
                            f"<div class='reference-item'>"
                            f"<strong>Page {text.get('page', 'N/A')}</strong> "
                            f"(Score: {text.get('score', 0):.2f})<br>"
                            f"{text.get('text', '')}</div>",
                            unsafe_allow_html=True
                        )

            # Image references
            if isinstance(doc_data.get('images'), list) and doc_data['images']:
                st.markdown("**Image References:**")
                num_images = len(doc_data['images'])
                if num_images > 0:
                    num_cols = min(3, num_images)
                    image_cols = st.columns([1] * num_cols)

                    for idx, img in enumerate(doc_data['images']):
                        if isinstance(img, dict) and 'path' in img:
                            col_idx = idx % num_cols
                            with image_cols[col_idx]:
                                if os.path.exists(img['path']):
                                    st.image(
                                        img['path'],
                                        caption=f"Page {img.get('page', 'N/A')} "
                                                f"(Score: {img.get('score', 0):.2f})"
                                    )

            # Excel references
            if isinstance(doc_data.get('excel_data'), list) and doc_data['excel_data']:
                valid_excel_data = [data for data in doc_data['excel_data']
                                    if isinstance(data, dict) and data.get('data')]
                if valid_excel_data:
                    st.markdown("**Excel References:**")
                    for excel_item in valid_excel_data:
                        data = excel_item.get('data', [])
                        sheet_name = excel_item.get('sheet', 'Unknown Sheet')

                        if data:
                            st.markdown(f"**Sheet: {sheet_name}**")
                            try:
                                if not isinstance(data, pd.DataFrame):
                                    df = pd.DataFrame(data)
                                else:
                                    df = data

                                st.dataframe(
                                    df,
                                    use_container_width=True,
                                    hide_index=True
                                )
                            except Exception as e:
                                if isinstance(data, list):
                                    for row in data:
                                        if isinstance(row, (list, tuple)):
                                            st.markdown(
                                                f"<div class='reference-item'>"
                                                f"{' | '.join(str(cell) for cell in row)}"
                                                f"</div>",
                                                unsafe_allow_html=True
                                            )
                                        else:
                                            st.markdown(
                                                f"<div class='reference-item'>{str(row)}</div>",
                                                unsafe_allow_html=True
                                            )

    # Footer
    st.markdown("---")
    st.markdown("*Generated by Document Analysis Dashboard*")


def main():
    st.set_page_config(
        page_title="Document Analysis Dashboard",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    initialize_session_state()

    st.title("📚 Document Analysis Dashboard")
    st.markdown("*Intelligent document analysis and insight generation*")

    with st.form("search_form", clear_on_submit=True):
        user_query = st.text_input(
            "🔍 Search Query:",
            placeholder="Enter your question or topic...",
            help="Be specific for better results"
        )

        cols = st.columns([1, 4])
        with cols[0]:
            search_button = st.form_submit_button("🔎 Analyze")

    if search_button and user_query:
        if len(user_query.strip()) < 3:
            st.warning("⚠️ Please enter a more detailed query")
            return

        with st.spinner("🔄 Analyzing documents..."):
            try:
                # Get analyzed and grouped results from document_searcher
                search_results = st.session_state.qa_system.searcher.search(
                    query=user_query,
                    limit=Config.SEARCH_LIMIT,
                    score_threshold=Config.SIMILARITY_THRESHOLD
                )

                if search_results:
                    # Get LLM processed response
                    llm_response, context = st.session_state.qa_system.get_chat_response(
                        query=user_query,
                        search_results=search_results
                    )
                    print("\n\nDEBUG: app.py: main: final response from LLM: ", llm_response)
                    # Create a new dictionary that includes both search results and LLM response
                    display_results_dict = {
                        'llm_response': llm_response,
                        **search_results  # Unpack the original search results
                    }

                    # Store LLM response in search_results
                    for filename in search_results:
                        if isinstance(search_results[filename], dict):  # Check if it's a dictionary
                            if 'summary' not in search_results[filename]:
                                search_results[filename]['summary'] = ''
                            if 'insights' not in search_results[filename]:
                                search_results[filename]['insights'] = ''
                            if 'data_analysis' not in search_results[filename]:
                                search_results[filename]['data_analysis'] = ''

                    display_results(display_results_dict)
                else:
                    st.info("ℹ️ No relevant content found. Try adjusting your query.")

            except Exception as e:
                st.error(f"❌ Analysis Error: {str(e)}")
                st.error("Please try again or contact support if the issue persists.")


if __name__ == "__main__":
    main()