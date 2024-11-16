import base64
import json
import os

import openai
import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image


def analyze_wine_label(images):
    """
    Analyze multiple wine label images using OpenAI's Vision API
    Returns structured data about the wine
    """
    # Process all images
    processed_images = []
    for image in images:
        # Reset file pointer to beginning
        image.seek(0)
        # Read image data directly from the uploaded file
        image_data = image.read()
        # Encode to base64
        encoded_image = base64.b64encode(image_data).decode("utf-8")

        # Validate base64 string
        try:
            # Test if we can decode it back
            base64.b64decode(encoded_image)
            processed_images.append(encoded_image)
        except Exception as e:
            st.error(f"Error encoding image: {str(e)}")
            return {"error": "Invalid image encoding"}

    # Construct the messages content
    content = [
        {
            "type": "text",
            "text": "whats on these images?",
        }
    ]

    # Add each image to the content
    for encoded_image in processed_images:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
            }
        )

    # Add debug output
    st.write("First few characters of encoded image:", encoded_image[:50])

    # Debug the API request
    st.write("Sending request with content:", content)

    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a wine label analysis expert. Analyze the wine label and provide structured information.",
                },
                {"role": "user", "content": content},
            ],
            functions=[
                {
                    "name": "analyze_wine_label",
                    "description": "Analyze wine label and return structured information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "wine_name": {"type": "string"},
                            "producer": {"type": "string"},
                            "region": {"type": "string"},
                            "vintage": {"type": "string"},
                            "grape_varieties": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "alcohol_content": {"type": "string"},
                            "additional_info": {"type": "string"},
                        },
                        "required": [
                            "wine_name",
                            "producer",
                            "region",
                            "vintage",
                            "grape_varieties",
                            "alcohol_content",
                        ],
                    },
                }
            ],
            function_call={"name": "analyze_wine_label"},
        )

        # Extract the function call arguments
        function_call = response.choices[0].message.function_call
        if function_call and function_call.arguments:
            wine_data = json.loads(function_call.arguments)
            return wine_data
        else:
            st.error("No valid data returned from the API.")
            return {"error": "No valid data returned from the API."}

    except Exception as e:
        st.error(f"API call error: {str(e)}")
        return {"error": f"API error: {str(e)}"}


def generate_search_queries_with_openai(wine_data):
    prompt = f"""
    You are a wine research assistant. Based on the structured wine label information provided, generate 3-5 Google search queries to gather more detailed information about the wine, its producer, the grape variety, the region or appellation, and any relevant technical details or tasting notes. Tailor the queries to maximize the depth and specificity of the information.
    
    ### Example:
    Wine Label Information:
    - Wine Name: Les Arènes Cornas 2021
    - Producer: M. Chapoutier
    - Region: Cornas, Rhône Valley, France
    - Vintage: 2021
    - Grape Varieties: Syrah
    - Alcohol Content: 13%
    - Additional Information: Contains sulfites; ideal for laying down; part of M. Chapoutier's Mon Héritage collection.

    Generated Queries:
    1. **"Les Arènes Cornas 2021 tasting notes"**  
       *Purpose*: To find professional and consumer tasting notes covering appearance, aroma, palate, and overall impressions.  
       *Expected Information*: Detailed sensory descriptions to fill in the Tasting Notes section.

    2. **"M. Chapoutier Les Arènes Cornas technical sheet"** or **"Les Arènes Cornas 2021 tech sheet PDF"**  
       *Purpose*: To access the official technical specifications provided by the winery.  
       *Expected Information*: In-depth Technical Production Information, including winemaking techniques, fermentation details, aging process, and more.

    3. **"M. Chapoutier winery history and philosophy"**  
       *Purpose*: To gather background information about the producer.  
       *Expected Information*: Insights for the Producer Information section, such as the winery’s history, size, winemaker’s philosophy, and sustainability practices.

    4. **"Cornas appellation characteristics"** or **"Cornas AOC details"**  
       *Purpose*: To learn about the specific appellation where the wine is produced.  
       *Expected Information*: Information for the Appellation Information section, including regulations, terroir details, and notable features.

    5. **"Syrah grape variety profile"** or **"Characteristics of Syrah grapes in Rhône Valley"**  
       *Purpose*: To understand more about the grape variety used in the wine.  
       *Expected Information*: Content for the Grape Information section, covering variety characteristics, typical flavor profiles, and viticulture aspects.

    ### Requirements for Search Queries:
    1. Create a query to find professional and consumer **tasting notes** for the specific wine, focusing on appearance, aroma, palate, and overall impressions.
    2. Create a query to locate the **technical sheet** or production details for the wine, including winemaking techniques, fermentation details, aging process, vineyard information, and viticulture practices.
    3. Create a query to gather background information about the **producer**, including their history, size, winemaker’s philosophy, and sustainability practices.
    4. Create a query to explore the **appellation or regional characteristics**, focusing on regulations, terroir details, and notable features.
    5. Optionally, include a query to understand more about the **grape variety**, its characteristics, and how it expresses itself in the wine’s region.

    ### Output:
    Provide a bulleted list of 3-5 Google search queries tailored to the provided wine label information, with each query followed by a brief explanation of its purpose and expected results.

    ### Wine Label Information:
    - Wine Name: {wine_data['wine_name']}
    - Producer: {wine_data['producer']}
    - Region: {wine_data['region']}
    - Vintage: {wine_data['vintage']}
    - Grape Varieties: {', '.join(wine_data['grape_varieties'])}
    - Alcohol Content: {wine_data['alcohol_content']}
    - Additional Information: {wine_data.get('additional_info', 'N/A')}

    """

    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a wine research assistant."},
                {"role": "user", "content": prompt},
            ],
            functions=[
                {
                    "name": "generate_search_queries",
                    "description": "Generate structured search queries for wine research",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "queries": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "query": {
                                            "type": "string",
                                            "description": "The actual search query",
                                        },
                                        "purpose": {
                                            "type": "string",
                                            "description": "The purpose of this search query",
                                        },
                                        "expected_results": {
                                            "type": "string",
                                            "description": "What information we expect to find",
                                        },
                                        "category": {
                                            "type": "string",
                                            "enum": [
                                                "tasting_notes",
                                                "technical_info",
                                                "producer_info",
                                                "region_info",
                                                "grape_info",
                                            ],
                                            "description": "The category of information this query targets",
                                        },
                                    },
                                    "required": [
                                        "query",
                                        "purpose",
                                        "expected_results",
                                        "category",
                                    ],
                                },
                            }
                        },
                        "required": ["queries"],
                    },
                }
            ],
            function_call={"name": "generate_search_queries"},
        )

        # Extract the function call arguments
        function_call = response.choices[0].message.function_call
        if function_call and function_call.arguments:
            queries_data = json.loads(function_call.arguments)

            # Format the output for display in Streamlit
            formatted_output = ""
            for query in queries_data["queries"]:
                formatted_output += f"""
### {query['category'].replace('_', ' ').title()}
**Search Query:** {query['query']}
**Purpose:** {query['purpose']}
**Expected Results:** {query['expected_results']}

"""
            return {
                "raw_data": queries_data,
                "formatted_output": formatted_output.strip(),
            }
        else:
            st.error("No valid queries generated.")
            return None

    except Exception as e:
        st.error(f"Error generating search queries: {str(e)}")
        return None


@st.cache_data(ttl=3600)  # Cache for 1 hour
def execute_single_search_query(api_key: str, search_query: str):
    """Execute a single search query and cache the results"""
    base_url = "https://api.scrapingdog.com/google"
    params = {
        "api_key": api_key,
        "query": search_query,
        "results": 10,
        "country": "us",
        "page": 0,
        "advance_search": "false",
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    return None

def execute_search_queries(queries_data):
    api_key = os.getenv("SCRAPINGDOG_API_KEY")
    search_results = {"results_by_category": {}, "raw_results": []}
    
    debug_section = st.expander("Debug Information", expanded=False)
    results_placeholder = st.empty()

    try:
        for query in queries_data["queries"]:
            category = query["category"]
            search_query = query["query"]

            with debug_section:
                st.write(f"Processing query: {search_query}")
                st.write(f"Category: {category}")

            with st.spinner(f"Searching for: {search_query}"):
                # Use cached function for API call
                results = execute_single_search_query(api_key, search_query)

                if results:
                    if category not in search_results["results_by_category"]:
                        search_results["results_by_category"][category] = []

                    processed_results = {
                        "query": search_query,
                        "purpose": query["purpose"],
                        "expected_results": query["expected_results"],
                        "search_results": results,
                    }

                    search_results["results_by_category"][category].append(processed_results)
                    search_results["raw_results"].append(processed_results)

                    # Update main display
                    with results_placeholder.container():
                        display_search_results(search_results)
                else:
                    st.error(f"Failed to execute query: {search_query}")

        return search_results

    except Exception as e:
        st.error(f"Error executing searches: {str(e)}")
        with debug_section:
            st.write("Exception details:")
            st.exception(e)
        return None


def display_search_results(search_results):
    """
    Display the search results in an organized manner
    """
    if not search_results:
        st.error("No search results to display")
        return

    st.subheader("Search Results")

    # Single debug expander for all raw data
    with st.expander("🔍 Debug: Raw Results Data", expanded=False):
        st.json(search_results)

    # Simple list of results by category
    for category, category_results in search_results["results_by_category"].items():
        st.markdown(f"### {category.replace('_', ' ').title()}")

        for result_group in category_results:
            st.markdown(f"**Query:** {result_group['query']}")

            # Process and display results
            search_results_data = result_group["search_results"]

            # Handle different response structures
            if (
                isinstance(search_results_data, dict)
                and "organic_results" in search_results_data
            ):
                results_to_show = search_results_data["organic_results"][:5]
            elif isinstance(search_results_data, list):
                results_to_show = search_results_data[:5]
            else:
                results_to_show = []
                st.warning("Unexpected results structure")

            # Display results in a clean format
            for idx, item in enumerate(results_to_show, 1):
                if isinstance(item, dict):
                    title = item.get("title", "No title")
                    link = item.get("link", "#")
                    st.markdown(f"{idx}. [{title}]({link})")

            st.markdown("---")


def main():
    st.title("Wine Label Analyzer")

    # Initialize session state for wine data and queries
    if "wine_data" not in st.session_state:
        st.session_state.wine_data = None
    if "analysis_complete" not in st.session_state:
        st.session_state.analysis_complete = False

    # Create two columns for layout
    col1, col2 = st.columns([2, 3])

    with col1:
        uploaded_files = st.file_uploader(
            "Choose wine label images (front and back)...",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            # Display images using PIL
            display_images = [Image.open(file) for file in uploaded_files]
            for i, image in enumerate(display_images):
                st.image(image, caption=f"Wine Label {i+1}", use_container_width=True)

    with col2:
        if uploaded_files:
            if st.button("Analyze Labels"):
                with st.spinner("Analyzing wine labels..."):
                    # Pass the original uploaded files instead of PIL Images
                    wine_data = analyze_wine_label(uploaded_files)

                    if "error" not in wine_data:
                        st.session_state.wine_data = wine_data
                        st.session_state.analysis_complete = True

                        # Display wine information card
                        st.subheader("Wine Information")
                        with st.container():
                            st.markdown(
                                """
                                <style>
                                .wine-info {
                                    padding: 20px;
                                    border-radius: 10px;
                                    background-color: #f5f5f5;
                                    margin: 10px 0;
                                }
                                </style>
                                """,
                                unsafe_allow_html=True,
                            )

                            st.markdown(
                                '<div class="wine-info">', unsafe_allow_html=True
                            )
                            st.markdown(f"**Wine Name:** {wine_data['wine_name']}")
                            st.markdown(f"**Producer:** {wine_data['producer']}")
                            st.markdown(f"**Region:** {wine_data['region']}")
                            st.markdown(f"**Vintage:** {wine_data['vintage']}")
                            st.markdown(
                                f"**Grape Varieties:** {', '.join(wine_data['grape_varieties']) if isinstance(wine_data['grape_varieties'], list) else wine_data['grape_varieties']}"
                            )
                            st.markdown(
                                f"**Alcohol Content:** {wine_data['alcohol_content']}"
                            )
                            if (
                                "additional_info" in wine_data
                                and wine_data["additional_info"] != "Not visible"
                            ):
                                st.markdown(
                                    f"**Additional Information:** {wine_data['additional_info']}"
                                )
                            st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.error("Could not analyze the wine label. Please try again.")

            # Show the Generate Search Queries button only after analysis is complete
            if st.session_state.analysis_complete:
                if st.button("Generate Search Queries"):
                    with st.spinner("Generating search queries..."):
                        queries = generate_search_queries_with_openai(
                            st.session_state.wine_data
                        )
                        if queries:
                            st.session_state.queries = (
                                queries  # Store queries in session state
                            )
                            st.subheader("Generated Search Queries")
                            st.markdown(queries["formatted_output"])

            # Add Execute Searches button after queries are generated
            if hasattr(st.session_state, "queries"):
                if st.button("Execute Searches"):
                    with st.spinner("Executing searches..."):
                        search_results = execute_search_queries(
                            st.session_state.queries["raw_data"]
                        )
                        if search_results:
                            st.session_state.search_results = (
                                search_results  # Store results in session state
                            )
                            display_search_results(search_results)


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Initialize OpenAI client
    openai.api_key = os.getenv("OPENAI_API_KEY")

    main()
