# GEONLP/app.py
from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
import re
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from openai import OpenAI
import folium
import plotly.express as px
from geopy.geocoders import Nominatim
import time
import folium
from folium.plugins import MarkerCluster
import os

geolocator = Nominatim(user_agent="disaster_mapper")


# -------------------- NLTK CONFIGURATION --------------------

nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(nltk_data_dir)
os.makedirs(nltk_data_dir, exist_ok=True)

nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)


try:
    nltk.data.find('tokenizers/punkt_tab')
    print("NLTK punkt_tab already installed")
except LookupError:
    print("Downloading NLTK punkt_tab...")
    nltk.download('punkt_tab')

try:
    nltk.data.find('tokenizers/punkt')
    print("NLTK punkt already installed")
except LookupError:
    print("Downloading NLTK punkt...")
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
    print("NLTK wordnet already installed")
except LookupError:
    print("Downloading NLTK wordnet...")
    nltk.download('wordnet')

# -------------------- OPENAI CONFIG --------------------
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=".env", override=True)

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

MODEL_NAME = "meta-llama/llama-3-8b-instruct"


app = Flask(__name__)

# -------------------- NLP FUNCTIONALITY --------------------
def process_reliefweb_url(url):
    """
    Extract and process content from a givnen URL
    Returns: title, date, full_text, cleaned_text, processed_text
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract title
        title_element = soup.find("h1")
        title = title_element.get_text(strip=True) if title_element else "No title found"
        
        # Extract date
        date_element = soup.find("time")
        date = date_element.get_text(strip=True) if date_element else "No date found"

        # Extract all report content
        full_text = ""
        
        # Look for ReliefWeb specific content containers
        content_selectors = [
            "div.report-body",
            "div.article-body", 
            "div#overview-content",
            "div.content",
            "article",
            "main",
            "div.node-content"
        ]
        
        for selector in content_selectors:
            elements = soup.select(selector)
            for element in elements:
                # Get all paragraphs from the element
                paragraphs = element.find_all("p")
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if text and len(text) > 10:  # Avoid short/empty paragraphs
                        full_text += text + " "
        
        #If still empty, get all paragraphs from the page
        if not full_text.strip():
            all_paragraphs = soup.find_all("p")
            for p in all_paragraphs:
                text = p.get_text(strip=True)
                if text and len(text) > 10:
                    full_text += text + " "
        
        # Clean the text - basic cleaning
        if full_text:
            # Remove URLs in parentheses
            full_text = re.sub(r'\(https?://[^\s)]+\)', '', full_text)
            # Remove HTML entities
            full_text = re.sub(r'&\w+;', '', full_text)
            # Normalize whitespace
            full_text = re.sub(r'\s+', ' ', full_text).strip()
        else:
            full_text = "No content could be extracted from the page."
        
        # Create version without punctuation
        cleaned_text = full_text
        # Remove all punctuation
        cleaned_text = re.sub(r'[^\w\s\d]', '', cleaned_text)
        # Remove extra spaces created by punctuation removal
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # Create a processed version for NLP/display (lowercase, lemmatized)
        processed_text = cleaned_text.lower()
        
        # Tokenize and lemmatize
        try:
            words = word_tokenize(processed_text)
            lemmatizer = WordNetLemmatizer()
            lemmatized_words = []
            for word in words:
                # Lemmatize nouns and verbs
                lemma = lemmatizer.lemmatize(word, pos='n')  # Noun
                lemma = lemmatizer.lemmatize(lemma, pos='v')  # Verb
                lemmatized_words.append(lemma)
            processed_text = " ".join(lemmatized_words)
        except Exception as e:
            print(f"Error in tokenization/lemmatization: {e}")
            # If NLP processing fails, use the cleaned lowercase text
            processed_text = processed_text.lower()
        
        return title, date, full_text, cleaned_text, processed_text
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching URL: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing URL: {str(e)}")

# -------------------- LLM INFORMATION EXTRACTION --------------------
def extract_info_with_llm(report_text):
    """
    Extract structured information from disaster report using LLM
    with adaptive token limits based on text length
    """
    # Calculate optimal token limits based on text length
    text_length = len(report_text)
    
    # OPTIMIZED FOR <500 CHARACTERS
    if text_length < 500:
        # Ultra-short text - minimal approach
        if text_length < 100:
            # Extremely short text (1-2 sentences)
            max_tokens = 250
            report_for_prompt = report_text  # Use all text
        
        elif text_length < 300:
            # Short text (paragraph)
            max_tokens = 350
            report_for_prompt = report_text  # Use all text
       
        else:
            # 300-500 characters
            max_tokens = 400
            report_for_prompt = report_text  # Use all text
        
    elif text_length < 1000:
        # Short text (500-1000 chars)
        max_tokens = 500
        report_for_prompt = report_text[:800] + "..."
        
    elif text_length < 3000:
        # Medium text
        max_tokens = 600
        report_for_prompt = report_text[:2500] + "..."
        
    elif text_length < 6000:
        # Large text
        max_tokens = 800
        report_for_prompt = report_text[:3500] + "..."
        
    else:
        # Very large text
        max_tokens = 1000
        report_for_prompt = report_text[:4000] + "..."
     
    print(f"Text length: {text_length} chars, Using max_tokens: {max_tokens}")
    
    # ULTRA-MINIMAL PROMPT FOR VERY SHORT TEXTS (<100 chars)
    if text_length < 100:
        prompt = f"""Text: {report_for_prompt}
        
JSON format:
{{
  "event_type": "",
  "location": "",
  "district": "",
  "provinces": [],
  "districts_list": [],
  "cities_list": [],
  "villages_list": [],
  "date": "",
  "disaster_scale": "",
  "alert_level": "",
  "impact": {{
    "deaths": null,
    "injured": null,
    "missing": null,
    "displaced": null,
    "affected_population": null,
    "shelters": null
  }},
  "event_data": {{
    "wind": null,
    "temperature": null,
    "rainfall": null,
    "pressure": null,
    "flood": null,
    "landslide": null,
    "magnitude": null,
    "depth": null
  }},
  "summary": ""
}}"""
        
        system_message = "Extract disaster info and return ONLY JSON."
    
    elif text_length < 3000:
        # Medium text - balanced prompt
        prompt = f"""Analyze this disaster report and extract information as JSON:

{report_for_prompt}

Required JSON structure:
{{
  "event_type": "disaster type",
  "location": "primary country",
  "district": "main district",
  "provinces": ["state1", "state2"],
  "districts_list": ["district1", "district2"],
  "cities_list": ["city1", "city2"],
  "villages_list": ["village1", "village2"],
  "date": "report date",
  "disaster_scale": "severity level",
  "alert_level": "warning level",
  "impact": {{
    "deaths": number,
    "injured": number,
    "missing": number,
    "displaced": number,
    "affected_population": number,
    "shelters": number
  }},
  "event_data": {{
    "wind": "wind speed or description",
    "temperature": "temperature",
    "rainfall": "rainfall amount",
    "pressure": "pressure",
    "flood": "flood level",
    "landslide": "landslide count",
    "magnitude": "earthquake magnitude",
    "depth": "earthquake depth"
  }},
  "summary": "2-3 sentence summary"
}}"""
        
        system_message = "Return complete JSON with disaster data."
    
    else:
        # Long text - detailed prompt
        prompt = f"""Comprehensive disaster analysis. Extract ALL information as JSON:

REPORT:
{report_for_prompt}

EXTRACT these fields comprehensively:

1. event_type: Type of disaster (flood, earthquake, storm, etc.)
2. location: Primary affected country
3. district: Primary affected district
4. provinces: Array of ALL states/provinces mentioned
5. districts_list: Array of ALL districts mentioned
6. cities_list: Array of ALL cities mentioned
7. villages_list: Array of ALL villages mentioned
8. date: Date mentioned
9. disaster_scale: Severity (minor, moderate, major, catastrophic)
10. alert_level: Any warnings/alerts
11. impact: Deaths, injured, missing, displaced, affected population, shelters
12. event_data: Wind, temperature, rainfall, pressure, flood level, landslides, magnitude, depth
13. summary: Detailed 2-3 sentence summary with key locations

Return ONLY valid JSON following this exact structure:
{{
    "event_type": "string or null",
    "location": "string or null",
    "district": "string or null",
    "provinces": [],
    "districts_list": [],
    "cities_list": [],
    "villages_list": [],
    "date": "string or null",
    "disaster_scale": "string or null",
    "alert_level": "string or null",
    "impact": {{
        "deaths": "number or null",
        "injured": "number or null",
        "missing": "number or null",
        "displaced": "number or null",
        "affected_population": "number or null",
        "shelters": "number or null"
    }},
    "event_data": {{
        "wind": "number or null",
        "temperature": "number or null",
        "rainfall": "number or null",
        "pressure": "number or null",
        "flood": "number or null",
        "landslide": "number or null",
        "magnitude": "number or null",
        "depth": "number or null"
    }},
    "summary": "string"
}}"""
        
        system_message = "Extract comprehensive disaster information. Return ONLY valid JSON."

    try:
        print(f"Sending request with max_tokens={max_tokens}")
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=max_tokens
        )

        content = response.choices[0].message.content.strip()
        print(f"Response length: {len(content)} characters")
        
        # Clean the response
        content = content.strip()
        
        # Remove markdown code blocks if present
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        # Try to parse JSON
        try:
            extracted_info = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON parse failed, trying to fix... Error: {e}")
            
            # Attempt to fix common JSON issues
            content = fix_incomplete_json(content)
            
            # Try parsing again
            extracted_info = json.loads(content)
        
        # Smart field initialization
        extracted_info = ensure_json_structure(extracted_info, text_length)
        
        return extracted_info
        
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error after fixes: {e}")
        print(f"Response preview (first 300 chars): {content[:300]}")
        
        # Try one more time with increased tokens if we have room
        if max_tokens < 1200 and text_length > 2000:
            print("Retrying with increased tokens...")
            return extract_info_with_llm_retry(report_text, max_tokens + 400)
        
        return get_minimal_structure(f"JSON parse failed: {str(e)[:50]}")
        
    except Exception as e:
        print(f"Error in LLM extraction: {e}")
        return get_minimal_structure(f"Extraction error: {str(e)[:50]}")


def fix_incomplete_json(content):
    """Attempt to fix incomplete JSON responses"""
    # Remove any text before the first {
    start_idx = content.find('{')
    if start_idx > 0:
        content = content[start_idx:]
    
    # Remove any text after the last }
    end_idx = content.rfind('}')
    if end_idx < len(content) - 1:
        content = content[:end_idx + 1]
    
    # Fix missing closing brackets
    open_braces = content.count('{')
    close_braces = content.count('}')
    
    if open_braces > close_braces:
        # Add missing closing braces
        content += '}' * (open_braces - close_braces)
    
    # Fix missing closing brackets in arrays
    open_brackets = content.count('[')
    close_brackets = content.count(']')
    
    if open_brackets > close_brackets:
        content += ']' * (open_brackets - close_brackets)
    
    return content

def ensure_json_structure(extracted_info, text_length):
    """Ensure all required fields exist with appropriate defaults"""
    
    # Basic fields with defaults
    defaults = {
        "event_type": None,
        "location": None,
        "district": None,
        "provinces": [],
        "districts_list": [],
        "cities_list": [],
        "villages_list": [],
        "date": None,
        "disaster_scale": None,
        "alert_level": None,
        "summary": f"Analysis of {text_length} character report"
    }
    
    for key, default in defaults.items():
        if key not in extracted_info:
            extracted_info[key] = default
    
    # Ensure impact structure
    if "impact" not in extracted_info or not isinstance(extracted_info["impact"], dict):
        extracted_info["impact"] = {}
    
    impact_defaults = {
        "deaths": None,
        "injured": None,
        "missing": None,
        "displaced": None,
        "affected_population": None,
        "shelters": None
    }
    
    for key, default in impact_defaults.items():
        if key not in extracted_info["impact"]:
            extracted_info["impact"][key] = default
    
    # Ensure event_data structure
    if "event_data" not in extracted_info or not isinstance(extracted_info["event_data"], dict):
        extracted_info["event_data"] = {}
    
    event_defaults = {
        "wind": None,
        "temperature": None,
        "rainfall": None,
        "pressure": None,
        "flood": None,
        "landslide": None,
        "magnitude": None,
        "depth": None
    }
    
    for key, default in event_defaults.items():
        if key not in extracted_info["event_data"]:
            extracted_info["event_data"][key] = default
    
    # Create combined geographical names list
    all_geo_names = []
    for array_name in ["provinces", "districts_list", "cities_list", "villages_list"]:
        if isinstance(extracted_info.get(array_name), list):
            all_geo_names.extend(extracted_info[array_name])
    
    if extracted_info.get("location"):
        all_geo_names.append(extracted_info["location"])
    if extracted_info.get("district"):
        all_geo_names.append(extracted_info["district"])
    
    # Remove duplicates and empty values
    all_geo_names = list(set([name for name in all_geo_names if name and str(name).strip()]))
    extracted_info["all_geographical_names"] = all_geo_names
    
    return extracted_info


def extract_info_with_llm_retry(report_text, retry_tokens):
    """Retry extraction with higher token limit"""
    if len(report_text) > 4000:
        report_for_prompt = report_text[:4000] + "..."
    else:
        report_for_prompt = report_text
    
    prompt = f"""Extract disaster data as JSON from: {report_for_prompt}
    
Return ONLY this JSON structure filled with data:
{{
  "event_type":"", "location":"", "district":"", "provinces":[], "districts_list":[], 
  "cities_list":[], "villages_list":[], "date":"", "disaster_scale":"", "alert_level":"",
  "impact":{{"deaths":null,"injured":null,"missing":null,"displaced":null,"affected_population":null,"shelters":null}},
  "event_data":{{"wind":null,"temperature":null,"rainfall":null,"pressure":null,"flood":null,"landslide":null,"magnitude":null,"depth":null}},
  "summary":""
}}"""
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Fill this JSON template. Return ONLY JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=retry_tokens
        )
        
        content = response.choices[0].message.content.strip()
        content = fix_incomplete_json(content)
        extracted_info = json.loads(content)
        return ensure_json_structure(extracted_info, len(report_text))
        
    except Exception as e:
        print(f"Retry also failed: {e}")
        return get_minimal_structure("Failed after retry")


def get_minimal_structure(error_message=""):
    """Return minimal valid JSON structure on complete failure"""
    return {
        "event_type": "Analysis failed",
        "location": None,
        "district": None,
        "provinces": [],
        "districts_list": [],
        "cities_list": [],
        "villages_list": [],
        "date": None,
        "disaster_scale": None,
        "alert_level": None,
        "impact": {
            "deaths": None,
            "injured": None,
            "missing": None,
            "displaced": None,
            "affected_population": None,
            "shelters": None
        },
        "event_data": {
            "wind": None,
            "temperature": None,
            "rainfall": None,
            "pressure": None,
            "flood": None,
            "landslide": None,
            "magnitude": None,
            "depth": None
        },
        "summary": f"Error: {error_message}" if error_message else "Could not extract information",
        "all_geographical_names": []
    }

### ------------------------ Goecoading ------------------

def get_country_coordinates(country_name):
    try:
        if not country_name:
            return [20, 0]

        location = geolocator.geocode(country_name, timeout=10)
        time.sleep(1)

        if location:
            return [location.latitude, location.longitude]

    except Exception as e:
        print(f"Country geocoding failed: {country_name} → {e}")

    return [20, 0]

def create_district_map_from_json(json_data):
    districts = json_data.get("all_geographical_names", [])
    country = json_data.get("location", "")

    if not districts:
        return None

    # Center map on country
    center = get_country_coordinates(country)
    m = folium.Map(location=center, zoom_start=7)

    marker_cluster = MarkerCluster().add_to(m)

    for district in districts:
        if not district:
            continue

        # IMPORTANT: district + country context
        query = f"{district}, {country}"

        try:
            location = geolocator.geocode(query, timeout=10)
            time.sleep(1)  # REQUIRED

            if not location:
                print(f"Not found: {query}")
                continue

            folium.Marker(
                location=[location.latitude, location.longitude],
                popup=f"{district}, {country}",
                tooltip=district,
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(marker_cluster)

        except Exception as e:
            print(f"Error geocoding {district}: {e}")

    return m._repr_html_()

#--------------add statistics data-----------

def compute_statistics(extracted_info):
    return {
        "event_type": extracted_info.get("event_type"),
        "alert_level": extracted_info.get("alert_level"),
        "disaster_scale": extracted_info.get("disaster_scale"),
        "date": extracted_info.get("date"),

        "geo_stats": {
            "provinces": len(extracted_info.get("provinces", [])),
            "districts": len(extracted_info.get("districts_list", [])),
            "cities": len(extracted_info.get("cities_list", [])),
            "villages": len(extracted_info.get("villages_list", [])),
            "total_locations": len(extracted_info.get("all_geographical_names", []))
        },

        "impact": extracted_info.get("impact", {})
    }

#------------- add bargraph for impacts--------------
def impact_chart(extracted_info):
    impact = extracted_info.get("impact", {})
    labels = []
    values = []

    for k, v in impact.items():
        if v is not None:
            labels.append(k.capitalize())
            values.append(v)

    if not values:
        return "<p>No impact data available</p>"

    # Use logarithmic scale if values differ by >100x
    max_val = max(values)
    min_val = min([v for v in values if v > 0] or [1])
    use_log = max_val / max(min_val,1) > 100

    # Define a distinct color for each impact type
    colors = ["#d32f2f", "#f57c00", "#1976d2", "#388e3c", "#7b1fa2"]  

    fig = px.bar(
        x=labels,
        y=values,
        title="Human Impact Summary",
        labels={"x": "Impact Type", "y": "Count"},
        color=labels, 
        color_discrete_sequence=colors,
        log_y=use_log
    )
   
    fig.update_layout(showlegend=False, height=550, margin=dict(t=50,b=50))

    return fig.to_html(full_html=False)


#------------ add geography bar chart-------------

def geography_chart(extracted_info):
    data = {
        "Provinces": len(extracted_info.get("provinces", [])),
        "Districts": len(extracted_info.get("districts_list", [])),
        "Cities": len(extracted_info.get("cities_list", [])),
        "Villages": len(extracted_info.get("villages_list", []))
    }

    labels = list(data.keys())
    values = list(data.values())

    if max(values) == 0:
        return "<p>No geographical data available</p>"

    # Decide whether to use log scale
    use_log = max(values) / max(min([v for v in values if v>0] or [1]), 1) > 10

    # Define colors for each bar
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig = px.bar(
        x=labels,
        y=values,
        title="Affected Administrative Areas",
        labels={"x": "Administrative Type", "y": "Count"},
        color=labels,  
        color_discrete_sequence=colors,
        log_y=use_log
    )

    fig.update_layout(showlegend=False, height=550, margin=dict(t=50,b=50))

    # Return HTML div for embedding
    return fig.to_html(full_html=False)


#------------- add pie chart-----------------
def impact_pie_chart(extracted_info):
    impact = extracted_info.get("impact", {})
    labels = []
    values = []

    for k, v in impact.items():
        if v is not None and v > 0:
            labels.append(k.capitalize())
            values.append(v)

    if not values:
        return "<p>No impact data available</p>"

    # Define distinct colors for each impact type
    colors = ["#d32f2f", "#f57c00", "#1976d2", "#388e3c", "#7b1fa2"]  # red, orange, blue, green, purple

    # Create Pie chart
    fig = px.pie(
        names=labels,
        values=values,
        title="Human Impact Distribution",
        color=labels,
        color_discrete_sequence=colors
    )

    # Show values on the slices
    fig.update_traces(text=values, textposition='outside')

    # Remove legend and adjust layout
    fig.update_layout(height=550)

    # Return HTML div with mode bar disabled
    return fig.to_html(full_html=False)


def spatial_pie(extracted_info):
    labels = ["Provinces", "Districts"]
    values = [
        len(extracted_info.get("provinces", [])),
        len(extracted_info.get("districts_list", [])),
    ]

    fig = px.pie(
        names=labels,
        values=values,
        title="Spatial Distribution of Impact"
    )

    fig.update_layout(height=550)
    fig.update_traces(text=values, textposition='outside')
    return fig.to_html(full_html=False)


# -------------------- MODIFIED FLASK ROUTE --------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    
    if request.method == "POST":
        url = request.form.get("url", "").strip()
        
        if not url:
            result = {"error": "Please enter a URL"}
        elif not url.startswith("http"):
            result = {"error": "Please enter a valid URL starting with http:// or https://"}
        else:
            try:
                print(f"Processing URL: {url}")
                
                # Get report content
                title, date, full_text, cleaned_text, processed_text = process_reliefweb_url(url)
                
                # Extract information using LLM
                extracted_info = extract_info_with_llm(full_text)

                # Generate map from the JSON data
                map_html = create_district_map_from_json(extracted_info)

                stats = compute_statistics(extracted_info)

                # Prepare result
                result = {
                    "title": title,
                    "date": date,
                    "full_text": full_text,
                    "cleaned_text": cleaned_text,
                    "text_preview": cleaned_text,
                    "processed_text": processed_text,
                    "extracted_info": extracted_info,  
                    "stats": {
                        "full_length": len(full_text),
                        "cleaned_length": len(cleaned_text),
                        "word_count": len(cleaned_text.split()),
                        "processed_length": len(processed_text)
                    },
                    "url": url,
                    "map_html": map_html, 
                    "stats": stats,
                    "impact_chart": impact_chart(extracted_info),
                    "geo_chart": geography_chart(extracted_info),
                    "spatial_chart": spatial_pie(extracted_info),
                    "spatial_chart2": impact_pie_chart(extracted_info)
                    
                }
                
                # Print all text and JSON to console
                print("\n" + "="*80)
                print(f"EXTRACTED TEXT FROM: {url}")
                print(f"TITLE: {title}")
                print(f"DATE: {date}")

                # Print the complete JSON output
                print("\nCOMPLETE JSON OUTPUT:")
                print(json.dumps(extracted_info, indent=2, ensure_ascii=False))
                
                # Print geographical extraction summary
                print("\nGEOGRAPHICAL EXTRACTION SUMMARY:")
                print(f"Country: {extracted_info.get('location', 'Not specified')}")
                print(f"Primary District: {extracted_info.get('district', 'Not specified')}")
                print(f"States/Provinces ({len(extracted_info.get('provinces', []))}): {', '.join(extracted_info.get('provinces', []))}")
                print(f"Districts ({len(extracted_info.get('districts_list', []))}): {', '.join(extracted_info.get('districts_list', []))}")
                print(f"Cities ({len(extracted_info.get('cities_list', []))}): {', '.join(extracted_info.get('cities_list', []))}")
                print(f"Villages ({len(extracted_info.get('villages_list', []))}): {', '.join(extracted_info.get('villages_list', []))}")
                print(f"TOTAL Geographical Names: {len(extracted_info.get('all_geographical_names', []))}")
                
                print("="*80 + "\n")
                
            except Exception as e:
                import traceback
                print(f"Error: {str(e)}")
                print(traceback.format_exc())
                result = {
                    "error": f"Error processing URL: {str(e)}",
                    "url": url
                }
    
    return render_template("index.html", result=result)

@app.route("/json-output")
def json_output():
    """Route to display raw JSON output"""
    url = request.args.get("url", "")
    
    if url:
        try:
            # Get report content
            full_text = process_reliefweb_url(url)
            
            # Extract information using LLM
            extracted_info = extract_info_with_llm(full_text)
            
            # Return JSON response
            return json.dumps(extracted_info, indent=2, ensure_ascii=False), 200, {'Content-Type': 'application/json'}
        except Exception as e:
            return json.dumps({"error": str(e)}), 400, {'Content-Type': 'application/json'}
    
    return json.dumps({"error": "No URL provided"}), 400, {'Content-Type': 'application/json'}

if __name__ == "__main__":
    print("Starting Global Disaster Report Analyzer with Comprehensive Geographical Extraction...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host="0.0.0.0", port=5000)