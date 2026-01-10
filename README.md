# GeoNLP Disaster Analyzer

## 1. Project Overview

**GeoNLP Disaster Analyzer** is a web application for **automatic extraction, analysis, and visualization** of disaster-related information from textual reports.  
It combines **web scraping, NLP, large language models (LLMs), geocoding, and interactive visualizations** to deliver actionable insights for disaster management.

## 2. Objectives

- Extract disaster information (type, location, date, scale, human impact, damage) from online reports.  
- Normalize and structure data using NLP and LLMs.  
- Generate **interactive maps** and **visual analytics**.  
- Provide a **user-friendly web interface** for real-time input and reporting.  

## 3. Methodology

Folloiwng shows the work flow of the project
<p align="center">
  <img src="https://github.com/UpekshaIndeewari/GeoNLP_Disaster_Analyzer/blob/main/Workflow_GEONLP.png" alt="GeoNLP Workflow" width="700">
</p>

### 3.1 Data Collection
- Scrape textual disaster reports from URLs (e.g., ReliefWeb, CNN, BBC etc).  
- Extract title, publication date, and main content using **BeautifulSoup**.

### 3.2 Text Preprocessing
- Clean text by removing HTML entities, URLs, and punctuation.  
- Tokenize and lemmatize words using **NLTK** (`punkt` & `wordnet`).  

### 3.3 LLM-based Information Extraction
- Use **OpenRouter API** with adaptive prompts based on text length.  
- Extract structured JSON including:  
  - Event type, location, districts, provinces, cities, villages.  
  - Date, disaster scale, alert level.  
  - Impact metrics: deaths, injured, displaced, shelters.  
  - Event-specific data: wind, temperature, rainfall, magnitude, etc.  
  - Summary (2-3 sentences).  
- Handle incomplete JSON responses and retry if needed.

### 3.4 Geospatial Mapping
- Geocode locations using **geopy**.  
- Plot interactive maps with **Folium** and marker clustering.

### 3.5 Visualization
- **Bar charts** for human impact and geographical distribution.  
- **Pie charts** for spatial distribution of affected areas.  
- **Logarithmic scaling** applied for highly uneven data.  
- Visuals rendered with **Plotly** for interactivity.

### 3.6 Web Interface
- Flask-based web app:  
  - Input disaster report URL.  
  - Display cleaned text, LLM-extracted JSON, interactive maps, and charts.  
- Route `/json-output` provides raw JSON for programmatic access.
- 
## 4. Tools & Technologies

- **Language:** Python 3.13  
- **Web Framework:** Flask  
- **NLP:** NLTK (tokenization, lemmatization)  
- **LLM API:** OpenRouter (meta-llama/llama-3-8b-instruct)  
- **Web Scraping:** BeautifulSoup, requests  
- **Geocoding & Mapping:** geopy, Folium, MarkerCluster  
- **Visualization:** Plotly  
- **Deployment:** Railway (supports environment variables)  
[Watch the video](https://drive.google.com/file/d/1ZZAK0y6AayI4VsxoiVEfwCRjWlQFAU1a/view?usp=drive_link)

## 5. Project Structure

