# GeoNLP Disaster Analyzer

## 1. Project Overview

**GeoNLP Disaster Analyzer** is a web application for **automatic extraction, analysis, and visualization** of disaster-related information from textual reports. It combines **web scraping, NLP, large language models (LLMs), geocoding, and interactive visualizations** to deliver actionable insights for disaster management.

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
  
## 4. Tools & Technologies

- **Language:** Python 3.13  
- **Web Framework:** Flask  
- **NLP:** NLTK (tokenization, lemmatization)  
- **LLM API:** OpenRouter (meta-llama/llama-3-8b-instruct)  
- **Web Scraping:** BeautifulSoup, requests  
- **Geocoding & Mapping:** geopy, Folium, MarkerCluster  
- **Visualization:** Plotly  
- **Deployment:** Railway (supports environment variables)  

## 5. Application Demo

<video src="https://private-user-images.githubusercontent.com/111135094/534197495-b319625e-5ddd-48d5-b623-fb0e4752b1a5.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjgwMzQyMDYsIm5iZiI6MTc2ODAzMzkwNiwicGF0aCI6Ii8xMTExMzUwOTQvNTM0MTk3NDk1LWIzMTk2MjVlLTVkZGQtNDhkNS1iNjIzLWZiMGU0NzUyYjFhNS5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTEwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDExMFQwODMxNDZaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1kZjg4OTE5NWI5OGRkZjMzOTEzZjU5ODkxMDgzOGYyY2M0MWE2M2EyN2Y3ZjkwMWE0ZjJhNWQzYWQzNTYyNzZmJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.DmWj2g963I10HlyonyxLFlro7rRk2dseiguIRenR8u4" controls width="700"></video>

This application is hosted. You can access it online at:

[**Live Application Link**](https://web-production-2deab1.up.railway.app/)

## 5. Project Structure

```text
GeoNLP_Disaster_Analyzer/
│
├── app.py                     # Main Flask application (routes & logic)                  
├── requirements.txt           # Python dependencies               
├── README.md                  # Project documentation
├── .gitignore                 # Ignore venv, cache, secrets
├── .env                       # Environment variables (NOT pushed)
│
├── venv/                    # Virtual Environment
│
├── templates/                 # HTML templates (Flask)
│   ├── index.html
│   ├── charts.html
│   └── map.html
│
└── static/                    # CSS / JS / assets
     └──style.css
```
## 6. How to Run the Project

Follow these steps to set up and run the GeoNLP Disaster Analyzer locally:

### Step 1: Clone the repository

```python
git clone https://github.com/UpekshaIndeewari/GeoNLP_Disaster_Analyzer.git
cd GeoNLP_Disaster_Analyzer
```
### Step 2: Create a virtual environment

```python
python -m venv venv
```
**On Windows:**

```python
venv\Scripts\activate
```
**On Linux / Mac:**

```python
source venv/bin/activate
```
You should see `venv` in your terminal prompt.

### Step 3: Install dependencies

```python
pip install -r requirements.txt
```
### Step 5: Run the application
```python
python app.py
```
### Step 6: Open in Browser
Go to http://127.0.0.1:5000/
 to access the web application.

## 7. Copyright & Licenses

License: MIT License
Copyright: Upeksha Indeewari Edirisooriya Kirihami Vidanelage
License Summary
This project is released under the MIT License, which permits you to:

Use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software.
Include the software in your own projects (commercial or non-commercial).

Conditions:
The copyright notice and license text must be included in all copies or substantial portions of the software.
The software is provided "as is", without warranty of any kind.
For full details, see the [MIT License](https://opensource.org/licenses/MIT) or include a `LICENSE `file in your repository.


