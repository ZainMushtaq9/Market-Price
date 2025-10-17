import os
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import threading
import time
import schedule
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from fuzzywuzzy import fuzz, process
import plotly.express as px
import plotly.graph_objects as go

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Configuration
DATA_FILE = "market_data_history.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
AMIS_URL = "https://www.amis.pk/daily%20market%20changes.aspx"

# Initialize LLMs
try:
    llm_translate = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, groq_api_key=GROQ_API_KEY)
    llm_report = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, groq_api_key=GROQ_API_KEY)
except:
    llm_translate = None
    llm_report = None

# Comprehensive aliases
CITY_ALIASES = {
    "multan": ["multan", "mtn", "mtln", "mltn", "mlt"],
    "lahore": ["lahore", "lhr", "lhore", "lahor", "lahur", "lhe"],
    "karachi": ["karachi", "khi", "krachi", "karchi", "krc"],
    "islamabad": ["islamabad", "isb", "isl", "islambad", "isbd"],
    "faisalabad": ["faisalabad", "fsd", "faisalbad", "lyallpur", "fsb"],
    "rawalpindi": ["rawalpindi", "rwp", "pindi", "rawlpindi", "rpl"],
    "peshawar": ["peshawar", "pesh", "pshawar", "psr"],
    "quetta": ["quetta", "qta", "queta", "qet"],
    "sialkot": ["sialkot", "skt", "slkt", "slt"],
    "gujranwala": ["gujranwala", "gjw", "gujrat", "grw"],
}

COMMODITY_ALIASES = {
    "potato": ["potato", "patato", "aloo", "allo", "alu", "آلو", "alloo"],
    "rice": ["rice", "chawal", "chawl", "چاول", "chwal", "rice"],
    "wheat": ["wheat", "gandum", "gandom", "گندم", "gehun"],
    "onion": ["onion", "pyaz", "piaz", "پیاز", "piaz"],
    "tomato": ["tomato", "tamatar", "tamater", "ٹماٹر", "tamater"],
    "chilli": ["chilli", "mirch", "mirchi", "مرچ", "chili"],
    "maize": ["maize", "makai", "maki", "makki", "مکئی", "corn"],
    "garlic": ["garlic", "lehsan", "لہسن", "lahsan"],
    "ginger": ["ginger", "adrak", "ادرک", "adrak"],
    "apple": ["apple", "seb", "سیب", "apel"],
    "cucumber": ["cucumber", "kheera", "khira", "کھیرا"],
    "spinach": ["spinach", "palak", "پالک", "saag"],
    "grapes": ["grapes", "angoor", "انگور", "grape"],
}

# ==================== TRANSLATION & FUZZY MATCHING ====================

def translate_query_to_english(query):
    """Translate Roman Urdu/Urdu query to English using LLM"""
    if llm_translate is None:
        return query, "en"
    
    try:
        # Check if query contains Urdu script
        has_urdu_script = any('\u0600' <= c <= '\u06FF' for c in query)
        
        prompt = ChatPromptTemplate.from_template(
            "You are a translator. Translate this query to English. Extract commodity and city names.\n"
            "Query: {query}\n\n"
            "Respond ONLY with JSON format (no markdown, no extra text):\n"
            '{{"english_query": "translated query", "detected_language": "english/roman_urdu/urdu", "commodities": ["commodity1"], "cities": ["city1"]}}\n\n'
            "IMPORTANT: If query has Urdu script (آ، ا، ب، etc), set detected_language to 'urdu'\n"
            "If query has roman urdu words (aloo, chawal, mein, ka), set detected_language to 'roman_urdu'\n\n"
            "Examples:\n"
            "Query: multan main allo ka kia rate ha\n"
            '{{"english_query": "what is potato rate in multan", "detected_language": "roman_urdu", "commodities": ["potato"], "cities": ["multan"]}}\n\n'
            "Query: کراچی، ملتان، لاہور: آلو کا ریٹ\n"
            '{{"english_query": "potato rate in karachi multan lahore", "detected_language": "urdu", "commodities": ["potato"], "cities": ["karachi", "multan", "lahore"]}}'
        )
        
        messages = prompt.format_messages(query=query)
        response = llm_translate.invoke(messages)
        
        import json
        response_text = response.content.strip()
        response_text = response_text.replace("```json", "").replace("```", "").strip()
        result = json.loads(response_text)
        
        # Override detection if Urdu script found
        if has_urdu_script:
            result["detected_language"] = "urdu"
        
        return result, result.get("detected_language", "en")
    except Exception as e:
        print(f"Translation error: {e}")
        # Check for Urdu script
        has_urdu_script = any('\u0600' <= c <= '\u06FF' for c in query)
        detected = "urdu" if has_urdu_script else "en"
        return {"english_query": query, "commodities": [], "cities": []}, detected

def fuzzy_match_item(word, reference_list, threshold=75):
    """Fuzzy match a word against reference list"""
    result = process.extractOne(word, reference_list, scorer=fuzz.ratio)
    if result and result[1] >= threshold:
        return result[0]
    return None

def extract_entities_with_fuzzy(translation_result, df):
    """Extract cities and commodities using fuzzy matching"""
    cities = []
    commodities = []
    
    if df.empty or "City" not in df.columns or "Commodity" not in df.columns:
        return cities, commodities
    
    all_cities = df["City"].dropna().unique().tolist()
    all_commodities = df["Commodity"].dropna().unique().tolist()
    
    # Get translated commodities and cities
    translated_commodities = translation_result.get("commodities", [])
    translated_cities = translation_result.get("cities", [])
    
    # Match commodities using aliases
    for trans_commodity in translated_commodities:
        trans_commodity = trans_commodity.lower()
        matched = False
        
        # Check aliases
        for standard, aliases in COMMODITY_ALIASES.items():
            if trans_commodity in aliases:
                # Find all commodities containing this standard name
                for commodity in all_commodities:
                    if standard in commodity.lower():
                        commodities.append(commodity)
                        matched = True
        
        # Fuzzy match if not found in aliases
        if not matched:
            fuzzy_matches = process.extract(trans_commodity, all_commodities, scorer=fuzz.partial_ratio, limit=3)
            for match, score in fuzzy_matches:
                if score >= 70:
                    commodities.append(match)
    
    # Match cities using aliases
    for trans_city in translated_cities:
        trans_city = trans_city.lower()
        matched = False
        
        # Check aliases
        for standard, aliases in CITY_ALIASES.items():
            if trans_city in aliases:
                for city in all_cities:
                    if standard in city.lower():
                        cities.append(city)
                        matched = True
                        break
        
        # Fuzzy match if not found
        if not matched:
            fuzzy_city = fuzzy_match_item(trans_city, all_cities, threshold=70)
            if fuzzy_city:
                cities.append(fuzzy_city)
    
    # Default to all cities if none specified
    if not cities:
        cities = all_cities
    
    commodities = list(set(commodities))
    cities = list(set(cities))
    
    return cities, commodities

# ==================== DATA SCRAPING ====================

def scrape_market_data():
    """Scrape market data from AMIS website"""
    try:
        r = requests.get(AMIS_URL, headers=HEADERS, timeout=15)
        r.raise_for_status()
    except requests.exceptions.SSLError:
        try:
            url = AMIS_URL.replace("https://", "http://")
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
        except:
            return None
    except:
        return None

    try:
        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table")
        
        if table is None:
            return None

        df = pd.read_html(str(table))[0]
        
        column_mapping = {
            "CityName": "City",
            "CropName": "Commodity",
            "Today's FQP/Average Price": "TodayPrice",
            "Yesterday's FQP/Average Price": "YesterdayPrice",
            "Change in Price": "Change",
            "Price Direction": "Direction"
        }
        
        for old, new in column_mapping.items():
            if old in df.columns:
                df = df.rename(columns={old: new})
        
        if "City" in df.columns:
            df["City"] = df["City"].astype(str).str.strip().str.lower()
        if "Commodity" in df.columns:
            df["Commodity"] = df["Commodity"].astype(str).str.strip().str.lower()
        
        df["scrape_time"] = datetime.now()
        
        return df
    except Exception as e:
        print(f"Error parsing: {e}")
        return None

def update_data_file():
    """Update historical data file"""
    print(f"🔄 Scraping at {datetime.now()}")
    new_data = scrape_market_data()
    
    if new_data is None or new_data.empty:
        print("⚠️ Failed to scrape data")
        return
    
    try:
        if os.path.exists(DATA_FILE):
            existing = pd.read_csv(DATA_FILE)
            existing["scrape_time"] = pd.to_datetime(existing["scrape_time"])
            combined = pd.concat([existing, new_data], ignore_index=True)
        else:
            combined = new_data
        
        combined.to_csv(DATA_FILE, index=False)
        print(f"✅ Data updated: {len(new_data)} rows added")
    except Exception as e:
        print(f"Error saving: {e}")

def scraping_scheduler():
    """Background thread for scheduled scraping"""
    update_data_file()
    
    schedule.every().day.at("04:00").do(update_data_file)
    schedule.every(2).hours.do(update_data_file)
    
    while True:
        schedule.run_pending()
        time.sleep(60)

# Start scraper thread once
if "scraper_started" not in st.session_state:
    scraper_thread = threading.Thread(target=scraping_scheduler, daemon=True)
    scraper_thread.start()
    st.session_state.scraper_started = True

# ==================== DATA LOADING ====================

def load_latest_data():
    """Load most recent data"""
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(DATA_FILE)
        df["scrape_time"] = pd.to_datetime(df["scrape_time"])
        
        latest_time = df["scrape_time"].max()
        df_latest = df[df["scrape_time"] == latest_time].copy()
        
        return df_latest
    except:
        return pd.DataFrame()

# ==================== QUERY PROCESSING ====================

def search_data_precise(cities, commodities, df):
    """Search data with precise commodity matching"""
    results = []
    
    if df.empty or not commodities:
        return results
    
    for city in cities:
        for commodity in commodities:
            # Exact match on the commodity
            matches = df[
                (df["City"] == city) & 
                (df["Commodity"] == commodity)
            ]
            
            for _, row in matches.iterrows():
                today = row.get("TodayPrice")
                yesterday = row.get("YesterdayPrice")
                
                if pd.notna(today) and pd.notna(yesterday):
                    # Convert per 100kg to per 40kg
                    today_per_40kg = float(today) * 0.4
                    yesterday_per_40kg = float(yesterday) * 0.4
                    change = today_per_40kg - yesterday_per_40kg
                    direction = "increased" if change > 0 else "decreased"
                    results.append({
                        "commodity": str(row['Commodity']).title(),
                        "city": str(row['City']).title(),
                        "today": round(today_per_40kg, 2),
                        "yesterday": round(yesterday_per_40kg, 2),
                        "change": abs(round(change, 2)),
                        "direction": direction
                    })
    
    return results

def generate_report_multilang(query: str, results: list, detected_lang: str):
    """Generate formatted report in detected language"""
    if not results:
        return "No data found for your query. Please check commodity or city names."
    
    if llm_report is None:
        result_text = "\n".join([f"{r['commodity']} in {r['city']}: Today Rs.{r['today']}/40kg, Yesterday Rs.{r['yesterday']}/40kg, {r['direction']} by Rs.{r['change']}/40kg" for r in results])
        return f"Raw results:\n{result_text}"
    
    try:
        lang_instruction = ""
        if detected_lang == "roman_urdu":
            lang_instruction = "Respond in Roman Urdu (English script with Urdu words like 'aaj', 'kal', 'ka', 'mein', 'rupees', 'per 40kg')."
        elif detected_lang == "urdu":
            lang_instruction = "Respond in PURE URDU language using Urdu script (آج، کل، کا، میں، روپے، فی 40 کلو). Use full Urdu sentences."
        else:
            lang_instruction = "Respond in English."
        
        results_text = "\n".join([f"{r['commodity']} in {r['city']}: Today Rs.{r['today']} per 40kg, Yesterday Rs.{r['yesterday']} per 40kg, {r['direction']} by Rs.{r['change']} per 40kg" for r in results])
        
        prompt = ChatPromptTemplate.from_template(
            "You are a market analyst. User asked: {query}\n\n"
            "Price data (ALL PRICES ARE PER 40KG):\n{data}\n\n"
            "IMPORTANT: {lang_instruction}\n"
            "Create a brief market report showing:\n"
            "- Commodity name\n"
            "- City name\n"
            "- Current price per 40kg\n"
            "- Price change per 40kg\n"
            "Keep it concise and clear. Mention 'per 40kg' or 'فی 40 کلو' in prices."
        )
        
        messages = prompt.format_messages(query=query, data=results_text, lang_instruction=lang_instruction)
        response = llm_report.invoke(messages)
        return response.content
    except Exception as e:
        result_text = "\n".join([f"{r['commodity']} in {r['city']}: Today Rs.{r['today']}/40kg, Yesterday Rs.{r['yesterday']}/40kg, {r['direction']} by Rs.{r['change']}/40kg" for r in results])
        return f"Error: {e}\n\nRaw results:\n{result_text}"

def create_visualizations(results):
    """Create tables from results"""
    if not results:
        return None
    
    df_results = pd.DataFrame(results)
    return df_results

# ==================== STREAMLIT UI ====================

st.set_page_config(page_title=" Aaj Ka Rate", page_icon="🌾", layout="wide")
st.title("🌾 Aaj Ka Rate")

# Auto-refresh check
if "last_check" not in st.session_state:
    st.session_state.last_check = datetime.now()

if (datetime.now() - st.session_state.last_check).seconds > 300:
    st.session_state.last_check = datetime.now()
    st.rerun()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant" and "table" in msg and msg["table"] is not None:
            if st.button(f"📊 Show Data Table", key=f"table_{idx}"):
                st.dataframe(msg["table"], use_container_width=True)

# Chat input
if user_input := st.chat_input("Ask about prices (e.g., 'allo ka rate', 'multan main allo ka rate')"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.spinner("Processing your query..."):
        df = load_latest_data()
        
        if df.empty:
            response = "⏳ No market data available yet. Data is being scraped. Please wait and try again."
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            # Translate query
            translation_result, detected_lang = translate_query_to_english(user_input)
            
            # Extract entities with fuzzy matching
            cities, commodities = extract_entities_with_fuzzy(translation_result, df)
            
            # Search data
            results = search_data_precise(cities, commodities, df)
            
            # Generate report
            response = generate_report_multilang(user_input, results, detected_lang)
            
            # Create table
            df_results = create_visualizations(results)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "table": df_results if results else None
            })
    
    with st.chat_message("assistant"):
        st.markdown(response)
        if results:
            if st.button("📊 Show Data Table", key="current_table"):
                st.dataframe(df_results, use_container_width=True)