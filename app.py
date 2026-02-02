"""
Market-Price Backend - Single File FastAPI Application
All scraping, database, chatbot, and API logic in one file
"""

import os
import sqlite3
import requests
import pandas as pd
from datetime import datetime, timedelta
import threading
import time
import hashlib
import random
import json
import re
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from bs4 import BeautifulSoup
from pydantic import BaseModel

# ==================== CONFIGURATION ====================

class Config:
    """Application configuration"""
    DATABASE_PATH = "database.db"
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    ]
    
    # Scraping sources with intervals (in minutes)
    SOURCES = [
        {
            "id": 1,
            "name": "AMIS Pakistan",
            "url": "https://www.amis.pk/daily%20market%20changes.aspx",
            "type": "sabzi",
            "interval_min": 120,  # 2 hours
            "interval_max": 180,  # 3 hours
        },
        # Add more sources here in future
    ]

# ==================== DATABASE MANAGER ====================

class DatabaseManager:
    """Handles all database operations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def init_database(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Prices table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    commodity TEXT NOT NULL,
                    category TEXT,
                    city TEXT NOT NULL,
                    mandi TEXT,
                    price REAL NOT NULL,
                    price_yesterday REAL,
                    unit TEXT DEFAULT 'per 40kg',
                    source TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    data_hash TEXT UNIQUE
                )
            """)
            
            # Sources table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sources (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    url TEXT NOT NULL,
                    scrape_interval INTEGER NOT NULL,
                    last_scraped TIMESTAMP,
                    next_scrape TIMESTAMP,
                    status TEXT DEFAULT 'active',
                    failure_count INTEGER DEFAULT 0
                )
            """)
            
            # Initialize sources
            for source in Config.SOURCES:
                cursor.execute("""
                    INSERT OR IGNORE INTO sources (id, name, url, scrape_interval)
                    VALUES (?, ?, ?, ?)
                """, (source["id"], source["name"], source["url"], source["interval_min"]))
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_commodity ON prices(commodity)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_city ON prices(city)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_updated ON prices(updated_at)")
    
    def insert_prices(self, prices_data: List[Dict]) -> int:
        """Insert price data with deduplication"""
        inserted = 0
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            for price in prices_data:
                # Create hash for deduplication
                hash_string = f"{price['commodity']}-{price['city']}-{price['price']}-{price.get('mandi', '')}"
                data_hash = hashlib.md5(hash_string.encode()).hexdigest()
                
                try:
                    cursor.execute("""
                        INSERT INTO prices (
                            commodity, category, city, mandi, price, 
                            price_yesterday, unit, source, data_hash
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        price['commodity'],
                        price.get('category'),
                        price['city'],
                        price.get('mandi'),
                        price['price'],
                        price.get('price_yesterday'),
                        price.get('unit', 'per 40kg'),
                        price['source'],
                        data_hash
                    ))
                    inserted += 1
                except sqlite3.IntegrityError:
                    # Duplicate data, skip
                    pass
        
        return inserted
    
    def get_latest_prices(self, hours: int = 24) -> List[Dict]:
        """Get latest prices within specified hours"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cutoff = datetime.now() - timedelta(hours=hours)
            
            cursor.execute("""
                SELECT * FROM prices 
                WHERE updated_at >= ?
                ORDER BY updated_at DESC
            """, (cutoff,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_prices_by_commodity(self, commodity: str) -> List[Dict]:
        """Get prices for specific commodity"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM prices 
                WHERE LOWER(commodity) LIKE ?
                ORDER BY updated_at DESC
                LIMIT 100
            """, (f"%{commodity.lower()}%",))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_prices_by_city(self, city: str) -> List[Dict]:
        """Get prices for specific city"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM prices 
                WHERE LOWER(city) LIKE ?
                ORDER BY updated_at DESC
                LIMIT 100
            """, (f"%{city.lower()}%",))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_all_commodities(self) -> List[str]:
        """Get unique commodity names"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT commodity FROM prices ORDER BY commodity")
            return [row[0] for row in cursor.fetchall()]
    
    def get_all_cities(self) -> List[str]:
        """Get unique city names"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT city FROM prices ORDER BY city")
            return [row[0] for row in cursor.fetchall()]
    
    def get_all_categories(self) -> List[str]:
        """Get unique categories"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT category FROM prices WHERE category IS NOT NULL ORDER BY category")
            return [row[0] for row in cursor.fetchall()]
    
    def update_source_scrape_time(self, source_id: int, success: bool = True):
        """Update source last scrape time"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if success:
                cursor.execute("""
                    UPDATE sources 
                    SET last_scraped = ?, failure_count = 0
                    WHERE id = ?
                """, (datetime.now(), source_id))
            else:
                cursor.execute("""
                    UPDATE sources 
                    SET failure_count = failure_count + 1
                    WHERE id = ?
                """, (source_id,))
    
    def get_sources_to_scrape(self) -> List[Dict]:
        """Get sources that need scraping"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM sources 
                WHERE status = 'active' 
                AND (last_scraped IS NULL 
                     OR datetime(last_scraped, '+' || scrape_interval || ' minutes') <= datetime('now'))
            """)
            return [dict(row) for row in cursor.fetchall()]

# ==================== SCRAPER AGENT ====================

class ScraperAgent:
    """Handles web scraping from multiple sources"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def get_random_headers(self) -> Dict:
        """Generate random headers for scraping"""
        return {
            "User-Agent": random.choice(Config.USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
    
    def scrape_amis(self, url: str) -> Optional[List[Dict]]:
        """Scrape AMIS Pakistan website"""
        try:
            # Random delay to avoid detection
            time.sleep(random.uniform(5, 20))
            
            # Try HTTPS first, fall back to HTTP
            try:
                response = requests.get(url, headers=self.get_random_headers(), timeout=15)
                response.raise_for_status()
            except requests.exceptions.SSLError:
                url = url.replace("https://", "http://")
                response = requests.get(url, headers=self.get_random_headers(), timeout=15)
                response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table")
            
            if not table:
                return None
            
            df = pd.read_html(str(table))[0]
            
            # Normalize column names
            column_mapping = {
                "CityName": "City",
                "CropName": "Commodity",
                "Today's FQP/Average Price": "TodayPrice",
                "Yesterday's FQP/Average Price": "YesterdayPrice",
            }
            
            for old, new in column_mapping.items():
                if old in df.columns:
                    df = df.rename(columns={old: new})
            
            # Process data
            prices = []
            for _, row in df.iterrows():
                if pd.notna(row.get("TodayPrice")):
                    # Convert per 100kg to per 40kg
                    today_per_40kg = float(row["TodayPrice"]) * 0.4
                    yesterday_per_40kg = float(row.get("YesterdayPrice", 0)) * 0.4 if pd.notna(row.get("YesterdayPrice")) else None
                    
                    prices.append({
                        "commodity": str(row["Commodity"]).strip().title(),
                        "category": self._categorize_commodity(str(row["Commodity"])),
                        "city": str(row["City"]).strip().title(),
                        "mandi": None,
                        "price": round(today_per_40kg, 2),
                        "price_yesterday": round(yesterday_per_40kg, 2) if yesterday_per_40kg else None,
                        "unit": "per 40kg",
                        "source": "AMIS Pakistan"
                    })
            
            return prices
        
        except Exception as e:
            print(f"Error scraping AMIS: {e}")
            return None
    
    def _categorize_commodity(self, commodity: str) -> str:
        """Categorize commodity based on name"""
        commodity_lower = commodity.lower()
        
        if any(word in commodity_lower for word in ["potato", "onion", "tomato", "chilli", "garlic", "ginger"]):
            return "Vegetables"
        elif any(word in commodity_lower for word in ["rice", "wheat", "maize", "grain"]):
            return "Grains"
        elif any(word in commodity_lower for word in ["apple", "banana", "mango", "orange", "grapes"]):
            return "Fruits"
        elif any(word in commodity_lower for word in ["cotton", "sugarcane"]):
            return "Crops"
        else:
            return "Other"
    
    def scrape_source(self, source: Dict) -> bool:
        """Scrape a specific source"""
        try:
            if source["name"] == "AMIS Pakistan":
                prices = self.scrape_amis(source["url"])
                
                if prices:
                    inserted = self.db.insert_prices(prices)
                    print(f"✅ Scraped {source['name']}: {inserted} new prices")
                    self.db.update_source_scrape_time(source["id"], success=True)
                    return True
                else:
                    self.db.update_source_scrape_time(source["id"], success=False)
                    return False
            
            return False
        
        except Exception as e:
            print(f"❌ Error scraping {source['name']}: {e}")
            self.db.update_source_scrape_time(source["id"], success=False)
            return False

# ==================== LANGUAGE AGENT ====================

class LanguageAgent:
    """Handles multilingual query understanding"""
    
    # Comprehensive language aliases
    CITY_ALIASES = {
        "multan": ["multan", "mtn", "mtln", "mltn", "mlt", "ملتان"],
        "lahore": ["lahore", "lhr", "lhore", "lahor", "lahur", "لاہور"],
        "karachi": ["karachi", "khi", "krachi", "karchi", "کراچی"],
        "islamabad": ["islamabad", "isb", "isl", "islambad", "اسلام آباد"],
        "faisalabad": ["faisalabad", "fsd", "faisalbad", "lyallpur", "فیصل آباد"],
        "rawalpindi": ["rawalpindi", "rwp", "pindi", "راولپنڈی"],
        "peshawar": ["peshawar", "pesh", "pshawar", "پشاور"],
        "quetta": ["quetta", "qta", "queta", "کوئٹہ"],
        "sialkot": ["sialkot", "skt", "slkt", "سیالکوٹ"],
        "gujranwala": ["gujranwala", "gjw", "gujrat", "گوجرانوالہ"],
    }
    
    COMMODITY_ALIASES = {
        "potato": ["potato", "patato", "aloo", "allo", "alu", "آلو", "alloo"],
        "rice": ["rice", "chawal", "chawl", "چاول", "chwal"],
        "wheat": ["wheat", "gandum", "gandom", "گندم", "gehun"],
        "onion": ["onion", "pyaz", "piaz", "پیاز"],
        "tomato": ["tomato", "tamatar", "tamater", "ٹماٹر"],
        "chilli": ["chilli", "mirch", "mirchi", "مرچ", "chili"],
        "maize": ["maize", "makai", "maki", "makki", "مکئی", "corn"],
        "garlic": ["garlic", "lehsan", "لہسن", "lahsan"],
        "ginger": ["ginger", "adrak", "ادرک"],
    }
    
    def detect_language(self, text: str) -> str:
        """Detect language from text"""
        # Check for Urdu script
        if any('\u0600' <= c <= '\u06FF' for c in text):
            return "urdu"
        
        # Check for Roman Urdu keywords
        roman_urdu_keywords = ["ka", "ki", "ke", "main", "mein", "ha", "hai", "kia", "kya", "aaj", "kal"]
        text_lower = text.lower()
        
        if any(keyword in text_lower.split() for keyword in roman_urdu_keywords):
            return "roman_urdu"
        
        return "english"
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for processing"""
        # Remove special characters but keep Urdu
        text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
        return text.lower().strip()
    
    def extract_cities(self, text: str) -> List[str]:
        """Extract city names from text"""
        normalized = self.normalize_text(text)
        words = normalized.split()
        
        found_cities = []
        for standard, aliases in self.CITY_ALIASES.items():
            for word in words:
                if word in aliases:
                    found_cities.append(standard.title())
                    break
        
        return list(set(found_cities))
    
    def extract_commodities(self, text: str) -> List[str]:
        """Extract commodity names from text"""
        normalized = self.normalize_text(text)
        words = normalized.split()
        
        found_commodities = []
        for standard, aliases in self.COMMODITY_ALIASES.items():
            for word in words:
                if word in aliases:
                    found_commodities.append(standard.title())
                    break
        
        return list(set(found_commodities))
    
    def translate_response(self, text: str, target_lang: str) -> str:
        """Simple rule-based translation for common phrases"""
        if target_lang == "urdu":
            translations = {
                "today": "آج",
                "yesterday": "کل",
                "price": "قیمت",
                "per 40kg": "فی 40 کلو",
                "increased": "بڑھ گئی",
                "decreased": "کم ہو گئی",
            }
        elif target_lang == "roman_urdu":
            translations = {
                "today": "aaj",
                "yesterday": "kal",
                "price": "rate",
                "per 40kg": "per 40kg",
                "increased": "barh gayi",
                "decreased": "kam ho gayi",
            }
        else:
            return text
        
        for eng, trans in translations.items():
            text = text.replace(eng, trans)
        
        return text

# ==================== CHATBOT AGENT ====================

class ChatbotAgent:
    """Rule-based chatbot for market queries"""
    
    def __init__(self, db_manager: DatabaseManager, language_agent: LanguageAgent):
        self.db = db_manager
        self.lang = language_agent
    
    def understand_query(self, query: str) -> Dict:
        """Parse user query and extract intent"""
        language = self.lang.detect_language(query)
        cities = self.lang.extract_cities(query)
        commodities = self.lang.extract_commodities(query)
        
        # Determine intent
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["compare", "comparison", "muqabla", "kon sasta", "kon mehenga"]):
            intent = "compare"
        elif any(word in query_lower for word in ["cheapest", "sasta", "kam", "lowest"]):
            intent = "cheapest"
        elif any(word in query_lower for word in ["expensive", "mehenga", "zyada", "highest"]):
            intent = "expensive"
        else:
            intent = "price"
        
        return {
            "language": language,
            "cities": cities,
            "commodities": commodities,
            "intent": intent,
            "original_query": query
        }
    
    def generate_response(self, parsed_query: Dict) -> str:
        """Generate response based on parsed query"""
        cities = parsed_query["cities"]
        commodities = parsed_query["commodities"]
        intent = parsed_query["intent"]
        language = parsed_query["language"]
        
        # If no cities specified, get all cities
        if not cities:
            cities = self.db.get_all_cities()[:5]  # Limit to top 5
        
        # If no commodities, try to get recent prices
        if not commodities:
            return self._format_no_commodity_response(language)
        
        # Get price data
        results = []
        for city in cities:
            for commodity in commodities:
                prices = self.db.get_latest_prices(hours=24)
                
                for price in prices:
                    if (commodity.lower() in price['commodity'].lower() and 
                        city.lower() in price['city'].lower()):
                        results.append(price)
        
        if not results:
            return self._format_no_data_response(language, commodities, cities)
        
        # Format response based on intent
        if intent == "compare":
            return self._format_comparison_response(results, language)
        elif intent == "cheapest":
            return self._format_cheapest_response(results, language)
        elif intent == "expensive":
            return self._format_expensive_response(results, language)
        else:
            return self._format_price_response(results, language)
    
    def _format_price_response(self, results: List[Dict], language: str) -> str:
        """Format standard price response"""
        response_parts = []
        
        for result in results[:10]:  # Limit to 10 results
            commodity = result['commodity']
            city = result['city']
            price = result['price']
            unit = result.get('unit', 'per 40kg')
            
            if language == "urdu":
                response_parts.append(f"{commodity} {city} میں: Rs.{price} {unit}")
            elif language == "roman_urdu":
                response_parts.append(f"{commodity} {city} mein: Rs.{price} {unit}")
            else:
                response_parts.append(f"{commodity} in {city}: Rs.{price} {unit}")
        
        return "\n".join(response_parts)
    
    def _format_comparison_response(self, results: List[Dict], language: str) -> str:
        """Format comparison response"""
        if not results:
            return "No data found for comparison."
        
        # Sort by price
        sorted_results = sorted(results, key=lambda x: x['price'])
        cheapest = sorted_results[0]
        expensive = sorted_results[-1]
        
        if language == "urdu":
            return f"سب سے سستا: {cheapest['commodity']} {cheapest['city']} میں Rs.{cheapest['price']}\nسب سے مہنگا: {expensive['commodity']} {expensive['city']} میں Rs.{expensive['price']}"
        elif language == "roman_urdu":
            return f"Sab se sasta: {cheapest['commodity']} {cheapest['city']} mein Rs.{cheapest['price']}\nSab se mehenga: {expensive['commodity']} {expensive['city']} mein Rs.{expensive['price']}"
        else:
            return f"Cheapest: {cheapest['commodity']} in {cheapest['city']} at Rs.{cheapest['price']}\nMost expensive: {expensive['commodity']} in {expensive['city']} at Rs.{expensive['price']}"
    
    def _format_cheapest_response(self, results: List[Dict], language: str) -> str:
        """Format cheapest price response"""
        if not results:
            return "No data found."
        
        cheapest = min(results, key=lambda x: x['price'])
        
        if language == "urdu":
            return f"سب سے سستا: {cheapest['commodity']} {cheapest['city']} میں Rs.{cheapest['price']} فی 40 کلو"
        elif language == "roman_urdu":
            return f"Sab se sasta: {cheapest['commodity']} {cheapest['city']} mein Rs.{cheapest['price']} per 40kg"
        else:
            return f"Cheapest: {cheapest['commodity']} in {cheapest['city']} at Rs.{cheapest['price']} per 40kg"
    
    def _format_expensive_response(self, results: List[Dict], language: str) -> str:
        """Format most expensive price response"""
        if not results:
            return "No data found."
        
        expensive = max(results, key=lambda x: x['price'])
        
        if language == "urdu":
            return f"سب سے مہنگا: {expensive['commodity']} {expensive['city']} میں Rs.{expensive['price']} فی 40 کلو"
        elif language == "roman_urdu":
            return f"Sab se mehenga: {expensive['commodity']} {expensive['city']} mein Rs.{expensive['price']} per 40kg"
        else:
            return f"Most expensive: {expensive['commodity']} in {expensive['city']} at Rs.{expensive['price']} per 40kg"
    
    def _format_no_commodity_response(self, language: str) -> str:
        """Response when no commodity specified"""
        if language == "urdu":
            return "براہ کرم کوئی چیز کا نام بتائیں، مثلاً آلو، چاول، یا پیاز"
        elif language == "roman_urdu":
            return "Koi cheez ka naam batayen, misal ke taur par aloo, chawal, ya pyaz"
        else:
            return "Please specify a commodity, for example: potato, rice, or onion"
    
    def _format_no_data_response(self, language: str, commodities: List[str], cities: List[str]) -> str:
        """Response when no data found"""
        commodity_str = ", ".join(commodities)
        city_str = ", ".join(cities)
        
        if language == "urdu":
            return f"{commodity_str} کی {city_str} میں قیمت دستیاب نہیں"
        elif language == "roman_urdu":
            return f"{commodity_str} ki {city_str} mein qeemat available nahi"
        else:
            return f"No price data available for {commodity_str} in {city_str}"

# ==================== SCHEDULER AGENT ====================

class SchedulerAgent:
    """Handles automatic background scraping"""
    
    def __init__(self, db_manager: DatabaseManager, scraper_agent: ScraperAgent):
        self.db = db_manager
        self.scraper = scraper_agent
        self.running = False
        self.thread = None
    
    def start(self):
        """Start background scraping"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._scrape_loop, daemon=True)
            self.thread.start()
            print("🚀 Scheduler started - automatic scraping enabled")
    
    def stop(self):
        """Stop background scraping"""
        self.running = False
    
    def _scrape_loop(self):
        """Main scraping loop"""
        # Initial scrape on startup
        self._run_scraping_cycle()
        
        while self.running:
            try:
                # Check which sources need scraping
                sources_to_scrape = self.db.get_sources_to_scrape()
                
                if sources_to_scrape:
                    for source in sources_to_scrape:
                        # Add randomized delay
                        random_delay = random.uniform(
                            source["interval_min"] * 60,
                            source.get("interval_max", source["interval_min"]) * 60
                        )
                        
                        print(f"⏰ Scraping {source['name']} (next in {random_delay/60:.1f} minutes)")
                        self.scraper.scrape_source(source)
                
                # Sleep for 5 minutes before checking again
                time.sleep(300)
            
            except Exception as e:
                print(f"❌ Scheduler error: {e}")
                time.sleep(300)
    
    def _run_scraping_cycle(self):
        """Run one complete scraping cycle"""
        print("🔄 Running initial scraping cycle...")
        for source in Config.SOURCES:
            self.scraper.scrape_source(source)

# ==================== FASTAPI APPLICATION ====================

# Initialize managers
db_manager = DatabaseManager(Config.DATABASE_PATH)
scraper_agent = ScraperAgent(db_manager)
language_agent = LanguageAgent()
chatbot_agent = ChatbotAgent(db_manager, language_agent)
scheduler_agent = SchedulerAgent(db_manager, scraper_agent)

# Start scheduler on application startup
scheduler_agent.start()

# Initialize FastAPI
app = FastAPI(
    title="Market Price API",
    description="Pakistan Market Price Information Platform",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    language: str
    detected_entities: Dict

# ==================== API ENDPOINTS ====================

@app.get("/")
async def read_root():
    """Redirect to index.html"""
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/prices")
async def get_prices(
    commodity: Optional[str] = Query(None),
    city: Optional[str] = Query(None),
    hours: int = Query(24, ge=1, le=168)
):
    """Get price data with optional filters"""
    try:
        if commodity:
            prices = db_manager.get_prices_by_commodity(commodity)
        elif city:
            prices = db_manager.get_prices_by_city(city)
        else:
            prices = db_manager.get_latest_prices(hours=hours)
        
        return JSONResponse(content={
            "success": True,
            "count": len(prices),
            "data": prices
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/categories")
async def get_categories():
    """Get all unique categories"""
    try:
        categories = db_manager.get_all_categories()
        return JSONResponse(content={
            "success": True,
            "categories": categories
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/cities")
async def get_cities():
    """Get all unique cities"""
    try:
        cities = db_manager.get_all_cities()
        return JSONResponse(content={
            "success": True,
            "cities": cities
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/commodities")
async def get_commodities():
    """Get all unique commodities"""
    try:
        commodities = db_manager.get_all_commodities()
        return JSONResponse(content={
            "success": True,
            "commodities": commodities
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Process chatbot query"""
    try:
        # Parse query
        parsed = chatbot_agent.understand_query(request.query)
        
        # Generate response
        response = chatbot_agent.generate_response(parsed)
        
        return JSONResponse(content={
            "success": True,
            "response": response,
            "language": parsed["language"],
            "detected_entities": {
                "cities": parsed["cities"],
                "commodities": parsed["commodities"],
                "intent": parsed["intent"]
            }
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    """Get platform statistics"""
    try:
        all_prices = db_manager.get_latest_prices(hours=24)
        
        return JSONResponse(content={
            "success": True,
            "stats": {
                "total_prices": len(all_prices),
                "total_cities": len(db_manager.get_all_cities()),
                "total_commodities": len(db_manager.get_all_commodities()),
                "last_updated": max([p["updated_at"] for p in all_prices]) if all_prices else None
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(content={
        "status": "healthy",
        "scheduler_running": scheduler_agent.running,
        "database_connected": os.path.exists(Config.DATABASE_PATH)
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
