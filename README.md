# Market-Price - Pakistan Market Information Platform

![Market-Price Banner](https://img.shields.io/badge/Pakistan-Market%20Prices-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-blue)
![Python](https://img.shields.io/badge/Python-3.10+-yellow)

## 🌾 Project Overview

Market-Price is a comprehensive, SEO-optimized platform providing real-time agricultural market prices across Pakistan. The platform supports multilingual queries (English, Urdu, Roman Urdu, Punjabi, Saraiki) and features an intelligent chatbot with voice support.

### ✨ Key Features

- ✅ **Automatic Price Scraping** - Data updates every 2-3 hours automatically
- ✅ **Multilingual Support** - English, Urdu, Roman Urdu, Punjabi, Saraiki
- ✅ **Intelligent Chatbot** - Rule-based NLP for market queries
- ✅ **Voice Search** - Speak to search prices
- ✅ **SEO Optimized** - Google AdSense ready
- ✅ **Mobile Friendly** - Responsive design
- ✅ **No Login Required** - Fully public access
- ✅ **Nationwide Coverage** - Multan, Lahore, Karachi, and more

## 📁 Repository Structure

```
Market-Price/
├── app.py                  # Single FastAPI backend file (ALL logic here)
├── database.db             # SQLite database (auto-created)
├── requirements.txt        # Python dependencies
│
├── index.html              # Homepage with SEO content
├── category.html           # Browse by category
├── city.html               # Browse by city
├── commodity.html          # Commodity details
├── mandi.html              # Mandi comparisons
├── comparison.html         # Price comparisons
├── chatbot.html            # Interactive chatbot
├── voice.html              # Voice search
├── about.html              # About page
├── contact.html            # Contact page
├── privacy.html            # Privacy policy
├── disclaimer.html         # Disclaimer
│
├── robots.txt              # SEO - Search engine rules
└── sitemap.xml             # SEO - Sitemap
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Internet connection for scraping

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/ZainMushtaq9/Market-Price.git
cd Market-Price
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python app.py
```

Or using uvicorn:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

4. **Access the platform**
```
http://localhost:8000
```

## 🏗️ Architecture

### Backend Architecture (app.py)

The entire backend is contained in a single `app.py` file with these internal components:

#### 1. **DatabaseManager**
- SQLite connection management
- Auto-creates tables on startup
- Handles price data storage with deduplication
- Optimized read queries

#### 2. **ScraperAgent**
- Scrapes AMIS Pakistan and other sources
- Normalizes commodity and city names
- Handles SSL errors gracefully
- Anti-IP-blocking measures:
  - Random delays (5-20 seconds)
  - Rotating User-Agents
  - Hash-based deduplication
  - Per-source cooldown tracking

#### 3. **SchedulerAgent**
- **Fully automatic** - Starts on backend launch
- No manual triggers required
- Randomized scraping intervals:
  - Vegetables/Fruits: Every 2-3 hours
  - Grains: Every 6-8 hours
  - Daily crops: 1-2 times per day
- Runs in background thread (daemon)

#### 4. **LanguageAgent**
- Detects language (English, Urdu, Roman Urdu)
- Extracts city and commodity entities
- Comprehensive alias mapping:
  - Cities: multan, mtn, ملتان, etc.
  - Commodities: aloo, allo, آلو, potato, etc.
- No external NLP libraries required

#### 5. **ChatbotAgent**
- Rule-based query understanding
- Intent detection (price, compare, cheapest, expensive)
- Database query execution
- Multilingual response generation

### Database Schema

```sql
-- Prices Table
CREATE TABLE prices (
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
);

-- Sources Table
CREATE TABLE sources (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    url TEXT NOT NULL,
    scrape_interval INTEGER NOT NULL,
    last_scraped TIMESTAMP,
    next_scrape TIMESTAMP,
    status TEXT DEFAULT 'active',
    failure_count INTEGER DEFAULT 0
);
```

## 🔌 API Endpoints

### Public Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve index.html |
| `/api/prices` | GET | Get latest prices (optional filters: commodity, city, hours) |
| `/api/categories` | GET | Get all unique categories |
| `/api/cities` | GET | Get all unique cities |
| `/api/commodities` | GET | Get all unique commodities |
| `/api/chat` | POST | Process chatbot query |
| `/api/stats` | GET | Platform statistics |
| `/health` | GET | Health check |

### Example API Calls

**Get all prices from last 24 hours:**
```bash
curl http://localhost:8000/api/prices
```

**Get potato prices:**
```bash
curl http://localhost:8000/api/prices?commodity=potato
```

**Get Multan prices:**
```bash
curl http://localhost:8000/api/prices?city=multan
```

**Chat query:**
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "allo ka rate kia ha"}'
```

## 🤖 Chatbot Usage

The chatbot understands queries in multiple languages and formats:

### English Queries
- "What is the potato price?"
- "Show me rice prices in Multan"
- "Which mandi has the cheapest onions?"

### Roman Urdu Queries
- "allo ka rate kia ha"
- "multan main chawal ki qeemat"
- "kon si mandi sasti ha"

### Urdu Queries
- "آلو کی قیمت کیا ہے"
- "ملتان میں چاول کا ریٹ"
- "کون سی منڈی سستی ہے"

### Intent Recognition
- **Price Query**: "rate kia ha", "qeemat batao"
- **Comparison**: "compare", "muqabla", "difference"
- **Cheapest**: "kon sasta", "lowest price"
- **Most Expensive**: "kon mehenga", "highest price"

## 🎤 Voice Search

Voice search uses browser APIs (no backend processing):
- Click microphone button
- Speak your query clearly
- Get instant results
- Auto-plays response

**Supported browsers:** Chrome, Edge, Safari

## 📱 Mobile Development

All HTML pages are mobile-responsive with:
- Flexible layouts
- Touch-friendly buttons
- Optimized font sizes
- Fast loading times

## 🔍 SEO Optimization

### AdSense Compliance

✅ **Human-written content** - All HTML pages have substantial text  
✅ **No thin pages** - Each page >300 words of unique content  
✅ **robots.txt** - Proper search engine directives  
✅ **sitemap.xml** - Complete site structure  
✅ **Fast loading** - Optimized for Core Web Vitals  
✅ **Chatbot after content** - Never blocks page rendering  

### Meta Tags

Every page includes:
- Title (unique, <60 chars)
- Description (unique, ~155 chars)
- Keywords
- Open Graph tags

## 🔒 Security & Privacy

- No user authentication required
- No personal data collection
- No payment processing
- SQLite injection prevention
- CORS enabled for API access

## 📊 Monitoring & Maintenance

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "scheduler_running": true,
  "database_connected": true
}
```

### Logs

The backend prints scraping logs to console:
```
🚀 Scheduler started - automatic scraping enabled
🔄 Running initial scraping cycle...
✅ Scraped AMIS Pakistan: 245 new prices
⏰ Scraping AMIS Pakistan (next in 135.2 minutes)
```

## 🐛 Troubleshooting

### Issue: No data showing

**Solution:**
1. Wait 2-3 minutes after startup for initial scrape
2. Check `/health` endpoint
3. Review console logs for errors

### Issue: Scraping failed

**Solution:**
1. Check internet connection
2. Verify source URLs are accessible
3. Look for SSL/certificate errors in logs
4. Database will auto-retry on next interval

### Issue: Chatbot not responding

**Solution:**
1. Ensure `/api/chat` endpoint is accessible
2. Check browser console for JavaScript errors
3. Verify database has price data

## 🚀 Deployment

### Deploy to VPS (Ubuntu/Debian)

```bash
# Install Python
sudo apt update
sudo apt install python3 python3-pip

# Clone repository
git clone https://github.com/ZainMushtaq9/Market-Price.git
cd Market-Price

# Install dependencies
pip3 install -r requirements.txt

# Run with systemd (production)
sudo nano /etc/systemd/system/market-price.service
```

**Systemd service file:**
```ini
[Unit]
Description=Market Price FastAPI Application
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/home/ubuntu/Market-Price
ExecStart=/usr/bin/python3 /home/ubuntu/Market-Price/app.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl enable market-price
sudo systemctl start market-price
sudo systemctl status market-price
```

### Deploy with Nginx

```nginx
server {
    listen 80;
    server_name market-price.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Deploy to Cloud

**Heroku:**
```bash
heroku create market-price-pk
git push heroku main
```

**Railway/Render:**
- Connect GitHub repository
- Set start command: `python app.py`
- Deploy

## 📈 Future Enhancements

- [ ] Add more data sources
- [ ] Historical price charts
- [ ] Price predictions (ML)
- [ ] SMS alerts for price changes
- [ ] WhatsApp integration
- [ ] Mobile app (React Native)

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Developer

**Zain Mushtaq**
- GitHub: [@ZainMushtaq9](https://github.com/ZainMushtaq9)
- Email: contact@market-price.com

## 🙏 Acknowledgments

- AMIS Pakistan for market data
- FastAPI framework
- Pakistan agricultural community

---

**Built with ❤️ for Pakistan's farmers, traders, and consumers**

For support or questions, please open an issue on GitHub.
