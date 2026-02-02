# 🚀 Quick Start Guide

Get Market-Price running in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install fastapi uvicorn requests beautifulsoup4 pandas pydantic lxml
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

## Step 2: Run the Application

**Option A: Using startup script (Recommended)**
```bash
chmod +x run.sh
./run.sh
```

**Option B: Direct Python**
```bash
python app.py
```

**Option C: Using uvicorn**
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## Step 3: Access the Platform

Open your browser and visit:
- **Homepage**: http://localhost:8000
- **Chatbot**: http://localhost:8000/chatbot.html
- **Voice Search**: http://localhost:8000/voice.html
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Step 4: Test the Chatbot

Try these queries in the chatbot:
- "allo ka rate kia ha"
- "multan main chawal ka rate"
- "potato price in Lahore"
- "compare prices"

## Step 5: Monitor Scraping

Watch the console for automatic scraping:
```
🚀 Scheduler started - automatic scraping enabled
🔄 Running initial scraping cycle...
✅ Scraped AMIS Pakistan: 245 new prices
⏰ Scraping AMIS Pakistan (next in 135.2 minutes)
```

## What Happens Automatically?

✅ **On First Run:**
1. Creates `database.db` SQLite file
2. Initializes tables (prices, sources)
3. Starts background scraper
4. Performs initial data scrape
5. Starts FastAPI server on port 8000

✅ **Every 2-3 Hours:**
- Automatically scrapes AMIS Pakistan
- Updates price database
- No manual intervention needed

✅ **On Every Request:**
- Serves HTML pages
- Processes API calls
- Handles chatbot queries
- Returns latest price data

## Project Structure

```
Market-Price/
├── app.py              ← ONLY backend file (all logic here)
├── database.db         ← Auto-created SQLite database
├── requirements.txt    ← Python dependencies
│
├── index.html          ← Homepage
├── chatbot.html        ← Interactive chatbot
├── voice.html          ← Voice search
├── category.html       ← Browse by category
├── city.html           ← Browse by city
├── comparison.html     ← Compare prices
│
├── about.html          ← About page
├── contact.html        ← Contact page
├── privacy.html        ← Privacy policy
├── disclaimer.html     ← Disclaimer
│
├── robots.txt          ← SEO configuration
├── sitemap.xml         ← SEO sitemap
│
├── README.md           ← Full documentation
├── DEPLOYMENT_GUIDE.md ← Production deployment guide
├── QUICKSTART.md       ← This file
└── run.sh              ← Startup script
```

## Common Issues

### "Module not found"
**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

### "Port 8000 already in use"
**Solution:** Use different port
```bash
uvicorn app:app --port 8001
```

### "No data showing"
**Solution:** Wait 2-3 minutes for initial scraping to complete

### Database file permissions
**Solution:** Make sure you have write permissions in the directory

## API Examples

**Get latest prices:**
```bash
curl http://localhost:8000/api/prices
```

**Get potato prices:**
```bash
curl http://localhost:8000/api/prices?commodity=potato
```

**Ask chatbot:**
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "allo ka rate"}'
```

## Next Steps

1. ✅ Test all features locally
2. 📖 Read [README.md](README.md) for full documentation
3. 🚀 Follow [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for production deployment
4. 🎨 Customize HTML pages for your branding
5. 📊 Add more data sources in `app.py`

## Key Features to Test

- [x] Homepage loads with market data
- [x] Chatbot responds to queries
- [x] Voice search works (Chrome/Edge)
- [x] Multilingual support (Urdu, Roman Urdu, English)
- [x] Price comparisons work
- [x] Automatic scraping runs in background
- [x] API endpoints return data
- [x] Mobile responsive design

## Development Tips

**View logs:**
- Check console output for scraping activity
- Look for "✅ Scraped" messages

**Database inspection:**
```bash
sqlite3 database.db "SELECT COUNT(*) FROM prices;"
sqlite3 database.db "SELECT DISTINCT city FROM prices;"
```

**Stop server:**
- Press `Ctrl+C` in terminal

**Restart with changes:**
```bash
./run.sh
```

## Production Checklist

Before deploying to production:

- [ ] Update domain in `sitemap.xml`
- [ ] Update domain in `robots.txt`
- [ ] Configure SSL certificate
- [ ] Set up process manager (systemd)
- [ ] Configure Nginx reverse proxy
- [ ] Enable firewall
- [ ] Set up monitoring
- [ ] Test all pages
- [ ] Submit sitemap to Google Search Console

## Support

- 📧 Email: support@market-price.com
- 💻 GitHub: https://github.com/ZainMushtaq9/Market-Price
- 📖 Documentation: README.md

---

**You're all set! Enjoy using Market-Price! 🌾**

Need help? Open an issue on GitHub or check the full README.md
