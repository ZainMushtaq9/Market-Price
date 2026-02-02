# Market-Price Deployment Guide

Complete guide for deploying the Market-Price platform to production.

## 🎯 Pre-Deployment Checklist

Before deploying, ensure:

- [ ] All HTML files are present
- [ ] `app.py` is configured correctly
- [ ] `requirements.txt` is complete
- [ ] `robots.txt` and `sitemap.xml` are created
- [ ] Domain name is ready (optional but recommended)
- [ ] SSL certificate provider selected (Let's Encrypt recommended)

## 🚀 Deployment Options

### Option 1: VPS Deployment (Recommended for Production)

#### Step 1: Server Setup

**Recommended VPS Providers:**
- DigitalOcean ($5-10/month)
- Linode ($5-10/month)
- Vultr ($5-10/month)
- AWS Lightsail ($3.50-10/month)

**Server Specifications:**
- OS: Ubuntu 22.04 LTS
- RAM: 1GB minimum (2GB recommended)
- Storage: 20GB minimum
- Bandwidth: 1TB/month

#### Step 2: Initial Server Configuration

```bash
# SSH into your server
ssh root@your-server-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y python3 python3-pip nginx git certbot python3-certbot-nginx

# Create application user
sudo adduser --system --group market-price

# Create application directory
sudo mkdir -p /var/www/market-price
sudo chown market-price:market-price /var/www/market-price
```

#### Step 3: Clone and Setup Application

```bash
# Switch to application user
sudo su - market-price

# Navigate to app directory
cd /var/www/market-price

# Clone repository
git clone https://github.com/ZainMushtaq9/Market-Price.git .

# Install Python dependencies
pip3 install -r requirements.txt

# Test the application
python3 app.py
# Press Ctrl+C after confirming it starts without errors
```

#### Step 4: Create Systemd Service

```bash
# Exit from market-price user
exit

# Create service file
sudo nano /etc/systemd/system/market-price.service
```

**Service file content:**
```ini
[Unit]
Description=Market Price FastAPI Application
After=network.target

[Service]
Type=simple
User=market-price
Group=market-price
WorkingDirectory=/var/www/market-price
Environment="PATH=/home/market-price/.local/bin:/usr/bin"
ExecStart=/usr/bin/python3 /var/www/market-price/app.py
Restart=always
RestartSec=10

# Logging
StandardOutput=append:/var/log/market-price/app.log
StandardError=append:/var/log/market-price/error.log

[Install]
WantedBy=multi-user.target
```

```bash
# Create log directory
sudo mkdir -p /var/log/market-price
sudo chown market-price:market-price /var/log/market-price

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable market-price
sudo systemctl start market-price

# Check status
sudo systemctl status market-price

# View logs
sudo tail -f /var/log/market-price/app.log
```

#### Step 5: Configure Nginx

```bash
# Create Nginx configuration
sudo nano /etc/nginx/sites-available/market-price
```

**Nginx configuration:**
```nginx
server {
    listen 80;
    server_name market-price.com www.market-price.com;  # Replace with your domain

    # Serve static files
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # API endpoints
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Static files optimization
    location ~* \.(jpg|jpeg|png|gif|ico|css|js)$ {
        proxy_pass http://127.0.0.1:8000;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/json;
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/market-price /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Reload Nginx
sudo systemctl reload nginx
```

#### Step 6: Setup SSL Certificate (HTTPS)

```bash
# Get SSL certificate from Let's Encrypt
sudo certbot --nginx -d market-price.com -d www.market-price.com

# Certificate auto-renewal test
sudo certbot renew --dry-run
```

#### Step 7: Configure Firewall

```bash
# Enable UFW
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw enable

# Check status
sudo ufw status
```

### Option 2: Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "app.py"]
```

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./database.db:/app/database.db
    restart: always
    environment:
      - PYTHONUNBUFFERED=1
```

Deploy:
```bash
docker-compose up -d
```

### Option 3: Platform-as-a-Service (PaaS)

#### Heroku

```bash
# Install Heroku CLI
curl https://cli-assets.heroku.com/install.sh | sh

# Login
heroku login

# Create app
heroku create market-price-pk

# Add Procfile
echo "web: python app.py" > Procfile

# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

#### Railway.app

1. Connect GitHub repository
2. Set start command: `python app.py`
3. Deploy automatically on push

#### Render.com

1. Connect GitHub repository
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `python app.py`
4. Deploy

## 📊 Post-Deployment

### Monitoring

**Check application health:**
```bash
curl https://market-price.com/health
```

**Monitor logs:**
```bash
sudo journalctl -u market-price -f
```

**Check scraper activity:**
```bash
sudo tail -f /var/log/market-price/app.log | grep "Scraped"
```

### Performance Optimization

**Enable HTTP/2:**
```nginx
listen 443 ssl http2;
```

**Configure caching:**
```nginx
location ~* \.(html|css|js)$ {
    expires 1h;
    add_header Cache-Control "public, must-revalidate";
}
```

**Database optimization:**
```bash
# Vacuum database monthly
sqlite3 database.db "VACUUM;"
```

### Security Hardening

**Disable root SSH:**
```bash
sudo nano /etc/ssh/sshd_config
# Set: PermitRootLogin no
sudo systemctl restart ssh
```

**Install fail2ban:**
```bash
sudo apt install fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

**Regular updates:**
```bash
# Create update script
sudo nano /usr/local/bin/update-market-price.sh
```

```bash
#!/bin/bash
cd /var/www/market-price
git pull
pip3 install -r requirements.txt
sudo systemctl restart market-price
```

```bash
chmod +x /usr/local/bin/update-market-price.sh

# Add to cron (weekly updates)
sudo crontab -e
# Add: 0 3 * * 0 /usr/local/bin/update-market-price.sh
```

## 🔍 Google AdSense Setup

1. **Wait for traffic** (100+ daily visitors recommended)
2. **Apply for AdSense**: https://www.google.com/adsense
3. **Add AdSense code** to HTML files:

```html
<!-- Add before </head> -->
<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-XXXXXXXXXX"
     crossorigin="anonymous"></script>
```

4. **Insert ad units** in HTML:

```html
<!-- Display Ad -->
<ins class="adsbygoogle"
     style="display:block"
     data-ad-client="ca-pub-XXXXXXXXXX"
     data-ad-slot="XXXXXXXXXX"
     data-ad-format="auto"></ins>
<script>
     (adsbygoogle = window.adsbygoogle || []).push({});
</script>
```

## 📈 SEO Optimization

### Submit Sitemap

**Google Search Console:**
1. Visit: https://search.google.com/search-console
2. Add property: market-price.com
3. Submit sitemap: https://market-price.com/sitemap.xml

**Bing Webmaster:**
1. Visit: https://www.bing.com/webmasters
2. Add site
3. Submit sitemap

### Update sitemap.xml

```bash
# Update sitemap with real domain
sudo nano /var/www/market-price/sitemap.xml
# Replace market-price.com with your actual domain
```

## 🐛 Troubleshooting

### Application won't start

```bash
# Check logs
sudo journalctl -u market-price -n 50

# Check Python path
which python3

# Test manually
cd /var/www/market-price
python3 app.py
```

### Database errors

```bash
# Check permissions
ls -la /var/www/market-price/database.db

# Reset database
rm database.db
# Restart app to recreate
sudo systemctl restart market-price
```

### Nginx errors

```bash
# Check configuration
sudo nginx -t

# View error logs
sudo tail -f /var/log/nginx/error.log
```

### Scraper not working

```bash
# Check network
ping amis.pk

# Test scraper manually
python3 -c "from app import scraper_agent; scraper_agent.scrape_amis('https://www.amis.pk/daily%20market%20changes.aspx')"
```

## 📞 Support

For deployment issues:
- Create GitHub issue: https://github.com/ZainMushtaq9/Market-Price/issues
- Email: support@market-price.com

## 🎉 Success Checklist

After deployment, verify:

- [ ] Website loads at your domain
- [ ] HTTPS is working
- [ ] All pages are accessible
- [ ] Chatbot responds to queries
- [ ] Voice search works
- [ ] Database is populating with prices
- [ ] Health endpoint returns healthy status
- [ ] Logs show regular scraping activity
- [ ] Mobile view is responsive
- [ ] Google can crawl your site (Search Console)

**Congratulations! Your Market-Price platform is now live! 🎊**
