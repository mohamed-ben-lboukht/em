# 🐅 My Tiger - Ultra-Optimized Installation Guide

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Node.js 16+ (optional, for development)
- Git

### 1. Clone & Install
```bash
# Clone the repository
git clone <repository-url>
cd my-tiger-optimized

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application
```bash
# Start the optimized server
python app.py

# Open your browser
open http://localhost:8000
```

## 🔧 Advanced Setup

### Environment Configuration
Create a `.env` file for production settings:
```bash
# Security
SECRET_KEY=your-ultra-secure-secret-key-here

# Performance
CACHE_TYPE=redis
REDIS_URL=redis://localhost:6379/0

# Monitoring
LOG_LEVEL=INFO
ENABLE_METRICS=true

# Rate Limiting
RATELIMIT_STORAGE_URL=redis://localhost:6379/1
```

### Redis Setup (Recommended)
For optimal performance, install Redis:

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl enable redis-server
sudo systemctl start redis-server
```

#### macOS
```bash
brew install redis
brew services start redis
```

#### Windows
```bash
# Using Chocolatey
choco install redis-64

# Or download from: https://github.com/microsoftarchive/redis/releases
```

### Production Deployment

#### Using Gunicorn (Recommended)
```bash
# Install Gunicorn
pip install gunicorn

# Run with optimizations
gunicorn --worker-class eventlet \
         --workers 4 \
         --bind 0.0.0.0:8000 \
         --max-requests 1000 \
         --max-requests-jitter 100 \
         --preload \
         --access-logfile - \
         --error-logfile - \
         app:app
```

#### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["gunicorn", "--worker-class", "eventlet", "--workers", "4", "--bind", "0.0.0.0:8000", "app:app"]
```

```bash
# Build and run
docker build -t my-tiger-optimized .
docker run -p 8000:8000 my-tiger-optimized
```

#### Docker Compose (with Redis)
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
    
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

## 🚀 Performance Tuning

### System Optimization
```bash
# Increase file descriptor limits
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf

# Optimize TCP settings
echo "net.core.somaxconn = 65536" >> /etc/sysctl.conf
echo "net.ipv4.tcp_max_syn_backlog = 65536" >> /etc/sysctl.conf
sysctl -p
```

### Python Optimizations
```bash
# Use optimized Python build
export PYTHONOPTIMIZE=2
export PYTHONDONTWRITEBYTECODE=1

# Memory optimization
export MALLOC_ARENA_MAX=2
```

### Application Configuration
```python
# config.py
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
    # Cache settings
    CACHE_TYPE = 'redis'
    CACHE_REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    CACHE_DEFAULT_TIMEOUT = 300
    
    # Rate limiting
    RATELIMIT_STORAGE_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/1')
    
    # Performance
    JSON_SORT_KEYS = False
    JSONIFY_PRETTYPRINT_REGULAR = False
    
    # Security
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
```

## 📊 Monitoring & Health Checks

### Built-in Endpoints
```bash
# Health check
curl http://localhost:8000/api/health

# Detailed metrics
curl http://localhost:8000/api/metrics
```

### Response Examples
```json
{
  "status": "healthy",
  "version": "2.0.0-optimized",
  "model": {
    "loaded": true,
    "type": "HybridOptimizedDetector"
  },
  "performance": {
    "uptime_seconds": 3600,
    "requests_total": 1000,
    "avg_response_time_ms": 45,
    "memory_usage_mb": 42.3
  }
}
```

### External Monitoring Setup

#### Prometheus Integration
```python
# Add to app.py for Prometheus metrics
from prometheus_client import Counter, Histogram, generate_latest

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests')
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')

@app.route('/metrics')
def prometheus_metrics():
    return generate_latest()
```

#### Grafana Dashboard
Import the provided dashboard configuration for visualization.

## 🔒 Security Configuration

### SSL/TLS Setup
```nginx
# Nginx configuration
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### Firewall Configuration
```bash
# UFW (Ubuntu)
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# Fail2ban for additional protection
sudo apt install fail2ban
```

## 🧪 Testing & Validation

### Performance Testing
```bash
# Load testing with Apache Bench
ab -n 1000 -c 10 http://localhost:8000/

# WebSocket load testing
npm install -g artillery
artillery quick --count 50 --num 10 ws://localhost:8000/socket.io/
```

### API Testing
```bash
# Health check
curl -X GET http://localhost:8000/api/health

# Keystroke data simulation
curl -X POST http://localhost:8000/api/test \
  -H "Content-Type: application/json" \
  -d '{"features": {"pp": [100, 150, 200], "pr": [50, 75, 60]}}'
```

### Browser Testing
- Open http://localhost:8000
- Start typing in the text area
- Verify real-time emotion detection
- Test export functionality
- Verify PWA installation

## 🔧 Troubleshooting

### Common Issues

#### Model Loading Errors
```bash
# Check model files exist
ls -la *.pkl *.joblib

# Verify model format
python -c "import joblib; print(joblib.load('xgboost.pkl'))"
```

#### Memory Issues
```bash
# Monitor memory usage
htop
# Or
ps aux | grep python

# Check Redis memory
redis-cli info memory
```

#### Performance Issues
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Check system resources
iostat 1
vmstat 1
```

#### Connection Issues
```bash
# Check port availability
netstat -tlnp | grep 8000

# Test WebSocket connection
wscat -c ws://localhost:8000/socket.io/
```

### Log Analysis
```bash
# View application logs
tail -f my_tiger.log

# Filter error logs
grep "ERROR" my_tiger.log

# Monitor real-time logs
journalctl -u my-tiger -f
```

## 📈 Scaling & High Availability

### Load Balancer Setup
```nginx
upstream my_tiger_backend {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
}

server {
    location / {
        proxy_pass http://my_tiger_backend;
    }
}
```

### Auto-scaling with Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-tiger
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-tiger
  template:
    metadata:
      labels:
        app: my-tiger
    spec:
      containers:
      - name: my-tiger
        image: my-tiger-optimized:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "64Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "500m"
```

## 🎯 Optimization Verification

### Performance Benchmarks
Run these commands to verify optimization effectiveness:

```bash
# Response time test
time curl -s http://localhost:8000/api/health > /dev/null

# Concurrent connections test
siege -c 100 -t 30s http://localhost:8000/

# Memory efficiency test
valgrind --tool=massif python app.py
```

### Success Metrics
- Response time < 100ms
- Memory usage < 50MB per worker
- 99% uptime
- Error rate < 0.1%
- Cache hit rate > 80%

## 📞 Support & Maintenance

### Backup Procedures
```bash
# Backup configuration
cp -r config/ backup/config_$(date +%Y%m%d)

# Backup logs
tar -czf logs_backup_$(date +%Y%m%d).tar.gz *.log

# Redis backup
redis-cli BGSAVE
```

### Updates & Maintenance
```bash
# Update dependencies
pip install -r requirements.txt --upgrade

# Restart service gracefully
sudo systemctl reload my-tiger

# Health check after update
curl http://localhost:8000/api/health
```

This installation guide provides everything needed to deploy and maintain the ultra-optimized My Tiger application in any environment from development to production.