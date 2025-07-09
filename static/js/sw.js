// My Tiger - Service Worker for PWA Support
const CACHE_NAME = 'my-tiger-v1.0.0';
const STATIC_CACHE = 'my-tiger-static-v1.0.0';
const DYNAMIC_CACHE = 'my-tiger-dynamic-v1.0.0';

// Assets to cache for offline use
const STATIC_ASSETS = [
  '/',
  '/static/css/style.css',
  '/static/js/app.js',
  '/static/images/mytiger.png',
  'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css',
  'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap',
  'https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.4/socket.io.js'
];

// Install event - cache static assets
self.addEventListener('install', event => {
  console.log('🚀 Service Worker: Installing...');
  
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then(cache => {
        console.log('📦 Service Worker: Caching static assets');
        return cache.addAll(STATIC_ASSETS);
      })
      .then(() => {
        console.log('✅ Service Worker: Static assets cached');
        return self.skipWaiting();
      })
      .catch(error => {
        console.error('❌ Service Worker: Failed to cache static assets', error);
      })
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
  console.log('🔄 Service Worker: Activating...');
  
  event.waitUntil(
    caches.keys()
      .then(cacheNames => {
        return Promise.all(
          cacheNames.map(cacheName => {
            if (cacheName !== STATIC_CACHE && cacheName !== DYNAMIC_CACHE) {
              console.log('🗑️ Service Worker: Deleting old cache', cacheName);
              return caches.delete(cacheName);
            }
          })
        );
      })
      .then(() => {
        console.log('✅ Service Worker: Activated');
        return self.clients.claim();
      })
  );
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Skip non-GET requests
  if (request.method !== 'GET') {
    return;
  }
  
  // Skip socket.io requests
  if (url.pathname.includes('socket.io')) {
    return;
  }
  
  // Handle different types of requests
  if (STATIC_ASSETS.some(asset => request.url.includes(asset))) {
    // Static assets - cache first strategy
    event.respondWith(cacheFirst(request));
  } else if (request.url.includes('/api/')) {
    // API requests - network first strategy
    event.respondWith(networkFirst(request));
  } else {
    // Other requests - stale while revalidate
    event.respondWith(staleWhileRevalidate(request));
  }
});

// Cache strategies
async function cacheFirst(request) {
  try {
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    const networkResponse = await fetch(request);
    const cache = await caches.open(STATIC_CACHE);
    cache.put(request, networkResponse.clone());
    
    return networkResponse;
  } catch (error) {
    console.error('Cache first strategy failed:', error);
    return new Response('Offline - Asset not available', { status: 503 });
  }
}

async function networkFirst(request) {
  try {
    const networkResponse = await fetch(request);
    
    if (networkResponse.ok) {
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    console.log('Network failed, trying cache:', error.message);
    const cachedResponse = await caches.match(request);
    
    if (cachedResponse) {
      return cachedResponse;
    }
    
    return new Response('Offline - Service not available', {
      status: 503,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: 'Service temporarily unavailable' })
    });
  }
}

async function staleWhileRevalidate(request) {
  const cache = await caches.open(DYNAMIC_CACHE);
  const cachedResponse = await cache.match(request);
  
  const fetchPromise = fetch(request).then(networkResponse => {
    if (networkResponse.ok) {
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  }).catch(() => cachedResponse);
  
  return cachedResponse || fetchPromise;
}

// Background sync for emotion data
self.addEventListener('sync', event => {
  if (event.tag === 'background-sync-emotions') {
    console.log('🔄 Background sync: Processing queued emotion data');
    event.waitUntil(syncEmotionData());
  }
});

async function syncEmotionData() {
  try {
    // Retrieve queued data from IndexedDB
    const queuedData = await getQueuedEmotionData();
    
    if (queuedData.length > 0) {
      console.log(`📤 Syncing ${queuedData.length} queued emotion records`);
      
      for (const data of queuedData) {
        try {
          await fetch('/api/emotion-data', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
          });
          
          // Remove from queue after successful sync
          await removeQueuedData(data.id);
        } catch (error) {
          console.error('Failed to sync emotion data:', error);
        }
      }
    }
  } catch (error) {
    console.error('Background sync failed:', error);
  }
}

// IndexedDB helpers for offline data storage
async function getQueuedEmotionData() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('MyTigerDB', 1);
    
    request.onerror = () => reject(request.error);
    request.onsuccess = () => {
      const db = request.result;
      const transaction = db.transaction(['emotionQueue'], 'readonly');
      const store = transaction.objectStore('emotionQueue');
      const getAll = store.getAll();
      
      getAll.onsuccess = () => resolve(getAll.result);
      getAll.onerror = () => reject(getAll.error);
    };
    
    request.onupgradeneeded = event => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains('emotionQueue')) {
        db.createObjectStore('emotionQueue', { keyPath: 'id' });
      }
    };
  });
}

async function removeQueuedData(id) {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('MyTigerDB', 1);
    
    request.onsuccess = () => {
      const db = request.result;
      const transaction = db.transaction(['emotionQueue'], 'readwrite');
      const store = transaction.objectStore('emotionQueue');
      const deleteRequest = store.delete(id);
      
      deleteRequest.onsuccess = () => resolve();
      deleteRequest.onerror = () => reject(deleteRequest.error);
    };
  });
}

// Push notification support
self.addEventListener('push', event => {
  if (event.data) {
    const data = event.data.json();
    
    const options = {
      body: data.body || 'My Tiger has detected an interesting emotion pattern!',
      icon: '/static/images/mytiger.png',
      badge: '/static/images/badge.png',
      vibrate: [200, 100, 200],
      data: data.data || {},
      actions: [
        {
          action: 'view',
          title: 'View Analysis',
          icon: '/static/images/view-icon.png'
        },
        {
          action: 'dismiss',
          title: 'Dismiss',
          icon: '/static/images/dismiss-icon.png'
        }
      ]
    };
    
    event.waitUntil(
      self.registration.showNotification(data.title || 'My Tiger', options)
    );
  }
});

// Handle notification clicks
self.addEventListener('notificationclick', event => {
  event.notification.close();
  
  if (event.action === 'view') {
    event.waitUntil(
      clients.openWindow('/')
    );
  }
});

// Periodic background sync for data cleanup
self.addEventListener('periodicsync', event => {
  if (event.tag === 'data-cleanup') {
    event.waitUntil(cleanupOldData());
  }
});

async function cleanupOldData() {
  try {
    console.log('🧹 Cleaning up old cached data');
    
    // Clean up old dynamic cache entries
    const cache = await caches.open(DYNAMIC_CACHE);
    const requests = await cache.keys();
    const now = Date.now();
    const maxAge = 7 * 24 * 60 * 60 * 1000; // 7 days
    
    for (const request of requests) {
      const response = await cache.match(request);
      if (response) {
        const cachedTime = response.headers.get('cached-time');
        if (cachedTime && (now - parseInt(cachedTime)) > maxAge) {
          await cache.delete(request);
        }
      }
    }
    
    console.log('✅ Cache cleanup completed');
  } catch (error) {
    console.error('Cache cleanup failed:', error);
  }
}

// Share target support
self.addEventListener('share', event => {
  event.waitUntil(
    handleShare(event)
  );
});

async function handleShare(event) {
  const { url, title, text } = event.data;
  
  // Handle shared emotion analysis data
  if (text && text.includes('emotion')) {
    await clients.openWindow(`/?shared=${encodeURIComponent(text)}`);
  } else {
    await clients.openWindow('/');
  }
}

console.log('🐅 My Tiger Service Worker: Loaded successfully');