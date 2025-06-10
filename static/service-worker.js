const CACHE_NAME = 'neural-style-transfer-cache-v1';
const urlsToCache = [
    '/',
    '/static/favicon.svg',
    '/static/manifest.json',
    '/templates/balba.html',
    // Add other static assets like CSS, JS, images if they are served directly
];

self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => {
                console.log('Opened cache');
                return cache.addAll(urlsToCache);
            })
    );
});

self.addEventListener('fetch', event => {
    event.respondWith(
        caches.match(event.request)
            .then(response => {
                // Cache hit - return response
                if (response) {
                    return response;
                }
                return fetch(event.request);
            })
    );
});