FROM nginx:1.28-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY index.html app.js styles.css favicon.svg /usr/share/nginx/html/
COPY assets /usr/share/nginx/html/assets
EXPOSE 80
