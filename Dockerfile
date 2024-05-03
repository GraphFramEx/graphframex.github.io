FROM nginx:alpine

COPY ./nginx.conf /etc/nginx/conf.d/default.conf
COPY ./root /usr/share/nginx/html/

RUN ls -la /usr/share/nginx/html/
