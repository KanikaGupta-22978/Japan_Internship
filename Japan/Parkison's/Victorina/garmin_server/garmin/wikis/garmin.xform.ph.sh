# for redirecting http traffic to https version of the site
server {
  listen 80;
  server_name garmin.xform.ph;
  return 301 https://$server_name$request_uri;
}

# for redirecting to non-www version of the site
server {
  listen 80;
  server_name www.garmin.xform.ph;
  return 301 https://garmin.xform.ph;
}

server {
  listen 443 ssl;
  server_name garmin.xform.ph;

  location / {
    passenger_enabled   on;
    rails_env           production;
    root                /home/ubuntu/garmin/current/public;
  }

  ssl on;
  ssl_certificate /etc/letsencrypt/live/garmin.xform.ph/fullchain.pem; # managed by Certbot
  ssl_certificate_key /etc/letsencrypt/live/garmin.xform.ph/privkey.pem; # managed by Certbot
  include /etc/letsencrypt/options-ssl-nginx.conf; # managed by Certbot
  ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem; # managed by Certbot

  # ssl_session_timeout  5m;
  # ssl_protocols  SSLv2 SSLv3 TLSv1;
  # ssl_ciphers  HIGH:!aNULL:!MD5;
  # ssl_prefer_server_ciphers   on;

  error_page 500 502 503 504 /500.html;
  client_max_body_size 4G;
  keepalive_timeout 10;
}
