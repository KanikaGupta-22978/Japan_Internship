Deployed on Aug 28, 2020 11:57 AM

HISTTIMEFORMAT="%F %T "

sudo apt-get update
sudo apt-get upgrade -y
  Note: Choose "keep the local version currently installed"

sudo apt-get install -y build-essential git zlib1g-dev libssl-dev libreadline-dev libyaml-dev libsqlite3-dev sqlite3 libxml2-dev libxslt1-dev libcurl4-openssl-dev libffi-dev libgdbm-dev libncurses5-dev automake libtool bison nodejs openssl htop ncdu curl software-properties-common yarn graphicsmagick gnupg2

# gpg --keyserver hkp://keys.gnupg.net --recv-keys 409B6B1796C275462A1703113804BB82D39DC0E3 7D2BAF1CF37B13E2069D6956105BD0E739499BDB
gpg2 --keyserver keyserver.ubuntu.com --recv-keys 409B6B1796C275462A1703113804BB82D39DC0E3 7D2BAF1CF37B13E2069D6956105BD0E739499BDB
curl -sSL https://get.rvm.io | bash -s stable
source ~/.rvm/scripts/rvm
rvm install 2.5.8
rvm use 2.5.8 --default

# NGINX & PASSENGER

sudo apt-get install -y dirmngr # gnupg
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 561F9B9CAC40B2F7
sudo apt-get install -y apt-transport-https ca-certificates

sudo sh -c 'echo deb https://oss-binaries.phusionpassenger.com/apt/passenger bionic main > /etc/apt/sources.list.d/passenger.list'
sudo apt-get update -y
sudo apt-get install -y nginx-extras libnginx-mod-http-passenger

if [ ! -f /etc/nginx/modules-enabled/50-mod-http-passenger.conf ]; then sudo ln -s /usr/share/nginx/modules-available/mod-http-passenger.load /etc/nginx/modules-enabled/50-mod-http-passenger.conf ; fi

# GIT

git config --global color.ui true
git config --global user.name "Garmin Server"
git config --global user.email "jnoelvictorino@gmail.com"

# In local machine

scp ~/.ssh/githubjnoelvictorino ubuntu@ec2-52-44-200-52.compute-1.amazonaws.com:/home/ubuntu/.ssh/.
# or
ssh-copy-id -i ~/.ssh/githubjnoelvictorino ubuntu@ec2-52-44-200-52.compute-1.amazonaws.com

# Back in remote server

sudo tee ~/.ssh/config << END
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/githubjnoelvictorino
END

# Needed for Capistrano
# chmod 700 ~/.ssh
# chmod 600 ~/.ssh/authorized_keys
# chmod 600 ~/.ssh/githubjnoelvictorino

# POSTGRESQL

sudo apt-get install -y postgresql postgresql-contrib libpq-dev

sudo -u postgres createuser -P garmin
# PW: garmin
sudo -u postgres createdb -O garmin garmin_production

# POSTGIS
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt/ $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'

wget --quiet -O - http://apt.postgresql.org/pub/repos/apt/ACCC4CF8.asc | sudo apt-key add -

sudo apt-get update -y

sudo apt-get install -y postgresql-10 postgresql-10-postgis-2.4 postgresql-10-postgis-2.4-scripts postgis

sudo -u postgres psql -c "CREATE EXTENSION postgis; CREATE EXTENSION postgis_topology;" garmin_production

sudo -u postgres psql -c "ALTER USER garmin WITH SUPERUSER;"

sudo sed -i 's/local   all             all                                     peer/local   all             all                                     md5/' /etc/postgresql/10/main/pg_hba.conf

# YARN
curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list
sudo apt-get -y update && sudo apt-get install -y --no-install-recommends yarn

sudo apt-get update
sudo apt-get upgrade -y
sudo reboot

# SSL
# Source: https://www.digitalocean.com/community/tutorials/how-to-secure-nginx-with-let-s-encrypt-on-ubuntu-18-04

sudo add-apt-repository ppa:certbot/certbot
sudo apt install -y python-certbot-nginx

# PRODUCTION GARMIN

sudo certbot --nginx -d garmin.xform.ph
# Enter email: jnoelvictorino@gmail.com
# Agree: A
# Email Subscription: N
# Choose 1 / 2: 2

sudo certbot renew --dry-run

sudo sed -i '93,$d' /etc/nginx/sites-enabled/default

sudo tee /etc/nginx/sites-available/garmin-production << END
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
END

sudo ln -s /etc/nginx/sites-available/garmin-production /etc/nginx/sites-enabled/.

sudo nginx -t
# Should be no errors
# Make sure HTTPS is allowed for AWS Security Group

sudo systemctl reload nginx

# FastAPI
sudo certbot --nginx -d garmin-fastapi.xform.ph
# Choose 1 / 2: 2

sudo certbot renew --dry-run

sudo sed -i '93,$d' /etc/nginx/sites-enabled/default

sudo tee /etc/nginx/sites-available/garmin-fastapi.xform.ph << END
# for redirecting http traffic to https version of the site
server {
  listen 80;
  server_name garmin-fastapi.xform.ph;
  return 301 https://$server_name$request_uri;
}

# for redirecting to non-www version of the site
server {
  listen 80;
  server_name www.garmin-fastapi.xform.ph;
  return 301 https://garmin-fastapi.xform.ph;
}

server {
  listen 443 ssl;
  server_name garmin-fastapi.xform.ph;

  location / {
    include proxy_params;
    proxy_pass http://127.0.0.1:3002;
  }

  ssl on;
  ssl_certificate /etc/letsencrypt/live/garmin-fastapi.xform.ph/fullchain.pem; # managed by Certbot
  ssl_certificate_key /etc/letsencrypt/live/garmin-fastapi.xform.ph/privkey.pem; # managed by Certbot
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
END

sudo ln -s /etc/nginx/sites-available/garmin-fastapi.xform.ph /etc/nginx/sites-enabled/.

sudo nginx -t
# Should be no errors
# Make sure HTTPS is allowed for AWS Security Group

sudo systemctl reload nginx



# In host machine
# Source: https://gorails.com/deploy/ubuntu/18.04#capistrano
# Source: http://waiyanyoon.com/deploying-rails-5-2-applications-with-encrypted-credentials-using-capistrano/

# Make sure to branch out feature/model_inference

docker-compose run --rm web bash
cap production deploy branch=feature/model_inference

# When error shows up, go back to remote server
tee /home/ubuntu/garmin/shared/config/database.yml << END
default: &default
  adapter: postgresql
  encoding: unicode
  host: localhost
  username: <%= ENV['POSTGRES_USER'] %>
  password: <%= ENV['POSTGRES_PASSWORD'] %>
  pool: <%= ENV.fetch("RAILS_MAX_THREADS") { 5 } %>

development:
  <<: *default
  database: <%= "#{ENV['DATABASE_NAME']}_development" %>

test:
  <<: *default
  database: <%= "#{ENV['DATABASE_NAME']}_test" %>

production:
  <<: *default
  database: garmin_production
  host: localhost
  username: garmin
  password: garmin
END

tee /home/ubuntu/garmin/shared/.env.production << END
# Prefix for docker images, containers, volumes and networks
INSTANCE=garmin

# Prefix for docker images, containers, volumes and networks
COMPOSE_PROJECT_NAME=garmin

# The database name will automatically get the Rails environment appended to it
DATABASE_NAME=garmin_production

# Required by the Postgres Docker image. This sets up the initial database when
# you first run it.
POSTGRES_USER=garmin
POSTGRES_PASSWORD=garmin

# What Rails environment are we in?
RAILS_ENV=production

GARMIN_KEY=5cc89373-0a5d-485f-86bb-53b34928c392
GARMIN_SECRET=aucCKoqKdvrtVQzVRnFQpD3aXsEfUhHFnYH
GARMIN_CALLBACK_URL=http://ec2-52-44-200-52.compute-1.amazonaws.com/users/auth/garmin/callback
GARMIN_URL=https://healthapi.garmin.com
END

tee /home/ubuntu/garmin/shared/config/master.key << END
3a62142cb3ae574850d013f6a9b67e25
END

tee /home/ubuntu/garmin/shared/config/sidekiq.yml << END
verbose: true
concurrency: 25
queues:
  - [mailers, 7]
  - [default, 5]
END

# Go back in host machine
cap production deploy branch=master

# Go back to remote server
cd /home/ubuntu/garmin/current
bundle exec rails db:seed RAILS_ENV=production

cd /old_garmin/home/ubuntu/backups/11042023_1718
/usr/lib/postgresql/12/bin/pg_restore -h localhost -U garmin -d garmin_production garmin_production_11042023_1718.tar

# Docker
# Reference: https://docs.docker.com/engine/install/ubuntu/

sudo apt autoremove

for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done

sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo docker run hello-world

sudo groupadd docker

sudo usermod -aG docker $USER

newgrp docker

docker run hello-world

sudo systemctl disable docker.service
sudo systemctl disable containerd.service