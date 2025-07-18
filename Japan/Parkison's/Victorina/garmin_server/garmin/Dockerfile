FROM surnet/alpine-wkhtmltopdf:3.9-0.12.5-small

FROM ruby:2.5.8-alpine3.10

RUN apk update && apk add build-base nodejs postgresql-dev postgresql

RUN apk add --update imagemagick-dev imagemagick

RUN apk add --update git nano openssh python

RUN apk add --update curl libcurl curl-dev

# RUN apk add --update mesa-dri-swrast

# RUN apk add --update wkhtmltopdf

RUN apk add --no-cache \
  libstdc++ \
  libx11 \
  libxrender \
  libxext \
  libssl1.1 \
  ca-certificates \
  fontconfig \
  freetype \
  ttf-dejavu \
  ttf-droid \
  ttf-freefont \
  ttf-liberation \
  ttf-ubuntu-font-family \
  && apk add --no-cache --virtual .build-deps \
  msttcorefonts-installer \
  \
  # Install microsoft fonts
  && update-ms-fonts \
  && fc-cache -f \
  \
  # Clean up when done
  && rm -rf /tmp/* \
  && apk del .build-deps

COPY --from=0 /bin/wkhtmltopdf /usr/bin/.

RUN apk add --update yarn

RUN apk add --update npm

RUN apk add --update libressl

RUN apk add --update bash tzdata && rm -rf /var/cache/apk/*

RUN npm install -g bower

RUN mkdir /app

WORKDIR /app

COPY . /app
