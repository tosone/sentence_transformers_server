# sentence_transformers_server

## Build

``` bash
docker build --build-arg=USE_CHINA_MIRROR=true --progress plain -t sentence-transformer:dev .
```

## Server

``` bash
docker run -it --rm -p 3000:3000 sentence-transformer:dev
```
