@echo off
set TAG=%1
if "%TAG%"=="" set TAG=my-app

set PLATFORM=%2
if "%PLATFORM%"=="" set PLATFORM=linux/amd64

set DOCKER_DEFAULT_PLATFORM=%PLATFORM%
docker build --platform %PLATFORM% -t %TAG% .
