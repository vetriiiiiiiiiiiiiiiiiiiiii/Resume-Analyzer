#!/bin/bash
docker build -t resume-analyzer .
docker run -p 5000:5000 resume-analyzer
