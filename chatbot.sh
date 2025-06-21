#!/bin/bash

API_ENDPOINT="http://localhost:8000/chat"

if ! curl -s "$API_ENDPOINT" > /dev/null 2>&1; then
    nohup python run_server.py > server.log 2>&1 &
    while true; do
        if curl -s "http://localhost:8000/health" > /dev/null 2>&1; then
            break
        fi
        sleep 2
    done
fi

nohup python run_gradio.py > gradio.log 2>&1 &
# http://localhost:7860

jq -c '.[]' data/test_realtime.json | while read -r item; do
  question=$(echo "$item" | jq -r '.user')

  payload=$(jq -n --arg q "$question" '{"question": $q}')

  api_response=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "$payload" \
    "$API_ENDPOINT")

  model_response=$(echo "$api_response" | jq -r '.answer // "응답 없음"')

  printf '{ "user": %s, "model": %s }\n' \
    "$(echo "$question" | jq -sRr @json)" \
    "$(echo "$model_response" | jq -sRr @json)"
done | jq -s '.' > outputs/realtime_output.json