curl http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "llama2",
"prompt": "San Francisco is a",
"max_tokens": 7,
"temperature": 0,
"stream": true
}'
