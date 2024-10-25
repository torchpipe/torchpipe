curl http://0.0.0.0:8080/v1/completions \
-H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "string",
  "prompt": "San Francisco is a",
  "priority": "default",
  "n": 1,
  "best_of": 0,
  "max_tokens": 7,
  "stream": true,
  "stream_options": {
    "include_usage": true
  },
  "logprobs": 0,
  "echo": false,
  "temperature": 0.0,
  "presence_penalty": 0,
  "frequency_penalty": 0,
  "repetition_penalty": 1,
  "top_p": 1,
  "top_k": -1,
  "skip_special_tokens": true,
  "ignore_eos": false,
  "stop": "string",
  "stop_token_ids": [
    0
  ]
}'