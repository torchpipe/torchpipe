
tensor([ 1.9443,  0.3882, -0.2551,  0.1792,  2.8262])

cp modeling_opt.py /usr/local/lib/python3.10/dist-packages/transformers/models/opt/modeling_opt.py 

cp  /usr/local/lib/python3.10/dist-packages/transformers/models/opt/modeling_opt.py .
```
vllm serve facebook/opt-125m

client.sh

data: {"id":"cmpl-d71dd12e4a614a40bb5a62075d24104e","object":"text_completion","created":1729747345,"model":"facebook/opt-125m","choices":[{"index":0,"text":" great","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}

data: {"id":"cmpl-d71dd12e4a614a40bb5a62075d24104e","object":"text_completion","created":1729747345,"model":"facebook/opt-125m","choices":[{"index":0,"text":" place","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}

data: {"id":"cmpl-d71dd12e4a614a40bb5a62075d24104e","object":"text_completion","created":1729747345,"model":"facebook/opt-125m","choices":[{"index":0,"text":" to","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}

data: {"id":"cmpl-d71dd12e4a614a40bb5a62075d24104e","object":"text_completion","created":1729747345,"model":"facebook/opt-125m","choices":[{"index":0,"text":" live","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}

data: {"id":"cmpl-d71dd12e4a614a40bb5a62075d24104e","object":"text_completion","created":1729747345,"model":"facebook/opt-125m","choices":[{"index":0,"text":".","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}

data: {"id":"cmpl-d71dd12e4a614a40bb5a62075d24104e","object":"text_completion","created":1729747345,"model":"facebook/opt-125m","choices":[{"index":0,"text":" ","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}

data: {"id":"cmpl-d71dd12e4a614a40bb5a62075d24104e","object":"text_completion","created":1729747345,"model":"facebook/opt-125m","choices":[{"index":0,"text":" I","logprobs":null,"finish_reason":"length","stop_reason":null}],"usage":null}

data: [DONE]

 python openai_api_server.py 

<!-- pip install -U scalellm -i https://whl.vectorch.com/cu118/torch2.4.0/ -->


 
  python openai_server_api.py 

curl http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "facebook/opt-125m",
"prompt": "San Francisco is a",
"max_tokens": 7,
"temperature": 0
}'

```