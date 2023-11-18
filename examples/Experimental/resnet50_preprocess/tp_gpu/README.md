


```
apt-get install thrift-compiler -y
pip install thrift

thrift -r -out ./ --gen py serve.thrift

python3 start.py

python client.py
```