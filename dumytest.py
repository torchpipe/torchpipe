import hami
from typing import Optional, List, Any
print(hami.__version__)




a=hami._C.dict({"1":"2"})
print(a)
print(a["1"])
b=a.pop("1")
print("1" in a)
# print(a["1"])

print(f"b={b}")
