import omniback
import os
import pytest


def test_dag():
    config = {
        'adag': {'next': "b,cdag"},
        'b': {'next': "ddag", "map": "adag[result:data,color:color]"},
        'cdag':  {'next': "ddag"},
        'ddag': {'map': "cdag[result:1],b[result:data,color:color]"}
    }
    model = omniback.pipe(config)
    inp = {'data': 1, 'color': 'rgb'}
    model(inp)
    print(inp)


def test_dag_com():
    model = omniback.pipe(os.path.join(os.path.dirname(__file__), "config/dag_com.toml"))
    inp = {'data': 1, 'color': 'rgb'}
    model(inp)
    print(inp)


if __name__ == "__main__":
    omniback.init("DebugLogger")
    test_dag_com()
