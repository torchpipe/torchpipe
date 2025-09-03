import hami

def test_dag():
    # todo
    config = {
        'a':{'next':"b,c"},
        'b': {'next': "d", "map": "a[result:data,color:color]"},
        'c':  {'next': "d"},
        'd': {'map':"c[result:1],b[result:data,color:color]"}   
    }
    model = hami.pipe(config)
    inp = {'data':1,'color':'rgb'}
    model(inp)
    print(inp)


def test_dag_com():
        
    model = hami.pipe("config/dag_com.toml")
    inp = {'data':1,'color':'rgb'}
    model(inp)
    print(inp)
if __name__ == "__main__":
    hami.init("DebugLogger")
    test_dag()
