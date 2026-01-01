import omniback


model = omniback.init("ParserTest")

TESTS = {"A(b,c=1)[D[E[f,g]]]": ['A', {'E::dependency': 'f,g', 'D::dependency': 'E', 'A::dependency': 'D', 'A::args': 'b,c=1'}],
        "A[C;D]": ['A', {'A::dependency': 'C;D'}],
        'A[(d,d)C]':['A', {'A::dependency': '(d,d)C'}],
        'A[(xx,de3[])B(arg)]':['A', {'A::dependency': '(xx,de3[])B(arg)'}],
        'A(a)[B(xx,de3{})[Z]]':['A', {'A::dependency': 'B', 'B::args':'xx,de3{}', 'B::dependency':'Z', 'A::args':'a'}],
        }

# TESTS ={  'A[(xx,de3[])B(arg)]':['A', {'A::dependency': '(xx,de3[])B(arg)'}],
#         }
def test_parsers():
    for config, result in TESTS.items():
        data = {'data':config}
        model(data)
        print( (data['result']))
        # for i in range(len(result)):
        assert data['result'][0] == result[0], f"{data['result'][0]} != {result[0]}"
        assert dict(data['result'][1]) == result[1], f"{data['result'][1]} != {result[1]}"

def test_container():
    TESTS = { "(ax,ay=2)X(x1,x2=3)[B(z)],Y(yx=1)[BB(zz=2)]" : ['ax,ay=2', 'X(x1,x2=3)[B(z)]', '', 'Y(yx=1)[BB(zz=2)]'],
    "(ax[3;3,d()],ay(d,d)=2)X(x1,x2=3)[B(z)],Y(],;;[yx=1)[BB()(zz=2)]" : ['ax[3;3,d()],ay(d,d)=2', 'X(x1,x2=3)[B(z)]', '', 'Y(],;;[yx=1)[BB()(zz=2)]']  }
    for config, result in TESTS.items():
        data = {'data':config}
        model(data)
        # print(data)
        # print(config, result, type(data['result']))
        assert list(data['result']) == result, f"{data['result'][0]} != {result[0]}"
if __name__ == "__main__":
    import time 
    # time.sleep(5)

    test_parsers()


