import omniback

# 初始化

def test_AtomicInt():
    # atomic_int = omniback.ffi.AtomicInt()

    # # 支持 += 任意值
    # atomic_int += 5  # 调用 __iadd__
    # assert (atomic_int.get() == 5)  # 输出: 15


    # assert (atomic_int.increment() == 6)
    # assert atomic_int.get() == 6
    print(omniback.__version__)
if __name__ == "__main__":
    test_AtomicInt()