import omniback


def test_AtomicInt():
    # atomic_int = omniback.ffi.AtomicInt()

    # # Support += any value
    # atomic_int += 5  # calls __iadd__
    # assert (atomic_int.get() == 5)  # output: 15

    # assert (atomic_int.increment() == 6)
    # assert atomic_int.get() == 6
    print(omniback.__version__)


if __name__ == "__main__":
    test_AtomicInt()
