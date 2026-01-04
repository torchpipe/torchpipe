import pytest
import omniback


def create_initialized_page_table():
    """Regular helper function to initialize page table (not a fixture)."""
    page_size = 16
    max_num_req = 10
    max_num_page = 1000
    page_table = omniback._C.default_page_table("")
    assert page_table is not None
    page_table.init(max_num_req,
                    max_num_page,
                    page_size)
    return page_table, max_num_req, max_num_page, page_size


@pytest.fixture
def initialized_page_table():
    return create_initialized_page_table()


def test_page_allocation(initialized_page_table):
    page_table, max_num_req, max_num_page, page_size = initialized_page_table

    # Test initial allocation
    a = page_table.alloc('1', 5)
    assert page_table.available_ids() == max_num_req - 1

    # Test page info for first allocation
    b = page_table.page_info('1')
    assert len(b.kv_page_indices) == 1
    assert b.kv_last_page_len == 5

    # Test allocation that spans multiple pages
    a2 = page_table.alloc('2', page_size + 5)
    b2 = page_table.page_info('2')
    assert len(b2.kv_page_indices) == 2
    assert b2.kv_last_page_len == 5

    # Test freeing pages
    page_table.free('1')
    assert page_table.available_ids() == max_num_req - 1
    assert page_table.available_pages() == max_num_page - 2


if __name__ == "__main__":
    # Run the test manually using the helper (not the fixture)
    pt, max_req, max_page, psize = create_initialized_page_table()
    test_page_allocation((pt, max_req, max_page, psize))
