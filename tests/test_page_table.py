import pytest
import hami

@pytest.fixture
def initialized_page_table():
    page_size = 16
    max_num_req = 10
    max_num_page = 1000
    page_table = hami.default_page_table()
    page_table.init(max_num_req=max_num_req, 
                   max_num_page=max_num_page, 
                   page_size=page_size)
    return page_table, max_num_req, max_num_page, page_size

def test_page_allocation(initialized_page_table):
    page_table, max_num_req, max_num_page, page_size = initialized_page_table
    
    # Test initial allocation
    a = page_table.alloc('1', num_tok=5)
    assert page_table.available_ids() == max_num_req - 1
    
    # Test page info for first allocation
    b = page_table.page_info('1')
    assert b.kv_page_indices.shape[0] == 1
    assert b.kv_last_page_len == 5
    
    # Test allocation that spans multiple pages
    a2 = page_table.alloc('2', num_tok=page_size + 5)
    b2 = page_table.page_info('2')
    assert b2.kv_page_indices.shape[0] == 2
    assert b2.kv_last_page_len == 5
    
    # Test freeing pages
    page_table.free('1')
    assert page_table.available_ids() == max_num_req - 1
    assert page_table.available_pages() == max_num_page - 2