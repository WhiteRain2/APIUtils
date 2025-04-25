import pytest
from apiutils import API  # Assuming your module is named api_module.py

# Setup for tests, possibly defining some standard APIs for testing
@pytest.fixture(scope="module")
def standard_apis():
    API.set_standard_apis(['module.class.method', 'module.class.method(arg1, arg2)'], ['Description for method', None])
    return API.get_standard_apis()

# Test instantiation of API
def test_api_instantiation():
    api_str = "module.class.method(arg1, arg2)"
    api = API(api_str)
    assert api.fullname == "module.class.method"
    assert api.args == ["arg1", "arg2"]
    assert api.method == "method"
    assert api.prefix == "module.class"
    assert str(api) == "module.class.method(arg1, arg2)"

# Test standard API check
def test_is_standard(standard_apis):
    api = API("module.class.method")
    assert api.is_standard is True
    api_with_args = API("module.class.method(arg1, arg2)")
    assert api_with_args.is_standard is True

# Test fetching possible standard APIs
def test_get_possible_standard_apis(standard_apis):
    api = API("module.class.method(arg3)")
    possible_apis = api.get_possible_standard_apis(matched_ps=2)
    assert len(possible_apis) == 1
    assert possible_apis[0].fullname == "module.class.method"

# Test error handling for invalid API strings
def test_invalid_api():
    with pytest.raises(ValueError):
        API("this is not a valid api string")

# Test the equality and hash functions
def test_api_equality_and_hash():
    api1 = API("module.class.method(arg1)")
    api2 = API("module.class.method(arg1)")
    assert api1 == api2
    assert hash(api1) == hash(api2)
    api3 = API("module.class.method(arg2)")
    assert api1 != api3

# Running the tests
# You can run these tests using the following command in your terminal:
# pytest test_api.py
