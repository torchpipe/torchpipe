# Dependency sorting

Generic topological sorting for sorting a list of dependencies in C++17

# Use
There are two interfaces to choose from: `dep_sort_stl` and `dep_sort`.

The first will use `std::vector`, `std::unordered_set` and `std::unordered_map`
to implement a highly efficent `O(V+E)` dependency resolver container.

There's also however a `dep_sort` class which lets you supplement your
own `vector`, `set` and `map` implementions. You can use drop in
replacements like Google's `densehash` or `sparsehash` and stuff like
Boost's `multimap` if working with highly dense or sparse data.

As a result this code has no real dependencies other than a working C++17
compiler.

# Example
```cpp
int main() {
  dep_sort_stl<std::string> dep;
  dep.add_node("a");
  dep.add_node("b");
  dep.add_node("c");
  dep.add_node("d");
  std::vector<std::string> as = { "b", "c" };
  std::vector<std::string> bs = { "c" };
  std::vector<std::string> cs = { "d" };
  dep.add_dependencies("a", as);
  dep.add_dependencies("b", bs);
  dep.add_dependencies("c", cs);
  const auto& result = dep.sort();
  if (!result.has_cycles()) {
    // print the sorted list
    for (const auto& value : result.sorted) {
      std::cout << value << std::endl;
    }
  } else {
    // print nodes that could not be sorted due to cycles
    for (const auto& value : result.unsorted) {
      std::cout << value << std::endl;
    }
  }
}
```

# Documentation

The interfaces are documented in `dep_sort.h`

----
# Modify

### modified from https://github.com/graphitemaster/dep_sort with:
- c++14 support
- additional namespace