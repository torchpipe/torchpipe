#ifndef DEP_SORT_H
#define DEP_SORT_H

#include <unordered_set>

namespace dep_sort {
namespace detail {
typedef decltype(sizeof 0) dep_size_type;
template <typename T, template <class...> typename S>
struct dep_relation {
  dep_size_type dependencies_;
  S<T> dependents_;
};
} // namespace detail

template <typename T, template <class...> typename A>
struct dep_sort_result {
  typedef T value_type;
  typedef A<value_type> array_type;

  // check if sort contains any cycles, returns true if |unsorted| contains
  // any elements
  bool has_cycles() const;

  // list of sorted nodes that are in dependency order
  array_type sorted;

  // list of unsorted nodes that could not be sorted due to cycles
  array_type unsorted;
};

template <
    typename T,
    template <class...>
    typename A,
    template <class...>
    typename S,
    template <class...>
    typename M>
struct dep_sort {
  typedef T value_type;
  typedef dep_sort_result<value_type, A> result_type;

  // add |node| to sort, returns false if |node| already exists
  bool add_node(const value_type& node);

  // add |dependency| to |node|, returns false if |node| and |dependency|
  // are the same or if |dependency| is already a dependency of |node|
  bool add_dependency(const value_type& node, const value_type& dependency);

  // add |dependencies| to |node|, returns false if |node| is the same
  // as any of the dependencies in |dependencies| or if any of the
  // dependencies are already a dependencey of |node|
  template <template <class...> typename C>
  bool add_dependencies(
      const value_type& node,
      const C<value_type>& dependencies);

  // clear all nodes and dependencies
  void clear();

  // sort the nodes and their dependencies
  result_type sort() const;

  // check if |node| exists, returns false if |node| does not exist
  bool has_node(const value_type& node) const;

  // check if |dependency| exists for |node|, returns false if |node| or
  // |dependency| do not exist
  bool has_dependency(const value_type& node, const value_type& dependency)
      const;

 private:
  void sort(result_type& result);

  typedef M<value_type, detail::dep_relation<value_type, S>> map_type;
  map_type map_;
};
} // namespace dep_sort
#include "dep_sort.inl"

#endif
