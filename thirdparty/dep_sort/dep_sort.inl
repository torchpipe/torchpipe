namespace dep_sort {

template <typename T, template <class...> typename A>
inline bool dep_sort_result<T, A>::has_cycles() const {
  return !unsorted.empty();
}

template <
    typename T,
    template <class...>
    typename A,
    template <class...>
    typename S,
    template <class...>
    typename M>
inline bool dep_sort<T, A, S, M>::add_node(const T& node) {
  if (has_node(node)) {
    return false;
  }
  map_.insert({node, {}});
  return true;
}

template <
    typename T,
    template <class...>
    typename A,
    template <class...>
    typename S,
    template <class...>
    typename M>
inline bool dep_sort<T, A, S, M>::add_dependency(
    const T& node,
    const T& dependency) {
  if (dependency == node) {
    return false;
  }
  auto& dependents = map_[dependency].dependents_;
  if (dependents.find(node) == dependents.end()) {
    dependents.insert(node);
    ++map_[node].dependencies_;
    return true;
  }
  return false;
}

template <
    typename T,
    template <class...>
    typename A,
    template <class...>
    typename S,
    template <class...>
    typename M>
template <template <class...> typename C>
inline bool dep_sort<T, A, S, M>::add_dependencies(
    const T& node,
    const C<T>& dependencies) {
  for (const auto& dependency : dependencies) {
    if (!add_dependency(node, dependency)) {
      return false;
    }
  }
  return true;
}

template <
    typename T,
    template <class...>
    typename A,
    template <class...>
    typename S,
    template <class...>
    typename M>
inline void dep_sort<T, A, S, M>::clear() {
  map_.clear();
}

template <
    typename T,
    template <class...>
    typename A,
    template <class...>
    typename S,
    template <class...>
    typename M>
inline typename dep_sort<T, A, S, M>::result_type dep_sort<T, A, S, M>::sort()
    const {
  dep_sort<T, A, S, M> copy(*this);
  dep_sort_result<T, A> result;
  copy.sort(result);
  return result;
}

template <
    typename T,
    template <class...>
    typename A,
    template <class...>
    typename S,
    template <class...>
    typename M>
inline bool dep_sort<T, A, S, M>::has_node(const value_type& node) const {
  return map_.find(node) != map_.end();
}

template <
    typename T,
    template <class...>
    typename A,
    template <class...>
    typename S,
    template <class...>
    typename M>
inline bool dep_sort<T, A, S, M>::has_dependency(
    const value_type& node,
    const value_type& dependency) const {
  if (has_node(node)) {
    const auto& dependencies = map_[node];
    return dependencies.find(dependency) != dependencies.end();
  }
  return false;
}

template <
    typename T,
    template <class...>
    typename A,
    template <class...>
    typename S,
    template <class...>
    typename M>
inline void dep_sort<T, A, S, M>::sort(result_type& result) {
  auto& sorted = result.sorted;
  auto& unsorted = result.unsorted;
  for (const auto& node_relations : map_) {
    const auto& node = node_relations.first;
    const auto& relations = node_relations.second;
    if (relations.dependencies_ == 0) {
      sorted.push_back(node);
    }
  }
  for (detail::dep_size_type i = 0; i < sorted.size(); i++) {
    for (const auto& node : map_[sorted[i]].dependents_) {
      if (--map_[node].dependencies_ == 0) {
        sorted.push_back(node);
      }
    }
  }
  for (const auto& node_relations : map_) {
    const auto& node = node_relations.first;
    const auto& relations = node_relations.second;
    if (relations.dependencies_ != 0) {
      unsorted.push_back(node);
    }
  }
}

} // namespace dep_sort