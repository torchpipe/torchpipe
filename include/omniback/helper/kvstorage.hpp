#include <unordered_map>
#include <shared_mutex>
#include <mutex>
#include <any>
#include <optional>
#include <stdexcept>
#include <memory>

template <typename GroupKey = std::string,
          typename EntryKey = std::string,
          typename Value = std::any>
class ConcurrentKVStore {
 private:
  struct EntryGroup {
    std::unique_ptr<std::unordered_map<EntryKey, Value>> entries;
    mutable std::mutex entry_mutex;
  };

  using StoreMap = std::unordered_map<GroupKey, EntryGroup>;
  StoreMap store_;
  mutable std::shared_mutex store_mutex_;

 public:
  ConcurrentKVStore() = default;

  /**
   * @brief Insert or update an entry in an existing request group
   * @throws std::runtime_error if request group doesn't exist
   */
  void insert_or_assign(const GroupKey& group_id,
                        const EntryKey& key,
                        const Value& value) {
    std::shared_lock read_lock(store_mutex_);
    auto group_it = store_.find(group_id);
    if (group_it == store_.end()) {
      throw std::runtime_error("Request group not found: " + group_id);
    }

    EntryGroup& group = group_it->second;
    std::scoped_lock group_lock(group.entry_mutex);
    read_lock.unlock();
    if (!group.entries) {
      group.entries = std::make_unique<std::unordered_map<EntryKey, Value>>();
    }

    group.entries->insert_or_assign(key, value);
  }

  /**
   * @brief Try to insert or update an entry in an existing request group,
   * returns false if group does not exist
   */
  bool try_insert_or_assign(const GroupKey& group_id,
                            const EntryKey& key,
                            const Value& value) {
    std::shared_lock read_lock(store_mutex_);
    auto group_it = store_.find(group_id);
    if (group_it == store_.end()) {
      return false;
    }

    EntryGroup& group = group_it->second;
    std::scoped_lock group_lock(group.entry_mutex);
    read_lock.unlock();

    group.entries->insert_or_assign(key, value);
    return true;
  }
  /**
   * @brief Insert or update an entry, creating request group if needed
   */
  void upsert(const GroupKey& group_id,
              const EntryKey& key,
              const Value& value) {
    // Optimistic read path
    {
      std::shared_lock read_lock(store_mutex_);
      auto group_it = store_.find(group_id);
      if (group_it != store_.end()) {
        EntryGroup& group = group_it->second;
        std::scoped_lock group_lock(group.entry_mutex);
        read_lock.unlock();
        if (!group.entries) {
          group.entries =
              std::make_unique<std::unordered_map<EntryKey, Value>>();
        }
        group.entries->insert_or_assign(key, value);
        return;
      }
    }

    // Write path for new group
    std::unique_lock write_lock(store_mutex_);
    EntryGroup& group = store_[group_id];
    std::scoped_lock group_lock(group.entry_mutex);
    write_lock.unlock();
    if (!group.entries) {
      group.entries = std::make_unique<std::unordered_map<EntryKey, Value>>();
    }
    group.entries->insert_or_assign(key, value);
  }

  /**
   * @brief Retrieve an entry from the store
   * @return std::nullopt if either request group or key doesn't exist
   */
  std::optional<Value> try_get(const GroupKey& group_id,
                               const EntryKey& key) const {
    std::shared_lock read_lock(store_mutex_);
    auto group_it = store_.find(group_id);
    if (group_it == store_.end()) {
      return std::nullopt;
    }

    const EntryGroup& group = group_it->second;
    std::scoped_lock group_lock(group.entry_mutex);
    read_lock.unlock();

    if (!group.entries) {
      return std::nullopt;  // Group exists, but entries map is not
                            // initialized (empty group)
    }

    auto entry_it = group.entries->find(key);
    return entry_it != group.entries->end()
               ? std::optional<Value>(entry_it->second)
               : std::nullopt;
  }
  /**
   * @brief Retrieve an entry from the store
   * @throws std::runtime_error if request group doesn't exist
   */
  Value get(const GroupKey& group_id, const EntryKey& key) const {
    std::shared_lock read_lock(store_mutex_);
    auto group_it = store_.find(group_id);
    if (group_it == store_.end()) {
      throw std::runtime_error("Request group not found: " + group_id);
    }

    const EntryGroup& group = group_it->second;
    std::scoped_lock group_lock(group.entry_mutex);
    read_lock.unlock();

    if (!group.entries) {
      throw std::runtime_error("Entry not found in group: " + group_id +
                               ", key: " + key);
    }

    auto entry_it = group.entries->find(key);
    if (entry_it == group.entries->end()) {
      throw std::runtime_error("Entry not found in group: " + group_id +
                               ", key: " + key);
    }
    return entry_it->second;
  }

  void add_group(const GroupKey& group_id) {
    std::unique_lock write_lock(store_mutex_);
    auto group_it = store_.find(group_id);
    if (group_it != store_.end()) {
      throw std::runtime_error("Request group already exists: " + group_id);
    }
    store_[group_id] = EntryGroup();
  }

  /**
   * @brief Remove an entire request group
   * @return true if group existed and was removed
   */
  bool remove_group(const GroupKey& group_id) {
    std::unique_lock write_lock(store_mutex_);
    auto group_it = store_.find(group_id);
    if (group_it == store_.end()) {
      return false;
    }

    EntryGroup& group = group_it->second;
    std::scoped_lock group_lock(group.entry_mutex);
    store_.erase(group_it);
    return true;
  }

  /**
   * @brief Remove specific key from a request group
   * @throws std::runtime_error if request group doesn't exist
   */
  void remove_entry(const GroupKey& group_id, const EntryKey& key) {
    std::shared_lock read_lock(store_mutex_);
    auto group_it = store_.find(group_id);
    if (group_it == store_.end()) {
      throw std::runtime_error("Request group not found: " + group_id);
    }

    EntryGroup& group = group_it->second;
    std::scoped_lock group_lock(group.entry_mutex);
    read_lock.unlock();
    if (!group.entries) {
      throw std::runtime_error("Entry not found in group: " + group_id +
                               ", key: " + key);
    }
    if (group.entries->find(key) == group.entries->end()) {
      throw std::runtime_error("Entry not found in group: " + group_id +
                               ", key: " + key);
    }

    group.entries->erase(key);
  }
  /**
   * @brief Try to remove specific key from a request group,
   * return false if group or entry not exist
   */
  bool try_remove_entry(const GroupKey& group_id, const EntryKey& key) {
    std::shared_lock read_lock(store_mutex_);
    auto group_it = store_.find(group_id);
    if (group_it == store_.end()) {
      return false;
    }

    EntryGroup& group = group_it->second;
    std::scoped_lock group_lock(group.entry_mutex);
    read_lock.unlock();
    if (!group.entries) {
      return false;
    }

    return group.entries->erase(key) > 0;
  }

  /**
   * @brief Check if a request group exists
   */
  bool contains_group(const GroupKey& group_id) const {
    std::shared_lock read_lock(store_mutex_);
    return store_.find(group_id) != store_.end();
  }

  /**
   * @brief Check if a specific entry exists
   * @throws std::runtime_error if request group doesn't exist
   */
  bool contains_entry(const GroupKey& group_id, const EntryKey& key) const {
    std::shared_lock read_lock(store_mutex_);
    auto group_it = store_.find(group_id);
    if (group_it == store_.end()) {
      throw std::runtime_error("Request group not found: " + group_id);
    }

    const EntryGroup& group = group_it->second;
    std::scoped_lock group_lock(group.entry_mutex);
    read_lock.unlock();
    if (!group.entries) {
      throw std::runtime_error("Entry not found in group: " + group_id +
                               ", key: " + key);
    }

    return group.entries->count(key);
  }

  /**
   * @brief Try to Check if a specific entry exists
   * return false if group or entry not exist
   */
  bool try_contains_entry(const GroupKey& group_id, const EntryKey& key) const {
    std::shared_lock read_lock(store_mutex_);
    auto group_it = store_.find(group_id);
    if (group_it == store_.end()) {
      return false;
    }

    const EntryGroup& group = group_it->second;
    std::scoped_lock group_lock(group.entry_mutex);
    read_lock.unlock();
    if (!group.entries) {
      return false;
    }

    return group.entries->count(key);
  }

  /**
   * @brief Get number of request groups in the store
   */
  size_t group_count() const {
    std::shared_lock read_lock(store_mutex_);
    return store_.size();
  }

  /**
   * @brief Get number of entries in a specific request group
   * @throws std::runtime_error if request group doesn't exist
   */
  size_t entry_count(const GroupKey& group_id) const {
    std::shared_lock read_lock(store_mutex_);
    auto group_it = store_.find(group_id);
    if (group_it == store_.end()) {
      throw std::runtime_error("Request group not found: " + group_id);
    }

    const EntryGroup& group = group_it->second;
    std::scoped_lock group_lock(group.entry_mutex);
    read_lock.unlock();
    if (!group.entries) {
      throw std::runtime_error("Request group is empty: " + group_id);
    }
    return group.entries->size();
  }
};