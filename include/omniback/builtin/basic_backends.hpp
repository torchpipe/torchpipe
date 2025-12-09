#pragma once
#include <functional>
#include "omniback/core/backend.hpp"

namespace omniback {

/**
 * @brief Backend with injectable dependency.
 *
 * This backend allows injecting a dependency, which can be used for chaining or
 * delegation.
 */
class OMNI_EXPORT DependencyV0 : public Backend {
 private:
  /**
   * @brief Injects a dependency into the backend or the backend's dependency
   * if alread exists.
   *
   * @param dependency Pointer to the dependency. Should not be nullptr.
   *
   * @throw std::invalid_argument If the provided dependency is nullptr.
   */
  void impl_inject_dependency(Backend* dependency) override final;

  virtual void impl_forward(
      const std::vector<dict>& input_output) override final {
    forward_with_dep(input_output, *injected_dependency_);
  }
  virtual void impl_forward_with_dep(
      const std::vector<dict>& input_output,
      Backend& dependency) override final {
    custom_forward_with_dep(input_output, dependency);
  }

  [[nodiscard]] size_t impl_max() const override {
    return injected_dependency_ ? injected_dependency_->max()
                                : std::numeric_limits<size_t>::max();
  }
  [[nodiscard]] size_t impl_min() const override {
    return injected_dependency_ ? injected_dependency_->min() : 1;
  }

  /**
   * @brief Optional support for A[B], where A::dependency=B is passed as a
   * parameter and injected as a dependency.
   *
   * The dependency instance is registered as {node_name}.{A}.{B}.
   */
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) override final;

  virtual void pre_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) {}
  virtual void post_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) {}

 public:
  ~DependencyV0() override;

 protected:
  void set_registered_name(const std::string& name) {
    registered_name_ = name;
  }
  void set_dependency_name(const std::string& name) {
    dependency_name_ = name;
  }

  void set_dependency_name(
      const std::unordered_map<std::string, std::string>& config,
      const std::string& default_cls_name,
      const std::string& default_dep_name);

  Backend* injected_dependency_{nullptr}; ///< The injected dependency.

 private:
  virtual void custom_forward_with_dep(
      const std::vector<dict>& input_output,
      Backend& dependency);
  std::string registered_name_{}; ///< The registered name of the backend.
  std::string dependency_name_{};
  std::shared_ptr<Backend> shared_owned_dependency_;
};

/**
 * @brief Backend with injectable dependency.
 *
 * This backend allows injecting a dependency, which can be used for chaining or
 * delegation.
 */
class OMNI_EXPORT Dependency : public Backend {
 protected:
  /**
   * @brief Injects a dependency into the backend or the backend's dependency
   * if alread exists.
   *
   * @param dependency Pointer to the dependency. Should not be nullptr.
   *
   * @throw std::invalid_argument If the provided dependency is nullptr.
   */
  void impl_inject_dependency(Backend* dependency) override;

 private:
  virtual void impl_forward(const std::vector<dict>& ios) override final {
    // Backend::forward_with_dep(input_output, injected_dependency_);
    if (!injected_dependency_)
      throw std::runtime_error("Dependency is not injected");
    forward_with_dep(ios, *injected_dependency_);
  }

  virtual void impl_forward_with_dep(const std::vector<dict>& ios, Backend& dep)
      override {
    dep.forward(ios);
  }

  [[nodiscard]] size_t impl_max() const override;
  [[nodiscard]] size_t impl_min() const override;

 protected:
  Backend* injected_dependency_{nullptr}; ///< The injected dependency.
};

/**
 * @brief A container that can hold multiple backends.
 * As it has multiple dependent backends, we treat it as a `container` of
 * backends and it is not allowed to modify its dependencies at runtime.
 * @note This class supports the syntax
 * `Container[(pre=1)B(post=z,post2=z2)[Z],C]`.
 */
class Container : public Backend {
 public:
  /**
   * @param *::dependency The name of the sub-backends, multiple backends
   * separated by commas.
   * @remark
   * 1. The initialization of sub-backends will be executed in **reverse
   * order**.
   * 2. This container supports expanding bracket compound syntax. It uses the
   * @ref str::brackets_split function to expand it as follows:
   * - backend = B[C]        =>     {backend=B, B::dependency=C}
   * - backend = D           =>     {backend=D}
   * - backend = B[E[Z1,Z2]] =>     {backend=B, B::dependency=E[Z1,Z2]}
   * - It supports prefix and suffix configuration syntax, i.e.,
   * (pre=1)A(post=z)[B]. The prefix configuration is stored in the protected
   * member variable base_config_, and the suffix configuration is passed to
   * A.init.
   */
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) override final;
  virtual void post_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) {};
  [[nodiscard]] size_t impl_max() const override final {
    return max_;
  }
  [[nodiscard]] size_t impl_min() const override final {
    return min_;
  }

 private:
  virtual std::pair<size_t, size_t> update_min_max(
      const std::vector<Backend*>& depends);
  virtual std::vector<size_t> set_init_order(size_t max_range) const;

 protected:
  std::vector<std::unique_ptr<Backend>> base_dependencies_;
  std::vector<std::string> backends_;
  std::vector<std::unordered_map<std::string, std::string>> base_config_;
  std::vector<std::function<void()>> lazy_init_func_;
  size_t max_{std::numeric_limits<std::size_t>::max()};
  size_t min_{1};
};

class OMNI_EXPORT List : public Backend {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) override final;
  void impl_forward(const std::vector<dict>& input_output) override final;
  [[nodiscard]] size_t impl_max() const override final {
    return 1;
  }
  [[nodiscard]] size_t impl_min() const override final {
    return std::numeric_limits<size_t>::max();
  }

 private:
  std::vector<std::unique_ptr<Backend>> backends_;

 public:
  ~List() {
    for (size_t i = 0; i < backends_.size(); ++i) {
      backends_[backends_.size() - i - 1].release();
    }
  }
};

} // namespace omniback