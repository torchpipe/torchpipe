#pragma once
#include <functional>
#include "hami/core/backend.hpp"

namespace hami {

/**
 * @brief Backend with injectable dependency.
 *
 * This backend allows injecting a dependency, which can be used for chaining or
 * delegation.
 */
class HAMI_EXPORT Dependency : public Backend {
   public:
    /**
     * @brief Injects a dependency into the backend or the backend's dependency
     * if alread exists.
     *
     * @param dependency Pointer to the dependency. Should not be nullptr.
     *
     * @throw std::invalid_argument If the provided dependency is nullptr.
     */
    void inject_dependency(Backend* dependency) override final;

    virtual void forward(const std::vector<dict>& input_output) override final {
        forward(input_output, injected_dependency_);
    }
    virtual void forward(const std::vector<dict>& input_output,
                         Backend* dependency) override final {
        if (dependency == nullptr) {
            throw std::invalid_argument("null dependency is not allowed");
        }
        forward_impl(input_output, dependency);
    }

    [[nodiscard]] size_t max() const override {
        return injected_dependency_ ? injected_dependency_->max()
                                    : std::numeric_limits<size_t>::max();
    }
    [[nodiscard]] size_t min() const override {
        return injected_dependency_ ? injected_dependency_->min() : 1;
    }

    /**
     * @brief Optional support for A[B], where A::dependency=B is passed as a
     * parameter and injected as a dependency.
     *
     * The dependency instance is registered as {node_name}.{A}.{B}.
     */
    void init(const std::unordered_map<std::string, std::string>& config,
              const dict& dict_config) override final;

    virtual void pre_init(
        const std::unordered_map<std::string, std::string>& config,
        const dict& dict_config) {}
    virtual void post_init(
        const std::unordered_map<std::string, std::string>& config,
        const dict& dict_config) {}

    ~Dependency() override;

   protected:
    void set_registered_name(const std::string& name) {
        registered_name_ = name;
    }
    void set_dependency_name(const std::string& name) {
        dependency_name_ = name;
    }

    Backend* injected_dependency_{nullptr};  ///< The injected dependency.

   private:
    virtual void forward_impl(const std::vector<dict>& input_output,
                              Backend* dependency);
    std::string registered_name_{};  ///< The registered name of the backend.
    std::string dependency_name_{};
    std::shared_ptr<Backend> shared_owned_dependency_;
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
    void init(const std::unordered_map<std::string, std::string>& config,
              const dict& dict_config) override final;
    virtual void post_init(
        const std::unordered_map<std::string, std::string>& config,
        const dict& dict_config) {};
    [[nodiscard]] size_t max() const override final { return max_; }
    [[nodiscard]] size_t min() const override final { return min_; }

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

/**
 * @brief A specialized backend that supports only single input and no
 * dependency.
 *
 * Enforces the constraint that only one input is allowed and no dependency can
 * be provided.
 */
class HAMI_EXPORT SingleBackend : public Backend {
   public:
    /**
     * @brief Returns the maximum number of inputs supported (always 1).
     *
     * @return Maximum number of inputs supported.
     */
    [[nodiscard]] size_t max() const override final { return 1; }

    /**
     * @brief Processes a single input/output.
     *
     * Must be implemented by derived classes.
     *
     * @param input_output Single input/output data to be processed.
     */
    virtual void forward(const dict& input_output) = 0;

    /**
     * @brief Processes input/output data with single input.
     *
     * @param input_output Input/output data to be processed.
     * @throws std::invalid_argument If more than one input is provided.
     */
    void forward(const std::vector<dict>& input_output) override final {
        if (input_output.size() != 1) {
            throw std::invalid_argument(
                "SingleBackend only supports single input");
        }
        forward(input_output[0]);
    }

    /**
     * @brief Overrides inject_dependency to disallow setting dependencies.
     *
     * @param dependency Pointer to the backend dependency (ignored).
     * @throws std::runtime_error Always, as SingleBackend does not support
     * dependencies.
     */
    void inject_dependency(Backend* dependency) override final {
        throw std::runtime_error(
            "SingleBackend does not support inject dependency");
    }
};

class HAMI_EXPORT List : public Backend {
   public:
    void init(const std::unordered_map<std::string, std::string>& config,
              const dict& dict_config) override final;
    void forward(const std::vector<dict>& input_output) override final;
    [[nodiscard]] size_t max() const override final { return 1; }
    [[nodiscard]] size_t min() const override final {
        return std::numeric_limits<size_t>::max();
    }

   private:
    std::vector<std::unique_ptr<Backend>> backends_;
};

}  // namespace hami