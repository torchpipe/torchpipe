#pragma once

#include <string>
#include <vector>

namespace hami {

class Sequential final : public Container {
   public:
    /**
     * @param Sequential::dependency The name of the sub-backend, multiple
     * backends separated by commas.
     * @remark
     * 1. The initialization of sub-backends will be executed in **reverse
     * order**;
     * 2. This container supports expanding bracket compound syntax. It use the
     * @ref str::brackets_split function to expand it as follows:
     * - backend = B[C]        =>     {backend=B, B::dependency=C}
     * - backend = D           =>     {backend=D}
     * - backend = B[E[Z1,Z2]] =>     {backend=B, B::dependency=E[Z1,Z2]}
     */
    virtual void post_init(const std::unordered_map<std::string, std::string>&,
                           const dict&) override {}

    /**
     * @brief Sequentially calls sub-backends.
     * @note During sequential calls, a swap operation will be performed: Assign
     * TASK_RESULT_KEY to TASK_DATA_KEY.
     */
    virtual void impl_forward(const std::vector<dict>&) override;
};

}  // namespace hami