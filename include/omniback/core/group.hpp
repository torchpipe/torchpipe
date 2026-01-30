#pragma once

#include <string>
#include <functional>

#include "omniback/helper/omniback_export.h"

namespace om{

bool OMNI_EXPORT register_backend_group(const std::string &backend_name, 
    const std::string &grp_name, 
    std::function<void()> callback);
void OMNI_EXPORT clear_groups();

bool OMNI_EXPORT try_callback_from_group(const std::string &backend_name);
}