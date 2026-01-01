#ifndef OMNI_HELPER_EXPORT__
#define OMNI_HELPER_EXPORT__

#define OMNI_EXPORT __attribute__((visibility("default")))
#define OMNI_LOCAL __attribute__((visibility("hidden")))

#endif