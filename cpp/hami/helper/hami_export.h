#ifndef HAMI_HELPER_EXPORT__
#define HAMI_HELPER_EXPORT__

#define HAMI_EXPORT __attribute__((visibility("default")))
#define HAMI_LOCAL __attribute__((visibility("hidden")))

#endif