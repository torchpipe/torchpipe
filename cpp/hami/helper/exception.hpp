#pragma once 

#include <stdexcept>

namespace hami::error {


class KeyNotFoundError : public std::runtime_error {
   public:
    using std::runtime_error::runtime_error;
};





class NoResultError : public KeyNotFoundError {
   public:
    NoResultError() : KeyNotFoundError("result is empty") {}
    using KeyNotFoundError::KeyNotFoundError;
};



}