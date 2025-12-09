

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "omniback/core/any.hpp"
#include "omniback/core/event.hpp"
#include "omniback/pybind/py_helper.hpp"
#include "omniback/helper/timer.hpp"

namespace omniback {

namespace py = pybind11;
using namespace pybind11::literals;

void init_event(py::module_& m) {
  py::class_<Event, std::shared_ptr<Event>> omniback_event(m, "Event");

  omniback_event.doc() =
      "omniback.Event provides an object similar to Python's threading.Event.";

  omniback_event.def(py::init<size_t>(), py::arg("max_num") = 1)
      .def(
          "wait",
          py::overload_cast<>(&Event::wait),
          py::call_guard<py::gil_scoped_release>(),
          "Wait for the event to be set without a timeout.")
      .def(
          "wait",
          py::overload_cast<size_t>(&Event::wait),
          py::arg("timeout"),
          py::call_guard<py::gil_scoped_release>(),
          "Wait for the event to be set with a timeout in milliseconds.")
      .def("set", &Event::set, "Set the event.")
      .def("is_set", &Event::is_set, "Check if the event is set.")
      .def("set_callback", &Event::set_callback, "Set a callback function.")
      .def(
          "set_exception_callback",
          [](Event& self, py::function py_callback) {
            auto p_cb = python::make_shared(py_callback);
            self.set_exception_callback([p_cb](std::exception_ptr eptr) {
              py::gil_scoped_acquire acquire; // 确保持有 GIL
              try {
                if (eptr) {
                  std::rethrow_exception(eptr); // 重新抛出异常
                }
              } catch (const std::exception& e) {
                // 将 C++ 异常转换为 Python 异常对象
                // py::object py_e = py::cast(e);
                (*p_cb)(e.what()); // 调用 Python 回调
              } catch (...) {
                (*p_cb)(py::cast("Unknown C++ exception"));
              }
              //   p_cb = nullptr;
            });
          })
      .def("try_throw", &Event::try_throw, "Try to throw an exception.")
      .def(
          "notify_all",
          py::overload_cast<>(&Event::notify_all),
          py::call_guard<py::gil_scoped_release>(),
          "Try to Notify.")
      .def(
          "set_final_callback",
          &Event::set_final_callback,
          "Set a callback function.")
      .attr("type_hash") = typeid(std::shared_ptr<Event>).hash_code();

  m.attr("TASK_EVENT_KEY") = py::cast(TASK_EVENT_KEY);
  m.def(
      "timestamp",
      &helper::timestamp,
      pybind11::call_guard<pybind11::gil_scoped_release>());
}

} // namespace omniback