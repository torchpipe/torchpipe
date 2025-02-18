

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "hami/core/any.hpp"
#include "hami/core/event.hpp"

namespace hami {

namespace py = pybind11;
using namespace pybind11::literals;

void init_event(py::module_& m) {
  py::class_<Event, std::shared_ptr<Event>> hami_event(m, "Event");

  hami_event.doc() = "hami.Event provides an object similar to Python's threading.Event.";

  hami_event.def(py::init<size_t>(), py::arg("max_num") = 1)
      .def("wait", py::overload_cast<>(&Event::wait), py::call_guard<py::gil_scoped_release>(),
           "Wait for the event to be set without a timeout.")
      .def("wait", py::overload_cast<size_t>(&Event::wait), py::arg("timeout"),
           py::call_guard<py::gil_scoped_release>(),
           "Wait for the event to be set with a timeout in milliseconds.")
      .def("set", &Event::set, "Set the event.")
      .def("is_set", &Event::is_set, "Check if the event is set.");
}

}  // namespace hami