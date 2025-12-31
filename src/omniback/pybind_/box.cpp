#include "omniback/builtin/box.hpp"
#include "omniback/pybind/box.hpp"
#include "omniback/pybind/py_register.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "omniback/pybind/py_helper.hpp"

namespace omniback {

namespace py = pybind11;
// PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

// PYBIND11_DECLARE_HOLDER_TYPE(omniback::Boxes,
// std::shared_ptr<omniback::Boxes>);

py::object cast(omniback::Boxes& boxes) {
  return py::cast(boxes);
}

void init_box(py::module_& m) {
  m.doc() = "Bounding box operations with XYXY coordinate format";

  // Box binding
  py::class_<Box, std::shared_ptr<Box>>(m, "Box")
      .def(py::init<>())
      .def_readwrite("id", &Box::id)
      .def_readwrite("score", &Box::score)
      .def_readwrite("x1", &Box::x1)
      .def_readwrite("y1", &Box::y1)
      .def_readwrite("x2", &Box::x2)
      .def_readwrite("y2", &Box::y2)
      // Computed properties
      .def_property_readonly("cx", &Box::cx)
      .def_property_readonly("cy", &Box::cy)
      .def_property_readonly("width", &Box::width)
      .def_property_readonly("height", &Box::height)
      .def("area", &Box::area)
      .def("__repr__", [](const Box& b) {
        return py::
            str("Box(id={}, score={:.2f}, xyxy=({:.1f},{:.1f},{:.1f},{:.1f}))")
                .format(b.id, b.score, b.x1, b.y1, b.x2, b.y2);
      });

  // Boxes binding
  py::class_<Boxes, std::shared_ptr<Boxes>>(m, "Boxes", py::module_local(false))
      .def(py::init<>())
      // Core operations
      .def("add", &Boxes::add, "Add a box object")
      .def("clear", &Boxes::clear, "Clear all boxes")
      .def("__len__", &Boxes::size)
      .def(
          "__getitem__",
          [](const Boxes& self, size_t i) {
            if (i >= self.size())
              throw py::index_error();
            return self.boxes[i];
          })
      // Coordinate-specific additions
      .def(
          "add_xyxy",
          &Boxes::add_xyxy,
          py::arg("x1"),
          py::arg("y1"),
          py::arg("x2"),
          py::arg("y2"),
          py::arg("score"),
          py::arg("id"),
          "Add box in XYXY format")
      .def(
          "add_cxcywh",
          &Boxes::add_cxcywh,
          py::arg("cx"),
          py::arg("cy"),
          py::arg("w"),
          py::arg("h"),
          py::arg("score"),
          py::arg("id"),
          "Add box in CXCYWH format")
      // Batch operations
      .def(
          "add_batch_cxcywh",
          [](Boxes& self,
             py::array_t<float> boxes,
             py::array_t<float> scores,
             py::array_t<int64_t> ids) {
            auto boxes_buf = boxes.request();
            auto scores_buf = scores.request();
            auto ids_buf = ids.request();

            const size_t n = boxes_buf.shape[0];
            if (boxes_buf.ndim != 2 || boxes_buf.shape[1] != 4) {
              throw std::runtime_error("Boxes must be shape [n,4]");
            }
            if (scores_buf.size != n) {
              throw std::runtime_error("Scores length must match boxes");
            }
            if (ids_buf.size != n) {
              throw std::runtime_error("IDs length must match boxes");
            }

            const float* boxes_ptr = static_cast<float*>(boxes_buf.ptr);
            const float* scores_ptr = static_cast<float*>(scores_buf.ptr);
            const int64_t* ids_ptr = static_cast<int64_t*>(ids_buf.ptr);

            {
              py::gil_scoped_release release;
              self.add_batch_cxcywh(boxes_ptr, scores_ptr, ids_ptr, n);
            }
          },
          py::arg("boxes"),
          py::arg("scores"),
          py::arg("ids"),
          "Add batch of boxes in (cx,cy,w,h) format")

      // 修复: 使用def_static绑定静态方法
      .def_static(
          "iou",
          &Boxes::iou,
          py::arg("a"),
          py::arg("b"),
          "Calculate IoU between two boxes")
      .def(
          "nms",
          &Boxes::nms,
          py::arg("iou_threshold") = 0.45f,
          py::arg("class_agnostic") = false,
          "Perform NMS with optional class handling",
          py::call_guard<py::gil_scoped_release>());
}
OMNI_ADD_HASH(Boxes);
OMNI_ADD_HASH(Box);
} // namespace omniback