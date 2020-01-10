#include <torch/csrc/autograd/python_engine.h>

#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/PtrWrapper.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/python_anomaly_mode.h>
#include <torch/csrc/autograd/python_function.h>
#include <torch/csrc/utils/auto_gil.h>
#include <ATen/core/EnableNamedTensor.h>

#ifndef _WIN32
#include <pthread.h>
#endif

#include <unordered_set>
#include <memory> // for unique_ptr

using namespace torch::autograd;

struct THPEngine {
    PyObject_HEAD
};

static torch::autograd::python::PythonEngine engine;

static Engine& get_python_engine() {
  return engine;
}

namespace torch { namespace autograd { namespace python {

void PythonEngine::thread_init(int device) {
  // Create a PyThreadState, but release the GIL. This lets AutoGIL calls
  // inside thread_main acquire the GIL without having to create a new
  // PyThreadState each time.
  AutoGIL gil;
  AutoNoGIL no_gil;
  Engine::thread_init(device);
}

void PythonEngine::thread_on_exception(NodeTask& task, std::exception& e) {
  auto python_err = dynamic_cast<python_error*>(&e);
  if (python_err) {
    python_err->persist();
  }
  Engine::thread_on_exception(task, e);
}

std::unique_ptr<AnomalyMetadata> PythonEngine::make_anomaly_metadata() {
  return std::unique_ptr<AnomalyMetadata>(new PyAnomalyMetadata());
}

variable_list PythonEngine::execute(
    const edge_list& roots,
    const variable_list& inputs,
    bool keep_graph,
    bool create_graph,
    const edge_list& outputs) {
  try {
    return Engine::execute(roots, inputs, keep_graph, create_graph, outputs);
  } catch (python_error& e) {
    e.restore();
    throw;
  }
}

}}} // namespace torch::autograd::python

PyObject *THPEngineClass = 0;

static bool _reinitialize_engine = false;

static void _maybe_reinitialize_engine_after_fork() {
  // This is "probably" thread-safe because the flag is set in a fork handler
  // before any threads are created, and this function is only called with the
  // GIL held. However, using fork + threads is playing with fire so this is
  // more of a "best effort" thing. For example, if the fork occurs while the
  // backwards threads hold a lock, we'll probably deadlock in the engine
  // destructor.
  if (_reinitialize_engine) {
    engine.~PythonEngine();
    new (&engine) torch::autograd::python::PythonEngine();
    _reinitialize_engine = false;
  }
}

// Implementation of torch._C._EngineBase.run_backward
PyObject *THPEngine_run_backward(THPEngine *self, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  _maybe_reinitialize_engine_after_fork();
  PyObject *tensors = 0;
  PyObject *grad_tensors = 0;
  unsigned char keep_graph = 0;
  unsigned char create_graph = 0;
  PyObject *inputs = 0;
  unsigned char allow_unreachable = 0;
  const char *accepted_kwargs[] = {
      "tensors", "grad_tensors", "keep_graph", "create_graph", "inputs",
      "allow_unreachable", 0
  };
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OObb|Ob", (char**)accepted_kwargs,
        &tensors, &grad_tensors, &keep_graph, &create_graph, &inputs, &allow_unreachable))
    return 0;

  THPUtils_assert(PyTuple_Check(tensors), "tensors argument is expected to "
      "be a tuple, but got %s", THPUtils_typename(tensors));
  THPUtils_assert(PyTuple_Check(grad_tensors), "grad_tensors argument is "
      "expected to be a tuple, but got %s", THPUtils_typename(grad_tensors));

  Py_ssize_t num_tensors = PyTuple_GET_SIZE(tensors);
  Py_ssize_t num_gradients = PyTuple_GET_SIZE(grad_tensors);
  THPUtils_assert(num_tensors == num_gradients, "got %ld tensors and %ld "
      "gradients", num_tensors, num_gradients);

  edge_list roots;
  roots.reserve(num_tensors);
  variable_list grads;
  grads.reserve(num_tensors);
  for (int i = 0; i < num_tensors; i++) {
    PyObject *_tensor = PyTuple_GET_ITEM(tensors, i);
    THPUtils_assert(THPVariable_Check(_tensor), "element %d of tensors "
        "tuple is not a Tensor", i);
    auto& variable = ((THPVariable*)_tensor)->cdata;
    auto gradient_edge = variable.gradient_edge();
    THPUtils_assert(gradient_edge.function,
        "element %d of tensors does not require grad and does not have a grad_fn", i);
    roots.push_back(std::move(gradient_edge));

    PyObject *grad = PyTuple_GET_ITEM(grad_tensors, i);
    if (THPVariable_Check(grad)) {
      const Variable& grad_var = ((THPVariable*)grad)->cdata;
#ifdef BUILD_NAMEDTENSOR
      if (grad_var.has_names()) {
        TORCH_WARN(
            "Autograd was passed a named grad tensor with dims ", grad_var.names(),
            ". Autograd does not yet support named tensor semantics, so all names ",
            "will be ignored. In practice all computed gradients will still be correct "
            "according to regular tensor semantics.");
      }
#endif
      grads.push_back(grad_var);
    } else {
      THPUtils_assert(grad == Py_None,
          "element %d of gradients tuple is not a Tensor or None", i);
      THPUtils_assert(!variable.requires_grad(),
          "element %d of gradients tuple is None, but the corresponding Tensor requires grad");
    }
  }

  std::vector<Edge> output_edges;
  if (inputs != 0) {
    int num_inputs = PyTuple_GET_SIZE(inputs);
    output_edges.reserve(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      PyObject *input = PyTuple_GET_ITEM(inputs, i);
      THPUtils_assert(THPVariable_Check(input),
          "all inputs have to be Tensors, but got %s", THPUtils_typename(input));
      THPVariable *input_var = (THPVariable*)input;
      const auto output_nr = input_var->cdata.output_nr();
      auto grad_fn = input_var->cdata.grad_fn();
      if (!grad_fn) {
          grad_fn = input_var->cdata.try_get_grad_accumulator();
      }
      THPUtils_assert(input_var->cdata.requires_grad(),
          "One of the differentiated Tensors does not require grad");
      if (!grad_fn) {
        output_edges.emplace_back();
      } else {
        output_edges.emplace_back(grad_fn, output_nr);
      }
    }
  }

  variable_list outputs;
  {
    AutoNoGIL no_gil;
    outputs = engine.execute(roots, grads, keep_graph, create_graph, output_edges);
  }

  if (inputs != 0) {
    int num_inputs = PyTuple_GET_SIZE(inputs);
    THPObjectPtr py_outputs {PyTuple_New(num_inputs)};
    if (!py_outputs) return 0;
    for (int i = 0; i < num_inputs; i++) {
      THPUtils_assert(allow_unreachable || outputs[i].defined(), "One of the "
                      "differentiated Tensors appears to not have been used "
                      "in the graph. Set allow_unused=True if this is the "
                      "desired behavior.");
      PyTuple_SET_ITEM(py_outputs.get(), i, THPVariable_Wrap(outputs[i]));
    }
    return py_outputs.release();
  } else {
    Py_RETURN_NONE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPEngine_queue_callback(PyObject *self, PyObject *_callback) {
  HANDLE_TH_ERRORS
  _maybe_reinitialize_engine_after_fork();
  std::shared_ptr<PyObject> callback(_callback, [](PyObject *obj) { AutoGIL gil; Py_DECREF(obj); });
  Py_INCREF(_callback);
  engine.queue_callback([callback]() {
    AutoGIL gil;
    THPObjectPtr result {PyObject_CallFunctionObjArgs(callback.get(), 0)};
    if (!result) throw python_error();
  });
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPEngine_is_checkpoint_valid(PyObject *self, PyObject *noargs) {
  HANDLE_TH_ERRORS
  if(engine.is_checkpoint_valid()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject *THPEngine_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  return type->tp_alloc(type, 0);
}

static struct PyMethodDef THPEngine_methods[] = {
  {(char*)"run_backward", (PyCFunction)(void(*)(void))THPEngine_run_backward, METH_VARARGS | METH_KEYWORDS, 0},
  {(char*)"queue_callback", (PyCFunction)THPEngine_queue_callback, METH_O, 0},
  {(char*)"is_checkpoint_valid", (PyCFunction)THPEngine_is_checkpoint_valid, METH_NOARGS, 0},
  {0}
};


PyTypeObject THPEngineType = {
  PyVarObject_HEAD_INIT(0, 0)
  "torch._C._EngineBase",                /* tp_name */
  sizeof(THPEngine),                     /* tp_basicsize */
  0,                                     /* tp_itemsize */
  0,                                     /* tp_dealloc */
  0,                                     /* tp_print */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  0,                                     /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  0,                                     /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  0,                                     /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
  0,                               /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  THPEngine_methods,                     /* tp_methods */
  0,                                     /* tp_members */
  0,                                     /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THPEngine_new                          /* tp_new */
};

static void child_atfork() {
  _reinitialize_engine = true;
}

bool THPEngine_initModule(PyObject *module)
{
#ifndef _WIN32
  if (pthread_atfork(0, 0, child_atfork) != 0) {
    throw std::runtime_error("unable to set pthread_atfork handler");
  }
#endif
  if (PyType_Ready(&THPEngineType) < 0)
    return false;
  Py_INCREF(&THPEngineType);
  PyModule_AddObject(module, "_ImperativeEngine", (PyObject *)&THPEngineType);
  set_default_engine_stub(get_python_engine);
  return true;
}
