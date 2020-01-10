#pragma once

#include <torch/csrc/python_headers.h>
#include <memory>
#include <typeinfo>

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/Exceptions.h>

namespace torch { namespace autograd {

struct THPCppFunction {
  PyObject_HEAD
  std::shared_ptr<Node> cdata;
};

template<typename Ctor>
PyObject* CppFunction_pynew(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  THPObjectPtr obj(type->tp_alloc(type, 0));
  if (!obj) return 0;
  THPCppFunction* f = (THPCppFunction*)obj.get();
  HANDLE_TH_ERRORS
  new (&f->cdata) std::shared_ptr<Node>(Ctor()(args));
  END_HANDLE_TH_ERRORS
  if (!f->cdata) {
    return 0;
  }
  return obj.release();
}

#define THP_FUNCTION_DEFAULT_METHODS \
  {(char*)"_register_hook_dict", (PyCFunction)THPCppFunction_register_hook_dict, METH_O, 0}, \
  {(char*)"register_hook", (PyCFunction)THPCppFunction_register_hook, METH_O, 0}, \
  {(char*)"name", (PyCFunction)THPCppFunction_name, METH_NOARGS, 0}

#define THP_FUNCTION_DEFAULT_PROPERTIES \
  {(char*)"next_functions", (getter)THPCppFunction_next_functions, 0, 0, 0}, \
  {(char*)"requires_grad", (getter)THPCppFunction_requires_grad, 0, 0, 0}, \
  {(char*)"metadata", (getter)THPCppFunction_metadata, 0, 0, 0}

PyObject* THPCppFunction_next_functions(THPCppFunction* self, PyObject* hook);
PyObject* THPCppFunction_metadata(THPCppFunction *self, void *_unused);
PyObject* THPCppFunction_requires_grad(THPCppFunction* self, void *_unused);
PyObject* THPCppFunction_register_hook_dict(PyObject* self, PyObject* _var);
PyObject* THPCppFunction_register_hook(PyObject* self, PyObject* hook);
PyObject* THPCppFunction_name(PyObject* self, PyObject *noargs);

PyTypeObject* _initFunctionPyTypeObject(PyTypeObject& type, const char* name,
  PyGetSetDef* function_properties, PyMethodDef* function_methods);

PyObject* registerFunctionHook(Node& fn, PyObject* hook);

template<typename Ctor>
PyTypeObject* createForwardFunctionPyTypeObject(PyTypeObject& type, const char* name,
  PyGetSetDef* function_properties=0, PyMethodDef* function_methods=0)
{
  type.tp_new = &CppFunction_pynew<Ctor>;
  return _initFunctionPyTypeObject(type, name, function_properties, function_methods);
}

void registerCppFunction(const std::type_info& type, PyTypeObject* pytype);
PyObject* functionToPyObject(const std::shared_ptr<Node>& cdata);

}} // namespace torch::autograd
