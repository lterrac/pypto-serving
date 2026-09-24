/**
 * Embeds CPython, loads the model through pypto_serving.bridge, runs a prefill
 * and decode steps on a thread, and prints the token ids. They must match
 * tests/fixtures/qwen3_generation.json.
 *
 *   bridge-embed <model_dir> [device]
 */

#include <Python.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

namespace
{

/// Print the pending Python exception, if any, and return false when there was one.
bool pythonOk(const char *what)
{
  if (PyErr_Occurred() == nullptr) { return true; }
  fprintf(stderr, "\n[bridge] python error during %s:\n", what);
  PyErr_Print();
  return false;
}

/// Build a Python list of ints from a vector.
PyObject *toPyList(const std::vector<long> &values)
{
  PyObject *list = PyList_New(static_cast<Py_ssize_t>(values.size()));
  if (list == nullptr) { return nullptr; }
  for (size_t i = 0; i < values.size(); ++i)
  {
    PyObject *item = PyLong_FromLong(values[i]);
    if (item == nullptr)
    {
      Py_DECREF(list);
      return nullptr;
    }
    PyList_SET_ITEM(list, static_cast<Py_ssize_t>(i), item); // steals the reference
  }
  return list;
}

/// Read a Python list of ints back into a vector.
bool fromPyList(PyObject *obj, std::vector<long> &out)
{
  if (!PyList_Check(obj)) { return false; }
  const Py_ssize_t n = PyList_Size(obj);
  out.clear();
  out.reserve(static_cast<size_t>(n));
  for (Py_ssize_t i = 0; i < n; ++i)
  {
    PyObject  *item  = PyList_GetItem(obj, i); // borrowed
    const long value = PyLong_AsLong(item);
    if (value == -1 && PyErr_Occurred() != nullptr) { return false; }
    out.push_back(value);
  }
  return true;
}

/// What the device lane produced, handed back to main.
struct LaneResult
{
  std::vector<long> tokens;
  bool              ok = false;
};

} // namespace

int main(int argc, char **argv)
{
  if (argc < 4)
  {
    fprintf(stderr,
            "usage: %s <model_dir> <device_id> <num_tokens> [repo_root]\n"
            "  repo_root is prepended to sys.path so `import pypto_serving.bridge` resolves.\n",
            argv[0]);
    return 2;
  }
  const std::string modelDir  = argv[1];
  const int         deviceId  = std::atoi(argv[2]);
  const int         numTokens = std::atoi(argv[3]);
  const std::string repoRoot  = (argc > 4) ? argv[4] : "";

  // ---------------------------------------------------------------------------
  // Phase 1: interpreter up, model up. Still single-threaded -- this is the
  // window in which simpler is allowed to fork.
  // ---------------------------------------------------------------------------
  Py_Initialize();

  if (!repoRoot.empty())
  {
    const std::string code = "import sys; sys.path.insert(0, r'" + repoRoot + "')";
    if (PyRun_SimpleString(code.c_str()) != 0)
    {
      fprintf(stderr, "[bridge] could not extend sys.path\n");
      return 1;
    }
  }

  printf("[bridge] importing pypto_serving.bridge (pulls torch + pypto) ...\n");
  fflush(stdout);
  PyObject *bridge = PyImport_ImportModule("pypto_serving.bridge");
  if (bridge == nullptr || !pythonOk("import")) { return 1; }

  printf("[bridge] open_model(%s, device=%d) -- compiles kernels and forks simpler's children\n", modelDir.c_str(), deviceId);
  fflush(stdout);
  PyObject *pages = PyObject_CallMethod(bridge, "open_model", "sisii", modelDir.c_str(), deviceId, "a2a3", 512, 128);
  if (pages == nullptr || !pythonOk("open_model")) { return 1; }
  const long numPages = PyLong_AsLong(pages);
  Py_DECREF(pages);
  printf("[bridge] model ready: %ld KV pages\n", numPages);

  PyObject *pageSizeObj = PyObject_CallMethod(bridge, "page_size", nullptr);
  if (pageSizeObj == nullptr || !pythonOk("page_size")) { return 1; }
  const long pageSize = PyLong_AsLong(pageSizeObj);
  Py_DECREF(pageSizeObj);

  PyObject *promptObj = PyObject_CallMethod(bridge, "encode", "s", "The capital of France is");
  if (promptObj == nullptr || !pythonOk("encode")) { return 1; }
  std::vector<long> promptTokens;
  if (!fromPyList(promptObj, promptTokens))
  {
    fprintf(stderr, "[bridge] encode() did not return a list of ints\n");
    return 1;
  }
  Py_DECREF(promptObj);

  printf("[bridge] prompt is %zu tokens, page_size=%ld\n", promptTokens.size(), pageSize);

  // One block table for the whole run: prompt + generated must fit.
  const long        maxSeq = static_cast<long>(promptTokens.size()) + numTokens;
  const long        blocks = (maxSeq + pageSize - 1) / pageSize;
  std::vector<long> blockIds;
  for (long b = 0; b < blocks; ++b) { blockIds.push_back(b); }

  // ---------------------------------------------------------------------------
  // Phase 2: drop the GIL and hand the work to a real thread. If simpler's fork
  // and an embedded interpreter cannot coexist with C++ threads, it fails here.
  // ---------------------------------------------------------------------------
  PyThreadState *mainState = PyEval_SaveThread();

  LaneResult  result;
  std::thread deviceLane([&]() {
    const PyGILState_STATE gil = PyGILState_Ensure();

    PyObject *tokenList = toPyList(promptTokens);
    PyObject *blockList = toPyList(blockIds);
    if (tokenList == nullptr || blockList == nullptr)
    {
      PyGILState_Release(gil);
      return;
    }

    PyObject *first = PyObject_CallMethod(bridge, "prefill", "OO", tokenList, blockList);
    Py_DECREF(tokenList);
    if (first == nullptr || !pythonOk("prefill"))
    {
      Py_DECREF(blockList);
      PyGILState_Release(gil);
      return;
    }
    long token = PyLong_AsLong(first);
    Py_DECREF(first);
    result.tokens.push_back(token);
    printf("[bridge]   prefill -> %ld\n", token);
    fflush(stdout);

    long seqLen = static_cast<long>(promptTokens.size());
    for (int step = 1; step < numTokens; ++step)
    {
      seqLen += 1; // the token just produced now occupies a slot
      PyObject *next = PyObject_CallMethod(bridge, "decode", "iiO", static_cast<int>(token), static_cast<int>(seqLen), blockList);
      if (next == nullptr || !pythonOk("decode")) { break; }
      token = PyLong_AsLong(next);
      Py_DECREF(next);
      result.tokens.push_back(token);
      printf("[bridge]   decode  -> %ld\n", token);
      fflush(stdout);
    }

    Py_DECREF(blockList);
    result.ok = result.tokens.size() == static_cast<size_t>(numTokens);
    PyGILState_Release(gil);
  });
  deviceLane.join();

  PyEval_RestoreThread(mainState);

  // ---------------------------------------------------------------------------
  // Phase 3: report and tear down.
  // ---------------------------------------------------------------------------
  printf("\n[bridge] token_ids: [");
  for (size_t i = 0; i < result.tokens.size(); ++i) { printf("%s%ld", i ? ", " : "", result.tokens[i]); }
  printf("]\n");

  PyObject *textList = toPyList(result.tokens);
  if (textList != nullptr)
  {
    PyObject *text = PyObject_CallMethod(bridge, "decode_text", "O", textList);
    Py_DECREF(textList);
    if (text != nullptr)
    {
      printf("[bridge] text: %s\n", PyUnicode_AsUTF8(text));
      Py_DECREF(text);
    }
    else { PyErr_Clear(); }
  }

  PyObject *closed = PyObject_CallMethod(bridge, "close", nullptr);
  if (closed != nullptr) { Py_DECREF(closed); }
  else { PyErr_Clear(); }
  Py_DECREF(bridge);

  printf("[bridge] %s\n", result.ok ? "OK" : "INCOMPLETE");
  // Py_Finalize is skipped: simpler's forked children and the device state do
  // not survive it.
  return result.ok ? 0 : 1;
}
