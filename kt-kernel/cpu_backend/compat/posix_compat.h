// SPDX-License-Identifier: Apache-2.0
//
// POSIX/Linux <-> Windows compatibility shim for kt-kernel.
//
// On Linux this header is a transparent passthrough to the real <numa.h>,
// <numaif.h>, <hwloc.h>, <sched.h>, and <pthread.h>.
//
// On Windows the same APIs are stubbed:
//   * single-NUMA assumption (numa_node_of_cpu always returns 0)
//   * no thread CPU pinning (let the Windows scheduler distribute)
//   * pthread_setname_np is a no-op
//   * posix_memalign / aligned_alloc map to _aligned_malloc / _aligned_free
//
// This is enough for AVX2 MoE inference correctness on a single-socket Windows
// box. Multi-NUMA Windows servers would need a real implementation backed by
// SetThreadAffinityMask / GetNumaNodeProcessorMaskEx.

#ifndef KT_CPU_BACKEND_COMPAT_POSIX_COMPAT_H
#define KT_CPU_BACKEND_COMPAT_POSIX_COMPAT_H

#ifndef _WIN32

// ---- Linux passthrough -----------------------------------------------------
#include <hwloc.h>
#include <numa.h>
#include <numaif.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <cstdlib>

// posix_memalign / std::aligned_alloc both work natively on glibc >= 2.16.
// Provide a name-stable wrapper so call sites don't need #ifdef.
inline void* kt_aligned_alloc(std::size_t alignment, std::size_t size) {
  void* p = nullptr;
  if (posix_memalign(&p, alignment, size) != 0) return nullptr;
  return p;
}
inline void kt_aligned_free(void* p) { std::free(p); }

#else  // _WIN32

// ---- Windows stubs ---------------------------------------------------------
#ifndef NOMINMAX
#define NOMINMAX  // prevent windows.h from defining min/max macros that break std::min/std::max
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN  // skip rare APIs (RPC, OLE, ...) we don't need
#endif
#include <malloc.h>  // _aligned_malloc / _aligned_free
#include <windows.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>

// --- numa stubs -------------------------------------------------------------
struct bitmask {
  unsigned long maskp[1];
  unsigned long size;
};

inline int numa_num_configured_nodes(void) { return 1; }
inline int numa_node_of_cpu(int /*cpu*/) { return 0; }
inline int numa_available(void) { return 0; }  // success on Windows stub

inline struct bitmask* numa_bitmask_alloc(unsigned int /*n*/) {
  auto* b = static_cast<struct bitmask*>(std::calloc(1, sizeof(struct bitmask)));
  return b;
}
inline struct bitmask* numa_bitmask_setbit(struct bitmask* b, unsigned int /*n*/) {
  if (b) b->maskp[0] |= 1ULL;
  return b;
}
inline void numa_bitmask_free(struct bitmask* b) {
  if (b) std::free(b);
}
inline void numa_bind(struct bitmask* /*b*/) {
  // No-op: Windows desktop is single NUMA node from our PoV.
}
inline int numa_run_on_node(int /*node*/) { return 0; }

// numaif.h equivalents
#define MPOL_BIND 0
#define MPOL_INTERLEAVE 1
inline long mbind(void* /*start*/, unsigned long /*len*/, int /*mode*/, const unsigned long* /*nodemask*/,
                  unsigned long /*maxnode*/, unsigned /*flags*/) {
  return 0;
}
inline long set_mempolicy(int /*mode*/, const unsigned long* /*nodemask*/, unsigned long /*maxnode*/) { return 0; }

// --- sched stubs ------------------------------------------------------------
inline int sched_getcpu(void) { return static_cast<int>(GetCurrentProcessorNumber()); }

// --- pthread stubs ----------------------------------------------------------
typedef HANDLE pthread_t;
inline int pthread_setname_np(pthread_t /*h*/, const char* /*name*/) {
  // The real Windows equivalent (SetThreadDescription) needs a wide string and
  // is purely diagnostic. Skipped for the AVX2 smoke path.
  return 0;
}
inline int pthread_getname_np(pthread_t /*h*/, char* buf, std::size_t buflen) {
  if (buf && buflen) buf[0] = '\0';
  return 0;
}

// --- aligned alloc ----------------------------------------------------------
// MSVC has no posix_memalign nor std::aligned_alloc (their aligned alloc must
// be paired with _aligned_free, not free()). Provide a uniform pair.
inline void* kt_aligned_alloc(std::size_t alignment, std::size_t size) {
  return _aligned_malloc(size, alignment);
}
inline void kt_aligned_free(void* p) { _aligned_free(p); }

inline int posix_memalign(void** out, std::size_t alignment, std::size_t size) {
  void* p = _aligned_malloc(size, alignment);
  if (!p) return 12;  // ENOMEM
  *out = p;
  return 0;
}

// --- hwloc minimal stubs ----------------------------------------------------
// We only stub the handful of hwloc calls used by worker_pool.cpp. All become
// no-ops; CPU pinning is dropped on Windows.
typedef struct hwloc_topology* hwloc_topology_t;
typedef struct hwloc_cpuset* hwloc_cpuset_t;
typedef hwloc_cpuset_t hwloc_bitmap_t;
typedef struct {
  hwloc_cpuset_t cpuset;
  hwloc_cpuset_t nodeset;
} hwloc_obj_struct;
typedef hwloc_obj_struct* hwloc_obj_t;

enum {
  HWLOC_OBJ_NUMANODE = 0,
  HWLOC_OBJ_CORE = 1,
  HWLOC_CPUBIND_STRICT = 0,
  HWLOC_CPUBIND_THREAD = 0,
  HWLOC_MEMBIND_BIND = 0,
  HWLOC_MEMBIND_THREAD = 0,
  HWLOC_MEMBIND_STRICT = 0,
  HWLOC_MEMBIND_BYNODESET = 0,
};

inline int hwloc_topology_init(hwloc_topology_t* t) {
  if (t) *t = nullptr;
  return 0;
}
inline int hwloc_topology_load(hwloc_topology_t /*t*/) { return 0; }
inline void hwloc_topology_destroy(hwloc_topology_t /*t*/) {}
inline hwloc_obj_t hwloc_get_obj_by_type(hwloc_topology_t /*t*/, int /*type*/, unsigned /*idx*/) { return nullptr; }
inline hwloc_obj_t hwloc_get_obj_inside_cpuset_by_type(hwloc_topology_t /*t*/, hwloc_cpuset_t /*set*/, int /*type*/,
                                                       unsigned /*idx*/) {
  return nullptr;
}
inline hwloc_bitmap_t hwloc_bitmap_alloc(void) { return nullptr; }
inline void hwloc_bitmap_free(hwloc_bitmap_t /*b*/) {}
inline int hwloc_bitmap_copy(hwloc_bitmap_t /*dst*/, hwloc_cpuset_t /*src*/) { return 0; }
inline void hwloc_bitmap_singlify(hwloc_bitmap_t /*b*/) {}
inline int hwloc_set_thread_cpubind(hwloc_topology_t /*t*/, pthread_t /*h*/, hwloc_cpuset_t /*set*/, int /*flags*/) {
  return 0;
}
inline int hwloc_get_thread_cpubind(hwloc_topology_t /*t*/, pthread_t /*h*/, hwloc_cpuset_t /*set*/, int /*flags*/) {
  return 0;
}
inline int hwloc_set_membind(hwloc_topology_t /*t*/, hwloc_cpuset_t /*set*/, int /*policy*/, int /*flags*/) { return 0; }

#endif  // _WIN32

#endif  // KT_CPU_BACKEND_COMPAT_POSIX_COMPAT_H
