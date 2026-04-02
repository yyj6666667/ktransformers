/**
 * @Description  : Cross-platform NUMA compatibility layer using hwloc.
 *                 Replaces direct libnuma calls so the code builds on both
 *                 Linux and Windows with a single hwloc dependency.
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_NUMA_COMPAT_H
#define CPUINFER_NUMA_COMPAT_H

#include <hwloc.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---- Memory alignment compat ------------------------------------------------
#ifdef _WIN32
#include <malloc.h>
static inline int compat_posix_memalign(void** memptr, size_t alignment, size_t size) {
  void* p = _aligned_malloc(size, alignment);
  if (!p) return errno;
  *memptr = p;
  return 0;
}
static inline void compat_aligned_free(void* ptr) { _aligned_free(ptr); }
#else
static inline int compat_posix_memalign(void** memptr, size_t alignment, size_t size) {
  return posix_memalign(memptr, alignment, size);
}
static inline void compat_aligned_free(void* ptr) { free(ptr); }
#endif

// ---- NUMA helpers using hwloc only -----------------------------------------

/**
 * Get the number of NUMA nodes on this system (replaces numa_num_configured_nodes).
 */
inline int hwloc_numa_num_nodes() {
  hwloc_topology_t topology;
  hwloc_topology_init(&topology);
  hwloc_topology_load(topology);
  int depth = hwloc_get_type_depth(topology, HWLOC_OBJ_NUMANODE);
  int count = (depth == HWLOC_TYPE_DEPTH_UNKNOWN) ? 1 : hwloc_get_nbobjs_by_depth(topology, depth);
  hwloc_topology_destroy(topology);
  return count;
}

/**
 * Get the current CPU's logical index (replaces sched_getcpu).
 */
inline int hwloc_get_current_cpu() {
  hwloc_topology_t topology;
  hwloc_topology_init(&topology);
  hwloc_topology_load(topology);

  hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
  hwloc_get_cpubind(topology, cpuset, HWLOC_CPUBIND_THREAD);
  int cpu = hwloc_bitmap_first(cpuset);
  hwloc_bitmap_free(cpuset);
  hwloc_topology_destroy(topology);
  return (cpu >= 0) ? cpu : 0;
}

/**
 * Get the NUMA node of a given CPU (replaces numa_node_of_cpu).
 */
inline int hwloc_numa_node_of_cpu(int cpu) {
  hwloc_topology_t topology;
  hwloc_topology_init(&topology);
  hwloc_topology_load(topology);

  hwloc_obj_t pu = hwloc_get_pu_obj_by_os_index(topology, (unsigned)cpu);
  int node = 0;
  if (pu) {
    hwloc_obj_t parent = pu->parent;
    while (parent) {
      if (parent->type == HWLOC_OBJ_NUMANODE) {
        node = (int)parent->logical_index;
        break;
      }
      // On many topologies NUMA nodes are not direct ancestors of PU but are
      // memory children of a Group/Package.  Walk up and check memory children.
      if (parent->memory_arity > 0) {
        for (unsigned i = 0; i < parent->memory_arity; i++) {
          hwloc_obj_t mem = parent->memory_first_child;
          while (mem) {
            if (mem->type == HWLOC_OBJ_NUMANODE) {
              node = (int)mem->logical_index;
              goto done;
            }
            mem = mem->next_sibling;
          }
        }
      }
      parent = parent->parent;
    }
  }
done:
  hwloc_topology_destroy(topology);
  return node;
}

/**
 * Get the NUMA node of the *current* CPU (common pattern replacing
 * numa_node_of_cpu(sched_getcpu())).
 */
inline int hwloc_current_numa_node() {
  return hwloc_numa_node_of_cpu(hwloc_get_current_cpu());
}

/**
 * Bind the calling thread's memory allocation policy to a NUMA node
 * (replaces the set_to_numa helper that used numa_bind).
 */
inline void hwloc_set_to_numa(int numa_id) {
  hwloc_topology_t topology;
  hwloc_topology_init(&topology);
  hwloc_topology_load(topology);

  hwloc_obj_t obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_NUMANODE, numa_id);
  if (!obj) {
    fprintf(stderr, "NUMA node %d not found.\n", numa_id);
    hwloc_topology_destroy(topology);
    return;
  }

  // Bind memory
  int ret = hwloc_set_membind(topology, obj->nodeset, HWLOC_MEMBIND_BIND,
                              HWLOC_MEMBIND_THREAD | HWLOC_MEMBIND_STRICT | HWLOC_MEMBIND_BYNODESET);
  if (ret != 0) {
    perror("hwloc_set_membind (set_to_numa)");
  }

  // Bind CPU
  if (obj->cpuset) {
    ret = hwloc_set_cpubind(topology, obj->cpuset, HWLOC_CPUBIND_THREAD);
    if (ret != 0) {
      perror("hwloc_set_cpubind (set_to_numa)");
    }
  }

  hwloc_topology_destroy(topology);
}

// ---- Thread naming compat ---------------------------------------------------
#ifdef _WIN32
#include <windows.h>
static inline void compat_set_thread_name(void* native_handle, const char* name) {
  // Convert char* to wchar_t* for SetThreadDescription
  int len = MultiByteToWideChar(CP_UTF8, 0, name, -1, NULL, 0);
  if (len > 0) {
    wchar_t* wname = (wchar_t*)_alloca(len * sizeof(wchar_t));
    MultiByteToWideChar(CP_UTF8, 0, name, -1, wname, len);
    SetThreadDescription((HANDLE)native_handle, wname);
  }
}
static inline void compat_get_thread_name(void* native_handle, char* name, size_t len) {
  PWSTR wname = NULL;
  HRESULT hr = GetThreadDescription((HANDLE)native_handle, &wname);
  if (SUCCEEDED(hr) && wname) {
    WideCharToMultiByte(CP_UTF8, 0, wname, -1, name, (int)len, NULL, NULL);
    LocalFree(wname);
  } else {
    if (len > 0) name[0] = '\0';
  }
}
#else
#include <pthread.h>
static inline void compat_set_thread_name(pthread_t native_handle, const char* name) {
  pthread_setname_np(native_handle, name);
}
static inline void compat_get_thread_name(pthread_t native_handle, char* name, size_t len) {
  pthread_getname_np(native_handle, name, len);
}
#endif

#endif  // CPUINFER_NUMA_COMPAT_H
