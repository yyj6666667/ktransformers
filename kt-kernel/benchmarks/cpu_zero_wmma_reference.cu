// Measurement instrument only: expose unchanged GPU MMA intermediate results
// so a CPU implementation can be checked instruction by instruction.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

__global__ void mma_chain(const __nv_bfloat16* a, const __nv_bfloat16* b,
                          float* out, float* steps, int m, int n, int k) {
  const int row = blockIdx.x * 16, col = blockIdx.y * 16;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::fill_fragment(acc, 0.f);
  for (int start = 0; start < k; start += 16) {
    wmma::load_matrix_sync(af, a + row*k + start, k);
    wmma::load_matrix_sync(bf, b + col*k + start, k);
    wmma::mma_sync(acc, af, bf, acc);
    if (steps)
      wmma::store_matrix_sync(steps + (start/16)*m*n + row*n + col, acc, n, wmma::mem_row_major);
  }
  wmma::store_matrix_sync(out + row*n + col, acc, n, wmma::mem_row_major);
}

extern "C" int run_mma_chain(const void* a, const void* b, float* out, float* steps,
                             int m, int n, int k, void* stream) {
  mma_chain<<<dim3(m/16,n/16),32,0,(cudaStream_t)stream>>>(
      (const __nv_bfloat16*)a, (const __nv_bfloat16*)b, out, steps, m, n, k);
  return (int)cudaGetLastError();
}
