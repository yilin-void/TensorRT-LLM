#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "cutlass/numeric_types.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "tensorrt_llm/kernels/preQuantScaleKernel.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/kernelLauncher.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace wo = tensorrt_llm::kernels::weight_only;

void simple_assert(bool flag)
{
    if (!flag)
    {
        throw std::runtime_error("assert failed");
    }
}

struct CudaBuffer
{
    void* _data;
    int _size;

    CudaBuffer(int size_in_bytes)
        : _size(size_in_bytes)
    {
        cudaMalloc(&_data, _size);
    }

    template <typename T = void>
    T* data()
    {
        return reinterpret_cast<T*>(_data);
    }

    void copy_to(void* dst)
    {
        cudaMemcpy(dst, _data, _size, cudaMemcpyDeviceToHost);
    }

    void copy_from(void* src)
    {
        cudaMemcpy(_data, src, _size, cudaMemcpyHostToDevice);
    }

    ~CudaBuffer()
    {
        cudaFree(_data);
    }
};

template <typename T>
float compare(void* _pa, void* _pb, int size, float scale)
{
    auto pa = reinterpret_cast<T*>(_pa);
    auto pb = reinterpret_cast<T*>(_pb);
    float max_diff = 0.f, tot_diff = 0.f;
    float max_val = 0.f;
    int diff_cnt = 0;
    float threshold = 1e-7;
    for (int n = 0; n < size; ++n)
    {
        float va = static_cast<float>(pa[n]);
        float vb = static_cast<float>(pb[n]);
        max_val = std::max(max_val, vb);
        float diff = std::abs(va - vb);
        if (diff > threshold)
        {
            max_diff = std::max(max_diff, diff);
            tot_diff += diff;
            ++diff_cnt;
        }
    }
    float diff_thres = max_val * scale;
#if defined(ENABLE_BF16)
    if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        // bfloat16 has fewer mantissa digits than float16(10 bits for fp16 but only 7 bits for bf16), so the cumulative
        // error will be larger.
        diff_thres *= 3.f;
    }
    else
#endif
    {
        diff_thres *= 1.5f;
    }
    printf("max diff %f (diff threshold %f), avg diff %f, diff cnt %d/%d\n", max_diff, diff_thres, tot_diff / diff_cnt,
        diff_cnt, size);
    return max_diff <= diff_thres;
}

template <typename T1, typename T2>
void random_fill(std::vector<T1>& vec, T2 minv, T2 maxv)
{
    std::mt19937 gen(rand());
    std::uniform_real_distribution<float> dis(static_cast<float>(minv), static_cast<float>(maxv));
    for (auto& v : vec)
    {
        v = static_cast<T1>(dis(gen));
    }
}

__global__ void get_timestamps_kernel(uint64_t* global_timestamp, uint64_t* sm0_timestamp) {
  uint32_t smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  if (smid == 0) {
    uint64_t gts, lts;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gts));
    lts = clock64();

    *global_timestamp = gts;
    *sm0_timestamp = lts;
  }
}

void get_timestamps(int sms, uint64_t* global_timestamp, uint64_t* sm0_timestamp, cudaStream_t stream) {
    get_timestamps_kernel<<<sms, 1, 0, stream>>>(global_timestamp, sm0_timestamp);
}

int get_sm_count() {
    const int device_id = 0;
    static int num_sms = 0;
    if (num_sms == 0) {
        TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id));
    }
    return num_sms;
}

static constexpr int kElemPerAccess = 32;

__global__ void test_kernel1(float* weight, half* scales, half* zeros, int n, int k, int gs) {
    int kid = threadIdx.x;
    int nid = blockIdx.x;
    float4 w32 = reinterpret_cast<float4*>(weight)[nid * k / kElemPerAccess + kid];
    // half s = scales[kid * kElemPerAccess / gs * n + nid];
    // half z = zeros[kid * kElemPerAccess / gs * n + nid];
    
    float v = w32.x + w32.y + w32.z + w32.w;// + static_cast<float>(s) + static_cast<float>(z);
    weight[nid * k / kElemPerAccess + kid] = v;
}

void launch_test_kernel1(void* weight, void* scales, void* zeros, int n, int k, int gs, cudaStream_t s) {
    dim3 grid(n);
    dim3 block(k / kElemPerAccess);
    test_kernel1<<<grid, block, 0, s>>>(reinterpret_cast<float*>(weight), reinterpret_cast<half*>(scales), reinterpret_cast<half*>(zeros), n, k, gs);
}

template <typename Func>
float run_kernel(Func func, wo::Params& params, int warmup, int iter)
{
    int arch = tensorrt_llm::common::getSMVersion();
    int sm_count = get_sm_count();
    uint64_t* timestamps = nullptr;
    TLLM_CUDA_CHECK(cudaHostAlloc(&timestamps, 4 * sizeof(uint64_t), cudaHostAllocMapped));
    simple_assert(wo::is_supported(arch, params.type));
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    for (int i = 0; i < warmup; ++i)
    {
        func(params.weight, params.scales, params.zeros, params.n, params.k, params.groupsize, s);
    }
    get_timestamps(sm_count, &timestamps[0], &timestamps[1], s);
    cudaEventRecord(begin, s);
    for (int i = 0; i < iter; ++i)
    {
        func(params.weight, params.scales, params.zeros, params.n, params.k, params.groupsize, s);
    }
    cudaEventRecord(end, s);
    get_timestamps(sm_count, &timestamps[2], &timestamps[3], s);
    cudaEventSynchronize(end);
    cudaDeviceSynchronize();
    float time;
    cudaEventElapsedTime(&time, begin, end);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaStreamDestroy(s);
    float elapsed_ns = static_cast<float>(timestamps[2] - timestamps[0]) / iter;
    float elapsed_clocks = static_cast<float>(timestamps[3] - timestamps[1]) / iter;
    float clock_rate = elapsed_clocks / elapsed_ns;
    float mem_bw = 0.5 * params.n * params.k / elapsed_ns;
    printf("n %d, k %d, groupsize %d, avg elapsed time %.3f us, avg elapsed clocks %.3f, clock rate %.3f GHz, memory bandwidth %.3f GB/s\n", params.n, params.k, params.groupsize, elapsed_ns / 1000.f, elapsed_clocks, clock_rate, mem_bw);
    TLLM_CUDA_CHECK(cudaFreeHost(timestamps));
    return time / iter;
}

template <typename Func>
bool benchmark_and_verify(Func func,int m, int n, int k, int groupsize, int warmup, int iter)
{
    std::srand(20240123);
    simple_assert(m <= 16);
    simple_assert(groupsize == 64 || groupsize == 128);
    static constexpr int ASizeInBits = 16;
    static constexpr int WSizeInBits = 4;
    int gs_factor = groupsize == 0 ? 1 : groupsize;

    CudaBuffer d_act(m * k * ASizeInBits / 8);
    CudaBuffer d_act_scale(k * ASizeInBits / 8);
    CudaBuffer d_weight(k * n * WSizeInBits / 8);
    CudaBuffer d_scales(n * k / gs_factor * ASizeInBits / 8);
    CudaBuffer d_zeros(n * k / gs_factor * ASizeInBits / 8);
    CudaBuffer d_bias(n * ASizeInBits / 8);
    CudaBuffer d_out(m * n * ASizeInBits / 8);
    std::vector<half> h_act(m * k), h_act_scale(k);
    std::vector<uint8_t> h_weight(k * n);
    std::vector<half> h_scales(n * k), h_zeros(n * k), h_bias(n);
    std::vector<half> h_out1(m * n), h_out2(m * n);

    random_fill(h_act, -1.f, 1.f);
    random_fill(h_act_scale, -1.f, 1.f);
    random_fill(h_scales, -1.f, 1.f);
    random_fill(h_zeros, -1.f, 1.f);
    random_fill(h_bias, -1.f, 1.f);

    for (uint8_t& v : h_weight)
    {
        v = rand() % 256;
    }

    d_act.copy_from(h_act.data());
    d_act_scale.copy_from(h_act_scale.data());
    d_weight.copy_from(h_weight.data());
    d_scales.copy_from(h_scales.data());
    d_zeros.copy_from(h_zeros.data());
    d_bias.copy_from(h_bias.data());

    void* p_act_scale = nullptr;
    void* p_zeros = nullptr;
    void* p_bias = nullptr;

    if (groupsize != 0)
    {
        p_zeros = d_zeros.data();
        p_bias = d_bias.data();
        p_act_scale = d_act_scale.data();
    }
    wo::Params params(d_act.data(), p_act_scale, d_weight.data(), d_scales.data(), p_zeros, p_bias, d_out.data(), 1.f,
        m, n, k, groupsize, wo::KernelType::FP16Int4Groupwise);
    run_kernel(func, params, warmup, iter);
    return true;
}

TEST(WeightOnly, Benchmark)
{
    int const arch = tensorrt_llm::common::getSMVersion();
    bool pass;
    static const char* str_ncu_profile = std::getenv("NCU_PROFILE");
    int warmup = 10, iter = 30;
    if (str_ncu_profile)
    {
        warmup = 0;
        iter = 1;
    }
    printf("warmup %d, iter %d\n", warmup, iter);
    std::vector<int> ms{1};
    std::vector<int> ns{4096};
    std::vector<int> ks{4096};
    for (auto m : ms)
    {
        for (auto n : ns)
        {
            for (auto k : ks)
            {
                pass = benchmark_and_verify(launch_test_kernel1, m, n, k, 64, warmup, iter);
                EXPECT_TRUE(pass);
            }
        }
    }
}
