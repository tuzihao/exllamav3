#include "gdn_128.cuh"

void recurrent(const at::Tensor& qkv, const at::Tensor& g, const at::Tensor& beta,
               at::Tensor& state, at::Tensor& out, int nk, int nv, int kd, int vd,
               const std::optional<at::Tensor>& slots, bool history)
{
    TORCH_CHECK(kd == 128 && vd == 128 && nk > 0 && nv % nk == 0, "GDN128 only");
    const c10::cuda::CUDAGuard guard(qkv.device());
    for (const auto& t : {qkv, g, beta, state, out}) {
        TORCH_CHECK(t.is_cuda() && t.device() == qkv.device() && t.is_contiguous(),
                    "Contiguous tensors on the same CUDA device required");
    }
    TORCH_CHECK(qkv.scalar_type() == at::kBFloat16 && g.scalar_type() == at::kFloat &&
                beta.scalar_type() == at::kBFloat16 && state.scalar_type() == at::kFloat &&
                out.scalar_type() == at::kBFloat16, "GDN dtype mismatch");
    TORCH_CHECK(qkv.dim() == 3 && state.dim() == 5 && out.dim() == 4, "GDN rank mismatch");
    int b = qkv.size(0), s = qkv.size(1), hs = state.size(1);
    TORCH_CHECK(b > 0 && s > 0 && qkv.size(2) == (2 * nk + nv) * 128, "GDN QKV shape mismatch");
    TORCH_CHECK(g.sizes() == at::IntArrayRef({b, s, nv}) && beta.sizes() == g.sizes(), "GDN gate shape mismatch");
    TORCH_CHECK(state.size(0) >= (slots ? 1 : b) && hs >= (history ? s : 1) &&
                state.size(2) == nv && state.size(3) == kd && state.size(4) == vd, "GDN state shape mismatch");
    TORCH_CHECK(out.sizes() == at::IntArrayRef({b, s, nv, vd}), "GDN output shape mismatch");
    TORCH_CHECK(!slots || (slots->device() == qkv.device() && slots->is_contiguous() &&
                slots->scalar_type() == at::kInt && slots->dim() == 1 && slots->numel() == b), "GDN slots mismatch");
    int vs = (b == 1 && nv <= 64) ? 4 : 1;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid(b, nv, vs), block(128, SUBK);
    const int* sp = slots ? slots->data_ptr<int>() : nullptr;
    float scale = 1.0f / sqrtf(kd);
#define ARGS (const bfloat16*)qkv.data_ptr(), g.data_ptr<float>(), (const bfloat16*)beta.data_ptr(), state.data_ptr<float>(), (bfloat16*)out.data_ptr(), b, s, nk, nv, kd, vd, scale, sp, hs
    if (history) {
        if (vs == 4) cuda_recurrent_gated_delta_rule_kernel_128<true, 4><<<grid, block, 0, stream>>>(ARGS);
        else cuda_recurrent_gated_delta_rule_kernel_128<true, 1><<<grid, block, 0, stream>>>(ARGS);
    } else {
        if (vs == 4) cuda_recurrent_gated_delta_rule_kernel_128<false, 4><<<grid, block, 0, stream>>>(ARGS);
        else cuda_recurrent_gated_delta_rule_kernel_128<false, 1><<<grid, block, 0, stream>>>(ARGS);
    }
#undef ARGS
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// One writer per output element; expert contributions are visited in a fixed order.
__global__ void gather_slots_kernel(const float* slots, const int64_t* order,
                                    float* output, int rows, int width, int top_k)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (col >= width || row >= rows) return;
    float sum = 0.0f;
    for (int k = 0; k < top_k; ++k)
        sum += slots[order[row * top_k + k] * width + col];
    output[row * width + col] += sum;
}

void gather_slots(const at::Tensor& slots, const at::Tensor& order, at::Tensor& output)
{
    TORCH_CHECK(slots.dim() == 2 && order.dim() == 2 && output.dim() == 2, "MoE rank mismatch");
    TORCH_CHECK(slots.scalar_type() == at::kFloat && output.scalar_type() == at::kFloat &&
                order.scalar_type() == at::kLong, "MoE dtype mismatch");
    for (const auto& t : {slots, order, output})
        TORCH_CHECK(t.is_cuda() && t.device() == slots.device() && t.is_contiguous(), "MoE device/stride mismatch");
    TORCH_CHECK(order.size(0) == output.size(0) && slots.size(1) == output.size(1) &&
                order.numel() == slots.size(0) && order.size(1) > 0, "MoE shape mismatch");
    const c10::cuda::CUDAGuard guard(slots.device());
    dim3 grid((output.size(1) + 255) / 256, output.size(0));
    gather_slots_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        slots.data_ptr<float>(), order.data_ptr<int64_t>(), output.data_ptr<float>(),
        output.size(0), output.size(1), order.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("recurrent", &recurrent);
    m.def("gather_slots", &gather_slots);
}
