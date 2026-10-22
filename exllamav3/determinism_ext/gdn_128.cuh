// GDN-128 from EXL3 0.0.37 (MIT), reduction backport of 900cf17.
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
using bfloat16 = __nv_bfloat16;
#define SUBK 4
template <bool save_history, int V_SPLIT>
__global__ __launch_bounds__(128 * SUBK)
void cuda_recurrent_gated_delta_rule_kernel_128
(
                                                // k_head_dim = v_head_dim = 128
    const bfloat16* __restrict__ mixed_qkv,     // [bsz, seqlen, (k_dim + k_dim + v_dim)]
    const float* __restrict__ g,                // [bsz, seqlen, (group * num_k_heads)]
    const bfloat16* __restrict__ beta,          // [bsz, seqlen, (group * num_k_heads)]
    float* __restrict__ recurrent_state,        // [num_slots, max_history + 1, (group * num_k_heads), 128, 128]
    bfloat16* __restrict__ core_attn_out,       // [bsz, seqlen, num_v_heads, 128]
    const int bsz,
    const int seqlen,
    const int num_k_heads,
    const int num_v_heads,
    const int k_head_dim,
    const int v_head_dim,
    const float scale,
    const int* __restrict__ slots,              // [bsz]
    const int history_stride                    // max_history + 1
)
{
    constexpr int HEAD_DIM = 128;
    constexpr int V_CHUNK_DIM = HEAD_DIM / V_SPLIT;
    constexpr int BTS = HEAD_DIM / SUBK;

    int group = num_v_heads / num_k_heads;
    constexpr size_t HEAD_STATE_SIZE = HEAD_DIM * HEAD_DIM;
    const size_t state_size = group * num_k_heads * HEAD_STATE_SIZE;
    const size_t slot_size = (size_t) history_stride * state_size;

    int bi = blockIdx.x;
    mixed_qkv +=        bi * seqlen * (3 * HEAD_DIM * num_k_heads + HEAD_DIM * (num_v_heads - num_k_heads));
    g +=                bi * seqlen * (group * num_k_heads);
    beta +=             bi * seqlen * (group * num_k_heads);
    int state_slot = slots ? slots[bi] : bi;
    float* slot_state = recurrent_state + (size_t) state_slot * slot_size;
    float* final_state = slot_state;
    core_attn_out +=    bi * seqlen * num_v_heads * HEAD_DIM;

    int t = threadIdx.x;
    int bt = threadIdx.y;
    int lane = t % 32;
    int warp = t / 32;
    int head = blockIdx.y;
    int k_head = head / group;
    int v_chunk = blockIdx.z;
    int v_start = v_chunk * V_CHUNK_DIM;

    __shared__ float sh_red[2][HEAD_DIM / 32];
    __shared__ float sh_k[HEAD_DIM];
    __shared__ float sh_q[HEAD_DIM];
    __shared__ float sh_dot1[SUBK][HEAD_DIM];
    __shared__ float sh_dot2[SUBK][HEAD_DIM];

    for (int s = 0; s < seqlen; ++s)
    {
        const bfloat16* gl_q = mixed_qkv + k_head * HEAD_DIM;
        const bfloat16* gl_k = mixed_qkv + (num_k_heads + k_head) * HEAD_DIM;
        const bfloat16* gl_v = mixed_qkv + (2 * num_k_heads * HEAD_DIM) + head * HEAD_DIM + v_start;
        bfloat16* out = core_attn_out + head * HEAD_DIM + v_start;

        float* gl_rs_r;
        float* gl_rs_w;
        if constexpr (save_history)
        {
            bool first = (s == 0);
            bool last = (s == seqlen - 1);
            float* history_r = first ? nullptr : slot_state + (size_t) s * state_size;
            float* history_w = last  ? final_state : slot_state + (size_t) (s + 1) * state_size;
            gl_rs_r = first ? final_state + head * HEAD_STATE_SIZE
                            : history_r   + head * HEAD_STATE_SIZE;
            gl_rs_w = history_w           + head * HEAD_STATE_SIZE;
        }
        else
        {
            gl_rs_r = final_state + head * HEAD_STATE_SIZE;
            gl_rs_w = gl_rs_r;
        }

        float q = __bfloat162float(gl_q[t]);
        float k = __bfloat162float(gl_k[t]);

        float sumq = q * q;
        float sumk = k * k;
        #pragma unroll
        for(int offset = 16; offset > 0; offset /= 2)
        {
            sumq += __shfl_xor_sync(0xffffffff, sumq, offset);
            sumk += __shfl_xor_sync(0xffffffff, sumk, offset);
        }
        if (lane == 0)
        {
            sh_red[0][warp] = sumq;
            sh_red[1][warp] = sumk;
        }
        __syncthreads();

        sumq = lane < HEAD_DIM / 32 ? sh_red[0][lane] : 0.0f;
        sumk = lane < HEAD_DIM / 32 ? sh_red[1][lane] : 0.0f;
        #pragma unroll
        for(int offset = 16; offset > 0; offset /= 2)
        {
            sumq += __shfl_xor_sync(0xffffffff, sumq, offset);
            sumk += __shfl_xor_sync(0xffffffff, sumk, offset);
        }

        q = q * rsqrtf(sumq + 1e-6f);
        k = k * rsqrtf(sumk + 1e-6f);
        sh_k[t] = k;
        sh_q[t] = q;

        __syncthreads();

        if (t < V_CHUNK_DIM)
        {
            float sum = 0.0f;
            float* sh_k_rd = sh_k + bt * BTS;
            float* rs_rd = gl_rs_r + v_start + t + bt * BTS * HEAD_DIM;

            #pragma unroll
            for (int i = 0; i < HEAD_DIM / 8 / SUBK; ++i)
            {
                #pragma unroll
                for (int j = 0; j < 8; ++j, rs_rd += HEAD_DIM, sh_k_rd++)
                    sum = sum + *sh_k_rd * *rs_rd;
            }
            sh_dot1[bt][t] = sum;
        }
        __syncthreads();

        if (t < V_CHUNK_DIM)
        {
            float g_h = __expf(g[head]);
            float beta_h = __bfloat162float(beta[head]);
            float dot1 = 0.0f;
            #pragma unroll
            for (int s = 0; s < SUBK; ++s) dot1 += sh_dot1[s][t];
            float v = __bfloat162float(gl_v[t]) - dot1 * g_h;
            float v_out = 0.0f;
            float* sh_k_rd = sh_k + bt * BTS;
            float* sh_q_rd = sh_q + bt * BTS;
            float* rs_r = gl_rs_r + v_start + t + bt * BTS * HEAD_DIM;
            float* rs_w = gl_rs_w + v_start + t + bt * BTS * HEAD_DIM;

            #pragma unroll
            for (int i = 0; i < HEAD_DIM / 8 / SUBK; ++i)
            {
                #pragma unroll
                for (int j = 0; j < 8; ++j, rs_r += HEAD_DIM, rs_w += HEAD_DIM, sh_k_rd++, sh_q_rd++)
                {
                    float state = *rs_r;
                    state = state * g_h + *sh_k_rd * v * beta_h;
                    *rs_w = state;
                    v_out = v_out + *sh_q_rd * state;
                }
            }
            sh_dot2[bt][t] = v_out;
        }
        __syncthreads();

        if (t < V_CHUNK_DIM && bt == 0)
        {
            float v_out = 0.0f;
            #pragma unroll
            for (int s = 0; s < SUBK; ++s) v_out += sh_dot2[s][t];
            out[t] = __float2bfloat16_rz(v_out * scale);
        }

        mixed_qkv +=        2 * HEAD_DIM * num_k_heads + HEAD_DIM * num_v_heads;
        g +=                num_v_heads;
        beta +=             num_v_heads;
        core_attn_out +=    num_v_heads * HEAD_DIM;
    }
}
