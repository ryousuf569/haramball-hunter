// physics/tti.py
#define V_MAX 5.0f
#define A_MAX 7.0f
#define REACTION_TIME 0.54f
#define INTERCEPT_UNCERTAINTY 0.45f

// physics/ppcf.py
#define ATTACKER_CONTROL_RATE 4.30f
#define DEFENDER_CONTROL_RATE (4.30f * 1.72f)
#define INTEGRATION_TIMESTEP 0.08
#define INTEGRATION_HORIZON 10.0
#define PPCF_CUTOFF 0.99f

// -(pi / sqrt(3)), the logistic slope in intercept_probability_vec
#define INTERCEPT_A (-1.8137993642342178f)

__device__ __forceinline__ float derive_t(float d_mag, float v_parallel){
    const float t_1 = (V_MAX - v_parallel) / A_MAX;
    const float d_1 = (v_parallel * t_1) + (0.5f * A_MAX * t_1 * t_1);

    if (d_1 > d_mag) {
        const float disc = v_parallel * v_parallel + 2.0f * A_MAX * d_mag;
        return (-v_parallel + sqrtf(disc)) / A_MAX;
    }
    return t_1 + (d_mag - d_1) / V_MAX;
}

__device__ __forceinline__ float tti(float px, float py, float vx, float vy, float tx, float ty){
    const float dx = tx - px;
    const float dy = ty - py;
    const float D = sqrtf(dx * dx + dy * dy);

    if (D == 0.0f) {
        return 0.0f;
    }

    const float ux = dx / D;
    const float uy = dy / D;
    const float u_mag = sqrtf(ux * ux + uy * uy);
    const float v_parallel = (vx * ux + vy * uy) / u_mag;

    float t;
    if (v_parallel >= 0.0f) {
        t = derive_t(D, v_parallel);
    } else {
        const float t_brake = fabsf(v_parallel) / A_MAX;
        const float d_brake = (fabsf(v_parallel) * t_brake) - (0.5f * A_MAX * t_brake * t_brake);
        t = derive_t(D + d_brake, 0.0f) + t_brake;
    }

    return t + REACTION_TIME;
}

__device__ __forceinline__ float intercept_probability(float T, float react_exp){
    const float b = (T - react_exp) / INTERCEPT_UNCERTAINTY;
    return 1.0f / (1.0f + expf(INTERCEPT_A * b));
}

extern "C" __global__
void ppcf_grid(const float* __restrict__ targets, const float* __restrict__ positions, const float* __restrict__ velocities,
               const float* __restrict__ lam, float* __restrict__ ppcf, int n_cells, int n_players) {
    extern __shared__ float smem[];
    float* s_px  = smem;
    float* s_py  = s_px + n_players;
    float* s_vx  = s_py + n_players;
    float* s_vy  = s_vx + n_players;
    float* s_lam = s_vy + n_players;
    float* s_tti = s_lam + n_players;

    for (int p = threadIdx.x; p < n_players; p += blockDim.x) {
        s_px[p]  = positions[2 * p];
        s_py[p]  = positions[2 * p + 1];
        s_vx[p]  = velocities[2 * p];
        s_vy[p]  = velocities[2 * p + 1];
        s_lam[p] = lam[p];
    }
    __syncthreads();

    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= n_cells) {
        return;
    }

    const float tx = targets[2 * cell];
    const float ty = targets[2 * cell + 1];

    float* my_ppcf = ppcf + (size_t)cell * n_players;

    for (int p = 0; p < n_players; ++p) {
        s_tti[p * blockDim.x + threadIdx.x] =
            tti(s_px[p], s_py[p], s_vx[p], s_vy[p], tx, ty);
        my_ppcf[p] = 0.0f;
    }

    float total = 0.0f;

    for (double t = 0.0; t < INTEGRATION_HORIZON; t += INTEGRATION_TIMESTEP) {
        const float t_f      = (float)t;
        const float snapshot = 1.0f - total;

        float step_sum = 0.0f;
        for (int p = 0; p < n_players; ++p) {
            const float f = intercept_probability(
                t_f, s_tti[p * blockDim.x + threadIdx.x]);
            const float increment =
                snapshot * f * s_lam[p] * (float)INTEGRATION_TIMESTEP;

            my_ppcf[p] += increment;
            step_sum += increment;
        }

        total += step_sum;

        if (total >= PPCF_CUTOFF) {
            break;
        }
    }
}

extern "C" __global__
void ppcf_ball_tti(const float* __restrict__ positions, const float* __restrict__ velocities, const float* __restrict__ ball_pos,
                   float* __restrict__ i_p, int n_players) {

    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (p >= n_players) {
        return;
    }

    i_p[p] = tti(positions[2 * p], positions[2 * p + 1], velocities[2 * p], velocities[2 * p + 1], ball_pos[0], ball_pos[1]);
}
