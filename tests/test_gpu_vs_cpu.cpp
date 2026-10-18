// GPU(MKX)バッチパイプライン vs CPU(MuJoCo C)参照の閾値付き比較。旧test_metal_synthの代替で、forward/collision/solver/eulerを通しで検証する。
// 検証: 1/10/T ステップ後のqpos・qvel誤差、NaN/Inf無し、同一初期状態envの決定性。
// Usage: test_gpu_vs_cpu <model.xml> [tol_1step] [tol_10step] [tol_T]   (T=100, 誤差は全envのmax|diff|)
//
// 軌道モード(--seconds指定時): mesh付き実ロボット向け。決定的な入力で数秒走らせ、関節角と指定bodyのワールド位置・姿勢(mj_kinematics)のCPU/GPU差の最大値を閾値判定する。
// Usage: test_gpu_vs_cpu <model.xml> --seconds S [--lift dz] [--ctrl zero|sin] [--bodies a,b,..]
//          [--tol-joint rad] [--tol-pos m] [--tol-rot rad] [--dump file] [--fps 30]
// --dumpはscripts/render_compare.py用: int32 nq, int32 nframes, float32 fps のあとにCPU qpos[nframes*nq], GPU qpos[nframes*nq]。

#include "test_utils.h"
#include <random>
#include <vector>

static const int B = 8;
static const int T = 100;

static double max_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return INFINITY;
        m = std::max(m, std::fabs((double)a[i] - b[i]));
    }
    return m;
}

static std::vector<std::string> split_csv(const char* s) {
    std::vector<std::string> out;
    std::string cur;
    for (const char* c = s; ; c++) {
        if (*c == ',' || !*c) { if (!cur.empty()) out.push_back(cur); cur.clear(); if (!*c) break; }
        else cur += *c;
    }
    return out;
}

static int run_trajectory(int argc, char** argv) {
    double seconds = 0, lift = 0, tol_joint = 1e-2, tol_pos = 1e-2, tol_rot = 1e-2, fps = 30;
    bool ctrl_sin = false;
    const char* dump = nullptr;
    std::vector<std::string> bodies;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char* { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", argv[i]); exit(1); } return argv[++i]; };
        if (a == "--seconds") seconds = atof(next());
        else if (a == "--lift") lift = atof(next());
        else if (a == "--ctrl") ctrl_sin = std::string(next()) == "sin";
        else if (a == "--bodies") bodies = split_csv(next());
        else if (a == "--tol-joint") tol_joint = atof(next());
        else if (a == "--tol-pos") tol_pos = atof(next());
        else if (a == "--tol-rot") tol_rot = atof(next());
        else if (a == "--dump") dump = next();
        else if (a == "--fps") fps = atof(next());
    }

    printf("=== test_gpu_vs_cpu trajectory (%s) ===\n", argv[1]);
    auto* model = mjmlx_load_model(argv[1]);
    if (!model) { fprintf(stderr, "FAIL: load\n"); return 1; }
    auto* mj = (const mjModel*)mjmlx_get_mj_model(model);
    auto info = mjmlx_model_info(model);
    int nq = info.nq, nv = info.nv, nu = info.nu;
    const int B = 2;
    int T = (int)std::lround(seconds / mj->opt.timestep);
    int stride = std::max(1, (int)std::lround(1.0 / (fps * mj->opt.timestep)));
    bool floating = mj->njnt > 0 && mj->jnt_type[0] == mjJNT_FREE;
    int jq0 = floating ? 7 : 0;  // 関節角として比較するqposの先頭
    std::vector<int> body_ids;
    for (auto& n : bodies) {
        int id = mj_name2id(mj, mjOBJ_BODY, n.c_str());
        if (id < 0) { fprintf(stderr, "FAIL: body %s not found\n", n.c_str()); return 1; }
        body_ids.push_back(id);
    }
    printf("nq=%d nv=%d nu=%d dt=%g T=%d floating=%d ctrl=%s lift=%g\n", nq, nv, nu, mj->opt.timestep, T, (int)floating, ctrl_sin ? "sin" : "zero", lift);

    MjmlxBatchedConfig cg = {};
    cg.num_envs = B; cg.use_gpu = 1; cg.integrator = MJMLX_INTEGRATOR_EULER;
    MjmlxBatchedConfig cc = cg; cc.use_gpu = 0;
    auto* gpu = mjmlx_batched_create(model, &cg);
    auto* cpu = mjmlx_batched_create(model, &cc);
    std::vector<int> mask(B, 1);
    mjmlx_batched_reset(cpu, mask.data());
    mjmlx_batched_reset(gpu, mask.data());
    std::vector<float> q(B * nq), v(B * nv);
    int a, b;
    mjmlx_batched_get_state(cpu, q.data(), v.data(), &a, &b);
    if (floating) for (int e = 0; e < B; e++) q[e * nq + 2] += (float)lift;
    mjmlx_batched_set_state(gpu, q.data(), v.data());
    mjmlx_batched_set_state(cpu, q.data(), v.data());

    mjData* d = mj_makeData(mj);
    auto fk = [&](const float* qp, std::vector<double>& pos, std::vector<double>& quat) {
        for (int i = 0; i < nq; i++) d->qpos[i] = qp[i];
        mj_kinematics(mj, d);
        for (int id : body_ids) {
            for (int k = 0; k < 3; k++) pos.push_back(d->xpos[3 * id + k]);
            for (int k = 0; k < 4; k++) quat.push_back(d->xquat[4 * id + k]);
        }
    };

    std::vector<float> ctrl(B * nu, 0.f), gq(B * nq), cq(B * nq), gv(B * nv), cv(B * nv);
    std::vector<float> dump_c, dump_g;
    double max_joint = 0, max_pos = 0, max_rot = 0;
    bool finite = true;
    for (int t = 1; t <= T; t++) {
        if (ctrl_sin && nu > 0) {
            // 位置サーボは現在の関節角(qpos0)を中心に、t=0でちょうど中心になる正弦波(ctrlrange幅の15%、トルク直接指令のmotorは3%)を送る。
            // ctrlrangeの中央から始めると初期に飽和する巨大ステップになり、過渡応答の差が支配的になるため。
            for (int j = 0; j < nu; j++) {
                double lo = -1, hi = 1, base = 0;
                if (mj->actuator_ctrllimited[j]) { lo = mj->actuator_ctrlrange[2 * j]; hi = mj->actuator_ctrlrange[2 * j + 1]; base = 0.5 * (lo + hi); }
                bool servo = mj->actuator_trntype[j] == mjTRN_JOINT && mj->actuator_biastype[j] == mjBIAS_AFFINE && mj->actuator_biasprm[mjNBIAS * j + 1] != 0;
                if (servo) base = mj->qpos0[mj->jnt_qposadr[mj->actuator_trnid[2 * j]]];
                double ph = 0.7 * j, w = 2 * M_PI * 0.5 * t * mj->opt.timestep;
                double val = base + (servo ? 0.15 : 0.03) * (hi - lo) * (std::sin(w + ph) - std::sin(ph));
                val = std::min(hi, std::max(lo, val));
                for (int e = 0; e < B; e++) ctrl[e * nu + j] = (float)val;
            }
        }
        mjmlx_batched_step(gpu, nu ? ctrl.data() : nullptr);
        mjmlx_batched_step(cpu, nu ? ctrl.data() : nullptr);
        if (t % stride != 0 && t != T) continue;
        mjmlx_batched_get_state(gpu, gq.data(), gv.data(), &a, &b);
        mjmlx_batched_get_state(cpu, cq.data(), cv.data(), &a, &b);
        for (float x : gq) finite &= std::isfinite(x);
        for (float x : cq) finite &= std::isfinite(x);
        if (!finite) break;
        double fj = 0;
        for (int e = 0; e < B; e++) for (int j = jq0; j < nq; j++) fj = std::max(fj, (double)std::fabs(gq[e * nq + j] - cq[e * nq + j]));
        std::vector<double> gp, gqt, cp, cqt;
        fk(gq.data(), gp, gqt);
        fk(cq.data(), cp, cqt);
        double fp = 0, fr = 0;
        for (size_t i = 0; i < body_ids.size(); i++) {
            double dp[3] = {gp[3 * i] - cp[3 * i], gp[3 * i + 1] - cp[3 * i + 1], gp[3 * i + 2] - cp[3 * i + 2]};
            fp = std::max(fp, std::sqrt(dp[0] * dp[0] + dp[1] * dp[1] + dp[2] * dp[2]));
            double w[3];
            mju_subQuat(w, &gqt[4 * i], &cqt[4 * i]);
            fr = std::max(fr, std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]));
        }
        max_joint = std::max(max_joint, fj); max_pos = std::max(max_pos, fp); max_rot = std::max(max_rot, fr);
        if (dump) {
            dump_c.insert(dump_c.end(), cq.begin(), cq.begin() + nq);
            dump_g.insert(dump_g.end(), gq.begin(), gq.begin() + nq);
        }
        if ((t - 1) * mj->opt.timestep - std::floor((t - 1) * mj->opt.timestep) > t * mj->opt.timestep - std::floor(t * mj->opt.timestep) || t == T || t == stride)
            printf("t=%5.2fs: joint=%.3e rad  body pos=%.3e m  rot=%.3e rad\n", t * mj->opt.timestep, fj, fp, fr);
    }
    printf("max over trajectory: joint=%.3e (tol %.1e)  pos=%.3e (tol %.1e)  rot=%.3e (tol %.1e)\n", max_joint, tol_joint, max_pos, tol_pos, max_rot, tol_rot);

    if (dump) {
        FILE* f = fopen(dump, "wb");
        int32_t hdr[2] = {nq, (int32_t)(dump_c.size() / nq)};
        float fp32 = (float)(1.0 / (stride * mj->opt.timestep));
        if (!f || fwrite(hdr, 4, 2, f) != 2 || fwrite(&fp32, 4, 1, f) != 1 || fwrite(dump_c.data(), 4, dump_c.size(), f) != dump_c.size() || fwrite(dump_g.data(), 4, dump_g.size(), f) != dump_g.size()) { fprintf(stderr, "FAIL: dump\n"); return 1; }
        fclose(f);
        printf("dumped %d frames (%.1f fps) to %s\n", hdr[1], fp32, dump);
    }

    TEST_BEGIN("traj_finite");
    CHECK(finite, "NaN/Inf in CPU or GPU state");
    TEST_END();
    TEST_BEGIN("traj_joint_angle");
    CHECK_LT(max_joint, tol_joint, "joint angle diff");
    TEST_END();
    TEST_BEGIN("traj_body_position");
    CHECK_LT(max_pos, tol_pos, "body position diff");
    TEST_END();
    TEST_BEGIN("traj_body_orientation");
    CHECK_LT(max_rot, tol_rot, "body orientation diff");
    TEST_END();

    mj_deleteData(d);
    mjmlx_batched_free(gpu);
    mjmlx_batched_free(cpu);
    TEST_EXIT();
}

int main(int argc, char** argv) {
    for (int i = 2; i < argc; i++) if (!strcmp(argv[i], "--seconds")) return run_trajectory(argc, argv);
    if (argc < 2) { fprintf(stderr, "Usage: test_gpu_vs_cpu <model.xml> [tol_1] [tol_10] [tol_T]\n"); return 1; }
    double tol1 = argc > 2 ? atof(argv[2]) : 1e-2;
    double tol10 = argc > 3 ? atof(argv[3]) : 5e-2;
    double tolT = argc > 4 ? atof(argv[4]) : 5e-1;

    printf("=== test_gpu_vs_cpu (%s) ===\n", argv[1]);
    auto* model = mjmlx_load_model(argv[1]);
    if (!model) { fprintf(stderr, "FAIL: load\n"); return 1; }
    auto info = mjmlx_model_info(model);
    int nq = info.nq, nv = info.nv, nu = info.nu;
    printf("nq=%d nv=%d nu=%d\n", nq, nv, nu);

    MjmlxBatchedConfig cg = {};
    cg.num_envs = B; cg.use_gpu = 1; cg.integrator = MJMLX_INTEGRATOR_EULER;
    MjmlxBatchedConfig cc = cg; cc.use_gpu = 0;
    auto* gpu = mjmlx_batched_create(model, &cg);
    auto* cpu = mjmlx_batched_create(model, &cc);

    // env0..B/2-1 と B/2..B-1 は同一の摂動(決定性チェック用)。env間の独立性は別envペアの結果で担保される。
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::vector<int> mask(B, 1);
    mjmlx_batched_reset(cpu, mask.data());
    mjmlx_batched_reset(gpu, mask.data());
    std::vector<float> q(B * nq), v(B * nv);
    int a, b;
    mjmlx_batched_get_state(cpu, q.data(), v.data(), &a, &b);
    for (int e = 0; e < B / 2; e++) {
        for (int j = 0; j < nq; j++) if (j < 3 || j >= 7) q[e * nq + j] += 0.01f * u(rng);
        for (int j = 0; j < nv; j++) v[e * nv + j] = 0.05f * u(rng);
        for (int j = 0; j < nq; j++) q[(e + B / 2) * nq + j] = q[e * nq + j];
        for (int j = 0; j < nv; j++) v[(e + B / 2) * nv + j] = v[e * nv + j];
    }
    mjmlx_batched_set_state(gpu, q.data(), v.data());
    mjmlx_batched_set_state(cpu, q.data(), v.data());

    std::vector<float> ctrl(B * nu), gq(B * nq), gv(B * nv), cq(B * nq), cv(B * nv);
    double eq[3] = {0, 0, 0}, ev[3] = {0, 0, 0};
    double det = 0;
    bool finite = true;
    for (int t = 1; t <= T; t++) {
        for (int e = 0; e < B / 2; e++) for (int j = 0; j < nu; j++) {
            ctrl[e * nu + j] = 0.1f * u(rng);
            ctrl[(e + B / 2) * nu + j] = ctrl[e * nu + j];
        }
        mjmlx_batched_step(gpu, nu ? ctrl.data() : nullptr);
        mjmlx_batched_step(cpu, nu ? ctrl.data() : nullptr);
        if (t != 1 && t != 10 && t != T) continue;
        mjmlx_batched_get_state(gpu, gq.data(), gv.data(), &a, &b);
        mjmlx_batched_get_state(cpu, cq.data(), cv.data(), &a, &b);
        int k = t == 1 ? 0 : t == 10 ? 1 : 2;
        eq[k] = max_diff(gq, cq);
        ev[k] = max_diff(gv, cv);
        printf("step %3d: qpos max|diff|=%.3e qvel max|diff|=%.3e\n", t, eq[k], ev[k]);
        for (float x : gq) finite &= std::isfinite(x);
        for (float x : gv) finite &= std::isfinite(x);
        if (t == T) {
            for (int e = 0; e < B / 2; e++) {
                for (int j = 0; j < nq; j++) det = std::max(det, (double)std::fabs(gq[e * nq + j] - gq[(e + B / 2) * nq + j]));
                for (int j = 0; j < nv; j++) det = std::max(det, (double)std::fabs(gv[e * nv + j] - gv[(e + B / 2) * nv + j]));
            }
        }
    }
    printf("determinism (同一env間の差): %.3e\n", det);

    TEST_BEGIN("gpu_finite_100step");
    CHECK(finite, "GPU state has NaN/Inf");
    TEST_END();

    TEST_BEGIN("gpu_vs_cpu_1step");
    CHECK_LT(eq[0], tol1, "qpos 1step"); CHECK_LT(ev[0], tol1 * 10, "qvel 1step");
    TEST_END();

    TEST_BEGIN("gpu_vs_cpu_10step");
    CHECK_LT(eq[1], tol10, "qpos 10step"); CHECK_LT(ev[1], tol10 * 10, "qvel 10step");
    TEST_END();

    TEST_BEGIN("gpu_vs_cpu_100step");
    CHECK_LT(eq[2], tolT, "qpos 100step"); CHECK_LT(ev[2], tolT * 10, "qvel 100step");
    TEST_END();

    TEST_BEGIN("gpu_multienv_determinism");
    CHECK_LT(det, 1e-5, "identical envs diverged");
    TEST_END();

    mjmlx_batched_free(gpu);
    mjmlx_batched_free(cpu);
    TEST_EXIT();
}
