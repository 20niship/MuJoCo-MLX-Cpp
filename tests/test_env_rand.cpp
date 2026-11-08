// env別の物理ランダム化(mjmlx_batched_set_env_params)の結果が、同じパラメータで書き換えたmjModelのMuJoCo Cと一致することを見る。
#include "test_utils.h"
#include <mujoco/mujoco.h>
#include <cmath>
#include <vector>

namespace {

// 箱(COMをずらした自由物体、床の上を滑る)と、位置アクチュエータで駆動するヒンジ腕(質量とゲインが効く)。
const char* XML = R"(
<mujoco><option gravity="0 0 -9.81" timestep="0.002" integrator="Euler" solver="Newton" iterations="50"/><worldbody>
  <geom name="floor" type="plane" size="5 5 0.1" friction="0.5 0.005 0.0001"/>
  <body name="box" pos="0 0 0.1"><freejoint/>
    <inertial pos="0.01 0 0" mass="2" diaginertia="0.01 0.012 0.014"/>
    <geom type="box" size="0.1 0.1 0.1" friction="0.5 0.005 0.0001"/></body>
  <body name="arm" pos="1.5 0 1"><joint name="h" type="hinge" axis="0 1 0" damping="0.05"/>
    <geom type="capsule" fromto="0 0 0 0.4 0 0" size="0.03" mass="1" contype="0" conaffinity="0"/></body>
</worldbody><actuator><position joint="h" kp="30" kv="1"/></actuator></mujoco>)";

constexpr int B = 4, STEPS = 150;

struct P { float mass, friction, gain; float com[3]; };  // 質量は全bodyに同じ倍率、COMずれは腕(body 2)だけ

} // namespace

int main() {
    printf("=== test_env_rand ===\n");
    auto path = write_temp_xml(XML);
    auto* model = mjmlx_load_model(path.c_str());
    char err[256] = "";
    mjModel* mj = mj_loadXML(path.c_str(), nullptr, err, sizeof err);
    unlink(path.c_str());
    if (!model || !mj) { fprintf(stderr, "FAIL: load (%s)\n", err); return 1; }
    auto info = mjmlx_model_info(model);
    const int nq = info.nq, nv = info.nv, nu = info.nu, nb = info.nbody;

    const P params[B] = {
        {1.0f, 1.0f, 1.0f, {0, 0, 0}},          // 恒等
        {1.7f, 1.0f, 1.0f, {0, 0, 0}},          // 質量だけ
        {1.0f, 0.4f, 1.0f, {-0.1f, 0, 0.03f}},  // 摩擦とCOMずれ
        {0.7f, 1.3f, 0.6f, {0.05f, 0, -0.02f}}, // 全部
    };

    std::vector<float> mass(B * nb), com(B * nb * 3, 0.0f), fric(B), gain(B * nu);
    for (int e = 0; e < B; e++) {
        for (int b = 0; b < nb; b++) mass[e * nb + b] = params[e].mass;
        for (int k = 0; k < 3; k++) com[(e * nb + 2) * 3 + k] = params[e].com[k];
        fric[e] = params[e].friction;
        for (int a = 0; a < nu; a++) gain[e * nu + a] = params[e].gain;
    }

    auto make = [&](int domain_rand) {
        MjmlxBatchedConfig cfg = {};
        cfg.num_envs = B; cfg.use_gpu = 1; cfg.integrator = MJMLX_INTEGRATOR_EULER; cfg.domain_rand = domain_rand;
        return mjmlx_batched_create(model, &cfg);
    };
    auto* sim = make(1);
    auto* plain = make(0);
    if (!sim || !plain) { fprintf(stderr, "FAIL: create\n"); return 1; }

    // 初期速度: 箱を滑らせる(摩擦とCOMずれが効く)。腕は静止、ctrl=0.5radの位置指令(ゲインと質量が効く)。
    std::vector<float> qv0(nv, 0.0f); qv0[0] = 1.5f;
    for (int e = 0; e < B; e++) { mjmlx_batched_set_env_qvel(sim, e, qv0.data(), nv); mjmlx_batched_set_env_qvel(plain, e, qv0.data(), nv); }
    int rc = mjmlx_batched_set_env_params(sim, nullptr, B, mass.data(), com.data(), fric.data(), gain.data());
    std::vector<float> ctrl(B * nu, 0.5f);
    for (int s = 0; s < STEPS; s++) { mjmlx_batched_step(sim, ctrl.data()); mjmlx_batched_step(plain, ctrl.data()); }

    int n;
    std::vector<float> q(mjmlx_batched_get_qpos(sim, &n), mjmlx_batched_get_qpos(sim, &n) + B * nq);
    std::vector<float> v(mjmlx_batched_get_qvel(sim, &n), mjmlx_batched_get_qvel(sim, &n) + B * nv);
    const float* qp = mjmlx_batched_get_qpos(plain, &n);

    int total = 0;
    TEST_BEGIN("set_env_params returns 0");
    CHECK(rc == 0, "set_env_params rc");
    TEST_END(); total++;

    // MuJoCo C(env別に書き換えたmjModel。mj_setConstは呼ばない: GPU側もinvweight等のモデル定数は据え置き)
    double worst = 0, cpu_x[B];
    for (int e = 0; e < B; e++) {
        mjModel* m = mj_copyModel(nullptr, mj);
        for (int b = 0; b < nb; b++) {
            m->body_mass[b] *= params[e].mass;
            for (int k = 0; k < 3; k++) m->body_inertia[b * 3 + k] *= params[e].mass;
        }
        for (int k = 0; k < 3; k++) m->body_ipos[2 * 3 + k] += params[e].com[k];
        for (int g = 0; g < m->ngeom; g++) for (int k = 0; k < 3; k++) m->geom_friction[g * 3 + k] *= params[e].friction;
        for (int a = 0; a < nu; a++) for (int k = 0; k < 3; k++) { m->actuator_gainprm[a * mjNGAIN + k] *= params[e].gain; m->actuator_biasprm[a * mjNBIAS + k] *= params[e].gain; }
        mjData* d = mj_makeData(m);
        d->qvel[0] = qv0[0];
        for (int s = 0; s < STEPS; s++) { for (int a = 0; a < nu; a++) d->ctrl[a] = 0.5; mj_step(m, d); }
        cpu_x[e] = d->qpos[0];
        // 腕(質量・COM・ゲイン)は接触が無いのでMuJoCo Cと厳密に比べる。
        double arm = std::fmax(std::fabs(d->qpos[nq - 1] - q[e * nq + nq - 1]), 0.1 * std::fabs(d->qvel[nv - 1] - v[e * nv + nv - 1]));
        if (getenv("ENVRAND_VERBOSE")) printf("env%d arm diff %.2e, box x gpu %.4f cpu %.4f\n", e, arm, q[e * nq], cpu_x[e]);
        char name[64]; snprintf(name, sizeof name, "env%d arm matches MuJoCo C", e);
        TEST_BEGIN(name);
        CHECK_LT((float)arm, 2e-3f, "arm qpos/qvel vs MuJoCo C with the same per-env mass/COM/gain");
        TEST_END(); total++;
        worst = std::fmax(worst, arm);
        mj_deleteData(d); mj_deleteModel(m);
    }
    printf("worst arm diff vs MuJoCo C: %.3e\n", worst);

    // 箱の摩擦: 接触solverのCPUとの既存の差(恒等でも約4e-3)は同じenv0基準の差分で相殺して、摩擦倍率による滑り距離の変化を比べる。
    TEST_BEGIN("friction scale changes the box slide like MuJoCo C");
    for (int e = 1; e < B; e++) {
        float g = q[e * nq] - q[0], c = (float)(cpu_x[e] - cpu_x[0]);
        if (getenv("ENVRAND_VERBOSE")) printf("env%d box slide delta gpu %.4f cpu %.4f\n", e, g, c);
        CHECK_LT(std::fabs(g - c), 1.2e-2f, "slide distance change vs env0 within 1.2e-2 of MuJoCo C");
    }
    TEST_END(); total++;

    TEST_BEGIN("identity params == non-randomized path");
    {
        float md = 0; for (int i = 0; i < nq; i++) md = std::fmax(md, std::fabs(qp[i] - q[i]));  // env0は恒等
        CHECK_LT(md, 1e-6f, "env0 (identity) matches the domain_rand=0 sim");
    }
    TEST_END(); total++;

    TEST_BEGIN("parameters change the result");
    {
        auto dist = [&](int e) { float dd = 0; for (int i = 0; i < nq; i++) dd = std::fmax(dd, std::fabs(q[e * nq + i] - q[i])); return dd; };
        CHECK_GT(dist(1), 1e-3f, "mass scale changes the arm response");
        CHECK_GT(dist(2), 1e-3f, "friction/COM offset change the box motion");
        CHECK_GT(dist(3), 1e-3f, "combined parameters differ from identity");
    }
    TEST_END(); total++;

    // env_idxで一部のenvだけ更新でき、範囲外・domain_rand無効は拒否される
    TEST_BEGIN("partial update by env_idx");
    {
        int idx[1] = {3}; float one_mass[3] = {1, 1, 1}, one_fric[1] = {1}, one_gain[1] = {1}; std::vector<float> zero_com(nb * 3, 0.0f);
        CHECK(mjmlx_batched_set_env_params(sim, idx, 1, one_mass, zero_com.data(), one_fric, one_gain) == 0, "rc");
        int bad[1] = {B};
        CHECK(mjmlx_batched_set_env_params(sim, bad, 1, one_mass, nullptr, nullptr, nullptr) == -2, "out of range env_idx rejected");
        CHECK(mjmlx_batched_set_env_params(plain, nullptr, B, mass.data(), nullptr, nullptr, nullptr) == -1, "domain_rand=0 sim rejects");
    }
    TEST_END(); total++;

    printf("\n=== %d checks groups ===\n", total);
    mjmlx_batched_free(sim); mjmlx_batched_free(plain);
    mj_deleteModel(mj); mjmlx_free_model(model);
    TEST_EXIT();
}
