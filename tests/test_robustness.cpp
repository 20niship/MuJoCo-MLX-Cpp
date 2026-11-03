// GPUバッチの健全性テスト: 自由度(nv)と拘束の構成を総当たりし、MuJoCo Cとの粗い比較に加えて「出力が壊れていないこと」を直接見る。
// CPU比較だけでは「どこかのenvが違う」としか分からない障害(一部スレッドが出力を書かない、barrierがずれて発散する、等)を捕まえるのが目的。
//  - 有限性/発散: 全env・全stepでqpos/qvelが有限かつ小さい(euler_のクランプ値1e4に張り付かない)。
//  - env内の独立性: 同一状態のenvが完全に一致する。単独(B=1)で回した結果がバッチ(B=8)のenvと一致する(Bや位置に依存しない)。
//  - 反復一致: 同じ入力で2回回すと同じ結果。
//  - MuJoCo C比: 数step後のqposが緩い閾値内。
// 構成: free(拘束なし) / limit(関節limit) / contact(振り子の先端が床に当たる) / tendon(固定tendonのlimit) / floss(関節frictionloss)。
// nvは8や32の倍数でない値(ワークグループの端数スレッド)を含める。solverの実装切替(MJMLX_SOLVER_SHARED/CG)やlavapipeのスレッド数はctestの環境変数で振る。
// Usage: test_robustness [--nv 1,7,8,...] [--cfg free,limit,contact,tendon,floss] [--case cfg:nv,nv ...] [--steps N]
//   --caseは構成ごとにnvを指定する(複数回可)。--cfg/--nvは直積。
#include "test_utils.h"
#include <random>
#include <sstream>
#include <vector>

static const int B = 8;  // 前半と後半は同一状態(決定性の確認用)
static int g_steps = 10;

static std::vector<int> parse_ints(const char* s) {
    std::vector<int> v;
    for (const char* p = s; *p;) { v.push_back(atoi(p)); while (*p && *p != ',') p++; if (*p == ',') p++; }
    return v;
}
static std::vector<std::string> parse_strs(const char* s) {
    std::vector<std::string> v;
    std::string cur;
    for (const char* p = s;; p++) { if (*p == ',' || !*p) { if (!cur.empty()) v.push_back(cur); cur.clear(); if (!*p) break; } else cur += *p; }
    return v;
}

// nv個のhingeを持つ振り子の鎖。軸はx/yを交互に。各構成で拘束の種類を変える。
static std::string chain_xml(int nv, const std::string& cfg) {
    const double len = 0.1, H = nv * len + 0.05;
    std::ostringstream x;
    x << "<mujoco><compiler angle=\"radian\"/><option timestep=\"0.002\" gravity=\"0 0 -9.81\" solver=\"Newton\" iterations=\"50\"/><worldbody>";
    if (cfg == "free") x << "<geom type=\"plane\" size=\"5 5 0.1\" contype=\"0\" conaffinity=\"0\"/>";           // 衝突ペア無し(solverを作らない)
    else if (cfg == "contact") x << "<geom type=\"plane\" size=\"5 5 0.1\" pos=\"0 0 " << (H - nv * len * std::cos(1.2) - 0.004) << "\"/>";  // 第1関節を1.2rad振った振り子の下端が最初から床に触れる
    else x << "<geom type=\"plane\" size=\"5 5 0.1\" pos=\"0 0 -50\"/>";                                        // ペアはあるが当たらない(solverは作る)
    std::string joint_attr;
    if (cfg == "limit") joint_attr = " limited=\"true\" range=\"-0.35 0.35\"";
    if (cfg == "floss") joint_attr = " frictionloss=\"0.05\"";
    for (int i = 0; i < nv; i++) {
        x << "<body pos=\"0 0 " << (i == 0 ? H : -len) << "\"><joint name=\"j" << i << "\" type=\"hinge\" axis=\"" << (i % 2 ? "1 0 0" : "0 1 0") << "\" damping=\"0.02\"" << joint_attr << "/>"
          << "<geom type=\"capsule\" fromto=\"0 0 0 0 0 " << -len << "\" size=\"0.02\" density=\"500\"" << (cfg == "free" ? " contype=\"0\" conaffinity=\"0\"" : "") << "/>";  // free以外は床との衝突ペアを作る(ペアが無いとsolverが作られずlimit等が効かない)
    }
    for (int i = 0; i < nv; i++) x << "</body>";
    x << "</worldbody>";
    if (cfg == "tendon") {
        x << "<tendon><fixed name=\"t\" limited=\"true\" range=\"-0.3 0.3\"><joint joint=\"j0\" coef=\"1\"/>";
        if (nv > 1) x << "<joint joint=\"j1\" coef=\"1\"/>";
        x << "</fixed></tendon>";
    }
    x << "</mujoco>";
    return x.str();
}

// 前半envはランダム状態(limitやtendonを超える角度を含む)、後半は前半の複製。contactは第1関節を大きく振って床に当てる。
static void make_state(int nq, int nv, const std::string& cfg, std::vector<float>& q, std::vector<float>& v, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    q.assign((size_t)B * nq, 0.f); v.assign((size_t)B * nv, 0.f);
    for (int e = 0; e < B / 2; e++) {
        for (int j = 0; j < nq; j++) q[(size_t)e * nq + j] = 0.5f * u(rng) * (cfg == "contact" ? 0.01f : 1.0f);
        if (cfg == "contact") q[(size_t)e * nq] = 1.2f;
        for (int j = 0; j < nv; j++) v[(size_t)e * nv + j] = 0.5f * u(rng);
    }
    for (int e = 0; e < B / 2; e++) {
        std::copy(q.begin() + (size_t)e * nq, q.begin() + (size_t)(e + 1) * nq, q.begin() + (size_t)(e + B / 2) * nq);
        std::copy(v.begin() + (size_t)e * nv, v.begin() + (size_t)(e + 1) * nv, v.begin() + (size_t)(e + B / 2) * nv);
    }
}

struct Run { std::vector<float> q, v; double max_abs_q = 0, max_abs_v = 0; bool finite = true; };

static Run run_sim(MjmlxModel* model, int nenv, bool gpu, const std::vector<float>& q0, const std::vector<float>& v0, int steps) {
    auto info = mjmlx_model_info(model);
    MjmlxBatchedConfig cfg = {};
    cfg.num_envs = nenv; cfg.use_gpu = gpu ? 1 : 0; cfg.integrator = MJMLX_INTEGRATOR_EULER;
    if (getenv("ROBUST_ITERS")) cfg.solver_iterations = atoi(getenv("ROBUST_ITERS"));
    auto* sim = mjmlx_batched_create(model, &cfg);
    Run r;
    r.q.resize((size_t)nenv * info.nq); r.v.resize((size_t)nenv * info.nv);
    mjmlx_batched_set_state(sim, q0.data(), v0.data());
    int a, b;
    for (int t = 0; t < steps; t++) {
        mjmlx_batched_step(sim, nullptr);
        mjmlx_batched_get_state(sim, r.q.data(), r.v.data(), &a, &b);
        for (float x : r.q) { r.finite &= std::isfinite(x); if (std::isfinite(x)) r.max_abs_q = std::max(r.max_abs_q, (double)std::fabs(x)); }
        for (float x : r.v) { r.finite &= std::isfinite(x); if (std::isfinite(x)) r.max_abs_v = std::max(r.max_abs_v, (double)std::fabs(x)); }
    }
    mjmlx_batched_free(sim);
    return r;
}

// MuJoCo C(mj_step)を各envに回した参照。
static Run run_mujoco_c(MjmlxModel* model, int nenv, const std::vector<float>& q0, const std::vector<float>& v0, int steps) {
    const mjModel* m = (const mjModel*)mjmlx_get_mj_model(model);
    mjData* d = mj_makeData(m);
    Run r;
    r.q.resize((size_t)nenv * m->nq); r.v.resize((size_t)nenv * m->nv);
    for (int e = 0; e < nenv; e++) {
        mj_resetData(m, d);
        for (int j = 0; j < m->nq; j++) d->qpos[j] = q0[(size_t)e * m->nq + j];
        for (int j = 0; j < m->nv; j++) d->qvel[j] = v0[(size_t)e * m->nv + j];
        for (int t = 0; t < steps; t++) mj_step(m, d);
        for (int j = 0; j < m->nq; j++) r.q[(size_t)e * m->nq + j] = (float)d->qpos[j];
        for (int j = 0; j < m->nv; j++) r.v[(size_t)e * m->nv + j] = (float)d->qvel[j];
    }
    mj_deleteData(m ? d : d);
    return r;
}

static double max_diff(const std::vector<float>& a, const std::vector<float>& b, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; i++) m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}

static void run_case(const std::string& name, const std::string& cfg, int nv) {
    std::string xml = chain_xml(nv, cfg);
    auto* model = mjmlx_load_model_from_string(xml.c_str());
    TEST_BEGIN(name.c_str());
    if (!model) { CHECK(false, "モデルを読めない"); } else {
    auto info = mjmlx_model_info(model);
    std::vector<float> q0, v0;
    make_state(info.nq, info.nv, cfg, q0, v0, 1234u + nv);

    Run g = run_sim(model, B, true, q0, v0, g_steps);
    CHECK(g.finite, "GPU状態にNaN/Inf");
    CHECK(g.max_abs_v < 1e3 && g.max_abs_q < 1e3, "GPU状態が発散(qvelのクランプ値1e4等)");

    // 同一状態のenvは完全に一致する。
    const size_t nq = info.nq, nvv = info.nv;
    double dup = 0;
    for (int e = 0; e < B / 2; e++) {
        for (size_t j = 0; j < nq; j++) dup = std::max(dup, (double)std::fabs(g.q[e * nq + j] - g.q[(e + B / 2) * nq + j]));
        for (size_t j = 0; j < nvv; j++) dup = std::max(dup, (double)std::fabs(g.v[e * nvv + j] - g.v[(e + B / 2) * nvv + j]));
    }
    CHECK(dup < 1e-6, "同一状態のenvが一致しない(env間で結果が違う)");

    // 反復一致。
    Run g2 = run_sim(model, B, true, q0, v0, g_steps);
    CHECK(max_diff(g.q, g2.q, g.q.size()) < 1e-6 && max_diff(g.v, g2.v, g.v.size()) < 1e-6, "同じ入力で2回回すと結果が違う");

    // 単独(B=1)の結果がバッチ内の同じenvと一致する(Bやバッチ内の位置に依存しない)。後半の1つで確認。
    for (int e : {B / 2 + 1}) {
        std::vector<float> q1(q0.begin() + (size_t)e * nq, q0.begin() + (size_t)(e + 1) * nq), v1(v0.begin() + (size_t)e * nvv, v0.begin() + (size_t)(e + 1) * nvv);
        Run s = run_sim(model, 1, true, q1, v1, g_steps);
        double d = 0;
        for (size_t j = 0; j < nq; j++) d = std::max(d, (double)std::fabs(s.q[j] - g.q[e * nq + j]));
        for (size_t j = 0; j < nvv; j++) d = std::max(d, (double)std::fabs(s.v[j] - g.v[e * nvv + j]));
        CHECK(d < 1e-4, "B=1とバッチで結果が違う(Bまたは位置依存)");
    }

    // MuJoCo Cとの粗い比較(拘束の有無で構成が変わっても桁違いに外れない)。
    Run c = run_mujoco_c(model, B, q0, v0, g_steps);
    double cvmax = 0; for (float x : c.v) cvmax = std::max(cvmax, (double)std::fabs(x));
    double dq = max_diff(g.q, c.q, g.q.size());
    if (getenv("ROBUST_VERBOSE")) printf("    %-16s nq=%zu  |q|max=%.2f |v|max=%.2f  dup=%.1e  gpu-vs-C q=%.2e  C|v|end=%.2f\n", name.c_str(), nq, g.max_abs_q, g.max_abs_v, dup, dq, cvmax);
    if (getenv("ROBUST_VERBOSE") && nq <= 4) for (int e = 0; e < 2; e++) { printf("      env%d q0:", e); for (size_t j = 0; j < nq; j++) printf(" %.3f", q0[e * nq + j]); printf("\n      GPU:"); for (size_t j = 0; j < nq; j++) printf(" %.3f", g.q[e * nq + j]); printf("\n      C  :"); for (size_t j = 0; j < nq; j++) printf(" %.3f", c.q[e * nq + j]); printf("\n"); }
    CHECK(dq < 5e-3, "MuJoCo Cとのqposが5e-3rad以上ずれた");
    }
    TEST_END();
    if (model) mjmlx_free_model(model);
}

int main(int argc, char** argv) {
    std::vector<int> nvs = {1, 3, 4, 7, 8, 9, 16, 17, 31, 32, 33};
    std::vector<std::string> cfgs = {"free", "limit", "contact", "tendon", "floss"};
    std::vector<std::pair<std::string, int>> cases;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--nv") && i + 1 < argc) nvs = parse_ints(argv[++i]);
        else if (!strcmp(argv[i], "--cfg") && i + 1 < argc) cfgs = parse_strs(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) g_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--case") && i + 1 < argc) {
            std::string c = argv[++i];
            size_t k = c.find(':');
            for (int nv : parse_ints(c.c_str() + k + 1)) cases.push_back({c.substr(0, k), nv});
        }
    }
    if (cases.empty()) for (const auto& cfg : cfgs) for (int nv : nvs) cases.push_back({cfg, nv});
    printf("=== test_robustness: %zu cases, %d steps ===\n", cases.size(), g_steps);
    for (const auto& c : cases) run_case(c.first + "_nv" + std::to_string(c.second), c.first, c.second);
    TEST_EXIT();
}
