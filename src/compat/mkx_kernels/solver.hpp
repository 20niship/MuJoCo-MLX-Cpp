#pragma once
// Vendored from 20niship/mlx-cross-platform tests/mujoco/solver.hpp at commit 3202339e12231f3a68f9ee67e0b6085a33c52286.

#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>

#include <mkx/ops/fast_kernel.hpp>

// MuJoCo-MLX-Cpp
// make_solver_source逐語GLSL移植。MetalのthreadgroupメモリはGLSLのshared変数、threadgroup_barrier(mem_device[|mem_threadgroup])はbarrier()+memoryBarrierBuffer()[+memoryBarrierShared()]に対応させ、use_pyramidal/refsafeはモデル設定で生成テキスト自体が変わるため元のC++条件分岐構造をそのまま踏襲した。

namespace mjmlx::mkx_kernels {

namespace detail_solver {
inline std::string solver_fmt_float(float x) {
  std::ostringstream ss;
  ss << x;
  std::string s = ss.str();
  if(s.find('.') == std::string::npos && s.find('e') == std::string::npos) s += ".0";
  return s;
}
} // namespace detail_solver

// ホスト側のscratch確保量(batched.cppのsolver_scratch_per_env)と一致させる責務を持つ、カーネルが1envあたり使うfloat数。
// 1envあたりの拘束行数の上限。小さいほどscratchが小さくキャッシュに載りやすい(超過分の接触は落ちる)。MJMLX_MAX_EFCで上書き。
inline int solver_max_efc() { static const int v = getenv("MJMLX_MAX_EFC") ? atoi(getenv("MJMLX_MAX_EFC")) : 256; return v; }
// 小さいモデル(nv<=32)は拘束行列などをworkgroup共有メモリに置く。globalメモリのレイテンシ(数百cycle)が直列部分を支配するため。
// 共有メモリに載せるため拘束行数の上限を下げる(超過分の接触は落ちる)。MJMLX_SOLVER_SHARED=0で無効。
inline bool solver_use_shared(int nv) { static const bool on = !getenv("MJMLX_SOLVER_SHARED") || atoi(getenv("MJMLX_SOLVER_SHARED")) != 0; return on && nv <= 32; }
inline int solver_max_efc_for(int nv) { return solver_use_shared(nv) ? std::min(solver_max_efc(), 96) : solver_max_efc(); }

inline int solver_scratch_floats(int nv, int max_efc) { return nv * nv + max_efc * nv + 7 * max_efc + 5 * nv + 3 * nv * nv; }

inline mkx::fast::Kernel<> make_solver_kernel(int nb, int nv, int nq, float timestep, bool use_pyramidal, bool refsafe, float impratio, int solver_iters, int cg_iters, int nlim, int nfl) {
  using namespace detail_solver;
  const float cg_tol = getenv("MJMLX_CGTOL") ? (float)atof(getenv("MJMLX_CGTOL")) : 1e-6f;
  const float gtol_rel = getenv("MJMLX_GTOL") ? (float)atof(getenv("MJMLX_GTOL")) : 1e-6f;  // Newton反復の打ち切り閾値(勾配ノルム2乗の初期値比)。0で無効
  const int max_efc              = solver_max_efc_for(nv);
  const bool use_shared          = solver_use_shared(nv);
  const int contact_stride       = 8;
  const int max_contacts_per_env = 128;

  int S_H             = 0;
  int S_J             = S_H + nv * nv;
  int S_D             = S_J + max_efc * nv;
  int S_AREF          = S_D + max_efc;
  int S_FORCE         = S_AREF + max_efc;
  int S_GRAD          = S_FORCE + max_efc;
  int S_SEARCH        = S_GRAD + nv;
  int S_QACC          = S_SEARCH + nv;
  int S_MA            = S_QACC + nv;
  int S_JAREF         = S_MA + nv;
  int S_ACTIVE        = S_JAREF + max_efc;
  int S_MV            = S_ACTIVE + max_efc;
  int S_JV            = S_MV + nv;
  int S_JACP          = S_JV + max_efc;
  int S_FLOSS         = S_JACP + nv * 3 * nv;  // 接触を並列に組むためスレッド(dof)ごとに3*nvを持つ
  int SCRATCH_PER_ENV = S_FLOSS + max_efc;
  if(SCRATCH_PER_ENV != solver_scratch_floats(nv, max_efc)) throw std::logic_error("mjmlx solver: scratchレイアウトとsolver_scratch_floatsが不一致");

  std::string header = R"GLSL(
vec3 msl_cross(vec3 a, vec3 b) { return cross(a, b); }
float msl_dot(vec3 a, vec3 b) { return dot(a, b); }
float msl_len(vec3 a) { return length(a); }
vec3 msl_norm(vec3 a) {
    float l = msl_len(a);
    return l > 1e-12 ? a / l : vec3(0.0, 0.0, 1.0);
}
// 拘束1行のコスト。frictionloss行(fl>0)は二次域(|D*x|<fl)と線形域(飽和)の折れ線、それ以外は片側(x<0のみ)の二次。
float efc_cost(float D, float x, float fl) {
    if (fl > 0.0f) {
        if (abs(D * x) < fl) return 0.5f * D * x * x;
        return fl * abs(x) - 0.5f * fl * fl / D;
    }
    return (x < 0.0f) ? 0.5f * D * x * x : 0.0f;
}
shared int tg_nefc;
shared int tg_ncp;
shared int tg_crow0[129];  // 接触ciの拘束行は [tg_crow0[ci], tg_crow0[ci+1])
shared float tg_ba;
shared float tg_g0;
shared float tg_rr0;
shared int tg_done;  // 1ならこのenvは収束したのでNewton反復を打ち切る(ワークグループ内で一様)
shared float tg_rr;
shared float tg_pAp;
)GLSL";

  if(use_shared) {
    std::ostringstream sh;
    sh << "shared float sh_H[" << nv * nv << "];\n"
       << "shared float sh_J[" << max_efc * nv << "];\n"
       << "shared float sh_D[" << max_efc << "];\n"
       << "shared float sh_aref[" << max_efc << "];\n"
       << "shared float sh_force[" << max_efc << "];\n"
       << "shared float sh_Jaref[" << max_efc << "];\n"
       << "shared float sh_act[" << max_efc << "];\n"
       << "shared float sh_floss[" << max_efc << "];\n"
       << "shared float sh_Jv[" << max_efc << "];\n"
       << "shared float sh_grad[" << nv << "];\n"
       << "shared float sh_search[" << nv << "];\n"
       << "shared float sh_qacc[" << nv << "];\n"
       << "shared float sh_Ma[" << nv << "];\n"
       << "shared float sh_Mv[" << nv << "];\n"
       << "shared float sh_red[" << 12 * nv << "];\n"
       << "shared float sh_qM[" << nv * nv << "];\n"
       << "shared float sh_qfs[" << nv << "];\n"
       << "shared float sh_cdof[" << nv * 6 << "];\n"
       << "shared float sh_scom[" << nb * 3 << "];\n";
    header += sh.str();
  }
  {
    std::ostringstream hk;
    hk << "void efc_kbi(float solref0, float solref1, float si0, float si1, float si2, float si3, float si4, float pos, out float k_val, out float b_val, out float imp_val) {\n"
       << "    float tc = solref0;\n";
    if(!refsafe) hk << "    tc = max(tc, 2.0f * " << solver_fmt_float(timestep) << "f);\n";
    hk << "    float dmin = clamp(si0, 0.0001f, 0.9999f);\n"
       << "    float dmax_v = clamp(si1, 0.0001f, 0.9999f);\n"
       << "    float width_v = max(si2, 1e-12f);\n"
       << "    float mid_v = clamp(si3, 0.0001f, 0.9999f);\n"
       << "    float power_v = max(si4, 1.0f);\n"
       << "    k_val = (tc > 0.0) ? 1.0f/(dmax_v*dmax_v*tc*tc*solref1*solref1) : -tc/(dmax_v*dmax_v);\n"
       << "    b_val = (solref1 > 0.0) ? 2.0f/(dmax_v*tc) : -solref1/dmax_v;\n"
       << "    float x = abs(pos) / width_v;\n"
       << "    float y;\n"
       << "    if (x < mid_v) y = pow(x, power_v) / pow(mid_v, power_v - 1.0f);\n"
       << "    else y = 1.0f - pow(1.0f - x, power_v) / pow(1.0f - mid_v, power_v - 1.0f);\n"
       << "    imp_val = clamp(dmin + y * (dmax_v - dmin), dmin, dmax_v);\n"
       << "    if (x > 1.0f) imp_val = dmax_v;\n"
       << "}\n";
    header += hk.str();
  }

  std::ostringstream ss;

  ss << "uint bid = gl_WorkGroupID.x;\n"
     << "uint NBATCH = gl_NumWorkGroups.x;\n"
     << "uint tid = gl_LocalInvocationID.x;\n"
     << "const int NV = " << nv << ";\n"
     << "const int NB = " << nb << ";\n"
     << "const int NQ = " << nq << ";\n"
     << "const int NLIM = " << nlim << ";\n"
     << "const int NFL = " << nfl << ";\n"
     << "const bool USE_PYR = " << (use_pyramidal ? "true" : "false") << ";\n"
     << "const int MAX_EFC_N = " << max_efc << ";\n"
     << "const int NSOLVE = " << solver_iters << ";\n"
     << "const int CON_STRIDE = " << contact_stride << ";\n"
     << "const int MAX_CON = " << max_contacts_per_env << ";\n"
     << "const float TIMESTEP = " << solver_fmt_float(timestep) << "f;\n"
     << "const float GTOL_REL = " << solver_fmt_float(gtol_rel) << "f;\n"
     << "const float MJMINVAL_CV = 1e-12f;\n"
     << "const float MJMINVAL_SV = 1e-14f;\n"
     << "const float MJMINIMP = 0.0001f;\n"
     << "const float MJMAXIMP = 0.9999f;\n\n";

  ss << "uint s_off = bid * " << SCRATCH_PER_ENV << ";\n";
  if(use_shared) {
    ss << "#define QMV(i)       sh_qM[(i)]\n";
    ss << "#define QFS(i)       sh_qfs[(i)]\n";
    ss << "#define CDOF(i)      sh_cdof[(i)]\n";
    ss << "#define SCOM(i)      sh_scom[(i)]\n";
    ss << "#define H(i)         sh_H[(i)]\n"
       << "#define J(r,c)       sh_J[(r)*NV+(c)]\n"
       << "#define efc_D(i)     sh_D[(i)]\n"
       << "#define efc_aref(i)  sh_aref[(i)]\n"
       << "#define efc_force(i) sh_force[(i)]\n"
       << "#define grad(i)      sh_grad[(i)]\n"
       << "#define search_d(i)  sh_search[(i)]\n"
       << "#define qacc(i)      sh_qacc[(i)]\n"
       << "#define Ma(i)        sh_Ma[(i)]\n"
       << "#define Jaref(i)     sh_Jaref[(i)]\n"
       << "#define act(i)       sh_act[(i)]\n"
       << "#define efc_floss(i) sh_floss[(i)]\n"
       << "#define Mv_arr(i)    sh_Mv[(i)]\n"
       << "#define Jv_arr(i)    sh_Jv[(i)]\n";
  } else {
    ss << "#define QMV(i)       qM_in[uint(i)*NBATCH+bid]\n";
    ss << "#define QFS(i)       qfrc_smooth_in[uint(i)*NBATCH+bid]\n";
    ss << "#define CDOF(i)      cdof_in[uint(i)*NBATCH+bid]\n";
    ss << "#define SCOM(i)      subtree_com_in[uint(i)*NBATCH+bid]\n";
    ss << "#define H(i)         solver_scratch[s_off + " << S_H << " + (i)]\n"
       << "#define J(r,c)       solver_scratch[s_off + " << S_J << " + (r)*NV+(c)]\n"
       << "#define efc_D(i)     solver_scratch[s_off + " << S_D << " + (i)]\n"
       << "#define efc_aref(i)  solver_scratch[s_off + " << S_AREF << " + (i)]\n"
       << "#define efc_force(i) solver_scratch[s_off + " << S_FORCE << " + (i)]\n"
       << "#define grad(i)      solver_scratch[s_off + " << S_GRAD << " + (i)]\n"
       << "#define search_d(i)  solver_scratch[s_off + " << S_SEARCH << " + (i)]\n"
       << "#define qacc(i)      solver_scratch[s_off + " << S_QACC << " + (i)]\n"
       << "#define Ma(i)        solver_scratch[s_off + " << S_MA << " + (i)]\n"
       << "#define Jaref(i)     solver_scratch[s_off + " << S_JAREF << " + (i)]\n"
       << "#define act(i)       solver_scratch[s_off + " << S_ACTIVE << " + (i)]\n"
       << "#define efc_floss(i) solver_scratch[s_off + " << S_FLOSS << " + (i)]\n"
       << "#define Mv_arr(i)    solver_scratch[s_off + " << S_MV << " + (i)]\n"
       << "#define Jv_arr(i)    solver_scratch[s_off + " << S_JV << " + (i)]\n";
  }
  if(use_shared) ss << "#define JACP_DECL float jp_l[NV*3];\n#define jacp_tmp(i)  jp_l[(i)]\n\n";  // スレッド専用なのでlocal配列(globalのscratchは散らばって遅い)
  else ss << "#define JACP_DECL\n#define jacp_tmp(i)  solver_scratch[s_off + " << S_JACP << " + int(tid)*NV*3 + (i)]\n\n";

  ss << "uint qp_off = bid * NQ;\n"
     << "uint qm_off = bid * NV * NV;\n"
     << "uint qfs_off = bid * NV;\n"
     << "uint cd_off = bid * NV * 6;\n"
     << "uint sc_off = bid * NB * 3;\n"
     << "uint qv_off = bid * NV;\n"
     << "uint con_off = bid * MAX_CON * CON_STRIDE;\n"
     << "uint qfc_off = bid * NV;\n\n";

  ss << "#define EFC_REFRESH(r) { float fl_ = efc_floss(r); float jr_ = Jaref(r); "
     << "if (fl_ > 0.0f) { float fd_ = efc_D(r) * jr_; if (abs(fd_) < fl_) { act(r) = 1.0f; efc_force(r) = -fd_; } else { act(r) = 0.0f; efc_force(r) = (jr_ > 0.0f) ? -fl_ : fl_; } } "
     << "else { act(r) = (jr_ < 0.0f) ? 1.0f : 0.0f; efc_force(r) = efc_D(r) * (-jr_) * act(r); } }\n";

  ss << "#define cg_r(i)   H(i)\n"
     << "#define cg_p(i)   H(NV + (i))\n"
     << "#define cg_Ap(i)  H(2*NV + (i))\n";

  ss << "const int CG_ITERS = " << cg_iters << ";\n"
     << "const float CG_TOL = " << solver_fmt_float(cg_tol) << "f;\n\n";

  if(!use_shared) {
    ss << "for (int ii = int(tid); ii < " << SCRATCH_PER_ENV << "; ii += NV) solver_scratch[s_off + ii] = 0.0f;\n"
       << "barrier(); memoryBarrierBuffer();\n\n";
  } else {
    // 共有メモリ版は全域のゼロ埋めをしない: 拘束行(J/D/aref/floss)は作る時に全要素を書き、qacc等は読む前に必ず書く。
    ss << "for (int ii = int(tid); ii < NV*NV; ii += NV) sh_qM[ii] = qM_in[uint(ii)*NBATCH+bid];\n"
       << "sh_qfs[tid] = qfrc_smooth_in[uint(int(tid))*NBATCH+bid];\n"
       << "for (int ii = int(tid); ii < NV*6; ii += NV) sh_cdof[ii] = cdof_in[uint(ii)*NBATCH+bid];\n"
       << "for (int ii = int(tid); ii < NB*3; ii += NV) sh_scom[ii] = subtree_com_in[uint(ii)*NBATCH+bid];\n"
       << "sh_qacc[tid] = 0.0f;\nbarrier();\n\n";
  }

  ss << "SABL0\n";
  const int nfl_eff = std::min(nfl, max_efc - 4);
  if(nfl_eff > 0) {
    // frictionloss行は常に先頭に並び、行ごとに独立なので全スレッドで並列に作る。
    ss << "const int NFL_EFF = " << nfl_eff << ";\n";
    ss << R"GLSL(
for (int fi = int(tid); fi < NFL_EFF; fi += NV) {
    int fp = fi * 10;
    int dof = int(fric_tab[fp+0]);
    float fl = fric_tab[fp+1];
    float k_f, b_f, imp_f;
    efc_kbi(fric_tab[fp+2], fric_tab[fp+3], fric_tab[fp+4], fric_tab[fp+5], fric_tab[fp+6], fric_tab[fp+7], fric_tab[fp+8], 0.0f, k_f, b_f, imp_f);
    float r_f = max(fric_tab[fp+9] * (1.0f - imp_f) / imp_f, MJMINVAL_CV);
    for (int di = 0; di < NV; di++) J(fi, di) = 0.0f;
    J(fi, dof) = 1.0f;
    efc_D(fi) = 1.0f / r_f;
    efc_aref(fi) = -b_f * qvel_in[qv_off + dof];
    efc_floss(fi) = fl;
}
barrier();
)GLSL";
  }
  ss << "if (tid == 0u) {\n"
     << "int ncon = int(contact_count_in[bid]);\n"
     << "int nefc = " << nfl_eff << ";\n\n";

  if(nlim > 0) {
    ss << R"GLSL(
for (int li = 0; li < NLIM; li++) {
    if (nefc >= MAX_EFC_N - 4) break;
    int lp = li * 13;
    int dof = int(lim_tab[lp+0]);
    int qa = int(lim_tab[lp+1]);
    float qv_j = qpos_in[qp_off + qa];
    float dist_lo = qv_j - lim_tab[lp+2];
    float dist_hi = lim_tab[lp+3] - qv_j;
    float pos = min(dist_lo, dist_hi) - lim_tab[lp+4];
    if (pos < 0.0f) {
        float sgn = (dist_lo < dist_hi) ? 1.0f : -1.0f;
        float k_l, b_l, imp_l;
        efc_kbi(lim_tab[lp+5], lim_tab[lp+6], lim_tab[lp+7], lim_tab[lp+8], lim_tab[lp+9], lim_tab[lp+10], lim_tab[lp+11], pos, k_l, b_l, imp_l);
        float r_l = max(lim_tab[lp+12] * (1.0f - imp_l) / imp_l, MJMINVAL_CV);
        for (int di = 0; di < NV; di++) J(nefc, di) = 0.0f;
        efc_floss(nefc) = 0.0f;
        J(nefc, dof) = sgn;
        efc_D(nefc) = 1.0f / r_l;
        efc_aref(nefc) = -b_l * (sgn * qvel_in[qv_off + dof]) - k_l * imp_l * pos;
        nefc++;
    }
}
)GLSL";
  }

  ss << R"GLSL(
int ncp = 0;
{
    // 各接触が使う拘束行数(ピラミッド錐でcondim>=3なら4行、他は1行、非アクティブは0)を先に数えて行オフセットを決める。
    int r_run = nefc;
    for (int ci = 0; ci < ncon && r_run < MAX_EFC_N - 4; ci++) {
        int ci_o = int(con_off) + ci * CON_STRIDE;
        int pp_ = int(contact_data_in[ci_o+7]) * 18;
        bool act_ = contact_data_in[ci_o+6] < pair_props[pp_+3];
        int cd_ = int(pair_props[pp_+2]);
        tg_crow0[ci] = r_run; ncp = ci + 1;
        r_run += act_ ? ((USE_PYR && cd_ >= 3) ? 4 : 1) : 0;
    }
    tg_crow0[ncp] = r_run; tg_ncp = ncp; tg_nefc = r_run;
}
} // end if (tid == 0)
barrier(); memoryBarrierBuffer(); memoryBarrierShared();
// 接触ごとの拘束行を並列(スレッド=接触)に組む。jacp_tmpはスレッド専用領域。
for (int ci = int(tid); ci < tg_ncp; ci += NV) {
    int row = tg_crow0[ci];
    if (tg_crow0[ci+1] == row) continue;
    for (int rr = row; rr < tg_crow0[ci+1]; rr++) efc_floss(rr) = 0.0f;
    int ci_off = int(con_off) + ci * CON_STRIDE;
    float c_pos_x = contact_data_in[ci_off+0];
    float c_pos_y = contact_data_in[ci_off+1];
    float c_pos_z = contact_data_in[ci_off+2];
    float c_norm_x = contact_data_in[ci_off+3];
    float c_norm_y = contact_data_in[ci_off+4];
    float c_norm_z = contact_data_in[ci_off+5];
    float c_dist = contact_data_in[ci_off+6];
    int pair_idx = int(contact_data_in[ci_off+7]);

    int pp = pair_idx * 18;
    int body1 = int(pair_props[pp+0]);
    int body2 = int(pair_props[pp+1]);
    int condim = int(pair_props[pp+2]);
    float pair_margin_v = pair_props[pp+3];
    float fri0 = pair_props[pp+4];
    float fri1 = pair_props[pp+5];
    float solref0 = pair_props[pp+9];
    float solref1_v = pair_props[pp+10];
    float si0 = pair_props[pp+11], si1 = pair_props[pp+12];
    float si2 = pair_props[pp+13], si3 = pair_props[pp+14], si4 = pair_props[pp+15];
    float invw_t = pair_props[pp+16];

    bool contact_active = (c_dist < pair_margin_v);
    float pos = c_dist;

    float tc = solref0;
)GLSL";
  if(!refsafe) ss << "    tc = max(tc, 2.0f * TIMESTEP);\n";
  ss << R"GLSL(
    float dmin = clamp(si0, MJMINIMP, MJMAXIMP);
    float dmax_v = clamp(si1, MJMINIMP, MJMAXIMP);
    float width_v = max(si2, MJMINVAL_CV);
    float mid_v = clamp(si3, MJMINIMP, MJMAXIMP);
    float power_v = max(si4, 1.0f);

    float k_val = (tc > 0.0) ? 1.0f/(dmax_v*dmax_v*tc*tc*solref1_v*solref1_v) : -tc/(dmax_v*dmax_v);
    float b_val = (solref1_v > 0.0) ? 2.0f/(dmax_v*tc) : -solref1_v/dmax_v;

    float imp_x_val = abs(pos) / width_v;
    float imp_y_val;
    if (imp_x_val < mid_v) {
        imp_y_val = pow(imp_x_val, power_v) / pow(mid_v, power_v - 1.0f);
    } else {
        imp_y_val = 1.0f - pow(1.0f - imp_x_val, power_v) / pow(1.0f - mid_v, power_v - 1.0f);
    }
    float imp_val = clamp(dmin + imp_y_val * (dmax_v - dmin), dmin, dmax_v);
    if (imp_x_val > 1.0f) imp_val = dmax_v;

    JACP_DECL
    for (int di = 0; di < NV*3; di++) jacp_tmp(di) = 0.0f;

    {
        int rid = int(body_rootid_buf[body2]);
        float ox = c_pos_x - SCOM(rid*3);
        float oy = c_pos_y - SCOM(rid*3+1);
        float oz = c_pos_z - SCOM(rid*3+2);
        for (int di = 0; di < NV; di++) {
            if (body_dof_masks_buf[body2*NV+di] < 0.5f) continue;
            float ax=CDOF(di*6),ay=CDOF(di*6+1),az=CDOF(di*6+2);
            float lx=CDOF(di*6+3),ly=CDOF(di*6+4),lz=CDOF(di*6+5);
            jacp_tmp(di*3+0) += lx + (ay*oz - az*oy);
            jacp_tmp(di*3+1) += ly + (az*ox - ax*oz);
            jacp_tmp(di*3+2) += lz + (ax*oy - ay*ox);
        }
    }
    {
        int rid = int(body_rootid_buf[body1]);
        float ox = c_pos_x - SCOM(rid*3);
        float oy = c_pos_y - SCOM(rid*3+1);
        float oz = c_pos_z - SCOM(rid*3+2);
        for (int di = 0; di < NV; di++) {
            if (body_dof_masks_buf[body1*NV+di] < 0.5f) continue;
            float ax=CDOF(di*6),ay=CDOF(di*6+1),az=CDOF(di*6+2);
            float lx=CDOF(di*6+3),ly=CDOF(di*6+4),lz=CDOF(di*6+5);
            jacp_tmp(di*3+0) -= lx + (ay*oz - az*oy);
            jacp_tmp(di*3+1) -= ly + (az*ox - ax*oz);
            jacp_tmp(di*3+2) -= lz + (ax*oy - ay*ox);
        }
    }
)GLSL";

  ss << "    float j_normal_arr[" << nv << "];\n"
     << "    for (int di = 0; di < NV; di++)\n"
     << "        j_normal_arr[di] = c_norm_x*jacp_tmp(di*3) + c_norm_y*jacp_tmp(di*3+1) + c_norm_z*jacp_tmp(di*3+2);\n";

  if(use_pyramidal) {
    ss << R"GLSL(
    vec3 nn = vec3(c_norm_x, c_norm_y, c_norm_z);
    vec3 tt1, tt2;
    if (abs(nn.z) < 0.999f) tt1 = msl_cross(nn, vec3(0,0,1));
    else tt1 = msl_cross(nn, vec3(0,1,0));
    tt1 = msl_norm(tt1); tt2 = msl_cross(nn, tt1);

    if (condim >= 3 && contact_active) {
        float mu0 = fri0, mu0_sq = mu0*mu0;
        float invw_py = invw_t + mu0_sq*invw_t;
        float r_first = max(invw_py*(1.0f-imp_val)/imp_val, MJMINVAL_CV);
)GLSL";
    ss << "        float r_py = max(2.0f*mu0_sq/" << solver_fmt_float(impratio) << "f*r_first, MJMINVAL_CV);\n";
    ss << R"GLSL(
        float D_py = 1.0f/r_py;
        float mu_vals[2] = float[](fri0, fri1);
        for (int tk = 0; tk < 2; tk++) {
            float mu_k = mu_vals[tk];
            vec3 tang = (tk==0) ? tt1 : tt2;
            for (int di = 0; di < NV; di++) {
                float jt = tang.x*jacp_tmp(di*3)+tang.y*jacp_tmp(di*3+1)+tang.z*jacp_tmp(di*3+2);
                J(row,di) = j_normal_arr[di] + mu_k*jt;
            }
            float jdot = 0.0f;
            for (int di = 0; di < NV; di++) jdot += J(row,di)*qvel_in[qv_off+di];
            efc_D(row) = D_py;
            efc_aref(row) = -b_val*jdot - k_val*imp_val*pos;
            row++;
            for (int di = 0; di < NV; di++) {
                float jt = tang.x*jacp_tmp(di*3)+tang.y*jacp_tmp(di*3+1)+tang.z*jacp_tmp(di*3+2);
                J(row,di) = j_normal_arr[di] - mu_k*jt;
            }
            jdot = 0.0f;
            for (int di = 0; di < NV; di++) jdot += J(row,di)*qvel_in[qv_off+di];
            efc_D(row) = D_py;
            efc_aref(row) = -b_val*jdot - k_val*imp_val*pos;
            row++;
        }
    } else if (contact_active) {
)GLSL";
  } else {
    ss << "    if (contact_active) {\n";
  }

  ss << R"GLSL(
        for (int di = 0; di < NV; di++) J(row,di) = j_normal_arr[di];
        float r = max(invw_t*(1.0f-imp_val)/imp_val, MJMINVAL_CV);
        efc_D(row) = 1.0f/r;
        float jdot = 0.0f;
        for (int di = 0; di < NV; di++) jdot += j_normal_arr[di]*qvel_in[qv_off+di];
        efc_aref(row) = -b_val*jdot - k_val*imp_val*pos;
        row++;
    }
}
barrier(); memoryBarrierBuffer(); memoryBarrierShared();
int nefc = tg_nefc;
SABL1

qacc(int(tid)) = 0.0f;
barrier(); memoryBarrierBuffer();

if (nefc == 0) {
    qfrc_constraint_out[qfc_off + int(tid)] = 0.0f;
    if (tid == 0u) for (int ci = 0; ci < MAX_CON; ci++) contact_fn_out[bid * uint(MAX_CON) + uint(ci)] = 0.0f;
    return;
}

// qacc=0から始めるので M*qacc=0、J*qacc=0(Jaref=-aref)。行ごとに独立なので全スレッドで並列に更新する。
Ma(int(tid)) = 0.0f;
for (int r2 = int(tid); r2 < nefc; r2 += NV) { Jaref(r2) = -efc_aref(r2); EFC_REFRESH(r2); }
barrier(); memoryBarrierBuffer();

{ float s = 0.0f; for (int r2 = 0; r2 < nefc; r2++) s += J(r2,int(tid))*efc_force(r2);
  grad(int(tid)) = Ma(int(tid)) - QFS(int(tid)) - s; }
barrier(); memoryBarrierBuffer();

SABL2
if (tid == 0u) { float gn = 0.0f; for (int i = 0; i < NV; i++) gn += grad(i)*grad(i); tg_g0 = gn; tg_done = 0; }
barrier(); memoryBarrierBuffer(); memoryBarrierShared();
for (int iter = 0; iter < NSOLVE; iter++) {
    search_d(int(tid)) = 0.0f;
    cg_r(int(tid)) = -grad(int(tid));
    cg_p(int(tid)) = -grad(int(tid));
    barrier(); memoryBarrierBuffer();

    if (tid == 0u) { float s = 0.0f; for (int i = 0; i < NV; i++) { float v = cg_r(i); s += v*v; } tg_rr = s; tg_rr0 = s; }
    barrier(); memoryBarrierBuffer(); memoryBarrierShared();

    for (int cg_it = 0; cg_it < CG_ITERS; cg_it++) {
        if (tg_rr < 1e-20f || tg_rr < CG_TOL * tg_rr0) break;  // 残差2乗が初期値の相対CG_TOL未満なら不正確なNewton方向で十分

        float mp = MJMINVAL_SV * cg_p(int(tid));
        for (int j = 0; j < NV; j++) mp += QMV(int(tid)*NV + j) * cg_p(j);

        for (int r2 = int(tid); r2 < nefc; r2 += NV) {
            float s = 0.0f; for (int j = 0; j < NV; j++) s += J(r2,j)*cg_p(j);
            Jv_arr(r2) = efc_D(r2)*act(r2)*s;
        }
        barrier(); memoryBarrierBuffer();

        float jt = 0.0f; for (int r2 = 0; r2 < nefc; r2++) jt += J(r2,int(tid))*Jv_arr(r2);
        cg_Ap(int(tid)) = mp + jt;
        barrier(); memoryBarrierBuffer();

        if (tid == 0u) { float s = 0.0f; for (int i = 0; i < NV; i++) s += cg_p(i)*cg_Ap(i); tg_pAp = s; }
        barrier(); memoryBarrierBuffer(); memoryBarrierShared();

        float alpha_cg = tg_rr / max(tg_pAp, 1e-30f);
        search_d(int(tid)) += alpha_cg * cg_p(int(tid));
        cg_r(int(tid)) -= alpha_cg * cg_Ap(int(tid));
        barrier(); memoryBarrierBuffer();

        float old_rr = tg_rr;
        if (tid == 0u) { float s = 0.0f; for (int i = 0; i < NV; i++) { float v = cg_r(i); s += v*v; } tg_rr = s; }
        barrier(); memoryBarrierBuffer(); memoryBarrierShared();

        float beta_cg = tg_rr / max(old_rr, 1e-30f);
        cg_p(int(tid)) = cg_r(int(tid)) + beta_cg * cg_p(int(tid));
        barrier(); memoryBarrierBuffer();
    }

    SABL3
    { float s = 0.0f; for (int j = 0; j < NV; j++) s += QMV(int(tid)*NV + j)*search_d(j); Mv_arr(int(tid)) = s; }
    barrier(); memoryBarrierBuffer();

LS_BEGIN
    if (tid == 0u) {
        for (int r2 = 0; r2 < nefc; r2++) {
            float s = 0.0f; for (int j = 0; j < NV; j++) s += J(r2,j)*search_d(j);
            Jv_arr(r2) = s;
        }
        float qg=0.0f,lg=0.0f,qc=0.0f,lc=0.0f;
        for (int i = 0; i < NV; i++) { qg += 0.5f*search_d(i)*Mv_arr(i); lg += search_d(i)*(Ma(i)-QFS(i)); }
        for (int r2 = 0; r2 < nefc; r2++) { qc += 0.5f*efc_D(r2)*Jv_arr(r2)*Jv_arr(r2)*act(r2); lc += -efc_force(r2)*Jv_arr(r2); }
        float dnom = 2.0f*(qg+qc);
        float an = clamp(-(lg+lc)/max(dnom, MJMINVAL_SV), -2.0f, 2.0f);
        // コストの導関数に対する二分法で厳密な1次元最小化を行う(区分二次の拘束で固定刻みの候補では1反復の進みが小さいため)。
        float abis = 0.0f;
        {
            float lo_a = 0.0f, hi_a = 2.0f;
            for (int bi = 0; bi < 10; bi++) {
                float mid = 0.5f*(lo_a+hi_a);
                float dd = lg + mid*2.0f*qg;
                for (int r2 = 0; r2 < nefc; r2++) {
                    float x = Jaref(r2) + mid*Jv_arr(r2);
                    float fl = efc_floss(r2); float Dd = efc_D(r2);
                    float dc = 0.0f;
                    if (fl > 0.0f) { dc = (abs(Dd*x) < fl) ? Dd*x : ((x > 0.0f) ? fl : -fl); }
                    else if (x < 0.0f) { dc = Dd*x; }
                    dd += dc*Jv_arr(r2);
                }
                if (dd > 0.0f) hi_a = mid; else lo_a = mid;
            }
            abis = 0.5f*(lo_a+hi_a);
        }
        float als[5] = float[](abis, an, 0.5f*an, 0.1f*an, 0.01f);
        float bc = 1e30f, ba_local = 0.0f;
        float cc2=0.0f; for (int r2 = 0; r2 < nefc; r2++) cc2 += efc_cost(efc_D(r2), Jaref(r2), efc_floss(r2));
        float cg=0.0f; for (int i = 0; i < NV; i++) cg += 0.5f*(Ma(i)-QFS(i))*qacc(i);
        bc = cc2+cg;
        for (int ai = 0; ai < 5; ai++) {
            float a = als[ai]; float tc2=0.0f;
            for (int r2 = 0; r2 < nefc; r2++) { float x = Jaref(r2)+a*Jv_arr(r2); tc2 += efc_cost(efc_D(r2), x, efc_floss(r2)); }
            float tg2 = cg + a*lg + 0.5f*a*a*(2.0f*qg); float tt = tc2+tg2;
            if (tt < bc) { bc=tt; ba_local=a; }
        }
        tg_ba = ba_local;
        // 1反復のコスト改善が相対1e-6未満(または改善なし)なら収束とみなす。静止付近のenvは1〜2反復で収束し、残りのNSOLVE回を丸ごと省ける。
        tg_done = (ba_local == 0.0f) ? 1 : 0;
        for (int r2 = 0; r2 < nefc; r2++) Jaref(r2) += ba_local*Jv_arr(r2);
        for (int r2 = 0; r2 < nefc; r2++) EFC_REFRESH(r2);
    }
LS_END
    SABL4
    barrier(); memoryBarrierBuffer(); memoryBarrierShared();
    float ba = tg_ba;

    qacc(int(tid)) += ba * search_d(int(tid));
    Ma(int(tid)) += ba * Mv_arr(int(tid));
    barrier(); memoryBarrierBuffer();

    { float s = 0.0f; for (int r2 = 0; r2 < nefc; r2++) s += J(r2,int(tid))*efc_force(r2);
      grad(int(tid)) = Ma(int(tid)) - QFS(int(tid)) - s; }
    barrier(); memoryBarrierBuffer();
    // 勾配ノルムが初期値の相対GTOL_REL(2乗比)未満なら収束とみなして打ち切る。静止付近のenvは少ない反復で収束し、残りのNSOLVE回を省ける。
    if (tid == 0u) { float gn = 0.0f; for (int i = 0; i < NV; i++) gn += grad(i)*grad(i); if (gn < GTOL_REL*tg_g0) tg_done = 1; }
    barrier(); memoryBarrierBuffer(); memoryBarrierShared();
    if (tg_done != 0) break;
}

{ float s = 0.0f; for (int r2 = 0; r2 < nefc; r2++) s += J(r2,int(tid))*efc_force(r2);
  qfrc_constraint_out[qfc_off + int(tid)] = s; }
barrier(); memoryBarrierBuffer();
// 接触ごとの法線方向の力(ピラミッド錐は全ファセットの力の和)。cfrc_ext算出用。
if (tid == 0u) {
    int nc_out = int(contact_count_in[bid]);
    for (int ci = 0; ci < nc_out && ci < MAX_CON; ci++) {  // cfrcはcontact_count個しか読まないので128個全部は書かない
        float fn = 0.0f;
        if (ci < tg_ncp) for (int r2 = tg_crow0[ci]; r2 < tg_crow0[ci+1]; r2++) fn += efc_force(r2);
        contact_fn_out[bid * uint(MAX_CON) + uint(ci)] = fn;
    }
}
)GLSL";

  const std::string kLineSearchParallel = R"GLSL(
    // 直線探索を全スレッドで並列に行う(共有メモリ版)。行(拘束)はスレッドに割り当て、合計は共有配列に書いてbarrier後に全スレッドが同じ順で足すので結果は全スレッドで一致する。
    for (int r2 = int(tid); r2 < nefc; r2 += NV) { float s = 0.0f; for (int j = 0; j < NV; j++) s += J(r2,j)*search_d(j); Jv_arr(r2) = s; }
    {
        float sd_ = search_d(int(tid));
        sh_red[int(tid)] = sd_*(Ma(int(tid)) - QFS(int(tid)));
        sh_red[NV+int(tid)] = 0.5f*sd_*Mv_arr(int(tid));
        sh_red[2*NV+int(tid)] = 0.5f*(Ma(int(tid)) - QFS(int(tid)))*qacc(int(tid));
        float qc_p = 0.0f, lc_p = 0.0f;
        for (int r2 = int(tid); r2 < nefc; r2 += NV) { qc_p += 0.5f*efc_D(r2)*Jv_arr(r2)*Jv_arr(r2)*act(r2); lc_p += -efc_force(r2)*Jv_arr(r2); }
        sh_red[3*NV+int(tid)] = qc_p; sh_red[4*NV+int(tid)] = lc_p;
    }
    barrier();
    float lg = 0.0f, qg = 0.0f, cg = 0.0f, qc = 0.0f, lc = 0.0f;
    for (int i = 0; i < NV; i++) { lg += sh_red[i]; qg += sh_red[NV+i]; cg += sh_red[2*NV+i]; qc += sh_red[3*NV+i]; lc += sh_red[4*NV+i]; }
    barrier();
    float dnom = 2.0f*(qg+qc);
    float an = clamp(-(lg+lc)/max(dnom, MJMINVAL_SV), -2.0f, 2.0f);
    // コストの導関数に対する二分法で厳密な1次元最小化を行う。
    float lo_a = 0.0f, hi_a = 2.0f;
    for (int bi = 0; bi < 10; bi++) {
        float mid = 0.5f*(lo_a+hi_a);
        float dd_p = 0.0f;
        for (int r2 = int(tid); r2 < nefc; r2 += NV) {
            float x = Jaref(r2) + mid*Jv_arr(r2);
            float fl = efc_floss(r2); float Dd = efc_D(r2);
            float dc = 0.0f;
            if (fl > 0.0f) { dc = (abs(Dd*x) < fl) ? Dd*x : ((x > 0.0f) ? fl : -fl); }
            else if (x < 0.0f) { dc = Dd*x; }
            dd_p += dc*Jv_arr(r2);
        }
        sh_red[int(tid)] = dd_p;
        barrier();
        float dd = lg + mid*2.0f*qg;
        for (int i = 0; i < NV; i++) dd += sh_red[i];
        barrier();
        if (dd > 0.0f) hi_a = mid; else lo_a = mid;
    }
    float abis = 0.5f*(lo_a+hi_a);
    float als[5] = float[](abis, an, 0.5f*an, 0.1f*an, 0.01f);
    {
        float cc_p = 0.0f; float tc_p0 = 0.0f, tc_p1 = 0.0f, tc_p2 = 0.0f, tc_p3 = 0.0f, tc_p4 = 0.0f;
        for (int r2 = int(tid); r2 < nefc; r2 += NV) {
            float Dd = efc_D(r2), fl = efc_floss(r2), jr = Jaref(r2), jv = Jv_arr(r2);
            cc_p += efc_cost(Dd, jr, fl);
            tc_p0 += efc_cost(Dd, jr + als[0]*jv, fl); tc_p1 += efc_cost(Dd, jr + als[1]*jv, fl); tc_p2 += efc_cost(Dd, jr + als[2]*jv, fl);
            tc_p3 += efc_cost(Dd, jr + als[3]*jv, fl); tc_p4 += efc_cost(Dd, jr + als[4]*jv, fl);
        }
        sh_red[int(tid)] = cc_p; sh_red[NV+int(tid)] = tc_p0; sh_red[2*NV+int(tid)] = tc_p1; sh_red[3*NV+int(tid)] = tc_p2; sh_red[4*NV+int(tid)] = tc_p3; sh_red[5*NV+int(tid)] = tc_p4;
    }
    barrier();
    float cc2 = 0.0f; float tcs[5] = float[](0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    for (int i = 0; i < NV; i++) { cc2 += sh_red[i]; for (int ai = 0; ai < 5; ai++) tcs[ai] += sh_red[(1+ai)*NV+i]; }
    float bc = cc2 + cg, ba_local = 0.0f;
    for (int ai = 0; ai < 5; ai++) {
        float a = als[ai];
        float tt = tcs[ai] + cg + a*lg + 0.5f*a*a*(2.0f*qg);
        if (tt < bc) { bc = tt; ba_local = a; }
    }
    for (int r2 = int(tid); r2 < nefc; r2 += NV) { Jaref(r2) += ba_local*Jv_arr(r2); EFC_REFRESH(r2); }
    if (tid == 0u) {
        tg_ba = ba_local;
        // 1反復のコスト改善が無ければ収束とみなす。
        tg_done = (ba_local == 0.0f) ? 1 : 0;
    }
    barrier();
)GLSL";
  {
    static const int sabl = getenv("MJMLX_SABL") ? atoi(getenv("MJMLX_SABL")) : 0;  // 計測用: 1=拘束構築直後、2=初期勾配直後でqfrc_constraint=0を書いて終了
    std::string body = ss.str();
    // 早期returnで依存が切れると前段がコンパイラに消されるので、各点の状態を出力に書いてから終了する。
    auto rep = [&](const std::string& key, bool on, const std::string& chk) {
        size_t p = body.find(key);
        if (p != std::string::npos) body.replace(p, key.size(), on ? "{ float chk_ = " + chk + "; qfrc_constraint_out[qfc_off + int(tid)] = chk_; return; }" : std::string());
    };
    const std::string rows = "0.0f; for (int r2 = 0; r2 < nefc; r2++) chk_ += J(r2,int(tid))*(efc_D(r2)+efc_aref(r2)+efc_force(r2))";
    rep("SABL0", sabl == 5, "sh_H[0] + sh_J[int(tid)] + solver_scratch[s_off + int(tid)]");
    rep("SABL1", sabl == 1, rows);
    rep("SABL2", sabl == 2, "grad(int(tid))");
    rep("SABL3", sabl == 3, "search_d(int(tid))");
    rep("SABL4", sabl == 4, "tg_ba + search_d(int(tid))");
    {
      size_t b0 = body.find("LS_BEGIN\n"), e0 = body.find("LS_END\n");
      if(b0 != std::string::npos && e0 != std::string::npos) {
        if(use_shared) body.replace(b0, e0 + 7 - b0, kLineSearchParallel);
        else { body.erase(e0, 7); body.erase(b0, 9); }
      }
    }
    // 共有メモリ版はデータがsharedにあり(barrier()がworkgroupメモリの可視性も保証する)、globalのメモリバリアは不要。memoryBarrierBufferは未完了のglobal load/storeを全部待つので高コスト。
    static const bool keep_mbb = getenv("MJMLX_KEEP_MBB") != nullptr;
    if(use_shared && !keep_mbb) {
      const std::string mbb = "memoryBarrierBuffer();";
      for(size_t q = body.find(mbb); q != std::string::npos; q = body.find(mbb, q)) body.erase(q, mbb.size());
    }
    ss.str(body);
  }
  return mkx::fast::compute_kernel("mjmlx_solver", {"qM_in", "qfrc_smooth_in", "cdof_in", "subtree_com_in", "qvel_in", "contact_data_in", "contact_count_in", "pair_props", "body_dof_masks_buf", "body_rootid_buf", "qpos_in", "lim_tab", "fric_tab"}, {"qfrc_constraint_out", "solver_scratch", "contact_fn_out"}, ss.str(), header);
}


// 接触ごとの法線力(solver出力contact_fn)をbodyごとの外力cfrc_ext[B,nb,6]=[torque(3),force(3)]に集約する。接触法線はgeom1->geom2向きなので、body2に+n*Fn、body1に-n*Fnを加える。トルクは0(環境側は力のノルムしか使わない)。
inline mkx::fast::Kernel<> make_cfrc_ext_kernel(int nb, int contact_stride, int max_contacts_per_env) {
  std::ostringstream ss;
  ss << "uint bid = gl_GlobalInvocationID.x;\n"
     << "const int NB = " << nb << ";\n"
     << "const int CON_STRIDE = " << contact_stride << ";\n"
     << "const int MAX_CON = " << max_contacts_per_env << ";\n"
     << R"GLSL(
uint out_off = bid * uint(NB) * 6u;
for (int i = 0; i < NB * 6; i++) cfrc_out[out_off + uint(i)] = 0.0f;
int ncon = int(contact_count_in[bid]);
for (int ci = 0; ci < ncon; ci++) {
    uint ci_off = bid * uint(MAX_CON) * uint(CON_STRIDE) + uint(ci * CON_STRIDE);
    vec3 n = vec3(contact_data_in[ci_off+3], contact_data_in[ci_off+4], contact_data_in[ci_off+5]);
    int pp = int(contact_data_in[ci_off+7]) * 18;
    int b1 = int(pair_props[pp+0]), b2 = int(pair_props[pp+1]);
    vec3 f = n * contact_fn_in[bid * uint(MAX_CON) + uint(ci)];
    for (int k = 0; k < 3; k++) {
        cfrc_out[out_off + uint(b2 * 6 + 3 + k)] += f[k];
        cfrc_out[out_off + uint(b1 * 6 + 3 + k)] -= f[k];
    }
}
)GLSL";
  return mkx::fast::compute_kernel("mjmlx_cfrc_ext", {"contact_data_in", "contact_count_in", "contact_fn_in", "pair_props"}, {"cfrc_out"}, ss.str(), "");
}

} // namespace mjmlx::mkx_kernels
