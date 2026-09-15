#pragma once
// Vendored from 20niship/mlx-cross-platform tests/mujoco/euler.hpp at commit 3202339e12231f3a68f9ee67e0b6085a33c52286.

#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <mkx/ops/fast_kernel.hpp>

// MuJoCo-MLX-Cpp make_euler_source(nv<=80版)の逐語GLSL移植: 質量行列をコレスキー分解してqacc=solve(qM,qfrc)、半陰的Euler積分でqvel/qposを更新。joint種別ごとの更新式はホスト側ループでGLSLソースへ展開(モデルごとにソース自体が変わる)。GLSLは配列初期化子がfloat[](...)構文限定な点に注意。

namespace mjmlx::mkx_kernels {

// GLSLの浮動小数点リテラルは小数点が必須("0f"は不正、"0.0f"が必要)なため整形する。
inline std::string euler_fmt_float(float x) {
  std::ostringstream ss;
  ss << x;
  std::string s = ss.str();
  if(s.find('.') == std::string::npos && s.find('e') == std::string::npos) s += ".0";
  return s;
}

inline mkx::fast::Kernel<> make_euler_kernel(int nv, int nq, float dt, const std::vector<int>& simple_qa, const std::vector<int>& simple_da, const std::vector<std::pair<int, int>>& free_joints, const std::vector<std::pair<int, int>>& ball_joints, const std::vector<float>& dof_damping_vals) {
  int n              = nv;
  std::string dt_str = euler_fmt_float(dt);
  std::ostringstream ss;

  ss << "uint batch_idx = gl_GlobalInvocationID.x;\n"
     << "const uint n = " << n << "u;\n"
     << "uint moff = batch_idx * n * n;\n"
     << "uint voff = batch_idx * n;\n"
     << "uint qoff = batch_idx * " << nq << "u;\n"

     << "float a[" << n << " * " << n << "];\n"
     << "for (uint i = 0u; i < n * n; i++) a[i] = qM[moff + i];\n";

  for(int i = 0; i < nv; i++) {
    if(dof_damping_vals[static_cast<size_t>(i)] != 0.0f) {
      ss << "a[" << i * n + i << "] += " << dt_str << "f * " << euler_fmt_float(dof_damping_vals[static_cast<size_t>(i)]) << "f;\n";
    }
  }
  for(int i = 0; i < n; i++) ss << "a[" << i * n + i << "] += 1e-6f;\n";

  ss << "float rhs[" << n << "];\n"
     << "for (uint i = 0u; i < n; i++) rhs[i] = qfrc_smooth[voff+i] + qfrc_constraint[voff+i];\n"

     << "float l[" << n << " * " << n << "];\n"
     << "for (uint i = 0u; i < n * n; i++) l[i] = 0.0f;\n"
     << "for (uint j = 0u; j < n; j++) {\n"
     << "  float s = 0.0f;\n"
     << "  for (uint k = 0u; k < j; k++) s += l[j*n+k]*l[j*n+k];\n"
     << "  float diagv = a[j*n+j] - s;\n"
     << "  l[j*n+j] = sqrt(max(diagv, 1e-6f));\n"
     << "  for (uint i = j+1u; i < n; i++) {\n"
     << "    float s2 = 0.0f;\n"
     << "    for (uint k = 0u; k < j; k++) s2 += l[i*n+k]*l[j*n+k];\n"
     << "    l[i*n+j] = (a[i*n+j] - s2) / l[j*n+j];\n"
     << "  }\n"
     << "}\n"

     << "float y[" << n << "];\n"
     << "for (uint i = 0u; i < n; i++) {\n"
     << "  float s = 0.0f;\n"
     << "  for (uint k = 0u; k < i; k++) s += l[i*n+k]*y[k];\n"
     << "  y[i] = (rhs[i] - s) / l[i*n+i];\n"
     << "}\n"

     << "float qacc[" << n << "];\n"
     << "for (int i = " << (n - 1) << "; i >= 0; i--) {\n"
     << "  float s = 0.0f;\n"
     << "  for (uint k = uint(i)+1u; k < n; k++) s += l[k*n+uint(i)]*qacc[k];\n"
     << "  qacc[i] = (y[i] - s) / l[i*n+i];\n"
     << "}\n"

     << "float new_qvel[" << n << "];\n"
     << "for (uint i = 0u; i < n; i++) {\n"
     << "  float v = qvel_in[voff+i] + qacc[i] * " << dt_str << "f;\n"
     << "  new_qvel[i] = clamp(v, -1e4f, 1e4f);\n"
     << "}\n"

     << "float new_qpos[" << nq << "];\n";

  std::set<int> touched;
  for(int qa : simple_qa) touched.insert(qa);
  for(auto& [qa, da] : free_joints) {
    (void)da;
    for(int i = 0; i < 7; i++) touched.insert(qa + i);
  }
  for(auto& [qa, da] : ball_joints) {
    (void)da;
    for(int i = 0; i < 4; i++) touched.insert(qa + i);
  }
  for(int i = 0; i < nq; i++) {
    if(touched.find(i) == touched.end()) ss << "new_qpos[" << i << "] = qpos_in[qoff + " << i << "];\n";
  }

  for(size_t i = 0; i < simple_qa.size(); i++) {
    ss << "new_qpos[" << simple_qa[i] << "] = qpos_in[qoff + " << simple_qa[i] << "] + " << dt << "f * new_qvel[" << simple_da[i] << "];\n";
  }

  for(auto& [qa, da] : free_joints) {
    ss << "new_qpos[" << qa << "] = qpos_in[qoff+" << qa << "] + " << dt_str << "f * new_qvel[" << da << "];\n"
       << "new_qpos[" << qa + 1 << "] = qpos_in[qoff+" << qa + 1 << "] + " << dt_str << "f * new_qvel[" << da + 1 << "];\n"
       << "new_qpos[" << qa + 2 << "] = qpos_in[qoff+" << qa + 2 << "] + " << dt_str << "f * new_qvel[" << da + 2 << "];\n"
       << "{\n"
       << "  float w[3] = float[](new_qvel[" << da + 3 << "], new_qvel[" << da + 4 << "], new_qvel[" << da + 5 << "]);\n"
       << "  float wnorm = sqrt(w[0]*w[0]+w[1]*w[1]+w[2]*w[2]);\n"
       << "  float angle = " << dt_str << "f * wnorm;\n"
       << "  float ha = angle * 0.5f;\n"
       << "  float sinha = (wnorm > 1e-12f) ? sin(ha)/wnorm : 0.5f*" << dt_str << "f;\n"
       << "  float cosha = cos(ha);\n"
       << "  float dq0=cosha, dq1=sinha*w[0], dq2=sinha*w[1], dq3=sinha*w[2];\n"
       << "  float q0=qpos_in[qoff+" << qa + 3 << "], q1=qpos_in[qoff+" << qa + 4 << "];\n"
       << "  float q2=qpos_in[qoff+" << qa + 5 << "], q3=qpos_in[qoff+" << qa + 6 << "];\n"
       << "  float nq0=q0*dq0-q1*dq1-q2*dq2-q3*dq3;\n"
       << "  float nq1=q0*dq1+q1*dq0+q2*dq3-q3*dq2;\n"
       << "  float nq2=q0*dq2-q1*dq3+q2*dq0+q3*dq1;\n"
       << "  float nq3=q0*dq3+q1*dq2-q2*dq1+q3*dq0;\n"
       << "  float qn=sqrt(nq0*nq0+nq1*nq1+nq2*nq2+nq3*nq3);\n"
       << "  float inv_qn=(qn>1e-12f)?1.0f/qn:1.0f;\n"
       << "  new_qpos[" << qa + 3 << "]=nq0*inv_qn;\n"
       << "  new_qpos[" << qa + 4 << "]=nq1*inv_qn;\n"
       << "  new_qpos[" << qa + 5 << "]=nq2*inv_qn;\n"
       << "  new_qpos[" << qa + 6 << "]=nq3*inv_qn;\n"
       << "}\n";
  }

  for(auto& [qa, da] : ball_joints) {
    ss << "{\n"
       << "  float w[3] = float[](new_qvel[" << da << "], new_qvel[" << da + 1 << "], new_qvel[" << da + 2 << "]);\n"
       << "  float wnorm = sqrt(w[0]*w[0]+w[1]*w[1]+w[2]*w[2]);\n"
       << "  float angle = " << dt_str << "f * wnorm;\n"
       << "  float ha = angle * 0.5f;\n"
       << "  float sinha = (wnorm > 1e-12f) ? sin(ha)/wnorm : 0.5f*" << dt_str << "f;\n"
       << "  float cosha = cos(ha);\n"
       << "  float dq0=cosha, dq1=sinha*w[0], dq2=sinha*w[1], dq3=sinha*w[2];\n"
       << "  float q0=qpos_in[qoff+" << qa << "], q1=qpos_in[qoff+" << qa + 1 << "];\n"
       << "  float q2=qpos_in[qoff+" << qa + 2 << "], q3=qpos_in[qoff+" << qa + 3 << "];\n"
       << "  float nq0=q0*dq0-q1*dq1-q2*dq2-q3*dq3;\n"
       << "  float nq1=q0*dq1+q1*dq0+q2*dq3-q3*dq2;\n"
       << "  float nq2=q0*dq2-q1*dq3+q2*dq0+q3*dq1;\n"
       << "  float nq3=q0*dq3+q1*dq2-q2*dq1+q3*dq0;\n"
       << "  float qn=sqrt(nq0*nq0+nq1*nq1+nq2*nq2+nq3*nq3);\n"
       << "  float inv_qn=(qn>1e-12f)?1.0f/qn:1.0f;\n"
       << "  new_qpos[" << qa << "]=nq0*inv_qn;\n"
       << "  new_qpos[" << qa + 1 << "]=nq1*inv_qn;\n"
       << "  new_qpos[" << qa + 2 << "]=nq2*inv_qn;\n"
       << "  new_qpos[" << qa + 3 << "]=nq3*inv_qn;\n"
       << "}\n";
  }

  ss << "for (uint i = 0u; i < " << nq << "u; i++) qpos_out[qoff+i] = new_qpos[i];\n"
     << "for (uint i = 0u; i < n; i++) qvel_out[voff+i] = new_qvel[i];\n"
     << "for (uint i = 0u; i < n; i++) qacc_out[voff+i] = qacc[i];\n";

  return mkx::fast::compute_kernel("mjmlx_euler", {"qM", "qfrc_smooth", "qfrc_constraint", "qvel_in", "qpos_in"}, {"qpos_out", "qvel_out", "qacc_out"}, ss.str(), "");
}

} // namespace mjmlx::mkx_kernels
