#pragma once
// Vendored from 20niship/mlx-cross-platform tests/mujoco/forward.hpp at commit 3202339e12231f3a68f9ee67e0b6085a33c52286.

#include <array>
#include <sstream>
#include <string>
#include <vector>

#include <mkx/ops/fast_kernel.hpp>

// MuJoCo-MLX-Cpp
// make_forward_source(CRB法での質量行列組み立て+RNEでのqfrc_bias算出)の逐語GLSL移植。mkxにはMuJoCoのModel型が無いため、元コードがポインタで直接読んでいたモデル定数はホスト配列を引数で渡す設計にした。act_bias1/act_bias2は元のMSLでも本体未参照の死んだ定数のため省略、初期化子リスト`{a,b,c}`はGLSL非対応のため`TYPE[](a,b,c)`構文に置換した。

namespace mjmlx::mkx_kernels {

namespace detail_forward {

inline std::string int_array_literal(const std::vector<int>& v) {
  std::ostringstream ss;
  ss << "int[](";
  for(size_t i = 0; i < v.size(); i++) ss << v[i] << (i + 1 < v.size() ? "," : "");
  ss << ")";
  return ss.str();
}

// GLSLの浮動小数点リテラルは小数点が必須("0f"は不正、"0.0f"が必要)なため整形する。
inline std::string fmt_float(float x) {
  std::ostringstream ss;
  ss << x;
  std::string s = ss.str();
  if(s.find('.') == std::string::npos && s.find('e') == std::string::npos) s += ".0";
  return s;
}

inline std::string float_array_literal(const std::vector<float>& v) {
  std::ostringstream ss;
  ss << "float[](";
  for(size_t i = 0; i < v.size(); i++) ss << fmt_float(v[i]) << "f" << (i + 1 < v.size() ? "," : "");
  ss << ")";
  return ss.str();
}

} // namespace detail_forward

inline mkx::fast::Kernel<> make_forward_kernel(int nb, int nv, int nq, int nu, int njnt, float dt, std::array<float, 3> gravity, const std::vector<int>& body_parentid, const std::vector<int>& body_rootid, const std::vector<float>& body_mass, const std::vector<float>& body_inertia,
                                             const std::vector<int>& dof_bodyid, const std::vector<int>& dof_parentid, const std::vector<float>& dof_damping, const std::vector<float>& dof_armature, const std::vector<float>& dof_stiffness, const std::vector<int>& dof_qposadr,
                                             const std::vector<float>& qpos_spring, const std::vector<float>& act_gain0, const std::vector<float>& act_bias0, const std::vector<int>& dof_jtype, const std::vector<int>& dof_rotaxis, const std::vector<int>& dof_jid,
                                             const std::vector<std::vector<int>>& body_dofs, int jnt_dofadr0) {
  using namespace detail_forward;

  int off_crb          = 0;
  int off_cdof         = nb * 10;
  int off_cdof_dot     = off_cdof + nv * 6;
  int off_cacc         = off_cdof_dot + nv * 6;
  int off_cfrc         = off_cacc + nb * 6;
  int off_subpos       = off_cfrc + nb * 6;
  int off_submass      = off_subpos + nb * 3;
  int off_qfrc_bias    = off_submass + nb;
  int off_qfrc_passive = off_qfrc_bias + nv;
  int scratch_per_env  = off_qfrc_passive + nv;

  std::ostringstream ss;

  ss << "uint bid = gl_GlobalInvocationID.x;\n"
     << "const int NB = " << nb << ";\n"
     << "const int NV = " << nv << ";\n"
     << "const int NQ = " << nq << ";\n"
     << "const int NU = " << nu << ";\n"
     << "const int NJNT = " << njnt << ";\n"
     << "const int SCRATCH_SZ = " << scratch_per_env << ";\n\n";

  ss << "uint xip_off = bid * NB * 3;\n"
     << "uint xim_off = bid * NB * 9;\n"
     << "uint xa_off  = bid * NJNT * 3;\n"
     << "uint xax_off = bid * NJNT * 3;\n"
     << "uint xm_off  = bid * NB * 9;\n"
     << "uint q_off   = bid * NQ;\n"
     << "uint v_off   = bid * NV;\n"
     << "uint u_off   = bid * NU;\n"
     << "uint qM_off  = bid * NV * NV;\n"
     << "uint qfs_off = bid * NV;\n"
     << "uint sc_off  = bid * NB * 3;\n"
     << "uint ci_off  = bid * NB * 10;\n"
     << "uint cv_off  = bid * NB * 6;\n"
     << "uint qa_off  = bid * NV;\n"
     << "uint s_off   = bid * SCRATCH_SZ;\n\n";

  ss << "const int S_CRB = " << off_crb << ";\n"
     << "const int S_CDOF = " << off_cdof << ";\n"
     << "const int S_CDOFD = " << off_cdof_dot << ";\n"
     << "const int S_CACC = " << off_cacc << ";\n"
     << "const int S_CFRC = " << off_cfrc << ";\n"
     << "const int S_SUBP = " << off_subpos << ";\n"
     << "const int S_SUBM = " << off_submass << ";\n"
     << "const int S_BIAS = " << off_qfrc_bias << ";\n"
     << "const int S_PASS = " << off_qfrc_passive << ";\n\n";

  ss << "const int body_par[" << nb << "] = " << int_array_literal(body_parentid) << ";\n";
  ss << "const int body_root[" << nb << "] = " << int_array_literal(body_rootid) << ";\n";
  ss << "const float body_mass[" << nb << "] = " << float_array_literal(body_mass) << ";\n";
  ss << "const float body_inert[" << nb * 3 << "] = " << float_array_literal(body_inertia) << ";\n";
  ss << "const int dof_bodyid[" << nv << "] = " << int_array_literal(dof_bodyid) << ";\n";
  ss << "const int dof_par[" << nv << "] = " << int_array_literal(dof_parentid) << ";\n";
  ss << "const float dof_damp[" << nv << "] = " << float_array_literal(dof_damping) << ";\n";
  ss << "const float dof_arm[" << nv << "] = " << float_array_literal(dof_armature) << ";\n";
  ss << "const float dof_stiff[" << nv << "] = " << float_array_literal(dof_stiffness) << ";\n";
  ss << "const int dof_qa[" << nv << "] = " << int_array_literal(dof_qposadr) << ";\n";
  ss << "const float qpos_spr[" << nq << "] = " << float_array_literal(qpos_spring) << ";\n";

  if(nu > 0) {
    ss << "const float act_g0[" << nu << "] = " << float_array_literal(act_gain0) << ";\n";
    ss << "const float act_b0[" << nu << "] = " << float_array_literal(act_bias0) << ";\n";
  }

  ss << "const int dof_jtype[" << nv << "] = " << int_array_literal(dof_jtype) << ";\n";
  ss << "const int dof_rotaxis[" << nv << "] = " << int_array_literal(dof_rotaxis) << ";\n";
  ss << "const int dof_jid[" << nv << "] = " << int_array_literal(dof_jid) << ";\n\n";

  // ═══ Phase A: subtree COM, cinert, cdof ═══
  ss << "// Phase A: subtree COM, cinert, cdof\n";

  ss << "for (int b = 0; b < NB; b++) {\n"
     << "  float m = body_mass[b];\n"
     << "  scratch[s_off + S_SUBM + b] = m;\n"
     << "  for (int k = 0; k < 3; k++)\n"
     << "    scratch[s_off + S_SUBP + b*3+k] = xipos[xip_off + b*3+k] * m;\n"
     << "}\n";

  ss << "for (int b = NB-1; b >= 1; b--) {\n"
     << "  int p = body_par[b];\n"
     << "  scratch[s_off + S_SUBM + p] += scratch[s_off + S_SUBM + b];\n"
     << "  for (int k = 0; k < 3; k++)\n"
     << "    scratch[s_off + S_SUBP + p*3+k] += scratch[s_off + S_SUBP + b*3+k];\n"
     << "}\n";

  ss << "for (int b = 0; b < NB; b++) {\n"
     << "  float sm = max(scratch[s_off + S_SUBM + b], 1e-8f);\n"
     << "  for (int k = 0; k < 3; k++)\n"
     << "    subtree_com_out[sc_off + b*3+k] = scratch[s_off + S_SUBP + b*3+k] / sm;\n"
     << "}\n\n";

  ss << "for (int b = 0; b < NB; b++) {\n"
     << "  float m = body_mass[b];\n"
     << "  int rid = body_root[b];\n"
     << "  float off[3];\n"
     << "  for (int k = 0; k < 3; k++)\n"
     << "    off[k] = xipos[xip_off + b*3+k] - subtree_com_out[sc_off + rid*3+k];\n"
     << "  float R[9]; for (int k=0;k<9;k++) R[k] = ximat[xim_off + b*9+k];\n"
     << "  float Ix = body_inert[b*3], Iy = body_inert[b*3+1], Iz = body_inert[b*3+2];\n"
     << "  float RI[9] = float[](R[0]*Ix,R[1]*Iy,R[2]*Iz, R[3]*Ix,R[4]*Iy,R[5]*Iz, R[6]*Ix,R[7]*Iy,R[8]*Iz);\n"
     << "  float Ig[9];\n"
     << "  for (int r=0;r<3;r++) for (int c=0;c<3;c++) {\n"
     << "    float s=0.0f; for (int k=0;k<3;k++) s += RI[r*3+k]*R[c*3+k];\n"
     << "    Ig[r*3+c] = s;\n"
     << "  }\n"
     << "  float d2 = off[0]*off[0]+off[1]*off[1]+off[2]*off[2];\n"
     << "  Ig[0] += m*(d2 - off[0]*off[0]); Ig[4] += m*(d2 - off[1]*off[1]); Ig[8] += m*(d2 - off[2]*off[2]);\n"
     << "  Ig[1] -= m*off[0]*off[1]; Ig[3] -= m*off[1]*off[0];\n"
     << "  Ig[2] -= m*off[0]*off[2]; Ig[6] -= m*off[2]*off[0];\n"
     << "  Ig[5] -= m*off[1]*off[2]; Ig[7] -= m*off[2]*off[1];\n"
     << "  cinert_out[ci_off + b*10+0] = Ig[0];\n"
     << "  cinert_out[ci_off + b*10+1] = Ig[4];\n"
     << "  cinert_out[ci_off + b*10+2] = Ig[8];\n"
     << "  cinert_out[ci_off + b*10+3] = Ig[1];\n"
     << "  cinert_out[ci_off + b*10+4] = Ig[2];\n"
     << "  cinert_out[ci_off + b*10+5] = Ig[5];\n"
     << "  cinert_out[ci_off + b*10+6] = off[0]*m;\n"
     << "  cinert_out[ci_off + b*10+7] = off[1]*m;\n"
     << "  cinert_out[ci_off + b*10+8] = off[2]*m;\n"
     << "  cinert_out[ci_off + b*10+9] = m;\n"
     << "}\n\n";

  ss << "for (int di = 0; di < NV; di++) {\n"
     << "  int bdi = dof_bodyid[di];\n"
     << "  int jt = dof_jtype[di];\n"
     << "  int ji = dof_jid[di];\n"
     << "  int ra = dof_rotaxis[di];\n"
     << "  int rid = body_root[bdi];\n"
     << "  float rc[3]; for (int k=0;k<3;k++) rc[k] = subtree_com_out[sc_off + rid*3+k];\n"
     << "  float c6[6] = float[](0.0f,0.0f,0.0f,0.0f,0.0f,0.0f);\n"
     << "  if (jt == 3) {\n"
     << "    float ax[3] = float[](xaxis[xax_off+ji*3], xaxis[xax_off+ji*3+1], xaxis[xax_off+ji*3+2]);\n"
     << "    float anc[3] = float[](xanchor[xa_off+ji*3], xanchor[xa_off+ji*3+1], xanchor[xa_off+ji*3+2]);\n"
     << "    float d[3]; for (int k=0;k<3;k++) d[k] = rc[k] - anc[k];\n"
     << "    c6[0]=ax[0]; c6[1]=ax[1]; c6[2]=ax[2];\n"
     << "    c6[3]=ax[1]*d[2]-ax[2]*d[1]; c6[4]=ax[2]*d[0]-ax[0]*d[2]; c6[5]=ax[0]*d[1]-ax[1]*d[0];\n"
     << "  } else if (jt == 0 && ra < 0) {\n"
     << "    int sub = di - " << (njnt > 0 ? jnt_dofadr0 : 0) << ";\n"
     << "    c6[3+sub] = 1.0f;\n"
     << "  } else if (jt == 0 && ra >= 0) {\n"
     << "    float rm[9]; for (int k=0;k<9;k++) rm[k] = xmat[xm_off + bdi*9+k];\n"
     << "    float col[3] = float[](rm[ra], rm[3+ra], rm[6+ra]);\n"
     << "    float anc[3] = float[](xanchor[xa_off+ji*3], xanchor[xa_off+ji*3+1], xanchor[xa_off+ji*3+2]);\n"
     << "    float d[3]; for (int k=0;k<3;k++) d[k] = rc[k] - anc[k];\n"
     << "    c6[0]=col[0]; c6[1]=col[1]; c6[2]=col[2];\n"
     << "    c6[3]=col[1]*d[2]-col[2]*d[1]; c6[4]=col[2]*d[0]-col[0]*d[2]; c6[5]=col[0]*d[1]-col[1]*d[0];\n"
     << "  } else if (jt == 2) {\n"
     << "    float ax[3] = float[](xaxis[xax_off+ji*3], xaxis[xax_off+ji*3+1], xaxis[xax_off+ji*3+2]);\n"
     << "    c6[3]=ax[0]; c6[4]=ax[1]; c6[5]=ax[2];\n"
     << "  } else if (jt == 1) {\n"
     << "    float rm[9]; for (int k=0;k<9;k++) rm[k] = xmat[xm_off + bdi*9+k];\n"
     << "    float col[3] = float[](rm[ra], rm[3+ra], rm[6+ra]);\n"
     << "    float anc[3] = float[](xanchor[xa_off+ji*3], xanchor[xa_off+ji*3+1], xanchor[xa_off+ji*3+2]);\n"
     << "    float d[3]; for (int k=0;k<3;k++) d[k] = rc[k] - anc[k];\n"
     << "    c6[0]=col[0]; c6[1]=col[1]; c6[2]=col[2];\n"
     << "    c6[3]=col[1]*d[2]-col[2]*d[1]; c6[4]=col[2]*d[0]-col[0]*d[2]; c6[5]=col[0]*d[1]-col[1]*d[0];\n"
     << "  }\n"
     << "  for (int k=0;k<6;k++) scratch[s_off + S_CDOF + di*6+k] = c6[k];\n"
     << "}\n\n";

  // ═══ Phase B: CRB → mass matrix ═══
  ss << "// Phase B: CRB + mass matrix\n";

  ss << "for (int b=0;b<NB;b++) for (int k=0;k<10;k++)\n"
     << "  scratch[s_off + S_CRB + b*10+k] = cinert_out[ci_off + b*10+k];\n";

  ss << "for (int k=0;k<10;k++) scratch[s_off + S_CRB + k] = 0.0f;\n";

  ss << "for (int b=NB-1;b>=1;b--) {\n"
     << "  int p = body_par[b];\n"
     << "  for (int k=0;k<10;k++) scratch[s_off + S_CRB + p*10+k] += scratch[s_off + S_CRB + b*10+k];\n"
     << "}\n\n";

  ss << "for (int i=0;i<NV*NV;i++) qM_out[qM_off+i] = 0.0f;\n";

  ss << "for (int i=0;i<NV;i++) {\n"
     << "  int bi = dof_bodyid[i];\n"
     << "  float I00=scratch[s_off+S_CRB+bi*10], I11=scratch[s_off+S_CRB+bi*10+1], I22=scratch[s_off+S_CRB+bi*10+2];\n"
     << "  float I01=scratch[s_off+S_CRB+bi*10+3], I02=scratch[s_off+S_CRB+bi*10+4], I12=scratch[s_off+S_CRB+bi*10+5];\n"
     << "  float px=scratch[s_off+S_CRB+bi*10+6], py=scratch[s_off+S_CRB+bi*10+7], pz=scratch[s_off+S_CRB+bi*10+8];\n"
     << "  float mass=scratch[s_off+S_CRB+bi*10+9];\n"
     << "  float w0=scratch[s_off+S_CDOF+i*6], w1=scratch[s_off+S_CDOF+i*6+1], w2=scratch[s_off+S_CDOF+i*6+2];\n"
     << "  float l0=scratch[s_off+S_CDOF+i*6+3], l1=scratch[s_off+S_CDOF+i*6+4], l2=scratch[s_off+S_CDOF+i*6+5];\n"
     << "  float cc0 = I00*w0+I01*w1+I02*w2 + (py*l2-pz*l1);\n"
     << "  float cc1 = I01*w0+I11*w1+I12*w2 + (pz*l0-px*l2);\n"
     << "  float cc2 = I02*w0+I12*w1+I22*w2 + (px*l1-py*l0);\n"
     << "  float cc3 = mass*l0 - (py*w2-pz*w1);\n"
     << "  float cc4 = mass*l1 - (pz*w0-px*w2);\n"
     << "  float cc5 = mass*l2 - (px*w1-py*w0);\n"
     << "  int j = i;\n"
     << "  while (j >= 0) {\n"
     << "    float dot = 0.0f;\n"
     << "    for (int k=0;k<6;k++) {\n"
     << "      float ck = (k==0?cc0:k==1?cc1:k==2?cc2:k==3?cc3:k==4?cc4:cc5);\n"
     << "      dot += ck * scratch[s_off+S_CDOF+j*6+k];\n"
     << "    }\n"
     << "    qM_out[qM_off + i*NV+j] = dot;\n"
     << "    qM_out[qM_off + j*NV+i] = dot;\n"
     << "    j = dof_par[j];\n"
     << "  }\n"
     << "}\n";

  ss << "for (int i=0;i<NV;i++) qM_out[qM_off + i*NV+i] += dof_arm[i];\n\n";

  // ═══ Phase C: COM velocity ═══
  ss << "// Phase C: COM velocity\n";

  ss << "for (int b=0;b<NB;b++) for (int k=0;k<6;k++) { cvel_out[cv_off+b*6+k]=0.0f; scratch[s_off+S_CDOFD+k]=0.0f; }\n"
     << "for (int di=0;di<NV;di++) for (int k=0;k<6;k++) scratch[s_off+S_CDOFD+di*6+k]=0.0f;\n";

  for(int b = 1; b < nb; b++) {
    int pid = body_parentid[static_cast<size_t>(b)];
    ss << "{\n"
       << "  float cv[6]; for (int k=0;k<6;k++) cv[k] = cvel_out[cv_off+" << pid << "*6+k];\n";

    for(int di : body_dofs[static_cast<size_t>(b)]) {
      ss << "  {\n"
         << "    float ua[3] = float[](cv[0],cv[1],cv[2]);\n"
         << "    float ul[3] = float[](cv[3],cv[4],cv[5]);\n"
         << "    float va[3],vl[3]; for (int k=0;k<3;k++) { va[k]=scratch[s_off+S_CDOF+" << di << "*6+k]; vl[k]=scratch[s_off+S_CDOF+" << di << "*6+3+k]; }\n"
         << "    scratch[s_off+S_CDOFD+" << di << "*6+0]=ua[1]*va[2]-ua[2]*va[1];\n"
         << "    scratch[s_off+S_CDOFD+" << di << "*6+1]=ua[2]*va[0]-ua[0]*va[2];\n"
         << "    scratch[s_off+S_CDOFD+" << di << "*6+2]=ua[0]*va[1]-ua[1]*va[0];\n"
         << "    scratch[s_off+S_CDOFD+" << di << "*6+3]=ul[1]*va[2]-ul[2]*va[1]+ua[1]*vl[2]-ua[2]*vl[1];\n"
         << "    scratch[s_off+S_CDOFD+" << di << "*6+4]=ul[2]*va[0]-ul[0]*va[2]+ua[2]*vl[0]-ua[0]*vl[2];\n"
         << "    scratch[s_off+S_CDOFD+" << di << "*6+5]=ul[0]*va[1]-ul[1]*va[0]+ua[0]*vl[1]-ua[1]*vl[0];\n"
         << "    float qv = qvel[v_off+" << di << "];\n"
         << "    for (int k=0;k<6;k++) cv[k] += scratch[s_off+S_CDOF+" << di << "*6+k] * qv;\n"
         << "  }\n";
    }

    ss << "  for (int k=0;k<6;k++) cvel_out[cv_off+" << b << "*6+k] = cv[k];\n"
       << "}\n";
  }
  ss << "\n";

  // ═══ Phase D: RNE (Newton-Euler) ═══
  ss << "// Phase D: RNE\n";

  ss << "for (int b=0;b<NB;b++) for (int k=0;k<6;k++) scratch[s_off+S_CACC+b*6+k]=0.0f;\n"
     << "scratch[s_off+S_CACC+3] = " << fmt_float(-gravity[0]) << "f;\n"
     << "scratch[s_off+S_CACC+4] = " << fmt_float(-gravity[1]) << "f;\n"
     << "scratch[s_off+S_CACC+5] = " << fmt_float(-gravity[2]) << "f;\n";

  for(int b = 1; b < nb; b++) {
    int pid = body_parentid[static_cast<size_t>(b)];
    ss << "{\n"
       << "  float ac[6]; for (int k=0;k<6;k++) ac[k] = scratch[s_off+S_CACC+" << pid << "*6+k];\n";
    for(int di : body_dofs[static_cast<size_t>(b)]) {
      ss << "  { float qv=qvel[v_off+" << di << "]; for (int k=0;k<6;k++) ac[k]+=scratch[s_off+S_CDOFD+" << di << "*6+k]*qv; }\n";
    }
    ss << "  for (int k=0;k<6;k++) scratch[s_off+S_CACC+" << b << "*6+k]=ac[k];\n"
       << "}\n";
  }

  ss << "for (int b=0;b<NB;b++) {\n"
     << "  float I00=cinert_out[ci_off+b*10], I11=cinert_out[ci_off+b*10+1], I22=cinert_out[ci_off+b*10+2];\n"
     << "  float I01=cinert_out[ci_off+b*10+3], I02=cinert_out[ci_off+b*10+4], I12=cinert_out[ci_off+b*10+5];\n"
     << "  float px=cinert_out[ci_off+b*10+6], py=cinert_out[ci_off+b*10+7], pz=cinert_out[ci_off+b*10+8], mass=cinert_out[ci_off+b*10+9];\n"
     << "  float aw0=scratch[s_off+S_CACC+b*6], aw1=scratch[s_off+S_CACC+b*6+1], aw2=scratch[s_off+S_CACC+b*6+2];\n"
     << "  float al0=scratch[s_off+S_CACC+b*6+3], al1=scratch[s_off+S_CACC+b*6+4], al2=scratch[s_off+S_CACC+b*6+5];\n"
     << "  float Ia0=I00*aw0+I01*aw1+I02*aw2+(py*al2-pz*al1);\n"
     << "  float Ia1=I01*aw0+I11*aw1+I12*aw2+(pz*al0-px*al2);\n"
     << "  float Ia2=I02*aw0+I12*aw1+I22*aw2+(px*al1-py*al0);\n"
     << "  float Ia3=mass*al0-(py*aw2-pz*aw1);\n"
     << "  float Ia4=mass*al1-(pz*aw0-px*aw2);\n"
     << "  float Ia5=mass*al2-(px*aw1-py*aw0);\n"
     << "  float vw0=cvel_out[cv_off+b*6], vw1=cvel_out[cv_off+b*6+1], vw2=cvel_out[cv_off+b*6+2];\n"
     << "  float vl0=cvel_out[cv_off+b*6+3], vl1=cvel_out[cv_off+b*6+4], vl2=cvel_out[cv_off+b*6+5];\n"
     << "  float Iv0=I00*vw0+I01*vw1+I02*vw2+(py*vl2-pz*vl1);\n"
     << "  float Iv1=I01*vw0+I11*vw1+I12*vw2+(pz*vl0-px*vl2);\n"
     << "  float Iv2=I02*vw0+I12*vw1+I22*vw2+(px*vl1-py*vl0);\n"
     << "  float Iv3=mass*vl0-(py*vw2-pz*vw1);\n"
     << "  float Iv4=mass*vl1-(pz*vw0-px*vw2);\n"
     << "  float Iv5=mass*vl2-(px*vw1-py*vw0);\n"
     << "  float mcf0=vw1*Iv2-vw2*Iv1+vl1*Iv5-vl2*Iv4;\n"
     << "  float mcf1=vw2*Iv0-vw0*Iv2+vl2*Iv3-vl0*Iv5;\n"
     << "  float mcf2=vw0*Iv1-vw1*Iv0+vl0*Iv4-vl1*Iv3;\n"
     << "  float mcf3=vw1*Iv5-vw2*Iv4;\n"
     << "  float mcf4=vw2*Iv3-vw0*Iv5;\n"
     << "  float mcf5=vw0*Iv4-vw1*Iv3;\n"
     << "  scratch[s_off+S_CFRC+b*6+0]=Ia0+mcf0;\n"
     << "  scratch[s_off+S_CFRC+b*6+1]=Ia1+mcf1;\n"
     << "  scratch[s_off+S_CFRC+b*6+2]=Ia2+mcf2;\n"
     << "  scratch[s_off+S_CFRC+b*6+3]=Ia3+mcf3;\n"
     << "  scratch[s_off+S_CFRC+b*6+4]=Ia4+mcf4;\n"
     << "  scratch[s_off+S_CFRC+b*6+5]=Ia5+mcf5;\n"
     << "}\n";

  ss << "for (int b=NB-1;b>=1;b--) {\n"
     << "  int p = body_par[b];\n"
     << "  for (int k=0;k<6;k++) scratch[s_off+S_CFRC+p*6+k] += scratch[s_off+S_CFRC+b*6+k];\n"
     << "}\n";

  ss << "for (int di=0;di<NV;di++) {\n"
     << "  int b=dof_bodyid[di]; float dot=0.0f;\n"
     << "  for (int k=0;k<6;k++) dot+=scratch[s_off+S_CDOF+di*6+k]*scratch[s_off+S_CFRC+b*6+k];\n"
     << "  scratch[s_off+S_BIAS+di]=dot;\n"
     << "}\n\n";

  // ═══ Phase E: passive, actuation, qfrc_smooth ═══
  ss << "// Phase E: passive, actuation, qfrc_smooth\n";

  ss << "for (int di=0;di<NV;di++) {\n"
     << "  float f = 0.0f;\n"
     << "  if (dof_stiff[di] != 0.0f) f -= dof_stiff[di] * (qpos[q_off+dof_qa[di]] - qpos_spr[dof_qa[di]]);\n"
     << "  if (dof_damp[di] != 0.0f) f -= dof_damp[di] * qvel[v_off+di];\n"
     << "  scratch[s_off+S_PASS+di] = f;\n"
     << "}\n";

  if(nu > 0) {
    ss << "for (int di=0;di<NV;di++) {\n"
       << "  float fa = 0.0f;\n"
       << "  for (int ai=0;ai<NU;ai++) {\n"
       << "    float mom = act_moment[ai*NV+di];\n"
       << "    if (mom != 0.0f) {\n"
       << "      float force = act_g0[ai] * ctrl[u_off+ai] + act_b0[ai];\n"
       << "      fa += mom * force;\n"
       << "    }\n"
       << "  }\n"
       << "  qfrc_actuator_out[qa_off+di] = fa;\n"
       << "}\n";
  } else {
    ss << "for (int di=0;di<NV;di++) qfrc_actuator_out[qa_off+di] = 0.0f;\n";
  }

  ss << "for (int di=0;di<NV;di++) {\n"
     << "  qfrc_smooth_out[qfs_off+di] = scratch[s_off+S_PASS+di] - scratch[s_off+S_BIAS+di] + qfrc_actuator_out[qa_off+di];\n"
     << "}\n";

  ss << "uint cd_off = bid * NV * 6;\n"
     << "for (int di=0;di<NV;di++) for (int k=0;k<6;k++)\n"
     << "  cdof_out[cd_off + di*6+k] = scratch[s_off+S_CDOF+di*6+k];\n";

  (void)dt;

  return mkx::fast::compute_kernel("mjmlx_forward", {"xipos", "ximat", "xanchor", "xaxis", "xmat", "qpos", "qvel", "ctrl", "act_moment"}, {"qM_out", "qfrc_smooth_out", "subtree_com_out", "cinert_out", "cvel_out", "qfrc_actuator_out", "scratch", "cdof_out"}, ss.str(), "");
}

} // namespace mjmlx::mkx_kernels
