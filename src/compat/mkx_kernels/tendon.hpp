#pragma once
// mj_tendon / mju_wrap(MuJoCo C)のGLSL移植。tendon構造はモデルで固定なので直線コードを生成し、動的なのはwrapの接点計算だけにする。

#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace mjmlx::mkx_kernels {

// mjtWrap
enum { TW_JOINT = 1, TW_PULLEY = 2, TW_SITE = 3, TW_SPHERE = 4, TW_CYLINDER = 5 };

struct TendonHost {
  int nten = 0;
  std::vector<int> adr, num;                       // (nten)
  std::vector<int> wtype, wobj;                    // (nwrap)
  std::vector<float> wprm;                         // (nwrap)
  std::vector<int> site_body;                      // (nsite)
  std::vector<float> site_pos;                     // (nsite*3)
  std::vector<int> geom_body;                      // (ngeom)
  std::vector<float> geom_radius;                  // (ngeom) geom_size[0]
  std::vector<int> jnt_qposadr, jnt_dofadr;        // (njnt)
  std::vector<float> stiffness, damping;           // (nten)
  std::vector<float> lengthspring;                 // (nten*2)
  std::vector<int> act_tendon;                     // (nu) tendon transmissionならtendon id、他は-1
  bool has_spatial = false;
  bool has_tendon_actuator = false;
};

// 1envあたりtendonが持つ値: [length, velocity, J(nv)] x nten
inline int tendon_row_floats(int nv) { return nv + 2; }
inline int tendon_scratch_floats(int nten, int nv) { return nten * tendon_row_floats(nv); }

namespace detail_tendon {

// 純粋な数学関数(バッファには触らない)。mju_wrapとその下請けのGLSL移植。
inline const char* wrap_header() {
  return R"GLSL(
const float TW_MINVAL = 1e-15;
bool tw_is_intersect(vec2 p1, vec2 p2, vec2 p3, vec2 p4) {
    float det = (p4.y-p3.y)*(p2.x-p1.x) - (p4.x-p3.x)*(p2.y-p1.y);
    if (abs(det) < TW_MINVAL) return false;
    float a = ((p4.x-p3.x)*(p1.y-p3.y) - (p4.y-p3.y)*(p1.x-p3.x)) / det;
    float b = ((p2.x-p1.x)*(p1.y-p3.y) - (p2.y-p1.y)*(p1.x-p3.x)) / det;
    return a >= 0.0 && a <= 1.0 && b >= 0.0 && b <= 1.0;
}
float tw_length_circle(vec2 p0, vec2 p1, int ind, float radius) {
    vec2 n0 = p0, n1 = p1;
    float l0 = length(n0), l1 = length(n1);
    n0 = l0 > TW_MINVAL ? n0 / l0 : vec2(1.0, 0.0);
    n1 = l1 > TW_MINVAL ? n1 / l1 : vec2(1.0, 0.0);
    float angle = acos(clamp(dot(n0, n1), -1.0, 1.0));
    float cr = p0.y*p1.x - p0.x*p1.y;
    if ((cr > 0.0 && ind != 0) || (cr < 0.0 && ind == 0)) angle = 6.28318530717959 - angle;
    return radius * angle;
}
// 戻り値: 円弧長、-1でwrapなし。pnt[0..1],pnt[2..3]が2つの接点。
float tw_wrap_circle(out vec4 pnt, vec4 e, bool has_side, vec2 side, float radius) {
    pnt = vec4(0.0);
    float sqlen0 = e.x*e.x + e.y*e.y;
    float sqlen1 = e.z*e.z + e.w*e.w;
    float sqrad = radius*radius;
    if (sqlen0 < sqrad || sqlen1 < sqrad || radius < TW_MINVAL) return -1.0;
    vec2 dif = vec2(e.z - e.x, e.w - e.y);
    float dd = dif.x*dif.x + dif.y*dif.y;
    if (dd < TW_MINVAL) return -1.0;
    float a = -(dif.x*e.x + dif.y*e.y) / dd;
    a = clamp(a, 0.0, 1.0);
    vec2 tmp = vec2(a*dif.x + e.x, a*dif.y + e.y);
    if (dot(tmp, tmp) > sqrad && (!has_side || dot(side, tmp) >= 0.0)) return -1.0;
    float sqrt0 = sqrt(sqlen0 - sqrad);
    float sqrt1 = sqrt(sqlen1 - sqrad);
    vec2 s00, s01, s10, s11;  // sol[i][0], sol[i][1]
    float good0, good1;
    for (int i = 0; i < 2; i++) {
        float sgn = (i == 0) ? 1.0 : -1.0;
        vec2 a0 = vec2((e.x*sqrad + sgn*radius*e.y*sqrt0)/sqlen0, (e.y*sqrad - sgn*radius*e.x*sqrt0)/sqlen0);
        vec2 a1 = vec2((e.z*sqrad - sgn*radius*e.w*sqrt1)/sqlen1, (e.w*sqrad + sgn*radius*e.z*sqrt1)/sqlen1);
        float g;
        if (has_side) {
            vec2 t = a0 + a1;
            float tl = length(t);
            t = tl > TW_MINVAL ? t / tl : vec2(1.0, 0.0);
            g = dot(t, side);
        } else {
            vec2 t = a0 - a1;
            g = -dot(t, t);
        }
        if (tw_is_intersect(e.xy, a0, e.zw, a1)) g = -10000.0;
        if (i == 0) { s00 = a0; s01 = a1; good0 = g; } else { s10 = a0; s11 = a1; good1 = g; }
    }
    int sel = (good0 > good1) ? 0 : 1;
    vec2 p0 = sel == 0 ? s00 : s10;
    vec2 p1 = sel == 0 ? s01 : s11;
    pnt = vec4(p0, p1);
    if (tw_is_intersect(e.xy, p0, e.zw, p1)) return -1.0;
    return tw_length_circle(p0, p1, sel, radius);
}
// 戻り値: 0でwrap、-1でwrapなし。
float tw_wrap_inside(out vec4 pnt, vec4 e, float radius) {
    pnt = vec4(0.0);
    float len0 = length(e.xy), len1 = length(e.zw);
    vec2 dif = vec2(e.z - e.x, e.w - e.y);
    float dd = dif.x*dif.x + dif.y*dif.y;
    if (len0 <= radius || len1 <= radius || radius < TW_MINVAL || len0 < TW_MINVAL || len1 < TW_MINVAL) return -1.0;
    if (dd > TW_MINVAL) {
        float a = -(dif.x*e.x + dif.y*e.y) / dd;
        if (a > 0.0 && a < 1.0) {
            vec2 t = e.xy + a*dif;
            if (length(t) <= radius) return -1.0;
        }
    }
    vec2 avg = 0.5*(e.xy + e.zw);
    float al = length(avg);
    avg = al > TW_MINVAL ? avg / al : vec2(1.0, 0.0);
    avg *= radius;
    pnt = vec4(avg, avg);
    float A = radius/len0, B = radius/len1;
    float cosG = (len0*len0 + len1*len1 - dd) / (2.0*len0*len1);
    if (cosG < -1.0 + 1e-6) return -1.0;
    else if (cosG > 1.0 - 1e-6) return 0.0;
    float G = acos(cosG);
    // float32ではzinit=1-1e-7が1に丸まりasin/sqrtが破綻するので、1に近すぎない値から始める
    float z = 1.0 - 1e-6;
    float f = asin(A*z) + asin(B*z) - 2.0*asin(z) + G;
    if (f > 0.0) return 0.0;
    int iter;
    for (iter = 0; iter < 20 && abs(f) > 1e-6; iter++) {
        float df = A/max(1e-12, sqrt(max(1.0 - z*z*A*A, 0.0))) + B/max(1e-12, sqrt(max(1.0 - z*z*B*B, 0.0))) - 2.0/max(1e-12, sqrt(max(1.0 - z*z, 0.0)));
        if (df > -1e-12) return 0.0;
        float z1 = z - f/df;
        if (z1 > z) return 0.0;
        z = z1;
        f = asin(clamp(A*z, -1.0, 1.0)) + asin(clamp(B*z, -1.0, 1.0)) - 2.0*asin(clamp(z, -1.0, 1.0)) + G;
        if (f > 1e-6) return 0.0;
    }
    if (iter >= 20) return 0.0;
    vec2 v; float ang;
    if (e.x*e.w - e.y*e.z > 0.0) { v = e.xy; ang = asin(z) - asin(A*z); }
    else { v = e.zw; ang = asin(z) - asin(B*z); }
    v = normalize(v);
    pnt.xy = radius * vec2(cos(ang)*v.x - sin(ang)*v.y, sin(ang)*v.x + cos(ang)*v.y);
    pnt.zw = pnt.xy;
    return 0.0;
}
// mju_wrap: 戻り値はwrap長(-1でwrapなし)。gRは列優先なのでローカル系への変換は v*gR。
float tw_wrap(out vec3 w0, out vec3 w1, vec3 x0, vec3 x1, vec3 gx, mat3 gR, float radius, bool sphere, bool has_side, vec3 side) {
    w0 = vec3(0.0); w1 = vec3(0.0);
    // gRは列優先で構築済み(gR*v=R*v)。ローカル系へは R^T * v = v*gR
    vec3 p0 = (x0 - gx) * gR;
    vec3 p1 = (x1 - gx) * gR;
    if (length(p0) < TW_MINVAL || length(p1) < TW_MINVAL) return -1.0;
    vec3 ax0, ax1;
    if (sphere) {
        ax0 = normalize(p0);
        vec3 nrm = cross(p0, p1);
        float nl = length(nrm);
        nrm = nl > TW_MINVAL ? nrm / nl : vec3(0.0);
        if (nl < TW_MINVAL) {
            int i = 0;
            if (abs(ax0.y) > abs(ax0.x) && abs(ax0.y) > abs(ax0.z)) i = 1;
            if (abs(ax0.z) > abs(ax0.x) && abs(ax0.z) > abs(ax0.y)) i = 2;
            vec3 a1 = vec3(1.0);
            if (i == 0) a1.x = 0.0; else if (i == 1) a1.y = 0.0; else a1.z = 0.0;
            nrm = cross(ax0, a1);
            float l2 = length(nrm);
            nrm = l2 > TW_MINVAL ? nrm / l2 : vec3(1.0, 0.0, 0.0);
        }
        ax1 = cross(nrm, ax0);
        float l3 = length(ax1);
        ax1 = l3 > TW_MINVAL ? ax1 / l3 : vec3(0.0, 1.0, 0.0);
    } else {
        ax0 = vec3(1.0, 0.0, 0.0);
        ax1 = vec3(0.0, 1.0, 0.0);
    }
    vec4 d = vec4(dot(p0, ax0), dot(p0, ax1), dot(p1, ax0), dot(p1, ax1));
    vec3 s = vec3(0.0);
    vec2 sd = vec2(0.0);
    if (has_side) {
        s = (side - gx) * gR;
        sd = vec2(dot(s, ax0), dot(s, ax1));
        float sl = length(sd);
        sd = sl > TW_MINVAL ? sd / sl : vec2(1.0, 0.0);
        sd *= radius;
    }
    vec4 pnt;
    float wlen;
    if (has_side && length(s) < radius) wlen = tw_wrap_inside(pnt, d, radius);
    else wlen = tw_wrap_circle(pnt, d, has_side, sd, radius);
    if (wlen < 0.0) return -1.0;
    vec3 r0 = ax0*pnt.x + ax1*pnt.y;
    vec3 r1 = ax0*pnt.z + ax1*pnt.w;
    if (!sphere) {
        float L0 = sqrt((p0.x-r0.x)*(p0.x-r0.x) + (p0.y-r0.y)*(p0.y-r0.y));
        float L1 = sqrt((p1.x-r1.x)*(p1.x-r1.x) + (p1.y-r1.y)*(p1.y-r1.y));
        float tot = max(L0 + wlen + L1, TW_MINVAL);
        r0.z = p0.z + (p1.z - p0.z)*L0 / tot;
        r1.z = p0.z + (p1.z - p0.z)*(L0 + wlen) / tot;
        float height = abs(r1.z - r0.z);
        wlen = sqrt(wlen*wlen + height*height);
    }
    w0 = gR * r0 + gx;
    w1 = gR * r1 + gx;
    return wlen;
}
)GLSL";
}

inline std::string flt(float x) {
  std::ostringstream ss;
  ss.precision(9);
  ss << x;
  std::string s = ss.str();
  if(s.find('.') == std::string::npos && s.find('e') == std::string::npos && s.find("inf") == std::string::npos && s.find("nan") == std::string::npos) s += ".0";
  return s + "f";
}

} // namespace detail_tendon

// forward kernelのヘッダに足すGLSL(wrap数学関数)。空間tendonが無ければ空。
inline std::string tendon_glsl_header(const TendonHost& th) { return th.has_spatial ? detail_tendon::wrap_header() : ""; }

// Phase T: tendon t は scratch[S_TEN + t*(NV+2)] から [length, velocity, J(NV)]。cdof計算後に呼ぶ。
inline void emit_tendon_phase(std::ostringstream& ss, const TendonHost& th, int nv, const std::vector<int>& body_parent, const std::vector<std::vector<int>>& body_dofs) {
  using detail_tendon::flt;
  const int row = tendon_row_floats(nv);
  auto S = [&](const std::string& idx) { return "scratch[SCR_IDX(" + idx + ")]"; };

  // body内の点pのヤコビアンJp(世界座標、subtree_com基準)を、bodyの祖先dof鎖に対して dif と内積して J行へ sgn*scale ずつ加算する。
  auto emit_jac_add = [&](int t, int body, const std::string& p, const std::string& dif, const std::string& scale) {
    if(body <= 0) return;  // worldは動かない
    ss << "    { vec3 rc = vec3(subtree_com_out[uint(body_root[" << body << "]*3)*NBATCH+bid], subtree_com_out[uint(body_root[" << body << "]*3+1)*NBATCH+bid], subtree_com_out[uint(body_root[" << body << "]*3+2)*NBATCH+bid]);\n";
    for(int a = body; a > 0; a = body_parent[static_cast<size_t>(a)])
      for(int di : body_dofs[static_cast<size_t>(a)]) {
        const std::string c = "S_CDOF+" + std::to_string(di) + "*6+";
        ss << "      { vec3 w = vec3(" << S(c + "0") << "," << S(c + "1") << "," << S(c + "2") << "); vec3 l = vec3(" << S(c + "3") << "," << S(c + "4") << "," << S(c + "5") << ");\n"
           << "        " << S("S_TEN+" + std::to_string(t * row + 2 + di)) << " += " << scale << " * dot(" << dif << ", l + cross(w, " << p << " - rc)); }\n";
      }
    ss << "    }\n";
  };

  ss << "// Phase T: tendon length / velocity / Jacobian\n";
  for(int t = 0; t < th.nten; t++) {
    const int adr = th.adr[static_cast<size_t>(t)], num = th.num[static_cast<size_t>(t)];
    const int base = t * row;
    ss << "{\n";
    ss << "  " << S("S_TEN+" + std::to_string(base)) << " = 0.0f;\n";
    ss << "  for (int d=0; d<NV; d++) " << S("S_TEN+" + std::to_string(base + 2) + "+d") << " = 0.0f;\n";
    if(th.wtype[static_cast<size_t>(adr)] == TW_JOINT) {
      for(int j = 0; j < num; j++) {
        int k = th.wobj[static_cast<size_t>(adr + j)];
        float prm = th.wprm[static_cast<size_t>(adr + j)];
        ss << "  " << S("S_TEN+" + std::to_string(base)) << " += " << flt(prm) << " * qpos[q_off+" << th.jnt_qposadr[static_cast<size_t>(k)] << "];\n"
           << "  " << S("S_TEN+" + std::to_string(base + 2 + th.jnt_dofadr[static_cast<size_t>(k)])) << " = " << flt(prm) << ";\n";
      }
    } else {
      // 使うsiteのworld座標(body xpos + xmat * site_pos)を先に出す。
      std::vector<int> used;
      for(int j = 0; j < num; j++)
        if(th.wtype[static_cast<size_t>(adr + j)] == TW_SITE) {
          int sid = th.wobj[static_cast<size_t>(adr + j)];
          bool dup = false;
          for(int u : used) dup |= (u == sid);
          if(!dup) used.push_back(sid);
        } else if(th.wtype[static_cast<size_t>(adr + j)] == TW_SPHERE || th.wtype[static_cast<size_t>(adr + j)] == TW_CYLINDER) {
          int side = static_cast<int>(std::lround(th.wprm[static_cast<size_t>(adr + j)]));
          bool dup = false;
          for(int u : used) dup |= (u == side);
          if(side >= 0 && !dup) used.push_back(side);
        }
      for(int sid : used) {
        int b = th.site_body[static_cast<size_t>(sid)];
        const float* sp = &th.site_pos[static_cast<size_t>(sid) * 3];
        ss << "  vec3 sp" << sid << " = vec3(xpos[uint(" << b * 3 << ")*NBATCH+bid], xpos[uint(" << b * 3 + 1 << ")*NBATCH+bid], xpos[uint(" << b * 3 + 2 << ")*NBATCH+bid])"
           << " + mat3(xmat[uint(" << b * 9 + 0 << ")*NBATCH+bid], xmat[uint(" << b * 9 + 3 << ")*NBATCH+bid], xmat[uint(" << b * 9 + 6 << ")*NBATCH+bid],"
           << " xmat[uint(" << b * 9 + 1 << ")*NBATCH+bid], xmat[uint(" << b * 9 + 4 << ")*NBATCH+bid], xmat[uint(" << b * 9 + 7 << ")*NBATCH+bid],"
           << " xmat[uint(" << b * 9 + 2 << ")*NBATCH+bid], xmat[uint(" << b * 9 + 5 << ")*NBATCH+bid], xmat[uint(" << b * 9 + 8 << ")*NBATCH+bid])"
           << " * vec3(" << flt(sp[0]) << "," << flt(sp[1]) << "," << flt(sp[2]) << ");\n";
      }
      ss << "  float ten_L = 0.0f;\n";
      float divisor = 1.0f;
      int j = 0;
      int seg = 0;
      while(j < num - 1) {
        const int type0 = th.wtype[static_cast<size_t>(adr + j)], type1 = th.wtype[static_cast<size_t>(adr + j + 1)];
        const int id0 = th.wobj[static_cast<size_t>(adr + j)];
        int id1 = th.wobj[static_cast<size_t>(adr + j + 1)];
        if(type0 == TW_PULLEY || type1 == TW_PULLEY) {
          if(type0 == TW_PULLEY) divisor = th.wprm[static_cast<size_t>(adr + j)];
          j++;
          continue;
        }
        const std::string inv = flt(1.0f / divisor);
        const int b0 = th.site_body[static_cast<size_t>(id0)];
        const std::string tag = std::to_string(t) + "_" + std::to_string(seg++);
        int wraptype = 0, wrapid = -1;
        if(type1 == TW_SPHERE || type1 == TW_CYLINDER) {
          wraptype = type1;
          wrapid = id1;
          id1 = th.wobj[static_cast<size_t>(adr + j + 2)];
          const int sideid = static_cast<int>(std::lround(th.wprm[static_cast<size_t>(adr + j + 1)]));
          const int gb = th.geom_body[static_cast<size_t>(wrapid)];
          const int b1 = th.site_body[static_cast<size_t>(id1)];
          ss << "  {\n"
             << "    vec3 gx = vec3(geom_xpos[uint(" << wrapid * 3 << ")*NBATCH+bid], geom_xpos[uint(" << wrapid * 3 + 1 << ")*NBATCH+bid], geom_xpos[uint(" << wrapid * 3 + 2 << ")*NBATCH+bid]);\n"
             << "    mat3 gR = mat3(geom_xmat[uint(" << wrapid * 9 + 0 << ")*NBATCH+bid], geom_xmat[uint(" << wrapid * 9 + 3 << ")*NBATCH+bid], geom_xmat[uint(" << wrapid * 9 + 6 << ")*NBATCH+bid],"
             << " geom_xmat[uint(" << wrapid * 9 + 1 << ")*NBATCH+bid], geom_xmat[uint(" << wrapid * 9 + 4 << ")*NBATCH+bid], geom_xmat[uint(" << wrapid * 9 + 7 << ")*NBATCH+bid],"
             << " geom_xmat[uint(" << wrapid * 9 + 2 << ")*NBATCH+bid], geom_xmat[uint(" << wrapid * 9 + 5 << ")*NBATCH+bid], geom_xmat[uint(" << wrapid * 9 + 8 << ")*NBATCH+bid]);\n"
             << "    vec3 w0, w1;\n"
             << "    float wlen = tw_wrap(w0, w1, sp" << id0 << ", sp" << id1 << ", gx, gR, " << flt(th.geom_radius[static_cast<size_t>(wrapid)]) << ", " << (wraptype == TW_SPHERE ? "true" : "false") << ", "
             << (sideid >= 0 ? "true" : "false") << ", " << (sideid >= 0 ? "sp" + std::to_string(sideid) : std::string("vec3(0.0)")) << ");\n"
             << "    if (wlen < 0.0f) {\n"
             << "      ten_L += distance(sp" << id0 << ", sp" << id1 << ") * " << inv << ";\n";
          if(b0 != b1) {
            ss << "      { vec3 dif = sp" << id1 << " - sp" << id0 << "; float dl = length(dif); dif = dl > 1e-15 ? dif / dl : vec3(1.0, 0.0, 0.0);\n";
            emit_jac_add(t, b1, "sp" + std::to_string(id1), "dif", inv);
            emit_jac_add(t, b0, "sp" + std::to_string(id0), "dif", "(-" + inv + ")");
            ss << "      }\n";
          }
          ss << "    } else {\n"
             << "      ten_L += (distance(sp" << id0 << ", w0) + wlen + distance(w1, sp" << id1 << ")) * " << inv << ";\n";
          // 区間0: site(id0)→w0, 区間1: w0→w1(同一body、寄与なし)、区間2: w1→site(id1)
          if(b0 != gb) {
            ss << "      { vec3 dif = w0 - sp" << id0 << "; float dl = length(dif); dif = dl > 1e-15 ? dif / dl : vec3(1.0, 0.0, 0.0);\n";
            emit_jac_add(t, gb, "w0", "dif", inv);
            emit_jac_add(t, b0, "sp" + std::to_string(id0), "dif", "(-" + inv + ")");
            ss << "      }\n";
          }
          if(gb != b1) {
            ss << "      { vec3 dif = sp" << id1 << " - w1; float dl = length(dif); dif = dl > 1e-15 ? dif / dl : vec3(1.0, 0.0, 0.0);\n";
            emit_jac_add(t, b1, "sp" + std::to_string(id1), "dif", inv);
            emit_jac_add(t, gb, "w1", "dif", "(-" + inv + ")");
            ss << "      }\n";
          }
          ss << "    }\n"
             << "  }\n";
        } else {
          const int b1 = th.site_body[static_cast<size_t>(id1)];
          ss << "  ten_L += distance(sp" << id0 << ", sp" << id1 << ") * " << inv << ";\n";
          if(b0 != b1) {
            ss << "  { vec3 dif = sp" << id1 << " - sp" << id0 << "; float dl = length(dif); dif = dl > 1e-15 ? dif / dl : vec3(1.0, 0.0, 0.0);\n";
            emit_jac_add(t, b1, "sp" + std::to_string(id1), "dif", inv);
            emit_jac_add(t, b0, "sp" + std::to_string(id0), "dif", "(-" + inv + ")");
            ss << "  }\n";
          }
        }
        j += (wraptype != 0 ? 2 : 1);
      }
      ss << "  " << S("S_TEN+" + std::to_string(base)) << " = ten_L;\n";
    }
    // velocity = J . qvel
    ss << "  { float tv = 0.0f; for (int d=0; d<NV; d++) tv += " << S("S_TEN+" + std::to_string(base + 2) + "+d") << " * qvel[v_off+d]; " << S("S_TEN+" + std::to_string(base + 1)) << " = tv; }\n";
    ss << "}\n";
  }
  ss << "\n";
}

// passive: tendonのバネ/ダンパ力を J^T f で S_PASSへ加算(S_PASSは先にdofごとのパッシブ力で初期化済みであること)。
inline void emit_tendon_passive(std::ostringstream& ss, const TendonHost& th, int nv) {
  using detail_tendon::flt;
  const int row = tendon_row_floats(nv);
  for(int t = 0; t < th.nten; t++) {
    const float k = th.stiffness[static_cast<size_t>(t)], c = th.damping[static_cast<size_t>(t)];
    if(k == 0.0f && c == 0.0f) continue;
    const std::string L = "scratch[SCR_IDX(S_TEN+" + std::to_string(t * row) + ")]";
    const std::string V = "scratch[SCR_IDX(S_TEN+" + std::to_string(t * row + 1) + ")]";
    ss << "{\n  float f = 0.0f;\n";
    if(k != 0.0f) {
      const float lo = th.lengthspring[static_cast<size_t>(t) * 2], hi = th.lengthspring[static_cast<size_t>(t) * 2 + 1];
      ss << "  float L_ = " << L << ";\n"
         << "  if (L_ > " << flt(hi) << ") f += " << flt(k) << " * (" << flt(hi) << " - L_);\n"
         << "  else if (L_ < " << flt(lo) << ") f += " << flt(k) << " * (" << flt(lo) << " - L_);\n";
    }
    if(c != 0.0f) ss << "  f -= " << flt(c) << " * " << V << ";\n";
    ss << "  for (int d=0; d<NV; d++) scratch[SCR_IDX(S_PASS+d)] += scratch[SCR_IDX(S_TEN+" << t * row << "+2+d)] * f;\n}\n";
  }
}

} // namespace mjmlx::mkx_kernels
