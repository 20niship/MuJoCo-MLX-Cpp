// batchedの出力はstep開始時点の状態なので、ここで求めるsensordataはmj_step後のd->sensordataと対応する(compiled_stepの外で遅延計算)。

#include "internal.h"
#include "mjmlx/mjmlx.h"

#include <cmath>
#include <cstdio>
#include <set>

namespace mjmlx {

namespace {

// body固定のローカルフレーム(site/geom/body慣性系)。worldのposeはbodyのxpos/xquatから求める。
struct ObjFrame {
    int body = 0;
    std::array<float, 3> pos{0, 0, 0};
    std::array<float, 4> quat{1, 0, 0, 0};
    std::array<float, 9> mat{1, 0, 0, 0, 1, 0, 0, 0, 1};
};

struct SensorDesc {
    int type = 0;
    int adr = 0, dim = 0;
    int objtype = 0, objid = 0;
    bool supported = true;
    ObjFrame frame;            // objtypeがsite/geom/body/xbodyのとき
    int qposadr = 0, dofadr = 0;  // joint系
    float gear = 1.0f;         // actuatorpos/vel
    // touch: siteの形状(site座標系の内外判定用)
    int site_type = 0;
    std::array<float, 3> site_size{0, 0, 0};
    // camprojection: カメラのフレームと内部パラメータ(画素単位)。対象siteはframe
    ObjFrame cam;
    float fx = 1, fy = 1, cx = 0, cy = 0;
};

// rangefinderが当てるgeom(typeが負なら不可視で当たらない)
struct RayGeom {
    int type = 0, bodyid = 0;
    std::array<float, 3> size{0, 0, 0};
    ObjFrame frame;
};

std::array<float, 9> quat_to_mat9(const std::array<float, 4>& q) {
    float w = q[0], x = q[1], y = q[2], z = q[3];
    return {1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)};
}

using V3 = std::array<double, 3>;
using M3 = std::array<double, 9>;
using Q4 = std::array<double, 4>;

M3 quat_to_mat(const Q4& q) {
    double w = q[0], x = q[1], y = q[2], z = q[3];
    return {1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)};
}
Q4 quat_mul(const Q4& a, const Q4& b) {
    return {a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]};
}
V3 mat_vec(const M3& R, const V3& v) { return {R[0] * v[0] + R[1] * v[1] + R[2] * v[2], R[3] * v[0] + R[4] * v[1] + R[5] * v[2], R[6] * v[0] + R[7] * v[1] + R[8] * v[2]}; }
V3 mat_t_vec(const M3& R, const V3& v) { return {R[0] * v[0] + R[3] * v[1] + R[6] * v[2], R[1] * v[0] + R[4] * v[1] + R[7] * v[2], R[2] * v[0] + R[5] * v[1] + R[8] * v[2]}; }
M3 mat_mul(const M3& A, const M3& B) {
    M3 C{};
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) for (int k = 0; k < 3; k++) C[i * 3 + j] += A[i * 3 + k] * B[k * 3 + j];
    return C;
}
V3 cross(const V3& a, const V3& b) { return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]}; }
V3 operator+(const V3& a, const V3& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
V3 operator-(const V3& a, const V3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }

// 世界座標のフレーム。lin_velは根のsubtree_com基準のcvelからフレーム原点の点速度に直したもの。
struct WorldFrame {
    V3 pos, lin_vel, ang_vel;
    M3 mat;
    Q4 quat;
};


// ── rangefinder / touch用の幾何(geomローカル座標、長軸はz) ──
// a t^2 + 2 b t + c = 0 の最小の非負根(無ければ-1)
double ray_quadratic(double a, double b, double c) {
    double det = b * b - a * c;
    if (det < 0 || a == 0) return -1;
    double sq = std::sqrt(det), t0 = (-b - sq) / a, t1 = (-b + sq) / a;
    return t0 >= 0 ? t0 : (t1 >= 0 ? t1 : -1);
}
double min_hit(double a, double b) { return a < 0 ? b : (b < 0 ? a : std::min(a, b)); }
// レイ p + t*d (t>=0)とgeomの交点のt。当たらなければ-1。
double ray_local(int type, const std::array<float, 3>& sz, const V3& p, const V3& d) {
    switch (type) {
        case mjGEOM_PLANE: {
            if (d[2] == 0) return -1;
            double t = -p[2] / d[2];
            if (t < 0) return -1;
            double x = p[0] + t * d[0], y = p[1] + t * d[1];
            if ((sz[0] > 0 && std::fabs(x) > sz[0]) || (sz[1] > 0 && std::fabs(y) > sz[1])) return -1;
            return t;
        }
        case mjGEOM_SPHERE:
            return ray_quadratic(d[0] * d[0] + d[1] * d[1] + d[2] * d[2], p[0] * d[0] + p[1] * d[1] + p[2] * d[2], p[0] * p[0] + p[1] * p[1] + p[2] * p[2] - (double)sz[0] * sz[0]);
        case mjGEOM_ELLIPSOID: {
            V3 ps{p[0] / sz[0], p[1] / sz[1], p[2] / sz[2]}, ds{d[0] / sz[0], d[1] / sz[1], d[2] / sz[2]};
            return ray_quadratic(ds[0] * ds[0] + ds[1] * ds[1] + ds[2] * ds[2], ps[0] * ds[0] + ps[1] * ds[1] + ps[2] * ds[2], ps[0] * ps[0] + ps[1] * ps[1] + ps[2] * ps[2] - 1.0);
        }
        case mjGEOM_BOX: {
            double tin = -1e30, tout = 1e30;
            for (int i = 0; i < 3; i++) {
                if (std::fabs(d[i]) < 1e-12) { if (std::fabs(p[i]) > sz[i]) return -1; continue; }
                double t0 = (-sz[i] - p[i]) / d[i], t1 = (sz[i] - p[i]) / d[i];
                if (t0 > t1) std::swap(t0, t1);
                tin = std::max(tin, t0); tout = std::min(tout, t1);
            }
            if (tin > tout || tout < 0) return -1;
            return tin >= 0 ? tin : tout;
        }
        case mjGEOM_CYLINDER: case mjGEOM_CAPSULE: {
            const double r = sz[0], h = sz[1];
            double best = -1;
            double a = d[0] * d[0] + d[1] * d[1];  // 側面(無限円筒)のうち|z|<=hの部分
            if (a > 1e-14) {
                double b = p[0] * d[0] + p[1] * d[1], c = p[0] * p[0] + p[1] * p[1] - r * r, det = b * b - a * c;
                if (det >= 0) {
                    double sq = std::sqrt(det);
                    for (double t : {(-b - sq) / a, (-b + sq) / a})
                        if (t >= 0 && std::fabs(p[2] + t * d[2]) <= h) { best = min_hit(best, t); break; }
                }
            }
            if (type == mjGEOM_CYLINDER) {
                if (std::fabs(d[2]) > 1e-12)
                    for (double zc : {-h, h}) {
                        double t = (zc - p[2]) / d[2];
                        if (t >= 0 && std::pow(p[0] + t * d[0], 2) + std::pow(p[1] + t * d[1], 2) <= r * r) best = min_hit(best, t);
                    }
            } else {
                for (double zc : {-h, h}) {  // 両端の半球
                    double q2 = p[2] - zc;
                    double aa = d[0] * d[0] + d[1] * d[1] + d[2] * d[2], bb = p[0] * d[0] + p[1] * d[1] + q2 * d[2], cc = p[0] * p[0] + p[1] * p[1] + q2 * q2 - r * r, det = bb * bb - aa * cc;
                    if (det < 0) continue;
                    double sq = std::sqrt(det);
                    for (double t : {(-bb - sq) / aa, (-bb + sq) / aa})
                        if (t >= 0 && (zc > 0 ? p[2] + t * d[2] >= h : p[2] + t * d[2] <= -h)) { best = min_hit(best, t); break; }
                }
            }
            return best;
        }
        default: return -1;
    }
}
// siteローカル座標の点が形状の内側か(mju_insideGeomと同じ)
bool inside_geom(int type, const std::array<float, 3>& s, const V3& p) {
    switch (type) {
        case mjGEOM_SPHERE: return p[0] * p[0] + p[1] * p[1] + p[2] * p[2] <= (double)s[0] * s[0];
        case mjGEOM_ELLIPSOID: return std::pow(p[0] / s[0], 2) + std::pow(p[1] / s[1], 2) + std::pow(p[2] / s[2], 2) <= 1.0;
        case mjGEOM_BOX: return std::fabs(p[0]) <= s[0] && std::fabs(p[1]) <= s[1] && std::fabs(p[2]) <= s[2];
        case mjGEOM_CYLINDER: return p[0] * p[0] + p[1] * p[1] <= (double)s[0] * s[0] && std::fabs(p[2]) <= s[1];
        case mjGEOM_CAPSULE: {
            double z = std::max(-(double)s[1], std::min((double)s[1], p[2]));
            return p[0] * p[0] + p[1] * p[1] + (p[2] - z) * (p[2] - z) <= (double)s[0] * s[0];
        }
        default: return false;
    }
}

}  // namespace

struct SensorCtx {
    int B = 0;
    int nbody = 0, nq = 0, nv = 0;
    int nsensordata = 0;
    float dt = 0.002f;
    std::array<float, 3> gravity{0, 0, -9.81f};
    std::array<float, 3> magnetic{0, -0.5f, 0};
    std::vector<SensorDesc> sensors;
    std::vector<int> body_rootid;
    // センサが参照するbodyだけをデバイス側で抜き出してから読み戻す(全bodyのxpos/cvel等の読み戻しは4096envで数十msかかる)
    std::vector<int> used_bodies, com_bodies;  // 昇順・重複なし
    std::vector<int> bmap, cmap;               // body id -> 上の配列内の位置(無ければ-1)
    std::optional<mx::array> used_idx, com_idx;
    bool needs_prev = false;  // accelerometerがあるときだけstep前の出力を保持する
    bool needs_contacts = false;  // touchがあるときstepが接触配列を出す
    std::vector<RayGeom> ray_geoms;  // rangefinderがあるときだけ
    std::vector<std::array<int, 2>> pair_bodies;  // 衝突ペア番号 -> (body1, body2)。touchの絞り込み用(初回に作る)

    // 直近stepへの入力のqpos/qvel(xpos等の出力はこの状態に対応する)
    std::optional<mx::array> in_qpos, in_qvel;
    // 1つ前のstepの出力(accelerometerの1step差分用)
    std::optional<mx::array> prev_xpos, prev_xquat, prev_subtree_com, prev_cvel;
    bool have_cur = false;
    std::vector<float> fresh;  // reset直後のenvは差分が無効
    std::vector<float> prev_valid;  // (B,) 1つ前のstep出力が有効なenvか
};

std::shared_ptr<SensorCtx> make_sensor_ctx(const mjModel* m, int num_envs) {
    if (!m || m->nsensor == 0) return nullptr;
    auto ctx = std::make_shared<SensorCtx>();
    ctx->B = num_envs;
    ctx->nbody = m->nbody;
    ctx->nq = m->nq;
    ctx->nv = m->nv;
    ctx->nsensordata = m->nsensordata;
    ctx->dt = (float)m->opt.timestep;
    for (int i = 0; i < 3; i++) { ctx->gravity[i] = (float)m->opt.gravity[i]; ctx->magnetic[i] = (float)m->opt.magnetic[i]; }
    ctx->body_rootid.assign(m->body_rootid, m->body_rootid + m->nbody);
    ctx->fresh.assign(num_envs, 1.0f);

    auto frame_of = [&](int objtype, int id, SensorDesc& d) {
        ObjFrame& f = d.frame;
        auto copy3 = [&](const mjtNum* s) { for (int i = 0; i < 3; i++) f.pos[i] = (float)s[i]; };
        auto copy4 = [&](const mjtNum* s) { for (int i = 0; i < 4; i++) f.quat[i] = (float)s[i]; };
        switch (objtype) {
            case mjOBJ_SITE:  f.body = m->site_bodyid[id]; copy3(m->site_pos + 3 * id); copy4(m->site_quat + 4 * id); break;
            case mjOBJ_GEOM:  f.body = m->geom_bodyid[id]; copy3(m->geom_pos + 3 * id); copy4(m->geom_quat + 4 * id); break;
            case mjOBJ_BODY:  f.body = id; copy3(m->body_ipos + 3 * id); copy4(m->body_iquat + 4 * id); break;  // MuJoCoのbodyは慣性系
            case mjOBJ_XBODY: f.body = id; break;
            case mjOBJ_CAMERA: f.body = m->cam_bodyid[id]; copy3(m->cam_pos + 3 * id); copy4(m->cam_quat + 4 * id); break;
            default: return false;
        }
        f.mat = quat_to_mat9(f.quat);
        return true;
    };

    for (int i = 0; i < m->nsensor; i++) {
        SensorDesc d;
        d.type = m->sensor_type[i];
        d.adr = m->sensor_adr[i];
        d.dim = m->sensor_dim[i];
        d.objtype = m->sensor_objtype[i];
        d.objid = m->sensor_objid[i];
        const char* nm = mj_id2name(m, mjOBJ_SENSOR, i);
        const char* why = "";

        auto joint_ok = [&]() {
            int jt = m->jnt_type[d.objid];
            d.qposadr = m->jnt_qposadr[d.objid];
            d.dofadr = m->jnt_dofadr[d.objid];
            return jt;
        };
        // frame系のreftype指定(別フレームからの相対値)は未対応
        auto world_ref = [&]() { return m->sensor_reftype[i] == mjOBJ_UNKNOWN || m->sensor_refid[i] < 0; };

        switch (d.type) {
            case mjSENS_JOINTPOS: { int jt = joint_ok(); d.supported = (jt == mjJNT_HINGE || jt == mjJNT_SLIDE); break; }
            case mjSENS_JOINTVEL: { int jt = joint_ok(); d.supported = (jt == mjJNT_HINGE || jt == mjJNT_SLIDE); break; }
            case mjSENS_BALLQUAT:
            case mjSENS_BALLANGVEL: { int jt = joint_ok(); d.supported = (jt == mjJNT_BALL); break; }
            case mjSENS_JOINTACTFRC: { int jt = joint_ok(); d.supported = (jt == mjJNT_HINGE || jt == mjJNT_SLIDE); break; }
            case mjSENS_ACTUATORPOS:
            case mjSENS_ACTUATORVEL: {
                int tt = m->actuator_trntype[d.objid];
                int jid = m->actuator_trnid[2 * d.objid];
                d.supported = (tt == mjTRN_JOINT || tt == mjTRN_JOINTINPARENT) &&
                              (m->jnt_type[jid] == mjJNT_HINGE || m->jnt_type[jid] == mjJNT_SLIDE);
                if (d.supported) {
                    d.qposadr = m->jnt_qposadr[jid];
                    d.dofadr = m->jnt_dofadr[jid];
                    d.gear = (float)m->actuator_gear[6 * d.objid];
                }
                break;
            }
            case mjSENS_GYRO:
            case mjSENS_VELOCIMETER:
            case mjSENS_ACCELEROMETER:
            case mjSENS_MAGNETOMETER:
                d.supported = frame_of(mjOBJ_SITE, d.objid, d);
                d.objtype = mjOBJ_SITE;
                if (d.type == mjSENS_ACCELEROMETER) ctx->needs_prev = true;
                break;
            case mjSENS_FRAMEPOS: case mjSENS_FRAMEQUAT: case mjSENS_FRAMEXAXIS: case mjSENS_FRAMEYAXIS:
            case mjSENS_FRAMEZAXIS: case mjSENS_FRAMELINVEL: case mjSENS_FRAMEANGVEL:
                d.supported = world_ref() && frame_of(d.objtype, d.objid, d);
                break;
            case mjSENS_SUBTREECOM:
                d.supported = true;
                break;
            case mjSENS_TOUCH: {
                d.supported = frame_of(mjOBJ_SITE, d.objid, d);
                d.objtype = mjOBJ_SITE;
                d.site_type = m->site_type[d.objid];
                for (int k = 0; k < 3; k++) d.site_size[k] = (float)m->site_size[3 * d.objid + k];
                d.supported = d.supported && (d.site_type == mjGEOM_SPHERE || d.site_type == mjGEOM_CAPSULE || d.site_type == mjGEOM_ELLIPSOID ||
                                              d.site_type == mjGEOM_CYLINDER || d.site_type == mjGEOM_BOX);
                if (d.supported) ctx->needs_contacts = true;
                break;
            }
            case mjSENS_RANGEFINDER: {
                d.supported = frame_of(mjOBJ_SITE, d.objid, d);
                d.objtype = mjOBJ_SITE;
                for (int g = 0; g < m->ngeom && d.supported; g++) {
                    int gt = m->geom_type[g];
                    if (m->geom_bodyid[g] == m->site_bodyid[d.objid] || m->geom_rgba[4 * g + 3] == 0) continue;
                    if (gt == mjGEOM_MESH || gt == mjGEOM_HFIELD || gt == mjGEOM_SDF) { d.supported = false; why = "model has mesh/hfield/SDF geoms (ray intersection not implemented for them)"; }
                }
                if (d.supported && ctx->ray_geoms.empty()) {
                    for (int g = 0; g < m->ngeom; g++) {
                        RayGeom rg;
                        rg.type = m->geom_rgba[4 * g + 3] == 0 ? -1 : m->geom_type[g];  // 不可視(alpha 0)のgeomは当たらない
                        rg.bodyid = m->geom_bodyid[g];
                        for (int k = 0; k < 3; k++) rg.size[k] = (float)m->geom_size[3 * g + k];
                        SensorDesc tmp;
                        frame_of(mjOBJ_GEOM, g, tmp);
                        rg.frame = tmp.frame;
                        ctx->ray_geoms.push_back(rg);
                    }
                }
                break;
            }
            case mjSENS_CAMPROJECTION: {
                int cam = m->sensor_refid[i];
                d.supported = frame_of(mjOBJ_SITE, d.objid, d) && m->sensor_reftype[i] == mjOBJ_CAMERA && cam >= 0;
                d.objtype = mjOBJ_SITE;
                if (d.supported && m->cam_mode[cam] != mjCAMLIGHT_FIXED) { d.supported = false; why = "only fixed cameras are supported (not track/trackcom/targetbody)"; }
                if (d.supported) {
                    SensorDesc tmp;
                    frame_of(mjOBJ_CAMERA, cam, tmp);
                    d.cam = tmp.frame;
                    const double rx = m->cam_resolution[2 * cam], ry = m->cam_resolution[2 * cam + 1];
                    const double sx = m->cam_sensorsize[2 * cam], sy = m->cam_sensorsize[2 * cam + 1];
                    // MuJoCoのcamprojectionは主点(principal)を使わず、画像中心を原点にする(sensorsize指定時の焦点距離だけ使う)
                    if (sx > 0 && sy > 0) {  // sensorsize + focal指定
                        d.fx = (float)(m->cam_intrinsic[4 * cam] / sx * rx);
                        d.fy = (float)(m->cam_intrinsic[4 * cam + 1] / sy * ry);
                        d.cx = (float)(rx / 2);
                        d.cy = (float)(ry / 2);
                    } else {  // fovy指定(正方画素)
                        d.fy = d.fx = (float)(0.5 * ry / std::tan(0.5 * m->cam_fovy[cam] * M_PI / 180.0));
                        d.cx = (float)(rx / 2);
                        d.cy = (float)(ry / 2);
                    }
                }
                break;
            }
            case mjSENS_FORCE: case mjSENS_TORQUE:
                d.supported = false;
                why = "needs the full contact wrench (friction/torsion, not exported by the GPU solver) and cfrc_int (RNE backward pass); not implemented";
                break;

            default:
                d.supported = false;
        }
        if (!d.supported)
            fprintf(stderr, "[mjmlx WARNING] sensor '%s' (type %d) is not supported on the GPU path; its sensordata stays 0.%s%s\n",
                    nm ? nm : "", d.type, *why ? " Reason: " : "", why);
        ctx->sensors.push_back(d);
    }

    std::set<int> used, coms;
    for (const auto& d : ctx->sensors) {
        if (!d.supported) continue;
        bool framed = d.type == mjSENS_GYRO || d.type == mjSENS_VELOCIMETER || d.type == mjSENS_ACCELEROMETER || d.type == mjSENS_MAGNETOMETER ||
                      (d.type >= mjSENS_FRAMEPOS && d.type <= mjSENS_FRAMEANGVEL);
        framed = framed || d.type == mjSENS_TOUCH || d.type == mjSENS_RANGEFINDER || d.type == mjSENS_CAMPROJECTION;
        if (framed) { used.insert(d.frame.body); coms.insert(ctx->body_rootid[d.frame.body]); }
        if (d.type == mjSENS_CAMPROJECTION) { used.insert(d.cam.body); coms.insert(ctx->body_rootid[d.cam.body]); }
        if (d.type == mjSENS_RANGEFINDER)
            for (const auto& rg : ctx->ray_geoms) { used.insert(rg.bodyid); coms.insert(ctx->body_rootid[rg.bodyid]); }
        if (d.type == mjSENS_SUBTREECOM) coms.insert(d.objid);
    }
    if (used.empty()) used.insert(0);
    if (coms.empty()) coms.insert(0);
    ctx->used_bodies.assign(used.begin(), used.end());
    ctx->com_bodies.assign(coms.begin(), coms.end());
    ctx->bmap.assign(m->nbody, -1);
    ctx->cmap.assign(m->nbody, -1);
    for (size_t i = 0; i < ctx->used_bodies.size(); i++) ctx->bmap[ctx->used_bodies[i]] = (int)i;
    for (size_t i = 0; i < ctx->com_bodies.size(); i++) ctx->cmap[ctx->com_bodies[i]] = (int)i;
    ctx->used_idx = mx::array(ctx->used_bodies.data(), mx::Shape{(int)ctx->used_bodies.size()}, mx::int32);
    ctx->com_idx = mx::array(ctx->com_bodies.data(), mx::Shape{(int)ctx->com_bodies.size()}, mx::int32);
    return ctx;
}

int sensor_nsensordata(const SensorCtx& ctx) { return ctx.nsensordata; }
bool sensor_needs_contacts(const SensorCtx& ctx) { return ctx.needs_contacts; }

void sensor_before_step(SensorCtx& ctx, const BatchedSim& sim) {
    ctx.in_qpos = sim.qpos;
    ctx.in_qvel = sim.qvel;
    if (!ctx.needs_prev) return;
    if (ctx.have_cur) {
        ctx.prev_xpos = sim.xpos;
        ctx.prev_xquat = sim.xquat;
        ctx.prev_subtree_com = sim.subtree_com;
        ctx.prev_cvel = sim.cvel;
        ctx.prev_valid.resize(ctx.B);
        for (int i = 0; i < ctx.B; i++) ctx.prev_valid[i] = ctx.fresh[i] ? 0.0f : 1.0f;
    }
    ctx.have_cur = true;
    std::fill(ctx.fresh.begin(), ctx.fresh.end(), 0.0f);
}

void sensor_mark_reset(SensorCtx& ctx, const int* reset_mask, int num_envs) {
    for (int i = 0; i < num_envs && i < (int)ctx.fresh.size(); i++)
        if (reset_mask[i]) ctx.fresh[i] = 1.0f;
}

namespace {

// 1envぶんのbatched出力(host上)
struct EnvKin {
    const float* xpos; const float* xquat; const float* com; const float* cvel;
    const SensorCtx& ctx;

    WorldFrame frame(const ObjFrame& f) const {
        const int bi = ctx.bmap[f.body];
        const float* bp = xpos + 3 * bi;
        const float* bq = xquat + 4 * bi;
        Q4 q{bq[0], bq[1], bq[2], bq[3]};
        M3 Rb = quat_to_mat(q);
        V3 lp{f.pos[0], f.pos[1], f.pos[2]};
        M3 Rl;
        for (int i = 0; i < 9; i++) Rl[i] = f.mat[i];
        Q4 lq{f.quat[0], f.quat[1], f.quat[2], f.quat[3]};
        WorldFrame w;
        V3 off = mat_vec(Rb, lp);
        w.pos = {bp[0] + off[0], bp[1] + off[1], bp[2] + off[2]};
        w.mat = mat_mul(Rb, Rl);
        w.quat = quat_mul(q, lq);
        const float* cv = cvel + 6 * bi;
        const float* c = com + 3 * ctx.cmap[ctx.body_rootid[f.body]];
        w.ang_vel = {cv[0], cv[1], cv[2]};
        V3 v{cv[3], cv[4], cv[5]};
        w.lin_vel = v + cross(w.ang_vel, w.pos - V3{c[0], c[1], c[2]});
        return w;
    }
};

}  // namespace

void sensor_compute(SensorCtx& ctx, const BatchedSim& sim, std::vector<float>& out) {
    const int B = ctx.B, nsd = ctx.nsensordata;
    const mx::array& in_qpos = ctx.in_qpos ? *ctx.in_qpos : sim.qpos;
    const mx::array& in_qvel = ctx.in_qvel ? *ctx.in_qvel : sim.qvel;
    bool use_prev = ctx.needs_prev && ctx.have_cur && ctx.prev_xpos.has_value();
    auto pick = [&](const mx::array& a, const mx::array& idx) { return mx::take(a, idx, 1); };
    const int nu_ = (int)ctx.used_bodies.size(), nc_ = (int)ctx.com_bodies.size();
    mx::array g_xpos = pick(sim.xpos, *ctx.used_idx), g_xquat = pick(sim.xquat, *ctx.used_idx),
              g_cvel = pick(sim.cvel, *ctx.used_idx), g_com = pick(sim.subtree_com, *ctx.com_idx);
    std::optional<mx::array> p_xpos, p_xquat, p_cvel, p_com;
    if (use_prev) {
        p_xpos = pick(*ctx.prev_xpos, *ctx.used_idx); p_xquat = pick(*ctx.prev_xquat, *ctx.used_idx);
        p_cvel = pick(*ctx.prev_cvel, *ctx.used_idx); p_com = pick(*ctx.prev_subtree_com, *ctx.com_idx);
        mx::eval({in_qpos, in_qvel, sim.qfrc_actuator, g_xpos, g_xquat, g_cvel, g_com, *p_xpos, *p_xquat, *p_cvel, *p_com});
    } else {
        mx::eval({in_qpos, in_qvel, sim.qfrc_actuator, g_xpos, g_xquat, g_cvel, g_com});
    }

    const float* qpos = in_qpos.data<float>();
    const float* qvel = in_qvel.data<float>();
    const float* qfa = sim.qfrc_actuator.size() > 0 ? sim.qfrc_actuator.data<float>() : nullptr;
    const float* pvalid = use_prev ? ctx.prev_valid.data() : nullptr;

    // data<float>()は呼ぶたびにデバイスから読み戻すので、ループの外で1回だけ取る
    const float* xpos = g_xpos.data<float>();
    const float* xquat = g_xquat.data<float>();
    const float* com = g_com.data<float>();
    const float* cvel = g_cvel.data<float>();
    const float* pxpos = use_prev ? p_xpos->data<float>() : nullptr;
    const float* pxquat = use_prev ? p_xquat->data<float>() : nullptr;
    const float* pcom = use_prev ? p_com->data<float>() : nullptr;
    const float* pcvel = use_prev ? p_cvel->data<float>() : nullptr;

    // touch: 直近substepの接触(位置・法線力・ペア)。法線力は全ファセットの和でmj_contactForceのforce[0]に対応する。
    const float *con_data = nullptr, *con_cnt = nullptr, *con_fn = nullptr;
    int max_con = 0;
    if (ctx.needs_contacts && sim.con_fn.size() > 0) {
        mx::eval({sim.con_data, sim.con_cnt, sim.con_fn});
        con_data = sim.con_data.data<float>(); con_cnt = sim.con_cnt.data<float>(); con_fn = sim.con_fn.data<float>();
        max_con = (int)(sim.con_fn.size() / B);
        if (ctx.pair_bodies.empty())
            for (const auto& cp : sim.model->cache.collision_pairs) ctx.pair_bodies.push_back({cp.body1, cp.body2});
    }

    out.assign((size_t)B * nsd, 0.0f);
    for (int b = 0; b < B; b++) {
        EnvKin now{xpos + (size_t)b * nu_ * 3, xquat + (size_t)b * nu_ * 4, com + (size_t)b * nc_ * 3, cvel + (size_t)b * nu_ * 6, ctx};
        std::optional<EnvKin> prev;
        if (use_prev && pvalid[b] > 0.5f)
            prev.emplace(EnvKin{pxpos + (size_t)b * nu_ * 3, pxquat + (size_t)b * nu_ * 4, pcom + (size_t)b * nc_ * 3, pcvel + (size_t)b * nu_ * 6, ctx});
        const float* qp = qpos + (size_t)b * ctx.nq;
        const float* qv = qvel + (size_t)b * ctx.nv;
        float* o = out.data() + (size_t)b * nsd;

        for (const auto& d : ctx.sensors) {
            if (!d.supported) continue;
            float* y = o + d.adr;
            auto put3 = [&](const V3& v) { for (int i = 0; i < 3; i++) y[i] = (float)v[i]; };
            switch (d.type) {
                case mjSENS_JOINTPOS: y[0] = qp[d.qposadr]; break;
                case mjSENS_JOINTVEL: y[0] = qv[d.dofadr]; break;
                case mjSENS_JOINTACTFRC: y[0] = qfa ? qfa[(size_t)b * ctx.nv + d.dofadr] : 0.0f; break;
                case mjSENS_ACTUATORPOS: y[0] = d.gear * qp[d.qposadr]; break;
                case mjSENS_ACTUATORVEL: y[0] = d.gear * qv[d.dofadr]; break;
                case mjSENS_BALLQUAT: {
                    const float* q = qp + d.qposadr;
                    double n = std::sqrt((double)q[0] * q[0] + (double)q[1] * q[1] + (double)q[2] * q[2] + (double)q[3] * q[3]);
                    for (int i = 0; i < 4; i++) y[i] = (float)(q[i] / n);
                    break;
                }
                case mjSENS_BALLANGVEL: for (int i = 0; i < 3; i++) y[i] = qv[d.dofadr + i]; break;
                case mjSENS_SUBTREECOM: for (int i = 0; i < 3; i++) y[i] = now.com[3 * ctx.cmap[d.objid] + i]; break;
                case mjSENS_FRAMEPOS: put3(now.frame(d.frame).pos); break;
                case mjSENS_FRAMEQUAT: {
                    Q4 q = now.frame(d.frame).quat;
                    double sgn = q[0] < 0 ? -1.0 : 1.0;
                    for (int i = 0; i < 4; i++) y[i] = (float)(sgn * q[i]);
                    break;
                }
                case mjSENS_FRAMEXAXIS: case mjSENS_FRAMEYAXIS: case mjSENS_FRAMEZAXIS: {
                    int k = d.type - mjSENS_FRAMEXAXIS;
                    M3 R = now.frame(d.frame).mat;
                    put3({R[k], R[3 + k], R[6 + k]});
                    break;
                }
                case mjSENS_FRAMELINVEL: put3(now.frame(d.frame).lin_vel); break;
                case mjSENS_FRAMEANGVEL: put3(now.frame(d.frame).ang_vel); break;
                case mjSENS_GYRO: { auto w = now.frame(d.frame); put3(mat_t_vec(w.mat, w.ang_vel)); break; }
                case mjSENS_VELOCIMETER: { auto w = now.frame(d.frame); put3(mat_t_vec(w.mat, w.lin_vel)); break; }
                case mjSENS_MAGNETOMETER: put3(mat_t_vec(now.frame(d.frame).mat, V3{ctx.magnetic[0], ctx.magnetic[1], ctx.magnetic[2]})); break;
                case mjSENS_ACCELEROMETER: {
                    // Isaac Lab IMUと同様、body固定点の世界速度の1step差分。MuJoCoと同じく重力込み(固有加速度)。
                    auto w = now.frame(d.frame);
                    V3 a{0, 0, 0};
                    if (prev) {
                        auto wp = prev->frame(d.frame);
                        for (int i = 0; i < 3; i++) a[i] = (w.lin_vel[i] - wp.lin_vel[i]) / ctx.dt;
                    }
                    put3(mat_t_vec(w.mat, a - V3{ctx.gravity[0], ctx.gravity[1], ctx.gravity[2]}));
                    break;
                }
                case mjSENS_TOUCH: {
                    if (!con_fn) break;
                    auto w = now.frame(d.frame);
                    const int sbody = d.frame.body, nc = std::min(max_con, (int)con_cnt[b]);
                    double sum = 0;
                    for (int ci = 0; ci < nc; ci++) {
                        const float fn = con_fn[(size_t)b * max_con + ci];
                        if (fn <= 0) continue;
                        const float* cd = con_data + ((size_t)b * max_con + ci) * 8;
                        const auto& pb = ctx.pair_bodies[(size_t)cd[7]];
                        if (pb[0] != sbody && pb[1] != sbody) continue;
                        V3 pl = mat_t_vec(w.mat, V3{cd[0], cd[1], cd[2]} - w.pos);
                        if (inside_geom(d.site_type, d.site_size, pl)) sum += fn;
                    }
                    y[0] = (float)sum;
                    break;
                }
                case mjSENS_RANGEFINDER: {
                    auto w = now.frame(d.frame);
                    V3 dir{w.mat[2], w.mat[5], w.mat[8]};  // siteのz軸
                    double best = -1;
                    for (const auto& rg : ctx.ray_geoms) {
                        if (rg.type < 0 || rg.bodyid == d.frame.body) continue;
                        auto g = now.frame(rg.frame);
                        double t = ray_local(rg.type, rg.size, mat_t_vec(g.mat, w.pos - g.pos), mat_t_vec(g.mat, dir));
                        if (t >= 0 && (best < 0 || t < best)) best = t;
                    }
                    y[0] = (float)best;
                    break;
                }
                case mjSENS_CAMPROJECTION: {
                    auto c = now.frame(d.cam);
                    auto t = now.frame(d.frame);
                    V3 pc = mat_t_vec(c.mat, t.pos - c.pos);  // カメラは-zを向き、xが右、yが上
                    y[0] = (float)(d.cx + d.fx * pc[0] / -pc[2]);
                    y[1] = (float)(d.cy - d.fy * pc[1] / -pc[2]);
                    break;
                }
                default: break;
            }
        }
    }
}

}  // namespace mjmlx

#define MJMLX_SENSOR_FIELD(expr) \
    if (!model || !model->mj_model || sensor_id < 0 || sensor_id >= model->mj_model->nsensor) return -1; \
    return expr

MJMLX_API int mjmlx_model_nsensor(const MjmlxModel* model) { return (model && model->mj_model) ? model->mj_model->nsensor : 0; }
MJMLX_API int mjmlx_model_nsensordata(const MjmlxModel* model) { return (model && model->mj_model) ? model->mj_model->nsensordata : 0; }
MJMLX_API const char* mjmlx_model_sensor_name(const MjmlxModel* model, int sensor_id) {
    if (!model || !model->mj_model || sensor_id < 0 || sensor_id >= model->mj_model->nsensor) return nullptr;
    return mj_id2name(model->mj_model, mjOBJ_SENSOR, sensor_id);
}
MJMLX_API int mjmlx_model_sensor_type(const MjmlxModel* model, int sensor_id) { MJMLX_SENSOR_FIELD(model->mj_model->sensor_type[sensor_id]); }
MJMLX_API int mjmlx_model_sensor_adr(const MjmlxModel* model, int sensor_id) { MJMLX_SENSOR_FIELD(model->mj_model->sensor_adr[sensor_id]); }
MJMLX_API int mjmlx_model_sensor_dim(const MjmlxModel* model, int sensor_id) { MJMLX_SENSOR_FIELD(model->mj_model->sensor_dim[sensor_id]); }

#undef MJMLX_SENSOR_FIELD
