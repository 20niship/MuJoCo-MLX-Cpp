#pragma once
// Vendored from 20niship/mlx-cross-platform tests/mujoco/collision.hpp at commit 3202339e12231f3a68f9ee67e0b6085a33c52286.

#include <sstream>
#include <string>

#include <mkx/ops/fast_kernel.hpp>

// MuJoCo-MLX-Cpp make_collision_source逐語GLSL移植。元コードもplane-mesh/mesh-meshの2組み合わせしか実装しておらず他のgeom型ペアはhas_contact=falseのまま、本体は形状非依存の固定テンプレートなので次元(NG/NUM_PAIRS)だけホスト側展開にした。

namespace mjmlx::mkx_kernels {

inline mkx::fast::Kernel<> make_collision_kernel(int ng, int npairs, int max_contacts_per_env = 128) {
  const int contact_stride = 8;

  std::string header = R"GLSL(
vec3 msl_cross(vec3 a, vec3 b) { return cross(a, b); }
float msl_dot(vec3 a, vec3 b) { return dot(a, b); }
float msl_len(vec3 a) { return length(a); }
vec3 msl_norm(vec3 a) {
    float l = msl_len(a);
    return l > 1e-12 ? a / l : vec3(0.0, 0.0, 1.0);
}

// Generic GJK/EPA support point so plane-X and X-Y contact detection work for any geom-type pair, not just mesh. MuJoCo local-frame convention: sphere size=(radius,_,_), capsule/cylinder size=(radius,halflength,_) axis along local +z, box size=(halfx,halfy,halfz), ellipsoid size=semi-axes.
vec3 mjmlx_support(int gtype, vec3 pos, float R[9], vec3 size,
                    int mesh_adr, int mesh_nverts, vec3 dir_world) {
    vec3 ld = vec3(R[0]*dir_world.x+R[3]*dir_world.y+R[6]*dir_world.z,
                   R[1]*dir_world.x+R[4]*dir_world.y+R[7]*dir_world.z,
                   R[2]*dir_world.x+R[5]*dir_world.y+R[8]*dir_world.z);
    vec3 lp;
    if (gtype == 2) { // sphere
        lp = msl_norm(ld) * size.x;
    } else if (gtype == 4) { // ellipsoid: support(d) = (a^2 dx, b^2 dy, c^2 dz) / |(a dx, b dy, c dz)|
        vec3 scaled = vec3(size.x*ld.x, size.y*ld.y, size.z*ld.z);
        float denom = msl_len(scaled);
        lp = (denom > 1e-12) ? vec3(size.x*scaled.x, size.y*scaled.y, size.z*scaled.z) / denom : vec3(0.0, 0.0, size.z);
    } else if (gtype == 3) { // capsule
        vec3 n = msl_norm(ld);
        float capz = (ld.z >= 0.0) ? size.y : -size.y;
        lp = vec3(n.x*size.x, n.y*size.x, capz + n.z*size.x);
    } else if (gtype == 5) { // cylinder: disk (xy) x interval (z)
        vec2 xy = vec2(ld.x, ld.y);
        float xyl = length(xy);
        vec2 xyn = (xyl > 1e-12) ? xy / xyl : vec2(1.0, 0.0);
        lp = vec3(xyn.x*size.x, xyn.y*size.x, (ld.z >= 0.0) ? size.y : -size.y);
    } else if (gtype == 6) { // box
        lp = vec3(ld.x >= 0.0 ? size.x : -size.x,
                  ld.y >= 0.0 ? size.y : -size.y,
                  ld.z >= 0.0 ? size.z : -size.z);
    } else if (gtype == 7) { // mesh: linear scan over vertices
        float bd = -1e30; int bi = 0;
        for (int vi = 0; vi < mesh_nverts; vi++) {
            vec3 v = vec3(mesh_verts[(mesh_adr+vi)*3], mesh_verts[(mesh_adr+vi)*3+1], mesh_verts[(mesh_adr+vi)*3+2]);
            float dd = msl_dot(v, ld);
            if (dd > bd) { bd = dd; bi = vi; }
        }
        lp = vec3(mesh_verts[(mesh_adr+bi)*3], mesh_verts[(mesh_adr+bi)*3+1], mesh_verts[(mesh_adr+bi)*3+2]);
    } else {
        lp = vec3(0.0, 0.0, 0.0);
    }
    return vec3(R[0]*lp.x+R[1]*lp.y+R[2]*lp.z+pos.x,
                R[3]*lp.x+R[4]*lp.y+R[5]*lp.z+pos.y,
                R[6]*lp.x+R[7]*lp.y+R[8]*lp.z+pos.z);
}
)GLSL";

  std::ostringstream ss;
  ss << "uint bid = gl_GlobalInvocationID.x;\n"
     << "const int NG = " << ng << ";\n"
     << "const int NUM_PAIRS = " << npairs << ";\n"
     << "const int MAX_CON = " << max_contacts_per_env << ";\n"
     << "const int STRIDE = " << contact_stride << ";\n\n"
     << "uint gx_off = bid * NG * 3;\n"
     << "uint gm_off = bid * NG * 9;\n"
     << "uint con_off = bid * MAX_CON * STRIDE;\n\n"
     << "for (int i = 0; i < MAX_CON * STRIDE; i++) contact_data[con_off + i] = 0.0f;\n"
     << "int ncon = 0;\n\n"
     << "const int PAIR_STRIDE = 6;\n\n";

  ss << R"GLSLBODY(
for (int p = 0; p < NUM_PAIRS; p++) {
    int pd = p * PAIR_STRIDE;
    int g1 = int(pair_data[pd+0]), g2 = int(pair_data[pd+1]);
    int t1 = int(pair_data[pd+2]), t2 = int(pair_data[pd+3]);
    float pair_margin = pair_data[pd+4];
    float rbound_sum = pair_data[pd+5];

    if (t1 != 0 && t2 != 0) {
        vec3 c1 = vec3(geom_xpos[gx_off+g1*3], geom_xpos[gx_off+g1*3+1], geom_xpos[gx_off+g1*3+2]);
        vec3 c2 = vec3(geom_xpos[gx_off+g2*3], geom_xpos[gx_off+g2*3+1], geom_xpos[gx_off+g2*3+2]);
        if (msl_len(c1 - c2) >= rbound_sum) continue;
    }

    float c_pos[3], c_norm[3], c_dist;
    bool has_contact = false;

    if (t1 == 0 || t2 == 0) {
        int plane_g = (t1 == 0) ? g1 : g2;
        int xg = (t1 == 0) ? g2 : g1;
        int xt = (t1 == 0) ? t2 : t1;
        vec3 ppos = vec3(geom_xpos[gx_off+plane_g*3], geom_xpos[gx_off+plane_g*3+1], geom_xpos[gx_off+plane_g*3+2]);
        float pR[9]; for (int k=0;k<9;k++) pR[k] = geom_xmat[gm_off+plane_g*9+k];
        vec3 normal = vec3(pR[2], pR[5], pR[8]);
        vec3 xpos_ = vec3(geom_xpos[gx_off+xg*3], geom_xpos[gx_off+xg*3+1], geom_xpos[gx_off+xg*3+2]);
        float xR[9]; for (int k=0;k<9;k++) xR[k] = geom_xmat[gm_off+xg*9+k];
        vec3 xsize = vec3(geom_size_buf[xg*3], geom_size_buf[xg*3+1], geom_size_buf[xg*3+2]);
        int mesh_id = (xt == 7) ? int(geom_dataid_buf[xg]) : 0;
        int adr = (xt == 7) ? int(mesh_vertadr_buf[mesh_id]) : 0;
        int nverts = (xt == 7) ? int(mesh_vertnum_buf[mesh_id]) : 0;

        vec3 best_w = mjmlx_support(xt, xpos_, xR, xsize, adr, nverts, -normal);
        float best_d = msl_dot(best_w - ppos, normal);
        if (best_d < pair_margin) {
            vec3 cp = best_w - normal * best_d;
            c_pos[0]=cp.x; c_pos[1]=cp.y; c_pos[2]=cp.z;
            c_norm[0]=normal.x; c_norm[1]=normal.y; c_norm[2]=normal.z;
            c_dist = best_d;
            has_contact = true;
        }
    } else {
        vec3 pos1 = vec3(geom_xpos[gx_off+g1*3], geom_xpos[gx_off+g1*3+1], geom_xpos[gx_off+g1*3+2]);
        vec3 pos2 = vec3(geom_xpos[gx_off+g2*3], geom_xpos[gx_off+g2*3+1], geom_xpos[gx_off+g2*3+2]);
        float R1[9]; for (int k=0;k<9;k++) R1[k] = geom_xmat[gm_off+g1*9+k];
        float R2[9]; for (int k=0;k<9;k++) R2[k] = geom_xmat[gm_off+g2*9+k];
        vec3 size1 = vec3(geom_size_buf[g1*3], geom_size_buf[g1*3+1], geom_size_buf[g1*3+2]);
        vec3 size2 = vec3(geom_size_buf[g2*3], geom_size_buf[g2*3+1], geom_size_buf[g2*3+2]);
        int mid1 = (t1 == 7) ? int(geom_dataid_buf[g1]) : 0;
        int mid2 = (t2 == 7) ? int(geom_dataid_buf[g2]) : 0;
        int adr1 = (t1 == 7) ? int(mesh_vertadr_buf[mid1]) : 0;
        int adr2 = (t2 == 7) ? int(mesh_vertadr_buf[mid2]) : 0;
        int nv1 = (t1 == 7) ? int(mesh_vertnum_buf[mid1]) : 0;
        int nv2 = (t2 == 7) ? int(mesh_vertnum_buf[mid2]) : 0;

        #define SUPPORT1(DIR) mjmlx_support(t1, pos1, R1, size1, adr1, nv1, DIR)
        #define SUPPORT2(DIR) mjmlx_support(t2, pos2, R2, size2, adr2, nv2, DIR)

        vec3 dir = pos2 - pos1;
        if (msl_len(dir) < 1e-12f) dir = vec3(1,0,0);

        vec3 sdiff[4], sa_pts[4], sb_pts[4];
        int sn = 0;
        vec3 sup_a = SUPPORT1(dir);
        vec3 sup_b = SUPPORT2(-dir);
        sdiff[0] = sup_a - sup_b; sa_pts[0] = sup_a; sb_pts[0] = sup_b;
        sn = 1; dir = -sdiff[0];
        if (msl_len(dir) < 1e-12f) dir = vec3(1,0,0);
        bool gjk_overlap = false;

        for (int iter = 0; iter < 32; iter++) {
            sup_a = SUPPORT1(dir);
            sup_b = SUPPORT2(-dir);
            vec3 new_sd = sup_a - sup_b;
            if (msl_dot(new_sd, dir) < 0.0) break;
            sdiff[sn] = new_sd; sa_pts[sn] = sup_a; sb_pts[sn] = sup_b;
            sn++;
            if (sn == 2) {
                vec3 A=sdiff[1], B=sdiff[0], AB=B-A, AO=-A;
                if (msl_dot(AB,AO)>0.0) { dir=msl_cross(msl_cross(AB,AO),AB); if(msl_len(dir)<1e-12f)dir=AO; }
                else { sdiff[0]=A;sa_pts[0]=sa_pts[1];sb_pts[0]=sb_pts[1];sn=1;dir=AO; }
            } else if (sn == 3) {
                vec3 A=sdiff[2],B=sdiff[1],C=sdiff[0],AB=B-A,AC=C-A,AO=-A;
                vec3 ABC=msl_cross(AB,AC);
                if (msl_dot(msl_cross(ABC,AC),AO)>0.0) { sdiff[0]=C;sdiff[1]=A;sa_pts[0]=sa_pts[0];sa_pts[1]=sa_pts[2];sb_pts[0]=sb_pts[0];sb_pts[1]=sb_pts[2];sn=2;dir=msl_cross(msl_cross(AC,AO),AC);if(msl_len(dir)<1e-12f)dir=AO; }
                else if (msl_dot(msl_cross(AB,ABC),AO)>0.0) { sdiff[0]=B;sdiff[1]=A;sa_pts[0]=sa_pts[1];sa_pts[1]=sa_pts[2];sb_pts[0]=sb_pts[1];sb_pts[1]=sb_pts[2];sn=2;dir=msl_cross(msl_cross(AB,AO),AB);if(msl_len(dir)<1e-12f)dir=AO; }
                else if (msl_dot(ABC,AO)>0.0) { dir=ABC; }
                else { vec3 t=sdiff[0];sdiff[0]=sdiff[1];sdiff[1]=t;t=sa_pts[0];sa_pts[0]=sa_pts[1];sa_pts[1]=t;t=sb_pts[0];sb_pts[0]=sb_pts[1];sb_pts[1]=t;dir=-ABC; }
            } else if (sn == 4) {
                vec3 A=sdiff[3],B=sdiff[2],C=sdiff[1],D=sdiff[0];
                vec3 AB=B-A,AC=C-A,AD=D-A,AO=-A;
                vec3 ABC=msl_cross(AB,AC),ACD=msl_cross(AC,AD),ADB=msl_cross(AD,AB);
                bool abc_o=msl_dot(ABC,AO)>0.0, acd_o=msl_dot(ACD,AO)>0.0, adb_o=msl_dot(ADB,AO)>0.0;
                if (!abc_o&&!acd_o&&!adb_o) { gjk_overlap=true; break; }
                if (abc_o) { sdiff[0]=C;sdiff[1]=B;sdiff[2]=A;sa_pts[0]=sa_pts[1];sa_pts[1]=sa_pts[2];sa_pts[2]=sa_pts[3];sb_pts[0]=sb_pts[1];sb_pts[1]=sb_pts[2];sb_pts[2]=sb_pts[3];sn=3;dir=ABC; }
                else if (acd_o) { sdiff[0]=D;sdiff[1]=C;sdiff[2]=A;sa_pts[2]=sa_pts[3];sb_pts[2]=sb_pts[3];sn=3;dir=ACD; }
                else { vec3 ts;sdiff[0]=B;sdiff[1]=D;sdiff[2]=A;ts=sa_pts[0];sa_pts[0]=sa_pts[2];sa_pts[2]=sa_pts[3];sa_pts[1]=ts;ts=sb_pts[0];sb_pts[0]=sb_pts[2];sb_pts[2]=sb_pts[3];sb_pts[1]=ts;sn=3;dir=ADB; }
            }
            if (msl_len(dir) < 1e-12f) dir = vec3(1,0,0);
        }

        if (!gjk_overlap) {
            vec3 p0=sdiff[0], p1=(sn>=2)?sdiff[1]:sdiff[0];
            vec3 seg=p1-p0; float seg_sq=msl_dot(seg,seg);
            float t=(seg_sq>1e-12f)?-msl_dot(p0,seg)/seg_sq:0.0f;
            t=clamp(t,0.0f,1.0f);
            float gap_d=msl_len(p0+seg*t);
            if (gap_d < pair_margin) {
                vec3 wa=sa_pts[0]+(sa_pts[min(sn-1,1)]-sa_pts[0])*t;
                vec3 wb=sb_pts[0]+(sb_pts[min(sn-1,1)]-sb_pts[0])*t;
                vec3 n=msl_norm(wa-wb);
                vec3 cp=(wa+wb)*0.5f;
                c_pos[0]=cp.x;c_pos[1]=cp.y;c_pos[2]=cp.z;
                c_norm[0]=n.x;c_norm[1]=n.y;c_norm[2]=n.z;
                c_dist=gap_d; has_contact=true;
            }
        } else {
            float bw=1e30f; vec3 bn=vec3(0,0,1),bpa,bpb;
            const float D=0.577350269f;
            vec3 sd[14]=vec3[](vec3(1,0,0),vec3(-1,0,0),vec3(0,1,0),vec3(0,-1,0),vec3(0,0,1),vec3(0,0,-1),
                vec3(D,D,D),vec3(-D,D,D),vec3(D,-D,D),vec3(D,D,-D),
                vec3(-D,-D,D),vec3(-D,D,-D),vec3(D,-D,-D),vec3(-D,-D,-D));
            for (int si=0;si<14;si++) {
                vec3 d=sd[si];
                vec3 pa=SUPPORT1(d), pb=SUPPORT2(-d);
                float w=msl_dot(pa-pb,d);
                if(w<bw){bw=w;bn=d;bpa=pa;bpb=pb;}
            }
            for (int fi=0;fi<4&&sn>=3;fi++) {
                vec3 e1,e2;
                if(fi==0){e1=sdiff[1]-sdiff[0];e2=sdiff[2]-sdiff[0];}
                else if(fi==1){e1=sdiff[2]-sdiff[0];e2=sdiff[min(sn-1,3)]-sdiff[0];}
                else if(fi==2){e1=sdiff[min(sn-1,3)]-sdiff[0];e2=sdiff[1]-sdiff[0];}
                else{e1=sdiff[2]-sdiff[1];e2=sdiff[min(sn-1,3)]-sdiff[1];}
                vec3 fn=msl_cross(e1,e2);
                if(msl_len(fn)<1e-8f)continue;
                fn=msl_norm(fn);
                for(int s=-1;s<=1;s+=2){
                    vec3 d=fn*float(s);
                    vec3 pa=SUPPORT1(d), pb=SUPPORT2(-d);
                    float w=msl_dot(pa-pb,d);
                    if(w<bw){bw=w;bn=d;bpa=pa;bpb=pb;}
                }
            }
            float pen=max(bw,0.0f);
            vec3 cp=(bpa+bpb)*0.5f;
            c_pos[0]=cp.x;c_pos[1]=cp.y;c_pos[2]=cp.z;
            c_norm[0]=bn.x;c_norm[1]=bn.y;c_norm[2]=bn.z;
            c_dist=-pen; has_contact=true;
        }
        #undef SUPPORT1
        #undef SUPPORT2
    }

    if (has_contact && ncon < MAX_CON) {
        int idx = int(con_off) + ncon * STRIDE;
        contact_data[idx+0]=c_pos[0]; contact_data[idx+1]=c_pos[1]; contact_data[idx+2]=c_pos[2];
        contact_data[idx+3]=c_norm[0]; contact_data[idx+4]=c_norm[1]; contact_data[idx+5]=c_norm[2];
        contact_data[idx+6]=c_dist;
        contact_data[idx+7]=float(p);
        ncon++;
    }
}

contact_count[bid] = float(ncon);
)GLSLBODY";

  return mkx::fast::compute_kernel("mjmlx_collision", {"geom_xpos", "geom_xmat", "mesh_verts", "pair_data", "mesh_vertadr_buf", "mesh_vertnum_buf", "geom_dataid_buf", "geom_size_buf"}, {"contact_data", "contact_count"}, ss.str(), header);
}

} // namespace mjmlx::mkx_kernels
