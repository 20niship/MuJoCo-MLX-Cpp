// Copyright 2026 Arghya Sur
// Licensed under the Apache License, Version 2.0

// Internal C++ header -- not part of public API.
// Defines the actual structs behind opaque C handles.

#pragma once

#include <mlx/mlx.h>
#include <mlx/linalg.h>
#include <mujoco/mujoco.h>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mjmlx/mjmlx_types.h"

namespace mx = mlx::core;

namespace mjmlx {

// ── Enums (matching Python types.py) ──────────────────────────

enum class JointType : int { FREE = 0, BALL = 1, SLIDE = 2, HINGE = 3 };
enum class GeomType : int {
    PLANE = 0, HFIELD = 1, SPHERE = 2, CAPSULE = 3,
    ELLIPSOID = 4, CYLINDER = 5, BOX = 6, MESH = 7
};
enum class IntegratorType : int { EULER = 0, RK4 = 1, IMPLICIT = 2, IMPLICITFAST = 3 };
enum class SolverType : int { PGS = 0, CG = 1, NEWTON = 2 };
enum class ConeType : int { PYRAMIDAL = 0, ELLIPTIC = 1 };
enum class GainType : int { FIXED = 0, AFFINE = 1, MUSCLE = 2 };
enum class BiasType : int { NONE = 0, AFFINE = 1, MUSCLE = 2 };

// Disable bit flags (matching MuJoCo C mjDisableBit)
namespace DisableBit {
    constexpr int CONSTRAINT   = (1 << 0);
    constexpr int EQUALITY     = (1 << 1);
    constexpr int FRICTIONLOSS = (1 << 2);
    constexpr int LIMIT        = (1 << 3);
    constexpr int CONTACT      = (1 << 4);
    constexpr int SPRING       = (1 << 4);  // note: shares bit with CONTACT in some MuJoCo versions
    constexpr int PASSIVE      = (1 << 5);
    constexpr int DAMPER       = (1 << 6);
    constexpr int GRAVITY      = (1 << 7);
    constexpr int CLAMPCTRL    = (1 << 8);
    constexpr int WARMSTART    = (1 << 9);
    constexpr int FILTERPARENT = (1 << 10);
    constexpr int ACTUATION    = (1 << 11);
    constexpr int REFSAFE      = (1 << 12);
    constexpr int EULERDAMP    = (1 << 18);
}

// ── Option ───────────────────────────────────────────────────

struct Option {
    float timestep = 0.002f;
    float tolerance = 1e-8f;
    float ls_tolerance = 0.01f;
    int iterations = 100;
    int ls_iterations = 50;
    mx::array gravity{mx::array({0.0f, 0.0f, -9.81f})};
    IntegratorType integrator = IntegratorType::EULER;
    SolverType solver = SolverType::NEWTON;
    ConeType cone = ConeType::PYRAMIDAL;
    int disableflags = 0;
    float impratio = 1.0f;
    mx::array wind{mx::zeros({3})};
    float viscosity = 0.0f;
    float density = 0.0f;
};

// ── Statistic ────────────────────────────────────────────────

struct Statistic {
    float meaninertia = 1.0f;
    float meanmass = 1.0f;
    float meansize = 0.1f;
    float extent = 1.0f;
    mx::array center{mx::zeros({3})};
};

// ── Contact ──────────────────────────────────────────────────

struct Contact {
    mx::array dist{mx::array({})};          // (ncon,)
    mx::array pos{mx::array({})};           // (ncon, 3)
    mx::array frame{mx::array({})};         // (ncon, 3, 3)
    mx::array dim{mx::array({}, mx::int32)};
    mx::array friction{mx::array({})};
    mx::array solref{mx::array({})};        // (ncon, 2)
    mx::array solimp{mx::array({})};        // (ncon, 5)
    mx::array includemargin{mx::array({})};
    mx::array solreffriction{mx::array({})};
    mx::array geom{mx::array({})};          // (ncon, 2)
    mx::array efc_address{mx::array({}, mx::int32)};
};

// ── Model ────────────────────────────────────────────────────

struct Model {
    // Counts
    int nq = 0, nv = 0, nu = 0, na = 0;
    int nbody = 0, njnt = 0, ngeom = 0, nsite = 0;
    int ncam = 0, nmesh = 0, nmocap = 0, ntendon = 0;
    int neq = 0, ncon = 0, ngravcomp = 0, npair = 0, nexclude = 0;

    Option opt;
    Statistic stat;

    // Reference configuration
    mx::array qpos0{mx::array({})};
    mx::array qpos_spring{mx::array({})};

    // Body properties
    mx::array body_parentid{mx::array({})};
    mx::array body_rootid{mx::array({})};
    mx::array body_weldid{mx::array({})};
    mx::array body_jntadr{mx::array({})};
    mx::array body_jntnum{mx::array({})};
    mx::array body_dofadr{mx::array({})};
    mx::array body_dofnum{mx::array({})};
    mx::array body_geomadr{mx::array({})};
    mx::array body_geomnum{mx::array({})};
    mx::array body_pos{mx::array({})};
    mx::array body_quat{mx::array({})};
    mx::array body_mass{mx::array({})};
    mx::array body_subtreemass{mx::array({})};
    mx::array body_inertia{mx::array({})};
    mx::array body_ipos{mx::array({})};
    mx::array body_iquat{mx::array({})};
    mx::array body_invweight0{mx::array({})};
    mx::array body_gravcomp{mx::array({})};
    mx::array body_mocapid{mx::array({})};

    // Joint properties
    mx::array jnt_type{mx::array({})};
    mx::array jnt_bodyid{mx::array({})};
    mx::array jnt_qposadr{mx::array({})};
    mx::array jnt_dofadr{mx::array({})};
    mx::array jnt_range{mx::array({})};
    mx::array jnt_limited{mx::array({})};
    mx::array jnt_axis{mx::array({})};
    mx::array jnt_pos{mx::array({})};
    mx::array jnt_stiffness{mx::array({})};
    mx::array jnt_margin{mx::array({})};
    mx::array jnt_solref{mx::array({})};
    mx::array jnt_solimp{mx::array({})};

    // DOF properties
    mx::array dof_bodyid{mx::array({})};
    mx::array dof_jntid{mx::array({})};
    mx::array dof_parentid{mx::array({})};
    mx::array dof_Madr{mx::array({})};
    mx::array dof_armature{mx::array({})};
    mx::array dof_damping{mx::array({})};
    mx::array dof_invweight0{mx::array({})};
    mx::array dof_frictionloss{mx::array({})};

    // Site properties
    mx::array site_bodyid{mx::array({})};
    mx::array site_pos{mx::array({})};
    mx::array site_quat{mx::array({})};

    // Actuator properties
    mx::array actuator_trntype{mx::array({})};
    mx::array actuator_trnid{mx::array({})};
    mx::array actuator_gaintype{mx::array({})};
    mx::array actuator_gainprm{mx::array({})};
    mx::array actuator_biastype{mx::array({})};
    mx::array actuator_biasprm{mx::array({})};
    mx::array actuator_dyntype{mx::array({})};
    mx::array actuator_dynprm{mx::array({})};
    mx::array actuator_gear{mx::array({})};
    mx::array actuator_ctrllimited{mx::array({})};
    mx::array actuator_ctrlrange{mx::array({})};
    mx::array actuator_forcelimited{mx::array({})};
    mx::array actuator_forcerange{mx::array({})};

    // Geom properties (for collision)
    mx::array geom_type{mx::array({})};
    mx::array geom_bodyid{mx::array({})};
    mx::array geom_pos{mx::array({})};
    mx::array geom_quat{mx::array({})};
    mx::array geom_size{mx::array({})};
    mx::array geom_friction{mx::array({})};
    mx::array geom_solmix{mx::array({})};
    mx::array geom_solref{mx::array({})};
    mx::array geom_solimp{mx::array({})};
    mx::array geom_margin{mx::array({})};
    mx::array geom_gap{mx::array({})};
    mx::array geom_contype{mx::array({})};
    mx::array geom_conaffinity{mx::array({})};
    mx::array geom_condim{mx::array({})};

    // Pair properties
    mx::array pair_geom1{mx::array({})};
    mx::array pair_geom2{mx::array({})};
    mx::array pair_dim{mx::array({})};
    mx::array pair_margin{mx::array({})};
    mx::array pair_gap{mx::array({})};
    mx::array pair_friction{mx::array({})};
    mx::array pair_solref{mx::array({})};
    mx::array pair_solimp{mx::array({})};

    // Equality constraint properties
    mx::array eq_type{mx::array({})};
    mx::array eq_obj1id{mx::array({})};
    mx::array eq_obj2id{mx::array({})};
    mx::array eq_data{mx::array({})};
    mx::array eq_solref{mx::array({})};
    mx::array eq_solimp{mx::array({})};

    // Exclude
    mx::array exclude_signature{mx::array({})};

    // ── Precomputed cache (for vmap-compatible pipeline) ─────
    // Populated once at load time. All data is read eagerly so
    // the vmap-traced path never calls eval()/data<>().

    struct ModelCache {
        bool initialized = false;

        // Tree topology: bodies grouped by tree depth (for level-parallel scatter-add)
        // tree_levels[0] = {0} (world), tree_levels[1] = {children of world}, etc.
        std::vector<std::vector<int>> tree_levels;

        // Collision pairs: precomputed (g1, g2) with properties
        struct CollisionPair {
            int g1, g2;
            int type1, type2;
            int body1, body2;
            float margin, gap;
            float friction[5];
            float solref[2];
            float solimp[5];
            float size1[3], size2[3];
            int condim;
        };
        std::vector<CollisionPair> collision_pairs;
        int max_ncon = 0;  // = collision_pairs.size()

        // Joint limit plan: which joints are limited (HINGE/SLIDE)
        struct LimitInfo {
            int jnt_idx;
            int dof_adr;
            float range_low, range_high;
            float solref[2];
            float solimp[5];
            float margin;
        };
        std::vector<LimitInfo> limits;
        int max_nl = 0;  // = limits.size()
        int max_nefc = 0; // = max_nl + max_ncon (fixed constraint budget)

        // DOF info: precomputed for vectorized cdof
        struct DofInfo {
            int dof_idx;
            int body_id;
            int jnt_type;  // JointType enum
            int jnt_idx;
            int qpos_adr;
        };
        std::vector<DofInfo> dof_info;

        // Body-DOF mapping: for each body, list of its DOF indices
        std::vector<std::vector<int>> body_dofs;

        // Joint integration plan (for Euler / Metal euler)
        std::vector<int> simple_qa, simple_da;  // HINGE/SLIDE
        std::vector<std::pair<int,int>> free_joints;   // (qa, da) for FREE
        std::vector<std::pair<int,int>> ball_joints;    // (qa, da) for BALL
        std::vector<float> dof_damping_vals;

        // Actuator plan
        struct ActuatorInfo {
            int act_idx;
            int jnt_idx;   // target joint
            int dof_adr;   // target DOF address
            int qpos_adr;  // qpos address for length
            float gain;
        };
        std::vector<ActuatorInfo> actuator_info;
        mx::array act_moment_const{mx::array(0.0f)};
        mx::array act_qpos_idxs{mx::array(0.0f)};
        mx::array act_gear{mx::array(0.0f)};

        // Precomputed passive force arrays
        mx::array passive_stiffness{mx::array(0.0f)};
        mx::array passive_qpos_idxs{mx::array(0.0f)};  

        // Plain C++ vectors for loop indexing (no eval needed)
        std::vector<int> body_parentid_vec;
        std::vector<int> body_rootid_vec;
        std::vector<int> dof_bodyid_vec;

        // CDoF plan (for vectorized cdof computation)
        struct CdofPlan {
            mx::array bids{mx::zeros({1}, mx::int32)};
            mx::array jidxs{mx::zeros({1}, mx::int32)};
            mx::array root_bids{mx::zeros({1}, mx::int32)};  // rootid[bids]
            mx::array is_hinge{mx::zeros({1})};
            mx::array is_slide{mx::zeros({1})};
            mx::array is_free_trans{mx::zeros({1})};
            mx::array is_free_rot{mx::zeros({1})};
            mx::array is_ball{mx::zeros({1})};
            mx::array free_trans_unit{mx::zeros({1})};
            mx::array rot_col0_mask{mx::zeros({1})};
            mx::array rot_col1_mask{mx::zeros({1})};
            mx::array rot_col2_mask{mx::zeros({1})};
        };
        CdofPlan cdof_plan;

        // Body DOF ancestor masks: for Jacobian computation in constraints
        // body_dof_masks[body_id] = (nv,) float where 1.0 if DOF i is ancestor of body
        std::vector<mx::array> body_dof_masks;

        // Dense mass matrix tree mask: (nv, nv) float
        mx::array make_m_mask{mx::zeros({1})};

        // Precomputed MLX arrays (model constants in the vmap graph)
        mx::array gravity_6d{mx::zeros({6})};
    };

    mutable ModelCache cache;

    // Populate cache (called once after loading)
    void init_cache() const;
};

// ── Data ─────────────────────────────────────────────────────

struct Data {
    // State
    mx::array qpos{mx::array({})};
    mx::array qvel{mx::array({})};
    mx::array qacc{mx::array({})};
    mx::array ctrl{mx::array({})};
    mx::array act{mx::array({})};

    // Derived quantities (computed by forward/step)
    mx::array xpos{mx::array({})};       // (nbody, 3) body positions
    mx::array xquat{mx::array({})};      // (nbody, 4) body quaternions
    mx::array xmat{mx::array({})};       // (nbody, 3, 3) body rotation matrices
    mx::array xipos{mx::array({})};      // (nbody, 3) body COM positions
    mx::array ximat{mx::array({})};      // (nbody, 3, 3) body COM rotations
    mx::array geom_xpos{mx::array({})};  // (ngeom, 3) geom positions
    mx::array geom_xmat{mx::array({})};  // (ngeom, 3, 3) geom rotations
    mx::array site_xpos{mx::array({})};  // (nsite, 3) site positions
    mx::array site_xmat{mx::array({})};  // (nsite, 3, 3) site rotations

    // Joint anchors/axes (computed by kinematics)
    mx::array xanchor{mx::array({})};    // (njnt, 3) joint anchors
    mx::array xaxis{mx::array({})};      // (njnt, 3) joint axes

    // Applied forces
    mx::array xfrc_applied{mx::array({})};  // (nbody, 6) external forces
    mx::array qfrc_applied{mx::array({})};  // (nv,) applied joint forces

    // Actuator
    mx::array actuator_length{mx::array({})};    // (nu,)
    mx::array actuator_moment{mx::array({})};    // (nu, nv)
    mx::array actuator_velocity{mx::array({})};  // (nu,)
    mx::array actuator_force{mx::array({})};     // (nu,)
    mx::array act_dot{mx::array({})};            // (na,)

    // Dynamics
    mx::array subtree_com{mx::array({})};    // (nbody, 3)
    mx::array cinert{mx::array({})};         // (nbody, 10)
    mx::array crb{mx::array({})};            // (nbody, 10)
    mx::array cdof{mx::array({})};           // (nv, 6)
    mx::array cvel{mx::array({})};           // (nbody, 6)
    mx::array cdof_dot{mx::array({})};       // (nv, 6)
    mx::array qM{mx::array({})};             // (nv, nv) or sparse
    mx::array qLD{mx::array({})};            // factored mass matrix
    mx::array qM_inv{mx::array({})};         // precomputed M^{-1} for GPU solves
    mx::array qLDiagInv{mx::array({})};      // inverse diagonal
    mx::array qfrc_bias{mx::array({})};      // (nv,) Coriolis + gravity
    mx::array qfrc_passive{mx::array({})};   // (nv,) spring/damper
    mx::array qfrc_actuator{mx::array({})};  // (nv,) actuator forces
    mx::array qfrc_gravcomp{mx::array({})};   // (nv,) gravity compensation
    mx::array qfrc_smooth{mx::array({})};    // (nv,) smooth forces (bias+passive+actuator)
    mx::array qacc_smooth{mx::array({})};    // (nv,) acceleration from smooth forces

    // Solver
    mx::array qfrc_constraint{mx::array({})};  // (nv,) constraint forces
    mx::array qacc_warmstart{mx::array({})};   // (nv,) warmstart acceleration

    // Constraint
    mx::array efc_J{mx::array({})};
    mx::array efc_D{mx::array({})};
    mx::array efc_aref{mx::array({})};
    mx::array efc_force{mx::array({})};
    mx::array efc_frictionloss{mx::array({})};
    int nefc = 0;
    int ne = 0, nf = 0, nl = 0;  // equality, friction, limit counts
    int ncon = 0;

    // Contact
    Contact contact;

    // Contact forces (per-body external forces from constraints)
    mx::array cfrc_ext{mx::array({})};   // (nbody, 6)

    // Time
    mx::array time{mx::array(0.0f)};
};

// ── BatchedSim ───────────────────────────────────────────────

struct BatchedSim {
    const Model* model;
    int num_envs;
    MjmlxBatchedConfig config;

    // Batched state: [num_envs, ...] arrays
    mx::array qpos{mx::array({})};
    mx::array qvel{mx::array({})};
    mx::array xpos{mx::array({})};

    // Observation fields: [num_envs, ...] arrays
    mx::array subtree_com{mx::array({})};   // (B, nbody, 3)
    mx::array cinert{mx::array({})};        // (B, nbody, 10)
    mx::array cvel{mx::array({})};          // (B, nbody, 6)
    mx::array qfrc_actuator{mx::array({})}; // (B, nv)
    mx::array cfrc_ext{mx::array({})};      // (B, nbody, 6)

    // Compiled+vmapped step function
    std::function<std::vector<mx::array>(const std::vector<mx::array>&)> compiled_step;
};

// ── Module functions (internal C++ API) ──────────────────────

// io.cpp
Model load_model(const char* xml_path);
Model load_model_filtered(const char* xml_path, bool foot_contacts_only);
Model load_model_from_string(const char* xml_string);
std::pair<Model, mjModel*> load_model_pair(const char* xml_path);
std::pair<Model, mjModel*> load_model_filtered_pair(const char* xml_path, bool foot_contacts_only);
std::pair<Model, mjModel*> load_model_from_string_pair(const char* xml_string);
Data make_data(const Model& model);

// math.cpp
mx::array cross(const mx::array& a, const mx::array& b);
mx::array norm(const mx::array& x);
mx::array normalize(const mx::array& x);
mx::array quat_mul(const mx::array& q1, const mx::array& q2);
mx::array quat_inv(const mx::array& q);
mx::array quat_to_mat(const mx::array& q);
mx::array rotate(const mx::array& vec, const mx::array& quat);
mx::array quat_integrate(const mx::array& q, const mx::array& v, float dt);
mx::array axis_angle_to_quat(const mx::array& axis, const mx::array& angle);
mx::array inert_mul(const mx::array& inert, const mx::array& vel);
mx::array motion_cross(const mx::array& u, const mx::array& v);
mx::array motion_cross_force(const mx::array& v, const mx::array& f);
std::pair<mx::array, mx::array> orthogonals(const mx::array& n);
mx::array closest_segment_point(const mx::array& p0, const mx::array& p1,
                                 const mx::array& point);
std::pair<mx::array, mx::array> closest_segment_to_segment_points(
    const mx::array& a0, const mx::array& a1,
    const mx::array& b0, const mx::array& b1);

// smooth.cpp
Data kinematics(const Model& m, Data d);
Data com_pos(const Model& m, Data d);
Data crb(const Model& m, Data d);
Data factor_m(const Model& m, Data d);
mx::array solve_m(const Model& m, const Data& d, const mx::array& rhs);
Data com_vel(const Model& m, Data d);
Data rne(const Model& m, Data d, bool flg_acc = false);
Data transmission(const Model& m, Data d);

// collision.cpp
Data collision(const Model& m, Data d);

// constraint.cpp
Data make_constraint(const Model& m, Data d);

// solver.cpp
Data solve(const Model& m, Data d);

// passive.cpp
Data passive(const Model& m, Data d);

// support.cpp
bool is_sparse(const Model& m);
std::pair<mx::array, mx::array> local_to_global(
    const mx::array& world_pos, const mx::array& world_quat,
    const mx::array& local_pos, const mx::array& local_quat);
mx::array make_m(const Model& m, const mx::array& a, const mx::array& b,
                 const mx::array& d_diag = mx::array({}));
mx::array full_m(const Model& m, const Data& d);
mx::array mul_m(const Model& m, const Data& d, const mx::array& vec);
mx::array xfrc_accumulate(const Model& m, const Data& d);
std::pair<mx::array, mx::array> jac(const Model& m, const Data& d,
                                     const mx::array& point, int body_id);

// forward.cpp
Data forward(const Model& m, Data d);
Data step(const Model& m, Data d);
Data step1(const Model& m, Data d);
Data step2(const Model& m, Data d);
Data rne_post_constraint(const Model& m, Data d);

// batched.cpp
std::function<std::vector<mx::array>(const std::vector<mx::array>&)>
make_batched_step(const Model& m, int num_envs, bool use_gpu, int solver_iterations_override = 0);

// ── Vmap-compatible functions (pure MLX graph, no eval/data) ──

// gpu_linalg (smooth_vmap.cpp) — vmap-compatible Cholesky for Newton solver
mx::array cholesky_gpu(const mx::array& A, int n);
mx::array cholesky_solve_gpu(const mx::array& L, const mx::array& b, int n);

// smooth_vmap.cpp
Data vmap_com_pos(const Model& m, Data d);
Data vmap_crb(const Model& m, Data d);
Data vmap_factor_m(const Model& m, Data d);
mx::array vmap_solve_m(const Model& m, const Data& d, const mx::array& rhs);
Data vmap_com_vel(const Model& m, Data d);
Data vmap_rne(const Model& m, Data d);
Data vmap_transmission(const Model& m, Data d);

// constraint_vmap.cpp
Data vmap_collision(const Model& m, Data d);
Data vmap_make_constraint(const Model& m, Data d);

// solver_vmap.cpp
Data vmap_solve(const Model& m, Data d);

// forward_vmap.cpp
Data vmap_forward(const Model& m, Data d);

// Batched math helpers (math.cpp)
mx::array batched_cross(const mx::array& a, const mx::array& b);
mx::array batched_inert_mul(const mx::array& inert, const mx::array& vel);
mx::array batched_motion_cross_force(const mx::array& v, const mx::array& f);

} // namespace mjmlx

// Expose internal types as the opaque C handles
struct MjmlxModel {
    mjmlx::Model model;
    mjModel* mj_model = nullptr;  // kept alive for name2id / field accessors
    ~MjmlxModel() { if (mj_model) mj_deleteModel(mj_model); }
};
struct MjmlxData { mjmlx::Data data; const mjmlx::Model* model_ref; };
struct MjmlxBatchedSim { mjmlx::BatchedSim sim; };
