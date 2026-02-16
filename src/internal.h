// Copyright 2026 Arghya Sur
// Licensed under the Apache License, Version 2.0

// Internal C++ header -- not part of public API.
// Defines the actual structs behind opaque C handles.

#pragma once

#include <mlx/mlx.h>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
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
    int neq = 0, ncon = 0, ngravcomp = 0, npair = 0;

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

    // Dynamics
    mx::array subtree_com{mx::array({})};    // (nbody, 3)
    mx::array cinert{mx::array({})};         // (nbody, 10)
    mx::array crb{mx::array({})};            // (nbody, 10)
    mx::array cdof{mx::array({})};           // (nv, 6)
    mx::array cvel{mx::array({})};           // (nbody, 6)
    mx::array cdof_dot{mx::array({})};       // (nv, 6)
    mx::array qM{mx::array({})};             // (nv, nv) or sparse
    mx::array qLD{mx::array({})};            // factored mass matrix
    mx::array qLDiagInv{mx::array({})};      // inverse diagonal
    mx::array qfrc_bias{mx::array({})};      // (nv,) Coriolis + gravity
    mx::array qfrc_passive{mx::array({})};   // (nv,) spring/damper
    mx::array qfrc_actuator{mx::array({})};  // (nv,) actuator forces
    mx::array qfrc_smooth{mx::array({})};    // (nv,) smooth forces (bias+passive+actuator)
    mx::array qacc_smooth{mx::array({})};    // (nv,) acceleration from smooth forces

    // Constraint
    mx::array efc_J{mx::array({})};
    mx::array efc_D{mx::array({})};
    mx::array efc_aref{mx::array({})};
    mx::array efc_force{mx::array({})};
    int nefc = 0;

    // Contact
    Contact contact;
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

    // Compiled+vmapped step function
    std::function<std::vector<mx::array>(const std::vector<mx::array>&)> compiled_step;
};

// ── ActorCritic ──────────────────────────────────────────────

struct ActorCritic {
    MjmlxActorCriticConfig config;

    // Network weights (variable number of hidden layers)
    std::vector<mx::array> weights;
    std::vector<mx::array> biases;

    // Actor head
    mx::array actor_w{mx::array({})};
    mx::array actor_b{mx::array({})};
    mx::array log_std{mx::array({})};

    // Critic head
    mx::array critic_w{mx::array({})};
    mx::array critic_b{mx::array({})};
};

// ── PPOTrainer ───────────────────────────────────────────────

struct PPOTrainer {
    BatchedSim* sim;
    ActorCritic* nn;
    MjmlxPPOConfig config;

    // Training state
    int global_step = 0;
    float last_policy_loss = 0;
    float last_value_loss = 0;
    float last_entropy = 0;
    float last_approx_kl = 0;
};

// ── Module functions (internal C++ API) ──────────────────────

// io.cpp
Model load_model(const char* xml_path);
Model load_model_from_string(const char* xml_string);
Data make_data(const Model& model);

// math.cpp
mx::array quat_mul(const mx::array& q1, const mx::array& q2);
mx::array quat_to_mat(const mx::array& q);
mx::array rotate(const mx::array& vec, const mx::array& quat);
mx::array normalize(const mx::array& x);

// forward.cpp
Data forward(const Model& m, Data d);
Data step(const Model& m, Data d);

// batched.cpp
std::function<std::vector<mx::array>(const std::vector<mx::array>&)>
make_batched_step(const Model& m, int num_envs, bool use_gpu);

} // namespace mjmlx

// Expose internal types as the opaque C handles
struct MjmlxModel { mjmlx::Model model; };
struct MjmlxData { mjmlx::Data data; const mjmlx::Model* model_ref; };
struct MjmlxBatchedSim { mjmlx::BatchedSim sim; };
struct MjmlxActorCritic { mjmlx::ActorCritic nn; };
struct MjmlxPPOTrainer { mjmlx::PPOTrainer trainer; };
