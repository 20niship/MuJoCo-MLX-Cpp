// Inline XML test models for MuJoCo-MLX-Cpp conformance tests.
// Organized by feature so each test file can #include this and pick the model it needs.
#pragma once

// ============================================================================
// Phase 1: Synth Foundation Models
// ============================================================================

// 3-body chain with gravcomp on middle body.
// Body 2 has gravcomp=0.5, so gravity is partially compensated.
static const char* GRAVCOMP_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="base" pos="0 0 1">
      <joint name="j0" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05 0.2" mass="1"/>
      <body name="mid" pos="0 0 -0.4" gravcomp="0.5">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.05 0.2" mass="1"/>
        <body name="tip" pos="0 0 -0.4">
          <joint name="j2" type="hinge" axis="0 1 0"/>
          <geom type="capsule" size="0.05 0.2" mass="1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j0" gear="50"/>
    <motor joint="j1" gear="50"/>
    <motor joint="j2" gear="50"/>
  </actuator>
</mujoco>
)";

// Full gravcomp=1.0 model (gravity fully compensated on all bodies).
static const char* GRAVCOMP_FULL_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="b1" pos="0 0 1" gravcomp="1">
      <joint name="j0" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05 0.2" mass="1"/>
      <body name="b2" pos="0 0 -0.4" gravcomp="1">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.05 0.2" mass="1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
)";

// Free body on a plane -- generates nonzero cfrc_ext on contact.
static const char* CFRC_EXT_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0"/>
    <body name="ball" pos="0 0 0.1">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

// Dense body tree with exclude directives to skip adjacent body collisions.
static const char* EXCLUDE_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="torso" pos="0 0 0.5">
      <freejoint/>
      <geom name="torso_g" type="capsule" size="0.1 0.15" mass="5"/>
      <body name="upper_arm" pos="0.2 0 0.15">
        <joint name="shoulder" type="hinge" axis="0 1 0"/>
        <geom name="ua_g" type="capsule" size="0.05 0.15" mass="1"/>
        <body name="lower_arm" pos="0 0 -0.3">
          <joint name="elbow" type="hinge" axis="0 1 0"/>
          <geom name="la_g" type="capsule" size="0.04 0.12" mass="0.5"/>
        </body>
      </body>
      <body name="upper_leg" pos="0 0 -0.15">
        <joint name="hip" type="hinge" axis="0 1 0"/>
        <geom name="ul_g" type="capsule" size="0.06 0.2" mass="2"/>
        <body name="lower_leg" pos="0 0 -0.4">
          <joint name="knee" type="hinge" axis="0 1 0"/>
          <geom name="ll_g" type="capsule" size="0.05 0.15" mass="1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <contact>
    <exclude body1="torso" body2="upper_arm"/>
    <exclude body1="torso" body2="upper_leg"/>
    <exclude body1="upper_arm" body2="lower_arm"/>
    <exclude body1="upper_leg" body2="lower_leg"/>
  </contact>
  <actuator>
    <motor joint="shoulder" gear="50"/>
    <motor joint="elbow" gear="30"/>
    <motor joint="hip" gear="100"/>
    <motor joint="knee" gear="50"/>
  </actuator>
</mujoco>
)";

// High-DOF tree model (~63 DOF, similar to Synth nv~69).
// 21 bodies, each with 3 hinge joints (x,y,z rotation).
static const char* HIGH_DOF_TREE_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <default>
    <geom type="capsule" size="0.03 0.08" mass="0.5"/>
    <joint damping="0.5" armature="0.01"/>
    <motor gear="20"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="root" pos="0 0 1.2">
      <freejoint/>
      <geom size="0.08 0.15" mass="10"/>
      <!-- Spine chain (5 segments including neck) -->
      <body name="spine1" pos="0 0 0.16">
        <joint name="sp1x" type="hinge" axis="1 0 0"/><joint name="sp1y" type="hinge" axis="0 1 0"/><joint name="sp1z" type="hinge" axis="0 0 1"/>
        <geom/><body name="spine2" pos="0 0 0.16">
          <joint name="sp2x" type="hinge" axis="1 0 0"/><joint name="sp2y" type="hinge" axis="0 1 0"/><joint name="sp2z" type="hinge" axis="0 0 1"/>
          <geom/><body name="spine3" pos="0 0 0.16">
            <joint name="sp3x" type="hinge" axis="1 0 0"/><joint name="sp3y" type="hinge" axis="0 1 0"/><joint name="sp3z" type="hinge" axis="0 0 1"/>
            <geom/><body name="neck" pos="0 0 0.14">
              <joint name="nkx" type="hinge" axis="1 0 0"/><joint name="nky" type="hinge" axis="0 1 0"/><joint name="nkz" type="hinge" axis="0 0 1"/>
              <geom size="0.025 0.06" mass="0.3"/><body name="head" pos="0 0 0.12">
                <joint name="hdx" type="hinge" axis="1 0 0"/><joint name="hdy" type="hinge" axis="0 1 0"/><joint name="hdz" type="hinge" axis="0 0 1"/>
                <geom type="sphere" size="0.08" mass="3"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <!-- Left arm chain with fingers -->
      <body name="l_shoulder" pos="0.2 0 0.15">
        <joint name="lsx" type="hinge" axis="1 0 0"/><joint name="lsy" type="hinge" axis="0 1 0"/><joint name="lsz" type="hinge" axis="0 0 1"/>
        <geom/><body name="l_elbow" pos="0 0 -0.2">
          <joint name="lex" type="hinge" axis="1 0 0"/><joint name="ley" type="hinge" axis="0 1 0"/><joint name="lez" type="hinge" axis="0 0 1"/>
          <geom/><body name="l_wrist" pos="0 0 -0.18">
            <joint name="lwx" type="hinge" axis="1 0 0"/><joint name="lwy" type="hinge" axis="0 1 0"/><joint name="lwz" type="hinge" axis="0 0 1"/>
            <geom size="0.02 0.05" mass="0.2"/>
            <body name="l_finger1" pos="0.02 0 -0.06">
              <joint name="lf1x" type="hinge" axis="1 0 0"/><joint name="lf1y" type="hinge" axis="0 1 0"/>
              <geom size="0.008 0.03" mass="0.05"/>
            </body>
            <body name="l_finger2" pos="-0.02 0 -0.06">
              <joint name="lf2x" type="hinge" axis="1 0 0"/><joint name="lf2y" type="hinge" axis="0 1 0"/>
              <geom size="0.008 0.03" mass="0.05"/>
            </body>
          </body>
        </body>
      </body>
      <!-- Right arm chain with fingers -->
      <body name="r_shoulder" pos="-0.2 0 0.15">
        <joint name="rsx" type="hinge" axis="1 0 0"/><joint name="rsy" type="hinge" axis="0 1 0"/><joint name="rsz" type="hinge" axis="0 0 1"/>
        <geom/><body name="r_elbow" pos="0 0 -0.2">
          <joint name="rex" type="hinge" axis="1 0 0"/><joint name="rey" type="hinge" axis="0 1 0"/><joint name="rez" type="hinge" axis="0 0 1"/>
          <geom/><body name="r_wrist" pos="0 0 -0.18">
            <joint name="rwx" type="hinge" axis="1 0 0"/><joint name="rwy" type="hinge" axis="0 1 0"/><joint name="rwz" type="hinge" axis="0 0 1"/>
            <geom size="0.02 0.05" mass="0.2"/>
            <body name="r_finger1" pos="0.02 0 -0.06">
              <joint name="rf1x" type="hinge" axis="1 0 0"/><joint name="rf1y" type="hinge" axis="0 1 0"/>
              <geom size="0.008 0.03" mass="0.05"/>
            </body>
            <body name="r_finger2" pos="-0.02 0 -0.06">
              <joint name="rf2x" type="hinge" axis="1 0 0"/><joint name="rf2y" type="hinge" axis="0 1 0"/>
              <geom size="0.008 0.03" mass="0.05"/>
            </body>
          </body>
        </body>
      </body>
      <!-- Left leg chain with toes -->
      <body name="l_hip" pos="0.1 0 -0.15">
        <joint name="lhx" type="hinge" axis="1 0 0"/><joint name="lhy" type="hinge" axis="0 1 0"/><joint name="lhz" type="hinge" axis="0 0 1"/>
        <geom size="0.04 0.15" mass="2"/><body name="l_knee" pos="0 0 -0.3">
          <joint name="lkx" type="hinge" axis="1 0 0"/><joint name="lky" type="hinge" axis="0 1 0"/><joint name="lkz" type="hinge" axis="0 0 1"/>
          <geom size="0.035 0.12" mass="1.5"/><body name="l_ankle" pos="0 0 -0.25">
            <joint name="lax" type="hinge" axis="1 0 0"/><joint name="lay" type="hinge" axis="0 1 0"/><joint name="laz" type="hinge" axis="0 0 1"/>
            <geom size="0.03 0.06" mass="0.5"/>
            <body name="l_toe" pos="0 0.06 -0.06">
              <joint name="ltx" type="hinge" axis="1 0 0"/>
              <geom size="0.02 0.04" mass="0.1"/>
            </body>
          </body>
        </body>
      </body>
      <!-- Right leg chain with toes -->
      <body name="r_hip" pos="-0.1 0 -0.15">
        <joint name="rhx" type="hinge" axis="1 0 0"/><joint name="rhy" type="hinge" axis="0 1 0"/><joint name="rhz" type="hinge" axis="0 0 1"/>
        <geom size="0.04 0.15" mass="2"/><body name="r_knee" pos="0 0 -0.3">
          <joint name="rkx" type="hinge" axis="1 0 0"/><joint name="rky" type="hinge" axis="0 1 0"/><joint name="rkz" type="hinge" axis="0 0 1"/>
          <geom size="0.035 0.12" mass="1.5"/><body name="r_ankle" pos="0 0 -0.25">
            <joint name="rax" type="hinge" axis="1 0 0"/><joint name="ray" type="hinge" axis="0 1 0"/><joint name="raz" type="hinge" axis="0 0 1"/>
            <geom size="0.03 0.06" mass="0.5"/>
            <body name="r_toe" pos="0 0.06 -0.06">
              <joint name="rtx" type="hinge" axis="1 0 0"/>
              <geom size="0.02 0.04" mass="0.1"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="sp1x"/><motor joint="sp1y"/><motor joint="sp1z"/>
    <motor joint="sp2x"/><motor joint="sp2y"/><motor joint="sp2z"/>
    <motor joint="sp3x"/><motor joint="sp3y"/><motor joint="sp3z"/>
    <motor joint="nkx"/><motor joint="nky"/><motor joint="nkz"/>
    <motor joint="hdx"/><motor joint="hdy"/><motor joint="hdz"/>
    <motor joint="lsx"/><motor joint="lsy"/><motor joint="lsz"/>
    <motor joint="lex"/><motor joint="ley"/><motor joint="lez"/>
    <motor joint="lwx"/><motor joint="lwy"/><motor joint="lwz"/>
    <motor joint="lf1x"/><motor joint="lf1y"/>
    <motor joint="lf2x"/><motor joint="lf2y"/>
    <motor joint="rsx"/><motor joint="rsy"/><motor joint="rsz"/>
    <motor joint="rex"/><motor joint="rey"/><motor joint="rez"/>
    <motor joint="rwx"/><motor joint="rwy"/><motor joint="rwz"/>
    <motor joint="rf1x"/><motor joint="rf1y"/>
    <motor joint="rf2x"/><motor joint="rf2y"/>
    <motor joint="lhx"/><motor joint="lhy"/><motor joint="lhz"/>
    <motor joint="lkx"/><motor joint="lky"/><motor joint="lkz"/>
    <motor joint="lax"/><motor joint="lay"/><motor joint="laz"/>
    <motor joint="ltx"/>
    <motor joint="rhx"/><motor joint="rhy"/><motor joint="rhz"/>
    <motor joint="rkx"/><motor joint="rky"/><motor joint="rkz"/>
    <motor joint="rax"/><motor joint="ray"/><motor joint="raz"/>
    <motor joint="rtx"/>
  </actuator>
</mujoco>
)";

// Model with unsupported features for validation testing.
static const char* VALIDATION_MESH_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81"/>
  <asset>
    <mesh name="cube" vertex="0 0 0  1 0 0  0 1 0  0 0 1  1 1 0  1 0 1  0 1 1  1 1 1"/>
  </asset>
  <worldbody>
    <body name="b1" pos="0 0 1">
      <freejoint/>
      <geom type="mesh" mesh="cube" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

static const char* VALIDATION_TENDON_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="b1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05 0.2" mass="1"/>
      <body name="b2" pos="0 0 -0.4">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.05 0.2" mass="1"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="t1">
      <joint joint="j1" coef="1"/>
      <joint joint="j2" coef="-1"/>
    </fixed>
  </tendon>
</mujoco>
)";

static const char* VALIDATION_EQUALITY_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="b1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05 0.2" mass="1"/>
    </body>
    <body name="b2" pos="0.5 0 1">
      <joint name="j2" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05 0.2" mass="1"/>
    </body>
  </worldbody>
  <equality>
    <joint joint1="j1" joint2="j2"/>
  </equality>
</mujoco>
)";

// ============================================================================
// Phase 2: Contact Friction Models
// ============================================================================

// Sphere on plane with condim=3 (sliding friction enabled).
static const char* FRICTION_CONDIM3_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" condim="3" friction="0.8 0.005 0.0001"/>
    <body name="ball" pos="0 0 0.15">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1" condim="3" friction="0.8 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
)";

// Sphere on plane with condim=6 (full rolling + torsional friction).
static const char* FRICTION_CONDIM6_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" condim="6" friction="0.8 0.005 0.0001"/>
    <body name="ball" pos="0 0 0.15">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1" condim="6" friction="0.8 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
)";

// Many spheres on a plane for contact stress testing.
static const char* CONTACT_STRESS_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="s00" pos="-0.4 -0.4 0.5"><freejoint/><geom type="sphere" size="0.08" mass="0.5"/></body>
    <body name="s01" pos="-0.2 -0.4 0.5"><freejoint/><geom type="sphere" size="0.08" mass="0.5"/></body>
    <body name="s02" pos=" 0.0 -0.4 0.5"><freejoint/><geom type="sphere" size="0.08" mass="0.5"/></body>
    <body name="s03" pos=" 0.2 -0.4 0.5"><freejoint/><geom type="sphere" size="0.08" mass="0.5"/></body>
    <body name="s04" pos=" 0.4 -0.4 0.5"><freejoint/><geom type="sphere" size="0.08" mass="0.5"/></body>
    <body name="s10" pos="-0.4 -0.2 0.5"><freejoint/><geom type="sphere" size="0.08" mass="0.5"/></body>
    <body name="s11" pos="-0.2 -0.2 0.5"><freejoint/><geom type="sphere" size="0.08" mass="0.5"/></body>
    <body name="s12" pos=" 0.0 -0.2 0.5"><freejoint/><geom type="sphere" size="0.08" mass="0.5"/></body>
    <body name="s13" pos=" 0.2 -0.2 0.5"><freejoint/><geom type="sphere" size="0.08" mass="0.5"/></body>
    <body name="s14" pos=" 0.4 -0.2 0.5"><freejoint/><geom type="sphere" size="0.08" mass="0.5"/></body>
    <body name="s20" pos="-0.4  0.0 0.5"><freejoint/><geom type="sphere" size="0.08" mass="0.5"/></body>
    <body name="s21" pos="-0.2  0.0 0.5"><freejoint/><geom type="sphere" size="0.08" mass="0.5"/></body>
    <body name="s22" pos=" 0.0  0.0 0.5"><freejoint/><geom type="sphere" size="0.08" mass="0.5"/></body>
    <body name="s23" pos=" 0.2  0.0 0.5"><freejoint/><geom type="sphere" size="0.08" mass="0.5"/></body>
    <body name="s24" pos=" 0.4  0.0 0.5"><freejoint/><geom type="sphere" size="0.08" mass="0.5"/></body>
    <body name="s30" pos="-0.4  0.2 0.5"><freejoint/><geom type="sphere" size="0.08" mass="0.5"/></body>
    <body name="s31" pos="-0.2  0.2 0.5"><freejoint/><geom type="sphere" size="0.08" mass="0.5"/></body>
    <body name="s32" pos=" 0.0  0.2 0.5"><freejoint/><geom type="sphere" size="0.08" mass="0.5"/></body>
    <body name="s33" pos=" 0.2  0.2 0.5"><freejoint/><geom type="sphere" size="0.08" mass="0.5"/></body>
    <body name="s34" pos=" 0.4  0.2 0.5"><freejoint/><geom type="sphere" size="0.08" mass="0.5"/></body>
  </worldbody>
</mujoco>
)";

// ============================================================================
// Phase 3: Collision Expansion Models
// ============================================================================

// Box above plane + box-box collision pair.
static const char* BOX_COLLISION_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="box1" pos="0 0 0.5">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
    <body name="box2" pos="0.15 0 0.8">
      <freejoint/>
      <geom type="box" size="0.08 0.08 0.08" mass="0.5"/>
    </body>
  </worldbody>
</mujoco>
)";

// Cylinder above plane + sphere-cylinder.
static const char* CYLINDER_COLLISION_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="cyl" pos="0 0 0.5">
      <freejoint/>
      <geom type="cylinder" size="0.08 0.15" mass="1"/>
    </body>
    <body name="ball" pos="0.2 0 0.8">
      <freejoint/>
      <geom type="sphere" size="0.06" mass="0.3"/>
    </body>
  </worldbody>
</mujoco>
)";

// Simple convex mesh (cube vertices) for GJK/EPA collision testing.
static const char* MESH_COLLISION_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <asset>
    <mesh name="cube" vertex="
      -0.1 -0.1 -0.1
       0.1 -0.1 -0.1
       0.1  0.1 -0.1
      -0.1  0.1 -0.1
      -0.1 -0.1  0.1
       0.1 -0.1  0.1
       0.1  0.1  0.1
      -0.1  0.1  0.1"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="mesh_body" pos="0 0 0.5">
      <freejoint/>
      <geom type="mesh" mesh="cube" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

// Multi-geom scene: sphere, capsule, box, cylinder all interacting.
static const char* MULTI_GEOM_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="sphere" pos="0 0 0.5">
      <freejoint/><geom type="sphere" size="0.08" mass="0.5"/>
    </body>
    <body name="capsule" pos="0.3 0 0.5">
      <freejoint/><geom type="capsule" size="0.06 0.1" mass="0.5"/>
    </body>
    <body name="box" pos="-0.3 0 0.5">
      <freejoint/><geom type="box" size="0.08 0.08 0.08" mass="0.5"/>
    </body>
    <body name="cylinder" pos="0 0.3 0.5">
      <freejoint/><geom type="cylinder" size="0.06 0.1" mass="0.5"/>
    </body>
  </worldbody>
</mujoco>
)";

// Ellipsoid on plane (for Phase 3.5).
static const char* ELLIPSOID_COLLISION_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="ell" pos="0 0 0.5">
      <freejoint/>
      <geom type="ellipsoid" size="0.12 0.08 0.06" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

// ============================================================================
// Phase 4: Constraint Models
// ============================================================================

// Two bodies welded together.
static const char* WELD_CONSTRAINT_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="b1" pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
    <body name="b2" pos="0.3 0 1">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
  <equality>
    <weld body1="b1" body2="b2" relpose="0.3 0 0 1 0 0 0"/>
  </equality>
</mujoco>
)";

// Two bodies connected at a point.
static const char* CONNECT_CONSTRAINT_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="b1" pos="0 0 1">
      <freejoint/>
      <geom type="capsule" size="0.05 0.15" mass="1"/>
    </body>
    <body name="b2" pos="0 0 0.7">
      <freejoint/>
      <geom type="capsule" size="0.05 0.15" mass="1"/>
    </body>
  </worldbody>
  <equality>
    <connect body1="b1" body2="b2" anchor="0 0 -0.15"/>
  </equality>
</mujoco>
)";

// Joint equality constraint.
static const char* JOINT_EQUALITY_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="b1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05 0.2" mass="1"/>
    </body>
    <body name="b2" pos="0.5 0 1">
      <joint name="j2" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05 0.2" mass="1"/>
    </body>
  </worldbody>
  <equality>
    <joint joint1="j1" joint2="j2"/>
  </equality>
</mujoco>
)";

// Joint with friction loss.
static const char* DOF_FRICTION_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="b1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0" frictionloss="2.0"/>
      <geom type="capsule" size="0.05 0.2" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

// ============================================================================
// Phase 5: Tendon Models
// ============================================================================

// Fixed tendon: linear joint combination.
static const char* FIXED_TENDON_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="b1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05 0.2" mass="1"/>
      <body name="b2" pos="0 0 -0.4">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.05 0.2" mass="1"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="t1">
      <joint joint="j1" coef="1"/>
      <joint joint="j2" coef="-1"/>
    </fixed>
  </tendon>
  <actuator>
    <motor joint="j1" gear="50"/>
  </actuator>
</mujoco>
)";

// Spatial tendon wrapping around a cylinder geom.
static const char* SPATIAL_TENDON_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <site name="s0" pos="0 0 1"/>
    <body name="b1" pos="0 0 0.8">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom name="wrap_cyl" type="cylinder" size="0.05 0.1" pos="0 0 0" euler="90 0 0"/>
      <site name="s1" pos="0 0 -0.2"/>
    </body>
  </worldbody>
  <tendon>
    <spatial name="sp1">
      <site site="s0"/>
      <geom geom="wrap_cyl"/>
      <site site="s1"/>
    </spatial>
  </tendon>
  <actuator>
    <motor joint="j1" gear="50"/>
  </actuator>
</mujoco>
)";

// Site transmission actuator.
static const char* SITE_TRANSMISSION_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="b1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05 0.2" mass="1"/>
      <site name="act_site" pos="0 0 -0.1"/>
    </body>
  </worldbody>
  <actuator>
    <motor site="act_site" gear="0 0 50 0 0 0"/>
  </actuator>
</mujoco>
)";

// ============================================================================
// Phase 6: Actuator Dynamics Models
// ============================================================================

// Filter actuator dynamics.
static const char* FILTER_ACTUATOR_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="b1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05 0.2" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <general joint="j1" gainprm="50" dyntype="filter" dynprm="0.02"/>
  </actuator>
</mujoco>
)";

// Integrator actuator dynamics.
static const char* INTEGRATOR_ACTUATOR_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="b1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05 0.2" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <general joint="j1" gainprm="50" dyntype="integrator"/>
  </actuator>
</mujoco>
)";

// ============================================================================
// Phase 7: Integrator Models
// ============================================================================

// Pendulum with RK4 integrator.
static const char* RK4_PENDULUM_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002" integrator="RK4"/>
  <worldbody>
    <body name="pend" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05 0.3" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

// Stiff spring system with implicitfast integrator.
static const char* IMPLICIT_STIFF_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.005" integrator="implicitfast"/>
  <worldbody>
    <body name="b1" pos="0 0 1">
      <joint name="j1" type="slide" axis="0 0 1" stiffness="10000" damping="10"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

// ============================================================================
// Existing baseline: simple pendulum (for quick sanity checks)
// ============================================================================

static const char* SIMPLE_PENDULUM_XML = R"(
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="pend" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05 0.3" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j1" gear="50"/>
  </actuator>
</mujoco>
)";
