// Copyright 2026 Arghya Sur
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "internal.h"

#include <stdexcept>

namespace mjmlx {

// TODO: Phase 1b - port from Python mjmlx._src.math
mx::array quat_mul(const mx::array& q1, const mx::array& q2) {
  (void)q1;
  (void)q2;
  throw std::runtime_error("not implemented");
}

// TODO: Phase 1b - port from Python mjmlx._src.math
mx::array quat_to_mat(const mx::array& q) {
  (void)q;
  throw std::runtime_error("not implemented");
}

// TODO: Phase 1b - port from Python mjmlx._src.math
mx::array rotate(const mx::array& vec, const mx::array& quat) {
  (void)vec;
  (void)quat;
  throw std::runtime_error("not implemented");
}

// TODO: Phase 1b - port from Python mjmlx._src.math
mx::array normalize(const mx::array& x) {
  (void)x;
  throw std::runtime_error("not implemented");
}

}  // namespace mjmlx
