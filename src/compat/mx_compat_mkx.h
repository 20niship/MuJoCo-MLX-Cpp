// namespace mx backed by mkx (Vulkan); N in mkx::array<T,N> is decorative (never read at runtime), so every proxy uses a fixed dummy rank; mkx shaders are float32-only regardless of logical dtype, so this wrapper tracks Dtype itself and converts only at host readback/astype.
#pragma once

#include <mkx/core/array.hpp>
#include <mkx/core/eval.hpp>
#include <mkx/core/op_node.hpp>
#include <mkx/core/types.hpp>
#include <mkx/ops/creation.hpp>
#include <mkx/ops/elementwise.hpp>
#include <mkx/ops/linalg.hpp>
#include <mkx/ops/random.hpp>
#include <mkx/ops/shape.hpp>
#include <mkx/ops/transforms.hpp>
#include <mkx/vulkan/vulkan_backend.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace mx {

using Backend = mkx::VulkanBackend;
template <class T> using Raw = mkx::array<T, 1>;

enum class Dtype { Float32, Int32, Bool };
inline constexpr Dtype float32 = Dtype::Float32;
inline constexpr Dtype int32 = Dtype::Int32;
inline constexpr Dtype bool_ = Dtype::Bool;

using Shape = std::vector<int>;

class Device {
public:
    static const Device cpu;
    static const Device gpu;
};
inline const Device Device::cpu = Device();
inline const Device Device::gpu = Device();

namespace detail {

inline mkx::Shape to_mkx_shape(const Shape& s) { return mkx::Shape(s.begin(), s.end()); }
inline Shape from_mkx_shape(const mkx::Shape& s) { return Shape(s.begin(), s.end()); }

// numpy-style right-aligned broadcast; mkx::broadcast_to already left-pads with 1s internally.
inline Shape broadcast_shapes(const Shape& a, const Shape& b) {
    size_t n = std::max(a.size(), b.size());
    Shape out(n);
    for (size_t i = 0; i < n; i++) {
        int ai = (i < n - a.size()) ? 1 : a[i - (n - a.size())];
        int bi = (i < n - b.size()) ? 1 : b[i - (n - b.size())];
        if (ai != 1 && bi != 1 && ai != bi) {
            std::string msg = "mx_compat_mkx: incompatible shapes for broadcast: [";
            for (size_t k = 0; k < a.size(); k++) msg += std::to_string(a[k]) + ",";
            msg += "] vs [";
            for (size_t k = 0; k < b.size(); k++) msg += std::to_string(b[k]) + ",";
            msg += "]";
            throw std::runtime_error(msg);
        }
        out[i] = std::max(ai, bi);
    }
    return out;
}

} // namespace detail

class array {
public:
    array() : node_(alloc_from_host(nullptr, mkx::Shape{0})), dtype_(Dtype::Float32) {}

    explicit array(float scalar) : node_(alloc_from_host(&scalar, mkx::Shape{}, 1)), dtype_(Dtype::Float32) {}
    explicit array(double scalar) : array(static_cast<float>(scalar)) {}
    explicit array(int scalar) : node_(alloc_from_host_val(static_cast<float>(scalar))), dtype_(Dtype::Int32) {}
    explicit array(bool scalar) : node_(alloc_from_host_val(scalar ? 1.0f : 0.0f)), dtype_(Dtype::Bool) {}

    template <class T = float>
    array(std::initializer_list<T> vals, Dtype dtype = default_dtype<T>()) : dtype_(dtype) {
        std::vector<float> host(vals.size());
        size_t i = 0;
        for (auto v : vals) host[i++] = static_cast<float>(v);
        node_ = alloc_from_host(host.data(), mkx::Shape{static_cast<int64_t>(vals.size())}, vals.size());
    }

    template <class T>
    array(const T* data, Shape shape, Dtype dtype = Dtype::Float32) : dtype_(dtype) {
        mkx::Shape ms = detail::to_mkx_shape(shape);
        size_t n = static_cast<size_t>(mkx::shape_size(ms));
        std::vector<float> host(n);
        for (size_t i = 0; i < n; i++) host[i] = static_cast<float>(data[i]);
        node_ = alloc_from_host(host.data(), ms, n);
    }

    explicit array(mkx::NodePtr node, Dtype dtype = Dtype::Float32) : node_(std::move(node)), dtype_(dtype) {}

    Shape shape() const { return detail::from_mkx_shape(node_->shape); }
    int shape(int axis) const {
        int n = static_cast<int>(node_->shape.size());
        return static_cast<int>(node_->shape[static_cast<size_t>(axis < 0 ? axis + n : axis)]);
    }
    Dtype dtype() const { return dtype_; }
    size_t ndim() const { return node_->shape.size(); }
    int64_t size() const { return mkx::shape_size(node_->shape); }

    mkx::NodePtr& node() { return node_; }
    const mkx::NodePtr& node() const { return node_; }

    template <class T> const T* data() const {
        ensure_readback();
        std::vector<T> out(cache_f_.size());
        for (size_t i = 0; i < cache_f_.size(); i++) out[i] = static_cast<T>(cache_f_[i]);
        cache_bytes_.resize(out.size() * sizeof(T));
        std::memcpy(cache_bytes_.data(), out.data(), cache_bytes_.size());
        return reinterpret_cast<const T*>(cache_bytes_.data());
    }

    template <class T> T item() const { return data<T>()[0]; }

private:
    template <class T> static Dtype default_dtype() {
        if constexpr (std::is_same_v<T, bool>) return Dtype::Bool;
        else if constexpr (std::is_integral_v<T>) return Dtype::Int32;
        else return Dtype::Float32;
    }

    static mkx::NodePtr alloc_from_host_val(float v) { return alloc_from_host(&v, mkx::Shape{}, 1); }

    static mkx::NodePtr alloc_from_host(const float* data, mkx::Shape shape, size_t n = 0) {
        auto node = mkx::make_node(mkx::OpType::Const, shape, mkx::Dtype::Float32);
        size_t count = static_cast<size_t>(mkx::shape_size(shape));
        size_t alloc_count = std::max<size_t>(count, 1); // Vulkan/MoltenVK rejects a 0-byte vkAllocateMemory
        auto* buf = Backend::alloc(alloc_count * sizeof(float));
        std::vector<float> zeros;
        if (data == nullptr) { zeros.assign(alloc_count, 0.0f); data = zeros.data(); }
        Backend::upload(buf, data, count * sizeof(float));
        node->gpu_buffer = buf;
        node->evaluated = true;
        // eval_node() (mlx_vulkan) sets free_gpu_buffer for its own allocations, but this constructor bypasses it entirely -- without this every host-constructed mx::array leaked its GPU buffer.
        auto* raw_buf = buf;
        node->free_gpu_buffer = [raw_buf]() { Backend::free(raw_buf); };
        return node;
    }

    void ensure_readback() const {
        if (cached_) return;
        Raw<float> proxy(node_);
        mkx::eval<Backend>(proxy);
        cache_f_ = proxy.template to_vector<Backend>();
        cached_ = true;
    }

    mkx::NodePtr node_;
    Dtype dtype_;
    mutable bool cached_ = false;
    mutable std::vector<float> cache_f_;
    mutable std::vector<uint8_t> cache_bytes_;
};

inline Raw<float> to_raw(const array& a) { return Raw<float>(a.node()); }
inline array from_raw(mkx::NodePtr n, Dtype dt) { return array(std::move(n), dt); }

namespace fast {
// Bridges mx::array/mx::Shape to the raw mkx::fast::Kernel::operator() call shape for the vendored kernels in src/compat/mkx_kernels/.
inline std::vector<Raw<float>> kernel_inputs(const std::vector<array>& arrs) {
    std::vector<Raw<float>> out;
    out.reserve(arrs.size());
    for (auto& a : arrs) out.push_back(to_raw(a));
    return out;
}
inline std::vector<mkx::Shape> kernel_shapes(const std::vector<Shape>& shapes) {
    std::vector<mkx::Shape> out;
    out.reserve(shapes.size());
    for (auto& s : shapes) out.push_back(detail::to_mkx_shape(s));
    return out;
}
inline std::vector<array> kernel_outputs(const std::vector<Raw<float>>& outs) {
    std::vector<array> result;
    result.reserve(outs.size());
    for (const auto& o : outs) result.push_back(array(o.node(), Dtype::Float32));
    return result;
}
} // namespace fast

// forward decl, defined below
inline array broadcast_to(const array& a, Shape target_shape);

namespace detail {

inline std::pair<array, array> broadcast_pair(const array& a, const array& b) {
    Shape sa = a.shape(), sb = b.shape();
    if (sa == sb) return {a, b};
    Shape bc = broadcast_shapes(sa, sb);
    array a2 = (sa == bc) ? a : broadcast_to(a, bc);
    array b2 = (sb == bc) ? b : broadcast_to(b, bc);
    return {a2, b2};
}

} // namespace detail

#define MX_BINARY(name, dtype_expr) \
    inline array name(const array& a, const array& b) { \
        auto [pa, pb] = detail::broadcast_pair(a, b); \
        return array(mkx::name(to_raw(pa), to_raw(pb)).node(), dtype_expr); \
    }

MX_BINARY(add, a.dtype())
MX_BINARY(subtract, a.dtype())
MX_BINARY(multiply, a.dtype())
MX_BINARY(divide, a.dtype())
MX_BINARY(power, a.dtype())
MX_BINARY(maximum, a.dtype())
MX_BINARY(minimum, a.dtype())
MX_BINARY(equal, Dtype::Bool)
MX_BINARY(greater, Dtype::Bool)
MX_BINARY(greater_equal, Dtype::Bool)
MX_BINARY(less, Dtype::Bool)
MX_BINARY(less_equal, Dtype::Bool)
MX_BINARY(logical_and, Dtype::Bool)
MX_BINARY(logical_or, Dtype::Bool)
#undef MX_BINARY

#define MX_UNARY(name) \
    inline array name(const array& a) { return array(mkx::name(to_raw(a)).node(), a.dtype()); }

MX_UNARY(negative)
MX_UNARY(abs)
MX_UNARY(sqrt)
MX_UNARY(square)
MX_UNARY(sign)
MX_UNARY(floor)
MX_UNARY(sin)
MX_UNARY(cos)
#undef MX_UNARY

inline array logical_not(const array& a) { return array(mkx::logical_not(to_raw(a)).node(), Dtype::Bool); }

inline array where(const array& cond, const array& x, const array& y) {
    Shape bc = detail::broadcast_shapes(detail::broadcast_shapes(cond.shape(), x.shape()), y.shape());
    array c2 = (cond.shape() == bc) ? cond : broadcast_to(cond, bc);
    array x2 = (x.shape() == bc) ? x : broadcast_to(x, bc);
    array y2 = (y.shape() == bc) ? y : broadcast_to(y, bc);
    return array(mkx::where(to_raw(c2), to_raw(x2), to_raw(y2)).node(), x.dtype());
}

inline array clip(const array& x, const array& lo, const array& hi) {
    Shape bc = detail::broadcast_shapes(detail::broadcast_shapes(x.shape(), lo.shape()), hi.shape());
    array x2 = (x.shape() == bc) ? x : broadcast_to(x, bc);
    array l2 = (lo.shape() == bc) ? lo : broadcast_to(lo, bc);
    array h2 = (hi.shape() == bc) ? hi : broadcast_to(hi, bc);
    return array(mkx::clip(to_raw(x2), to_raw(l2), to_raw(h2)).node(), x.dtype());
}

inline array reshape(const array& a, Shape new_shape) {
    return array(mkx::reshape<float, 1, 1>(to_raw(a), detail::to_mkx_shape(new_shape)).node(), a.dtype());
}

inline array flatten(const array& a) { return array(mkx::flatten(to_raw(a)).node(), a.dtype()); }

inline array transpose(const array& a, std::vector<int> perm) {
    return array(mkx::transpose(to_raw(a), std::move(perm)).node(), a.dtype());
}
inline array transpose(const array& a) {
    std::vector<int> perm(a.ndim());
    for (size_t i = 0; i < perm.size(); i++) perm[i] = static_cast<int>(perm.size() - 1 - i);
    return transpose(a, perm);
}

inline array broadcast_to(const array& a, Shape target_shape) {
    return array(mkx::broadcast_to(to_raw(a), detail::to_mkx_shape(target_shape)).node(), a.dtype());
}

inline array tile(const array& a, Shape reps) {
    std::vector<int64_t> mreps(reps.begin(), reps.end());
    return array(mkx::tile(to_raw(a), std::move(mreps)).node(), a.dtype());
}

inline array slice(const array& a, Shape starts, Shape stops) {
    std::vector<int64_t> mstarts(starts.begin(), starts.end());
    std::vector<int64_t> mstops(stops.begin(), stops.end());
    return array(mkx::slice(to_raw(a), std::move(mstarts), std::move(mstops)).node(), a.dtype());
}

inline array concatenate(const std::vector<array>& parts, int axis = 0) {
    array acc = parts[0];
    for (size_t i = 1; i < parts.size(); i++) acc = array(mkx::concatenate(to_raw(acc), to_raw(parts[i]), axis).node(), acc.dtype());
    return acc;
}

inline array stack(const std::vector<array>& parts, int axis = 0) {
    auto insert_dim = [](Shape s, int ax) { s.insert(s.begin() + ax, 1); return s; };
    array acc = reshape(parts[0], insert_dim(parts[0].shape(), axis));
    for (size_t i = 1; i < parts.size(); i++) {
        array next = reshape(parts[i], insert_dim(parts[i].shape(), axis));
        acc = concatenate({acc, next}, axis);
    }
    return acc;
}

inline array take(const array& data, const array& indices, int axis) {
    return array(mkx::take(to_raw(data), Raw<float>(indices.node()), axis).node(), data.dtype());
}
inline array take(const array& data, const array& indices) { return take(flatten(data), indices, 0); }

inline array diag(const array& v) { return array(mkx::diag<float>(Raw<float>(v.node())).node(), v.dtype()); }
inline array copy(const array& a) { return array(mkx::copy(to_raw(a)).node(), a.dtype()); }

inline int normalize_axis(int axis, size_t ndim) { return axis < 0 ? axis + static_cast<int>(ndim) : axis; }

inline array sum(const array& a) { return array(mkx::sum(to_raw(a)).node(), a.dtype()); }
inline array sum(const array& a, int axis) {
    return array(mkx::sum_axis(to_raw(a), normalize_axis(axis, a.ndim())).node(), a.dtype());
}
inline array sum(const array& a, std::vector<int> axes) { return sum(a, axes[0]); }
inline array sum(const array& a, int axis, bool keepdims) {
    int ax = normalize_axis(axis, a.ndim());
    array r = sum(a, ax);
    if (!keepdims) return r;
    Shape s = r.shape();
    s.insert(s.begin() + ax, 1);
    return reshape(r, s);
}

inline array reduce_max(const array& a) { return array(mkx::reduce_max(to_raw(a)).node(), a.dtype()); }
inline array max(const array& a) { return reduce_max(a); }
inline array argmax(const array& a) { return array(mkx::argmax(to_raw(a)).node(), Dtype::Int32); }
inline array argmin(const array& a) { return array(mkx::argmin(to_raw(a)).node(), Dtype::Int32); }

inline array matmul(const array& a, const array& b) {
    return array(mkx::matmul(mkx::array<float, 2>(a.node()), mkx::array<float, 2>(b.node())).node(), a.dtype());
}

inline array zeros(Shape shape, Dtype dtype = Dtype::Float32) {
    (void)dtype;
    size_t n = static_cast<size_t>(mkx::shape_size(detail::to_mkx_shape(shape)));
    std::vector<float> host(n, 0.0f);
    return array(host.data(), std::move(shape), dtype);
}
inline array ones(Shape shape, Dtype dtype = Dtype::Float32) {
    size_t n = static_cast<size_t>(mkx::shape_size(detail::to_mkx_shape(shape)));
    std::vector<float> host(n, 1.0f);
    return array(host.data(), std::move(shape), dtype);
}
inline array zeros_like(const array& a) { return zeros(a.shape(), a.dtype()); }

inline array eye(int64_t n, Dtype dtype = Dtype::Float32) {
    std::vector<float> host(static_cast<size_t>(n * n), 0.0f);
    for (int64_t i = 0; i < n; i++) host[static_cast<size_t>(i * n + i)] = 1.0f;
    return array(host.data(), Shape{static_cast<int>(n), static_cast<int>(n)}, dtype);
}

inline array arange(float start, float stop, float step = 1.0f) {
    std::vector<float> host;
    for (float v = start; v < stop; v += step) host.push_back(v);
    return array(host.data(), Shape{static_cast<int>(host.size())}, Dtype::Float32);
}

// mkx's native tril/triu have no diagonal offset; build the offset mask from row/col index arrays instead.
inline array row_col_diff(int n0, int n1) {
    array ridx = reshape(arange(0.0f, static_cast<float>(n0), 1.0f), Shape{n0, 1});
    array cidx = reshape(arange(0.0f, static_cast<float>(n1), 1.0f), Shape{1, n1});
    return subtract(broadcast_to(cidx, Shape{n0, n1}), broadcast_to(ridx, Shape{n0, n1}));
}
inline array tril(const array& a, int k = 0) {
    array mask = less_equal(row_col_diff(static_cast<int>(a.shape()[0]), static_cast<int>(a.shape()[1])), array(static_cast<float>(k)));
    return array(multiply(a, mask).node(), a.dtype());
}
inline array triu(const array& a, int k = 0) {
    array mask = greater_equal(row_col_diff(static_cast<int>(a.shape()[0]), static_cast<int>(a.shape()[1])), array(static_cast<float>(k)));
    return array(multiply(a, mask).node(), a.dtype());
}

// astype truncation/threshold is composed from existing float ops (shaders are float32-only regardless of dtype), so this stays on GPU.
inline array astype(const array& a, Dtype dt) {
    if (a.dtype() == dt) return a;
    if (dt == Dtype::Bool) {
        array z = zeros_like(a);
        array ne = logical_or(greater(a, z), less(a, z));
        return array(ne.node(), Dtype::Bool);
    }
    if (dt == Dtype::Int32) {
        array neg = less(a, array(0.0f));
        array trunc_pos = floor(a);
        array trunc_neg = negative(floor(negative(a)));
        return array(where(neg, trunc_neg, trunc_pos).node(), Dtype::Int32);
    }
    return array(a.node(), Dtype::Float32);
}

inline array arange(float start, float stop, float step, Dtype dtype) { return astype(arange(start, stop, step), dtype); }

// mkx::eval<Backend, Arrays...> shares one topo_sort + one Backend::wait_idle() across the whole pack; calling it per-array (as this shim used to) turned every mx::eval(a,b,c) into N GPU syncs instead of MLX's one.
template <class... Arrays> void eval(Arrays&... arrs) {
    auto proxies = std::make_tuple(Raw<float>(arrs.node())...);
    std::apply([](auto&... ps) { mkx::eval<Backend>(ps...); }, proxies);
}
inline void eval(std::initializer_list<array> arrs) {
    std::vector<Raw<float>> proxies;
    proxies.reserve(arrs.size());
    for (const auto& a : arrs) proxies.emplace_back(a.node());

    std::unordered_set<mkx::OpNode*> visited;
    std::vector<mkx::NodePtr> order;
    for (auto& p : proxies) mkx::detail::topo_sort(p.node(), visited, order);

    static std::unordered_map<size_t, Backend::Pipeline> pipeline_cache;
    for (auto& node : order) mkx::detail::eval_node<Backend>(*node, pipeline_cache);
    Backend::wait_idle();
}

template <class Fn> auto compile(Fn fn) { return fn; }

namespace linalg {

inline array cholesky(const array& a, bool upper, Device device) {
    (void)upper;
    (void)device;
    return array(mkx::cholesky<float>(mkx::array<float, 2>(a.node())).node(), a.dtype());
}

// mkx::solve_triangular only does forward (lower) substitution; upper=true is done eagerly on host since mkx has no GPU path for it either way.
inline array solve_triangular(const array& a, const array& b, bool upper, Device device) {
    (void)device;
    if (!upper) {
        return array(mkx::solve_triangular<float>(mkx::array<float, 2>(a.node()), Raw<float>(b.node())).node(), b.dtype());
    }
    int n = static_cast<int>(a.shape().back());
    const float* U = a.data<float>();
    const float* rhs = b.data<float>();
    std::vector<float> x(static_cast<size_t>(n));
    for (int i = n - 1; i >= 0; i--) {
        float s = rhs[i];
        for (int j = i + 1; j < n; j++) s -= U[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)] * x[static_cast<size_t>(j)];
        x[static_cast<size_t>(i)] = s / U[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(i)];
    }
    return array(x.data(), Shape{n}, Dtype::Float32);
}

inline array cross(const array& a, const array& b) {
    return array(mkx::cross(to_raw(a), to_raw(b)).node(), a.dtype());
}

} // namespace linalg

namespace random {
inline array normal(Shape shape) {
    static uint32_t seed = 1u;
    auto k = mkx::random::key(seed++);
    return array(mkx::random::normal<float, 1>(k, detail::to_mkx_shape(shape)).node(), Dtype::Float32);
}
} // namespace random

} // namespace mx
