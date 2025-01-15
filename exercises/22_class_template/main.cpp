#include <iostream>
#include <cstring> // 用于 std::memcpy

template<class T>
struct Tensor4D {
    unsigned int shape[4];  // 张量的形状
    T *data;                // 张量的数据

    // 构造函数：初始化张量的形状和数据
    Tensor4D(unsigned int const shape_[4], T const *data_) {
        unsigned int size = 1;
        // TODO: 填入正确的 shape 并计算 size
        for (int i = 0; i < 4; ++i) {
            shape[i] = shape_[i];
            size *= shape[i];  // 计算张量的总元素数量
        }
        data = new T[size];
        std::memcpy(data, data_, size * sizeof(T));  // 复制数据
    }

    // 析构函数：释放内存
    ~Tensor4D() {
        delete[] data;
    }

    // 为了保持简单，禁止复制和移动
    Tensor4D(Tensor4D const &) = delete;
    Tensor4D(Tensor4D &&) noexcept = delete;

    // 这个加法需要支持“单向广播”。
    // 具体来说，`others` 可以具有与 `this` 不同的形状，形状不同的维度长度必须为 1。
    // `others` 长度为 1 但 `this` 长度不为 1 的维度将发生广播计算。
    // 例如，`this` 形状为 `[1, 2, 3, 4]`，`others` 形状为 `[1, 2, 1, 4]`，
    // 则 `this` 与 `others` 相加时，3 个形状为 `[1, 2, 1, 4]` 的子张量各自与 `others` 对应项相加。
    Tensor4D &operator+=(Tensor4D const &others) {
        // TODO: 实现单向广播的加法
        // 确保两个张量的形状兼容
        for (int i = 0; i < 4; ++i) {
            if (shape[i] != others.shape[i] && others.shape[i] != 1) {
                throw std::invalid_argument("Incompatible shapes for broadcasting.");
            }
        }

        // 执行加法运算并支持广播
        unsigned int size = 1;
        for (int i = 0; i < 4; ++i) {
            size *= shape[i];  // 计算张量总元素数
        }

        for (unsigned int i = 0; i < size; ++i) {
            unsigned int indices[4];
            unsigned int temp = i;

            // 计算各维度的下标
            for (int j = 3; j >= 0; --j) {
                indices[j] = temp % shape[j];
                temp /= shape[j];
            }

            // 检查是否需要广播并执行加法
            unsigned int other_index = 0;
            for (int j = 0; j < 4; ++j) {
                if (others.shape[j] == 1) {
                    other_index *= shape[j];
                } else {
                    other_index = other_index * shape[j] + indices[j];
                }
            }

            data[i] += others.data[other_index];  // 执行加法
        }

        return *this;
    }
};

// ---- 不要修改以下代码 ----
#define ASSERT(cond, msg) if (!(cond)) std::cerr << msg << std::endl;

int main(int argc, char **argv) {
    {
        unsigned int shape[]{1, 2, 3, 4};
        // clang-format off
        int data[]{
            1,  2,  3,  4,
            5,  6,  7,  8,
            9, 10, 11, 12,

            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24};
        // clang-format on
        auto t0 = Tensor4D<int>(shape, data);
        auto t1 = Tensor4D<int>(shape, data);
        t0 += t1;
        for (auto i = 0u; i < sizeof(data) / sizeof(*data); ++i) {
            ASSERT(t0.data[i] == data[i] * 2, "Tensor doubled by plus its self.");
        }
    }
    {
        unsigned int s0[]{1, 2, 3, 4};
        // clang-format off
        float d0[]{
            1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,

            4, 4, 4, 4,
            5, 5, 5, 5,
            6, 6, 6, 6};
        // clang-format on
        unsigned int s1[]{1, 2, 3, 1};
        // clang-format off
        float d1[]{
            6,
            5,
            4,

            3,
            2,
            1};
        // clang-format on

        auto t0 = Tensor4D<float>(s0, d0);
        auto t1 = Tensor4D<float>(s1, d1);
        t0 += t1;

        // 验证输出
        for (unsigned int i = 0; i < sizeof(d0) / sizeof(d0[0]); ++i) {
            ASSERT(t0.data[i] == d0[i] + d1[i % 3], "Tensor addition with broadcasting.");
        }
    }
    return 0;
}