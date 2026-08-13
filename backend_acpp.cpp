#include "backend.h"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

/*
 * AdaptiveCpp/SYCL compute backend.
 *
 * GPT-2 weights do not change during inference, so each matrix and bias is
 * copied to device memory once and cached by its host address. The input and
 * output vectors are much smaller; they use reusable scratch allocations.
 *
 * This is not meant to be the final high-performance design. It is a clear
 * first GPU path that avoids the expensive mistake of uploading an entire
 * weight matrix for every token.
 */

namespace {

struct DeviceBuffer {
    float *ptr = nullptr;
    size_t count = 0;
};

std::string lower_copy(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return text;
}

bool parse_index(const std::string &text, size_t &index) {
    if (text.empty()) {
        return false;
    }

    char *end = nullptr;
    const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
    if (!end || *end != '\0') {
        return false;
    }

    index = static_cast<size_t>(value);
    return true;
}

std::vector<sycl::device> all_devices() {
    return sycl::device::get_devices();
}

sycl::device choose_device(const char *device_spec) {
    const std::string spec = lower_copy(
        device_spec && *device_spec ? device_spec : "auto");

    const auto devices = all_devices();
    if (devices.empty()) {
        throw std::runtime_error("no SYCL devices are available");
    }

    size_t index = 0;
    if (parse_index(spec, index)) {
        if (index >= devices.size()) {
            throw std::runtime_error("device index is out of range");
        }
        return devices[index];
    }

    if (spec == "auto") {
        for (const auto &device : devices) {
            if (device.is_gpu()) {
                return device;
            }
        }
        return devices.front();
    }

    if (spec == "gpu") {
        for (const auto &device : devices) {
            if (device.is_gpu()) {
                return device;
            }
        }
        throw std::runtime_error("no SYCL GPU is available");
    }

    if (spec == "cpu") {
        for (const auto &device : devices) {
            if (device.is_cpu()) {
                return device;
            }
        }
        throw std::runtime_error("no SYCL CPU device is available");
    }

    for (const auto &device : devices) {
        const std::string name = lower_copy(
            device.get_info<sycl::info::device::name>());
        if (name.find(spec) != std::string::npos) {
            return device;
        }
    }

    throw std::runtime_error("no SYCL device matches the requested name");
}

size_t choose_work_group_size(size_t max_work_group_size) {
    const size_t limit = std::min<size_t>(max_work_group_size, 256);
    size_t size = 1;

    while ((size << 1) <= limit) {
        size <<= 1;
    }

    return size;
}

}  // namespace

struct ChrisBackend {
    sycl::device device;
    sycl::queue queue;
    std::string device_name;

    /* Host address -> persistent USM device allocation. */
    std::unordered_map<const float *, DeviceBuffer> readonly_buffers;

    float *scratch_x = nullptr;
    float *scratch_y = nullptr;
    size_t scratch_x_count = 0;
    size_t scratch_y_count = 0;
    size_t work_group_size = 64;

    explicit ChrisBackend(const sycl::device &selected_device)
        : device(selected_device),
          queue(selected_device, sycl::property::queue::in_order{}),
          device_name(selected_device.get_info<sycl::info::device::name>()) {
        work_group_size = choose_work_group_size(
            selected_device.get_info<sycl::info::device::max_work_group_size>());
    }

    ~ChrisBackend() {
        try {
            queue.wait_and_throw();
        } catch (...) {
            /* Destructors must not throw. Any runtime error was reported earlier. */
        }

        for (auto &entry : readonly_buffers) {
            if (entry.second.ptr) {
                sycl::free(entry.second.ptr, queue);
            }
        }

        if (scratch_x) {
            sycl::free(scratch_x, queue);
        }
        if (scratch_y) {
            sycl::free(scratch_y, queue);
        }
    }

    float *upload_readonly(const float *host_data, size_t count) {
        if (!host_data || !count) {
            return nullptr;
        }

        const auto existing = readonly_buffers.find(host_data);
        if (existing != readonly_buffers.end()) {
            if (existing->second.count != count) {
                throw std::runtime_error(
                    "the same host pointer was registered with a different size");
            }
            return existing->second.ptr;
        }

        float *device_data = sycl::malloc_device<float>(count, queue);
        if (!device_data) {
            throw std::bad_alloc();
        }

        queue.memcpy(device_data, host_data, count * sizeof(float)).wait();
        readonly_buffers.emplace(host_data, DeviceBuffer{device_data, count});
        return device_data;
    }

    void ensure_scratch(size_t x_count, size_t y_count) {
        if (x_count > scratch_x_count) {
            if (scratch_x) {
                sycl::free(scratch_x, queue);
            }

            scratch_x = sycl::malloc_device<float>(x_count, queue);
            if (!scratch_x) {
                throw std::bad_alloc();
            }
            scratch_x_count = x_count;
        }

        if (y_count > scratch_y_count) {
            if (scratch_y) {
                sycl::free(scratch_y, queue);
            }

            scratch_y = sycl::malloc_device<float>(y_count, queue);
            if (!scratch_y) {
                throw std::bad_alloc();
            }
            scratch_y_count = y_count;
        }
    }
};

extern "C" ChrisBackend *chris_backend_create(const char *device_spec) {
    try {
        return new ChrisBackend(choose_device(device_spec));
    } catch (const std::exception &error) {
        std::fprintf(stderr, "error: AdaptiveCpp backend: %s\n", error.what());
        return nullptr;
    }
}

extern "C" void chris_backend_destroy(ChrisBackend *backend) {
    delete backend;
}

extern "C" void chris_backend_list_devices(void) {
    try {
        const auto devices = all_devices();

        for (size_t i = 0; i < devices.size(); ++i) {
            const auto &device = devices[i];
            const char *kind = device.is_gpu()
                                   ? "GPU"
                                   : (device.is_cpu() ? "CPU" : "other");
            const std::string name =
                device.get_info<sycl::info::device::name>();
            const std::string vendor =
                device.get_info<sycl::info::device::vendor>();
            const auto global_memory =
                device.get_info<sycl::info::device::global_mem_size>();

            std::printf(
                "[%zu] %s | %s | %s | %.2f GiB\n",
                i,
                kind,
                vendor.c_str(),
                name.c_str(),
                static_cast<double>(global_memory) /
                    (1024.0 * 1024.0 * 1024.0));
        }
    } catch (const std::exception &error) {
        std::fprintf(stderr,
                     "error: failed to enumerate SYCL devices: %s\n",
                     error.what());
    }
}

extern "C" const char *chris_backend_name(const ChrisBackend *backend) {
    (void)backend;
    return "adaptivecpp";
}

extern "C" const char *chris_backend_device_name(const ChrisBackend *backend) {
    return backend ? backend->device_name.c_str() : "<none>";
}

extern "C" int chris_backend_is_accelerated(const ChrisBackend *backend) {
    return backend && backend->device.is_gpu();
}

extern "C" int chris_backend_preload(ChrisBackend *backend,
                                      const float *host_data,
                                      size_t element_count) {
    if (!backend || !host_data || !element_count) {
        return -1;
    }

    try {
        backend->upload_readonly(host_data, element_count);
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr,
                     "error: AdaptiveCpp preload failed: %s\n",
                     error.what());
        return -1;
    }
}

extern "C" int chris_backend_matvec(ChrisBackend *backend,
                                     float *out,
                                     const float *weights,
                                     const float *x,
                                     const float *bias,
                                     uint32_t in_features,
                                     uint32_t out_features) {
    if (!backend || !out || !weights || !x || !in_features || !out_features) {
        return -1;
    }

    try {
        const size_t input_size = in_features;
        const size_t output_size = out_features;

        if (output_size > std::numeric_limits<size_t>::max() / input_size) {
            throw std::runtime_error("matvec shape exceeds SIZE_MAX");
        }

        const float *device_weights =
            backend->upload_readonly(weights, input_size * output_size);
        const float *device_bias =
            bias ? backend->upload_readonly(bias, output_size) : nullptr;

        backend->ensure_scratch(input_size, output_size);

        auto &queue = backend->queue;
        queue.memcpy(backend->scratch_x, x, input_size * sizeof(float));

        const size_t work_group_size = backend->work_group_size;
        float *device_x = backend->scratch_x;
        float *device_y = backend->scratch_y;

        queue.submit([&](sycl::handler &handler) {
            sycl::local_accessor<float, 1> partial(
                sycl::range<1>(work_group_size), handler);

            handler.parallel_for(
                sycl::nd_range<1>(
                    sycl::range<1>(output_size * work_group_size),
                    sycl::range<1>(work_group_size)),
                [=](sycl::nd_item<1> item) {
                    const size_t output_row = item.get_group_linear_id();
                    const size_t local_id = item.get_local_linear_id();
                    const size_t row_offset = output_row * input_size;

                    float sum = 0.0f;
                    for (size_t i = local_id; i < input_size;
                         i += work_group_size) {
                        sum += device_weights[row_offset + i] * device_x[i];
                    }

                    partial[local_id] = sum;
                    item.barrier(sycl::access::fence_space::local_space);

                    /*
                     * work_group_size is a power of two, so the reduction can
                     * halve the active range on every pass.
                     */
                    for (size_t stride = work_group_size >> 1;
                         stride > 0;
                         stride >>= 1) {
                        if (local_id < stride) {
                            partial[local_id] += partial[local_id + stride];
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    if (local_id == 0) {
                        device_y[output_row] =
                            partial[0] +
                            (device_bias ? device_bias[output_row] : 0.0f);
                    }
                });
        });

        queue.memcpy(out, backend->scratch_y, output_size * sizeof(float));
        queue.wait_and_throw();
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr,
                     "error: AdaptiveCpp matvec failed: %s\n",
                     error.what());
        return -1;
    }
}
