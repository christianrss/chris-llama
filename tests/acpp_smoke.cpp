#include <sycl/sycl.hpp>

#include <array>
#include <iostream>

int main() {
    sycl::queue queue{sycl::default_selector_v};

    std::array<int, 4> values{1, 2, 3, 4};
    int* data = sycl::malloc_shared<int>(values.size(), queue);
    if (!data) {
        std::cerr << "failed to allocate SYCL shared memory\n";
        return 1;
    }

    for (std::size_t i = 0; i < values.size(); ++i) {
        data[i] = values[i];
    }

    queue.parallel_for(sycl::range<1>{values.size()}, [=](sycl::id<1> index) {
        data[index[0]] *= 2;
    });
    queue.wait_and_throw();

    bool ok = true;
    for (std::size_t i = 0; i < values.size(); ++i) {
        ok = ok && data[i] == values[i] * 2;
    }

    std::cout << "device: "
              << queue.get_device().get_info<sycl::info::device::name>()
              << '\n';

    sycl::free(data, queue);
    return ok ? 0 : 2;
}
