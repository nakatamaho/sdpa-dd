#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <mpblas_dd.h>
#include <chrono>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

void Rgemm_NN_omp(mplapackint m, mplapackint n, mplapackint k, dd_real alpha, dd_real *A, mplapackint lda, dd_real *B, mplapackint ldb, dd_real beta, dd_real *C, mplapackint ldc);

void generate_random_matrix(mplapackint rows, mplapackint cols, dd_real *matrix) {
    unsigned int seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
    std::mt19937 mt(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (mplapackint i = 0; i < rows; ++i) {
        for (mplapackint j = 0; j < cols; ++j) {
            double random_value1 = dist(mt);
            double random_value2 = dist(mt);
            matrix[i + j * rows] = dd_real(random_value1) + dd_real(random_value2) * 1e-16;
        }
    }
}

template <typename Func> double benchmark(Func func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

std::pair<double, double> calculate_mean_and_variance(const std::vector<double> &values) {
    double mean = 0.0;
    double variance = 0.0;

    for (double value : values) {
        mean += value;
    }
    mean /= values.size();

    for (double value : values) {
        variance += (value - mean) * (value - mean);
    }
    variance /= values.size();

    return {mean, variance};
}

int main() {
#ifdef _OPENMP
    std::cout << "OpenMP is enabled.\n";
    std::cout << "Number of threads: " << omp_get_max_threads() << "\n";
#else
    std::cout << "OpenMP is not enabled.\n";
#endif

    std::vector<mplapackint> sizes = {256, 512, 768, 1024};
#ifdef _OPENMP
    int num_cores = omp_get_num_procs();
#else
    int num_cores = 1;
#endif

//    std::vector<int> thread_counts = {1, 2, 4, 8, 16, 32};
    std::vector<int> thread_counts = { num_cores, num_cores/2 };

    const int num_trials = 10;
    std::mt19937 mt(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (auto m : sizes) {
        for (auto n : sizes) {
            for (auto k : sizes) {
                double flop_count = static_cast<double>(m) * n * (2.0 * k + 1);

                std::vector<dd_real> A(m * k);
                std::vector<dd_real> B(k * n);
                std::vector<dd_real> C(m * n);

                dd_real alpha = dd_real(dist(mt)) + dd_real(dist(mt)) * 1e-16; // double-double型
                dd_real beta = dd_real(dist(mt)) + dd_real(dist(mt)) * 1e-16;

                generate_random_matrix(m, k, A.data());
                generate_random_matrix(k, n, B.data());
                generate_random_matrix(m, n, C.data());

                std::cout << "Benchmarking m=" << m << ", n=" << n << ", k=" << k << ":\n";

                for (auto threads : thread_counts) {
#ifdef _OPENMP
                    omp_set_num_threads(threads);
#endif
                    std::vector<double> flops_results;

                    for (int trial = 0; trial < num_trials; ++trial) {
                        double elapsed = benchmark([&]() { Rgemm_NN_omp(m, n, k, alpha, A.data(), m, B.data(), k, beta, C.data(), m); });

                        double flops = flop_count / elapsed / 1e6; // MFLOPS
                        flops_results.push_back(flops);
                    }
                    auto [mean_flops, variance_flops] = calculate_mean_and_variance(flops_results);
                    std::cout << "FLOPS for each trial: ";
                    for (const auto &flops : flops_results) {
                        std::cout << flops << " ";
                    }
                    std::cout << "\n";

                    std::cout << "Threads: " << threads << ", Mean FLOPS: " << mean_flops << " MFLOPS"
                              << ", Variance: " << variance_flops << "\n";
                }
                std::cout << "---------------------------------\n";
            }
        }
    }

    return 0;
}
