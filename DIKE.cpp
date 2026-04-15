/**
 * @file test.cpp
 * @brief Distributed PageRank using Hybrid MPI + RDMA communication
 *
 * Architecture:
 *   - One server node (coordinator): aggregates and broadcasts PageRank vectors
 *   - Multiple compute nodes: partition graph, compute local PageRank
 *
 * Communication model:
 *   - Intra-node parallelism : OpenMP
 *   - Intra-cluster (same node, multi-process): MPI (Allgatherv / Isend-Irecv)
 *   - Inter-cluster (node-to-node): RDMA write_with_imm
 *
 * Usage:
 *   mpirun -np <procs> ./pagerank <graph_file> <delimiter> <cmd> <alpha>
 */

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <math.h>
#include <mutex>
#include <numeric>
#include <queue>
#include <string>
#include <time.h>
#include <vector>

#include <myRDMA.hpp>
#include <pagerank.hpp>
#include "tcp.hpp"

// ── Tunable constants ──────────────────────────────────────────────────────
#define DAMPING_FACTOR  0.85
#define MAX_ITERATIONS  20
#define NUM_NODES       5
#define TCP_PORT        40145
#define SERVER_IP       "192.168.4.5"

// IP address table for the cluster (index 0 = coordinator/server)
static const std::string NODE_IPS[NUM_NODES] = {
    SERVER_IP,
    "192.168.4.10",
    "192.168.4.12",
    "192.168.4.13",
    "192.168.4.9"
};

// ── Helpers ────────────────────────────────────────────────────────────────
static inline bool is_server(const std::string& ip) {
    return ip == SERVER_IP;
}

// ── Main ───────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    using namespace std;

    TCP      tcp;
    Pagerank pagerank;
    myRDMA   myrdma;

    int rank, size;
    int start, end;
    int nn[NUM_NODES] = {};

    long double network_time     = 0;
    long double compute_time     = 0;
    long double avg_compute_time = 0;
    long double mpi_time         = 0;
    long double rdma_time        = 0;

    struct timespec begin1, end1, begin2, end2;

    vector<vector<size_t>>  sliced_graph;
    vector<double> send[NUM_NODES];
    vector<double> recv1[NUM_NODES];
    vector<double>* send_first = &send[1];
    vector<double>* send_end   = &send[NUM_NODES - 1];

    // ── MPI Initialization ─────────────────────────────────────────────────
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int recvcounts[size];
    int displs[size];

    const string my_ip = tcp.check_my_ip();

    if (rank == 0) {
        cout << "[INFO] IP: " << my_ip << endl;
        cout << "=====================================================" << endl;
        cout << (is_server(my_ip) ? "[INFO] NETWORK CONFIGURATION"
                                  : "[INFO] CREATE GRAPH") << endl;
    }

    // ── Graph Partitioning ─────────────────────────────────────────────────
    vector<int> num_outgoing;
    int num_of_vertex;

    pagerank.create_vertex_weight(
        argv[1], argv[2],
        num_outgoing, num_of_vertex,
        start, end, nn,
        NUM_NODES, size, const_cast<string*>(NODE_IPS), my_ip, rank,
        displs, recvcounts, send, recv1,
        argv[3], argv[4]
    );
    num_of_vertex = num_outgoing.size();

    if (rank == 0 && !is_server(my_ip))
        cout << "[INFO] TOTAL VERTEX: " << num_of_vertex << endl;

    pagerank.create_sliced_graph(argv[1], argv[2], start, end, sliced_graph, rank);

    // Report memory usage
    size_t total_edge_bytes = 0;
    for (auto& v : sliced_graph) {
        v.shrink_to_fit();
        total_edge_bytes += v.size() * sizeof(size_t);
    }
    size_t outgoing_bytes = sizeof(size_t) * num_outgoing.size();

    if (rank == 0 && !is_server(my_ip)) {
        int num_edge = 0;
        for (int i = start; i < end; i++) num_edge += num_outgoing[i];
        cout << "[INFO] FINISH CREATE GRAPH" << endl;
        cout << "[INFO] Vertex: " << (end - start) << ", Edge: " << num_edge << endl;
        cout << "[INFO] GRAPH MEMORY: "
             << total_edge_bytes << " + " << outgoing_bytes
             << " = " << (total_edge_bytes + outgoing_bytes) << " bytes" << endl;
    }

    // Coordinator does not need vertex out-degree data
    if (is_server(my_ip)) {
        num_outgoing.clear();
        num_outgoing.shrink_to_fit();
    }

    // ── RDMA Initialization ────────────────────────────────────────────────
    if (rank == 0) {
        if (!is_server(my_ip)) {
            cout << "=====================================================" << endl;
            cout << "[INFO] NETWORK CONFIGURATION" << endl;
        }
        myrdma.initialize_rdma_connection_vector(
            my_ip.c_str(), const_cast<string*>(NODE_IPS),
            NUM_NODES, TCP_PORT, send, recv1, num_of_vertex
        );
        myrdma.create_rdma_info(send, recv1);
        myrdma.send_info_change_qp();
    }

    const size_t div_buff_size = sizeof(double) * send[0].size();

    // Assign send buffer pointer based on role
    vector<double> div_send;
    double* send_buf_ptr;

    if (!is_server(my_ip) && size > 1) {
        div_send.resize(end - start);
        send_buf_ptr = div_send.data();
    } else {
        send_buf_ptr = send[0].data();
    }

    double* recv_buffer_ptr = recv1[0].data();

    // Global synchronization barrier before iterations
    int check = 1;
    int check1[size];
    MPI_Allgather(&check, 1, MPI_INT, check1, 1, MPI_INT, MPI_COMM_WORLD);
    if (rank == 0) myrdma.rdma_comm("write_with_imm", "1");
    MPI_Allgather(&check, 1, MPI_INT, check1, 1, MPI_INT, MPI_COMM_WORLD);

    if (rank == 0 || is_server(my_ip))
        cout << "=====================================================" << endl;

    // ── PageRank Iterations ────────────────────────────────────────────────
    double diff    = 1.0;
    double dangling_pr;
    vector<double> prev_pr;

    const double df_inv            = 1.0 - DAMPING_FACTOR;
    const double inv_num_of_vertex = 1.0 / num_of_vertex;

    clock_gettime(CLOCK_MONOTONIC, &begin2);

    for (size_t step = 0; step < MAX_ITERATIONS; step++) {
        if (rank == 0 || is_server(my_ip))
            cout << "================ STEP " << step + 1 << " ================" << endl;

        dangling_pr = 0.0;

        // Dangling node accumulation / convergence check (from previous step)
        if (step != 0) {
            clock_gettime(CLOCK_MONOTONIC, &begin1);
            if (!is_server(my_ip)) {
                for (size_t i = 0; i < (size_t)num_of_vertex; i++)
                    if (num_outgoing[i] == 0) dangling_pr += recv_buffer_ptr[i];
            } else {
                diff = 0.0;
                for (size_t i = 0; i < (size_t)num_of_vertex; i++)
                    diff += fabs(prev_pr[i] - send_buf_ptr[i]);
            }
            clock_gettime(CLOCK_MONOTONIC, &end1);
            compute_time += (end1.tv_sec - begin1.tv_sec)
                          + (end1.tv_nsec - begin1.tv_nsec) / 1e9;
        }

        // ── Local PageRank computation ─────────────────────────────────────
        if (!is_server(my_ip)) {
            if (rank == 0) cout << "[INFO] COMPUTE PAGERANK ";

            clock_gettime(CLOCK_MONOTONIC, &begin1);
            for (size_t i = 0; i < (size_t)(end - start); i++) {
                double tmp = 0.0;
                const size_t  sz  = sliced_graph[i].size();
                const size_t* ptr = sliced_graph[i].data();
                for (size_t j = 0; j < sz; j++) {
                    size_t from = ptr[j];
                    tmp += recv_buffer_ptr[from] / num_outgoing[from];
                }
                send_buf_ptr[i] = (tmp + dangling_pr * inv_num_of_vertex)
                                  * DAMPING_FACTOR + df_inv * inv_num_of_vertex;
            }
            clock_gettime(CLOCK_MONOTONIC, &end1);
            compute_time += (end1.tv_sec - begin1.tv_sec)
                          + (end1.tv_nsec - begin1.tv_nsec) / 1e9;
            if (rank == 0) cout << "- SUCCESS" << endl;

            // Intra-node MPI gather
            clock_gettime(CLOCK_MONOTONIC, &begin1);
            if (size > 1)
                MPI_Allgatherv(send_buf_ptr, div_send.size(), MPI_DOUBLE,
                               send[0].data(), recvcounts, displs, MPI_DOUBLE,
                               MPI_COMM_WORLD);
            clock_gettime(CLOCK_MONOTONIC, &end1);
            {
                long double t = (end1.tv_sec - begin1.tv_sec)
                              + (end1.tv_nsec - begin1.tv_nsec) / 1e9;
                if (rank == 0) { network_time += t; mpi_time += t; }
            }
        } else {
            prev_pr = send[0];
        }

        // ── RDMA push: compute → coordinator ──────────────────────────────
        clock_gettime(CLOCK_MONOTONIC, &begin1);
        if (is_server(my_ip)) {
            send[0].clear();
            myrdma.recv_t("send");
            cout << "[INFO] START RECEIVE - SUCCESS" << endl;

            for (size_t i = 0; i < NUM_NODES - 1; i++) {
                size_t seg = nn[i];
                send[0].insert(send[0].end(),
                               make_move_iterator(recv1[i].begin()),
                               make_move_iterator(recv1[i].begin() + seg));
            }
            if (diff < 1e-5) send_buf_ptr[0] += 1.0;  // convergence sentinel
            fill(send_first, send_end, send[0]);
            cout << "[INFO] START AGGREGATE - SUCCESS" << endl;
        } else {
            if (rank == 0) {
                cout << "[INFO] START SEND_RDMA - SUCCESS" << endl;
                myrdma.rdma_write_vector(0, div_buff_size);
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &end1);
        {
            long double t = (end1.tv_sec - begin1.tv_sec)
                          + (end1.tv_nsec - begin1.tv_nsec) / 1e9;
            if (rank == 0) { network_time += t; rdma_time += t; }
        }

        // ── RDMA push: coordinator → compute ──────────────────────────────
        if (is_server(my_ip)) {
            vector<thread> workers;
            workers.reserve(NUM_NODES - 1);
            for (size_t i = 0; i < NUM_NODES - 1; i++)
                workers.emplace_back(&myRDMA::rdma_write_pagerank, &myrdma, (int)i);
            for (auto& w : workers) w.join();
            cout << "[INFO] START SEND - SUCCESS" << endl;
        } else {
            MPI_Request request;
            if (rank == 0) {
                clock_gettime(CLOCK_MONOTONIC, &begin1);
                myrdma.rdma_recv_pagerank(0, div_buff_size);
                cout << "[INFO] START RECEIVE_RDMA - SUCCESS" << endl;
                clock_gettime(CLOCK_MONOTONIC, &end1);
                long double t = (end1.tv_sec - begin1.tv_sec)
                              + (end1.tv_nsec - begin1.tv_nsec) / 1e9;
                rdma_time    += t;
                network_time += t;
            }

            // Intra-node broadcast of updated PR vector
            clock_gettime(CLOCK_MONOTONIC, &begin1);
            if (size > 1) {
                if (rank == 0) {
                    for (int dest = 1; dest < size; dest++)
                        MPI_Isend(recv_buffer_ptr, num_of_vertex, MPI_DOUBLE,
                                  dest, 32548, MPI_COMM_WORLD, &request);
                } else {
                    MPI_Irecv(recv_buffer_ptr, num_of_vertex, MPI_DOUBLE,
                              0, 32548, MPI_COMM_WORLD, &request);
                    MPI_Wait(&request, MPI_STATUS_IGNORE);
                }
                MPI_Allgather(&check, 1, MPI_INT, check1, 1, MPI_INT, MPI_COMM_WORLD);
            }
            clock_gettime(CLOCK_MONOTONIC, &end1);
            {
                long double t = (end1.tv_sec - begin1.tv_sec)
                              + (end1.tv_nsec - begin1.tv_nsec) / 1e9;
                if (rank == 0) {
                    network_time     += t;
                    mpi_time         += t;
                    avg_compute_time += compute_time;

                    printf("\nCOMPUTE PAGERANK:       %LFs.\n", compute_time);
                    printf("NETWORK (RDMA):         %Lfs.\n",  rdma_time);
                    printf("STEP %zu EXECUTION TIME: %Lfs.\n",
                           step + 1, compute_time + network_time);
                    network_time = 0;
                    compute_time = 0;
                }
            }
        }

        if (is_server(my_ip) && rank == 0)
            cout << "[INFO] DIFF: " << diff << endl;
    }

    clock_gettime(CLOCK_MONOTONIC, &end2);
    long double total_time = (end2.tv_sec - begin2.tv_sec)
                           + (end2.tv_nsec - begin2.tv_nsec) / 1e9;

    // ── Top-N PageRank output ──────────────────────────────────────────────
    ofstream outfile("output.txt");

    if (!is_server(my_ip) && rank == 0) {
        cout    << "=====================================================" << endl;
        outfile << "=====================================================" << endl;
        cout    << "[INFO] SORTING PAGERANK VALUE." << endl;
        outfile << "[INFO] SORTING PAGERANK VALUE." << endl;

        const int topN = 20;
        vector<pair<double, int>> result;
        result.reserve(num_of_vertex);
        for (int i = 0; i < num_of_vertex; i++)
            result.emplace_back(recv1[0][i], i);

        partial_sort(result.begin(), result.begin() + topN,
                     result.end(), greater<pair<double, int>>());

        cout.precision(numeric_limits<double>::digits10);
        for (int i = 0; i < topN; i++) {
            cout    << "PR[" << result[i].second << "]: " << result[i].first << "\n";
            outfile << "PR[" << result[i].second << "]: " << result[i].first << "\n";
        }
        cout    << "=====================================================" << endl;
        outfile << "=====================================================" << endl;
    }

    // ── Summary ────────────────────────────────────────────────────────────
    if (rank == 0 || is_server(my_ip)) {
        printf("[INFO] AVG COMPUTE TIME:     %LFs.\n", avg_compute_time / MAX_ITERATIONS);
        printf("[INFO] AVG MPI TIME:         %Lfs.\n", mpi_time         / MAX_ITERATIONS);
        printf("[INFO] AVG RDMA TIME:        %Lfs.\n", rdma_time        / MAX_ITERATIONS);
        printf("[INFO] TOTAL EXECUTION TIME: %Lfs.\n", total_time);
        cout << "=====================================================" << endl;

        if (!is_server(my_ip)) {
            outfile << "[INFO] AVG COMPUTE TIME:     " << avg_compute_time / MAX_ITERATIONS << "s.\n";
            outfile << "[INFO] AVG MPI TIME:         " << mpi_time         / MAX_ITERATIONS << "s.\n";
            outfile << "[INFO] AVG RDMA TIME:        " << rdma_time        / MAX_ITERATIONS << "s.\n";
            outfile << "[INFO] TOTAL EXECUTION TIME: " << total_time << "s.\n";
            outfile << "=====================================================" << endl;
        }
    }

    outfile.close();
    MPI_Finalize();
    myrdma.exit_rdma();

    return 0;
}
