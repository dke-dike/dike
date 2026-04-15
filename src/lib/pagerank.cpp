/**
 * @file pagerank.cpp
 * @brief Graph partitioning and PageRank computation routines
 *
 * Implements:
 *  - Weighted graph partitioning across cluster nodes (create_vertex_weight)
 *  - Sliced sub-graph construction per partition (create_sliced_graph)
 *  - Core PageRank math (calc_pagerank_value)
 *  - Standard graph ingestion (create_graph)
 */

#include "pagerank.hpp"
#include "../../includes/network/myRDMA.hpp"
#include "../../includes/network/tcp.hpp"

#include <numeric>
#include <omp.h>
#include <time.h>

// ── File-scope state shared between legacy API functions ──────────────────
static TCP     tcp1;
static myRDMA  myrdma1;
static Pagerank pagerank_g;

static std::mutex mutx;
static vector<double> send_buffer[4];
static vector<double> recv_buffer[4];
static int    n, n1;
static vector<int> n2;
static vector<int> nn;

// ── Internal math helpers ──────────────────────────────────────────────────

static double calculateStandardDeviation(const vector<int>& data, int count) {
    double sum = 0.0;
    for (int i = 0; i < count; i++) sum += data[i];
    double mean = sum / count;

    double sq_diff = 0.0;
    for (int i = 0; i < count; i++)
        sq_diff += (data[i] - mean) * (data[i] - mean);

    return sqrt(sq_diff / count);
}

static vector<string> split(const string& str, char delim) {
    istringstream iss(str);
    string token;
    vector<string> result;
    while (getline(iss, token, delim))
        result.push_back(token);
    return result;
}

// ── Graph insertion helpers ────────────────────────────────────────────────

template <class Vector, class T>
bool Pagerank::insert_into_vector(Vector& v, const T& t) {
    auto it = lower_bound(v.begin(), v.end(), t);
    if (it == v.end() || t < *it) {
        v.insert(it, t);
        return true;
    }
    return false;
}

bool Pagerank::add_arc(size_t from, size_t to,
                       vector<vector<size_t>>* graph,
                       vector<int>& num_outgoing) {
    size_t max_dim = max(from, to) + 1;
    if (graph->size() < max_dim) {
        graph->resize(max_dim);
        if (num_outgoing.size() < max_dim)
            num_outgoing.resize(max_dim, 0);
    }

    bool inserted = insert_into_vector((*graph)[to], from);
    if (inserted) num_outgoing[from]++;
    return inserted;
}

bool Pagerank::add_arc1(size_t from, size_t to, vector<int>& num_outgoing) {
    size_t max_dim = max(from, to) + 1;
    if (num_outgoing.size() < max_dim)
        num_outgoing.resize(max_dim, 0);
    num_outgoing[from]++;
    return true;
}

// ── Degree-distribution analysis (debug / profiling) ──────────────────────

void Pagerank::check_power_law_degree(vector<int>& num_outgoing) {
    int max_degree = *max_element(num_outgoing.begin(), num_outgoing.end());
    vector<double> freq(max_degree + 1, 0.0);
    for (int deg : num_outgoing) freq[deg]++;

    ofstream out("graph_deviation.txt");
    if (!out.is_open()) {
        cerr << "[ERROR] Cannot open graph_deviation.txt" << endl;
        return;
    }
    for (int i = 0; i <= max_degree; i++) {
        if (freq[i] != 0.0) {
            out << i << " " << freq[i] << "\n";
        }
    }
    cout << "[INFO] Degree distribution saved to graph_deviation.txt" << endl;
}

// ── Sliced sub-graph construction ─────────────────────────────────────────
/**
 * Reads the edge list and builds the incoming-edge adjacency for
 * vertices [start, end) assigned to this process.
 */
void Pagerank::create_sliced_graph(string path, string del,
                                   int start, int end,
                                   vector<vector<size_t>>& sliced_graph,
                                   int rank) {
    ifstream infile(path.c_str());
    if (!infile.is_open()) {
        cerr << "[ERROR] Cannot open graph file: " << path << endl;
        exit(1);
    }

    sliced_graph.resize(end - start);

    const char delim_ch = (del == " ") ? ' ' : '\t';
    string line;
    size_t line_num = 0;

    while (getline(infile, line)) {
        size_t pos = line.find(delim_ch);
        if (pos == string::npos) continue;

        size_t x = strtoul(line.c_str(), nullptr, 10);
        size_t y = strtoul(line.c_str() + pos + 1, nullptr, 10);

        if ((int)y >= start && (int)y < end)
            insert_into_vector(sliced_graph[y - start], x);

        ++line_num;
        if (line_num % 50000000 == 0 && rank == 0)
            cout << "[INFO] READ " << line_num << " LINES." << endl;
    }
}

// ── Vertex-weighted graph partitioning ────────────────────────────────────
/**
 * Reads the full edge list to build out-degree counts, computes a
 * normalized vertex weight for the chosen partitioning strategy (cmd),
 * and assigns contiguous vertex ranges to each cluster node and MPI rank.
 *
 * Partitioning strategies (cmd argument):
 *   1 - Equi-vertex  (weight = 1)
 *   2 - Square root  (weight = sqrt(degree))
 *   3 - Log          (weight = log(degree + 1))
 *   4 - Equi-edge    (weight = degree + 1)
 *   5 - Log+e        (weight = log(degree + e))
 *   6 - Log+E        (weight = log(degree + E))  [unused alias]
 *   7 - DEWP-Prime   (sigmoid-weighted power law)
 *   default - Adaptive (two-tier sqrt with z-score)
 */
void Pagerank::create_vertex_weight(
    string path, string del,
    vector<int>& num_outgoing, int& num_of_vertex,
    int& start, int& end,
    int* nn_out, int num_of_node, int size,
    string* node, string my_ip, int rank,
    int* displs, int* recvcounts,
    vector<double>* send, vector<double>* recv1,
    string cmd, string alpha1)
{
    if (rank == 0) {
        const char* weight_names[] = {
            nullptr,
            "EQUI-VERTEX",  // 1
            "SQUARE ROOT",  // 2
            "LOG",          // 3
            "EQUI-EDGE",    // 4
            "EQUI-EDGE",    // 5
            "LOG + E",      // 6
            "DEWP-PRIME"    // 7
        };
        int cmd_i = stoi(cmd);
        if (cmd_i >= 1 && cmd_i <= 7)
            cout << "[INFO] WEIGHT = " << weight_names[cmd_i] << endl;
        else {
            cerr << "[ERROR] Invalid partitioning command (expected 1-7)" << endl;
            exit(1);
        }
    }

    // ── Pass 1: build out-degree histogram ───────────────────────────────
    ifstream infile(path.c_str());
    if (!infile.is_open()) {
        cerr << "[ERROR] Cannot open graph file: " << path << endl;
        exit(1);
    }

    const char delim_ch = (del == " ") ? ' ' : '\t';
    string line;
    size_t line_num = 0;
    int    num_vertex = 0;

    while (getline(infile, line)) {
        size_t pos = line.find(delim_ch);
        if (pos == string::npos) continue;

        size_t from = strtoul(line.c_str(), nullptr, 10);
        size_t to   = strtoul(line.c_str() + pos + 1, nullptr, 10);

        int mx = (int)max(from, to);
        if (num_vertex <= mx) num_vertex = mx + 1;
        add_arc1(from, to, num_outgoing);

        ++line_num;
        if (line_num % 50000000 == 0 && rank == 0)
            cout << "[INFO] READ " << line_num << " LINES." << endl;
    }
    infile.close();

    const int   total_edges = line_num;
    const double inv_nodes  = 1.0 / (num_of_node - 1);

    // ── Pass 2: compute vertex weights and node-level partition ──────────
    int start_arr[num_of_node - 1];
    int end_arr[num_of_node - 1];
    start_arr[0] = 0;

    // Coordinator node (index 0) does not compute — skip weight calc for it
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    if (my_ip != "192.168.0.102") {  // Not a legacy coordinator address
        vector<double> vertex_weight;
        vertex_weight.reserve(num_vertex);
        double sum_weight = 0.0;
        const double avg_deg = 40.0;

        for (int i = 0; i < num_vertex; i++) {
            double w;
            int deg = num_outgoing[i];
            if (cmd == "1") {
                w = 1.0;
            } else if (cmd == "2") {
                w = (num_vertex == 2394385) ? sqrt(deg + 1.0) : sqrt(deg);
            } else if (cmd == "3") {
                w = log(deg + 1.0);
            } else if (cmd == "4") {
                w = deg + 1.0;
            } else if (cmd == "5") {
                w = log(deg + 2.71828);
            } else if (cmd == "7") {
                // DEWP-Prime: sigmoid-gated power-law weight
                double alpha    = stod(alpha1);
                double n_diff   = pow((double)deg, alpha);
                double sigmoid  = 1.0 / (1.0 + exp(-5.0 * (deg - avg_deg)));
                w = 1.0 + sqrt(n_diff) * sigmoid;
            } else {
                // Adaptive two-tier strategy (default/cmd==6)
                const double AVG    = 14.2326;
                const double THRESH = 127.0;
                double z = deg - AVG;
                if (deg <= (int)round(AVG))
                    w = 1.0;
                else if (deg <= THRESH)
                    w = sqrt(deg);
                else
                    w = sqrt(deg + z);
            }
            vertex_weight.push_back(w);
            sum_weight += w;
        }

        // Normalize
        for (int i = 0; i < num_vertex; i++)
            vertex_weight[i] /= sum_weight;

        // Assign node-level partition boundaries (proportional cumulative weight)
        double cumsum  = 0.0;
        size_t part_idx = 0;
        for (int i = 0; i < num_vertex; i++) {
            cumsum += vertex_weight[i];
            if (cumsum >= inv_nodes) {
                end_arr[part_idx] = i;
                cumsum = 0.0;
                if ((int)part_idx < num_of_node - 2)
                    start_arr[part_idx + 1] = i;
                ++part_idx;
            }
            if ((int)part_idx == num_of_node - 2) break;
        }
        end_arr[num_of_node - 2] = num_vertex;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("[INFO] PARTITIONING EXECUTION TIME: %Lfs.\n",
           (long double)(t1.tv_sec - t0.tv_sec) +
           (long double)(t1.tv_nsec - t0.tv_nsec) / 1e9);

    // ── Assign this node's vertex range ──────────────────────────────────
    int div_num_of_vertex = 0;

    if (my_ip != node[0]) {
        // Compute node
        for (int i = 1; i < num_of_node; i++) {
            if (node[i] == my_ip) {
                start             = start_arr[i - 1];
                end               = end_arr[i - 1];
                div_num_of_vertex = end - start;
            }
        }

        // Resize send/recv buffers for this compute node
        send[0].resize(div_num_of_vertex);
        recv1[0].resize(num_vertex, 1.0 / num_vertex);
        for (int i = 1; i < num_of_node; i++) {
            send[i].resize(1);  send[i].shrink_to_fit();
            recv1[i].resize(1); recv1[i].shrink_to_fit();
        }

        // Further partition across MPI ranks on this node
        if (size > 1) {
            vector<double> proc_weight;
            double proc_sum = 0.0;
            for (int i = start; i < end; i++) {
                double w = sqrt(num_outgoing[i]);
                proc_weight.push_back(w);
                proc_sum += w;
            }
            for (double& w : proc_weight) w /= proc_sum;

            const double inv_size = 1.0 / size;
            int start_p[size]; start_p[0] = start;
            int end_p[size];
            size_t pidx = 0;
            double psum = 0.0;

            for (int i = start; i < end; i++) {
                psum += proc_weight[i - start];
                if (psum >= inv_size) {
                    end_p[pidx] = i;
                    psum = 0.0;
                    if ((int)pidx < size - 1) start_p[pidx + 1] = i;
                    ++pidx;
                }
                if ((int)pidx == size - 1) break;
            }
            end_p[size - 1] = end;

            for (int i = 0; i < size; i++) {
                if (rank == i) {
                    start = start_p[i];
                    end   = end_p[i];
                }
                displs[i]     = start_p[i] - start_p[0];
                recvcounts[i] = end_p[i] - start_p[i];
            }
        }
    } else {
        // Coordinator node: set up recv buffers sized to each partition
        for (int i = 0; i < num_of_node - 1; i++) {
            int part_size = end_arr[i] - start_arr[i];
            send[i].resize(num_vertex, 1.0 / num_vertex);
            recv1[i].resize(part_size);
            nn_out[i] = part_size;
        }
    }
}

// ── Full graph ingestion (single-node, non-partitioned) ───────────────────

void Pagerank::create_graph(string path, string del,
                            vector<vector<size_t>>* graph,
                            vector<int>& num_outgoing) {
    ifstream infile(path.c_str());
    if (!infile.is_open()) {
        cerr << "[ERROR] Cannot open graph file: " << path << endl;
        exit(1);
    }

    const char delim_ch = (del == " ") ? ' ' : '\t';
    string line;
    size_t line_num = 0;

    while (getline(infile, line)) {
        size_t pos = line.find(delim_ch);
        if (pos == string::npos) continue;

        size_t from = strtoul(line.c_str(), nullptr, 10);
        size_t to   = strtoul(line.c_str() + pos + 1, nullptr, 10);
        add_arc(from, to, graph, num_outgoing);
        ++line_num;
    }

    pagerank_g.num_of_vertex = (int)graph->size();
    cout << "[INFO] Loaded " << line_num << " edges, "
         << pagerank_g.num_of_vertex << " vertices." << endl;
}

// ── Legacy graph_partition (kept for compatibility) ───────────────────────

void Pagerank::graph_partition(vector<vector<size_t>>* /*graph*/,
                               vector<vector<size_t>>& /*sliced_graph*/,
                               vector<int>& num_outgoing,
                               int num_of_vertex,
                               int& start, int& end,
                               int* nn_out, int num_of_node, int size,
                               string* node, string my_ip, int rank,
                               int* displs, int* recvcounts,
                               vector<double>* send, vector<double>* recv1) {
    // Equi-weight (sqrt) partition — simplified version
    vector<double> w;
    double wsum = 0.0;
    for (int i = 0; i < num_of_vertex; i++) {
        double wi = sqrt(num_outgoing[i]);
        w.push_back(wi);
        wsum += wi;
    }
    for (double& wi : w) wi /= wsum;

    const double inv = 1.0 / (num_of_node - 1);
    int start_arr[num_of_node - 1]; start_arr[0] = 0;
    int end_arr[num_of_node - 1];
    size_t idx = 0;
    double cs  = 0.0;

    for (int i = 0; i < num_of_vertex; i++) {
        cs += w[i];
        if (cs >= inv) {
            end_arr[idx] = i;
            cs = 0.0;
            if ((int)idx < num_of_node - 2) start_arr[idx + 1] = i;
            ++idx;
        }
        if ((int)idx == num_of_node - 2) break;
    }
    end_arr[num_of_node - 2] = num_of_vertex;

    if (my_ip != node[0]) {
        for (int i = 1; i < num_of_node; i++) {
            if (node[i] == my_ip) {
                start = start_arr[i - 1];
                end   = end_arr[i - 1];
            }
        }
        send[0].resize(end - start);
        recv1[0].resize(num_of_vertex, 1.0 / num_of_vertex);
    } else {
        for (int i = 0; i < num_of_node - 1; i++) {
            int ps = end_arr[i] - start_arr[i];
            send[i].resize(num_of_vertex, 1.0 / num_of_vertex);
            recv1[i].resize(ps);
            nn_out[i] = ps;
        }
    }
}

// ── Core PageRank calculation ─────────────────────────────────────────────

void Pagerank::calc_pagerank_value(int start, int end, double dangling_pr, double /*unused*/) {
    const int num_v = pagerank_g.num_of_vertex;
    const double df_val  = df;
    const double df_inv  = 1.0 - df_val;
    const double inv_nv  = 1.0 / num_v;

    const auto& graph        = pagerank_g.graph;
    const auto& num_outgoing = pagerank_g.num_outgoing;

    double* recv_ptr = recv_buffer[0].data();
    double* send_ptr = send_buffer[0].data();

    for (int i = start; i < end; i++) {
        double tmp = 0.0;
        const size_t  sz  = graph[i].size();
        const size_t* ptr = graph[i].data();
        for (size_t j = 0; j < sz; j++) {
            size_t from = ptr[j];
            tmp += recv_ptr[from] / num_outgoing[from];
        }
        send_ptr[i - start] = (tmp + dangling_pr * inv_nv) * df_val
                             + df_inv * inv_nv;
    }
}

// ── Legacy run_pagerank (single-node mode) ────────────────────────────────

void Pagerank::run_pagerank(int iter) {
    const string& server_ip = pagerank_g.server_ip;
    const string& my_ip     = pagerank_g.my_ip;
    const int num_of_vertex = pagerank_g.num_of_vertex;
    const int s = pagerank_g.start1;
    const int e = pagerank_g.end1;
    const auto& num_outgoing = pagerank_g.num_outgoing;

    double* recv_ptr = recv_buffer[0].data();
    double* send_ptr = send_buffer[0].data();

    double diff      = 1.0;
    double dangling_pr;
    vector<double> prev_pr;
    struct timespec t0, t1;

    for (int step = 0; step < iter; step++) {
        cout << "===== STEP " << step + 1 << " =====" << endl;
        dangling_pr = 0.0;

        if (step != 0) {
            if (my_ip != server_ip) {
                for (int i = 0; i < num_of_vertex; i++)
                    if (num_outgoing[i] == 0) dangling_pr += recv_ptr[i];
            } else {
                diff = 0.0;
                for (int i = 0; i < num_of_vertex; i++)
                    diff += fabs(prev_pr[i] - send_ptr[i]);
                pagerank_g.diff = diff;
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &t0);
        if (my_ip != server_ip)
            calc_pagerank_value(s, e, dangling_pr, 0.0);
        else
            prev_pr = send_buffer[0];
        clock_gettime(CLOCK_MONOTONIC, &t1);
        printf("[INFO] STEP %d COMPUTE: %Lfs.\n", step + 1,
               (long double)(t1.tv_sec - t0.tv_sec) +
               (long double)(t1.tv_nsec - t0.tv_nsec) / 1e9);

        gather_pagerank("send");
        scatter_pagerank();

        if (my_ip == server_ip)
            cout << "[INFO] DIFF: " << diff << endl;

        if (diff < 1e-5 || recv_ptr[0] > 1.0) break;
    }
}

// ── Gather / scatter (legacy single-node API) ─────────────────────────────

static void fill_send_buffer(int num_of_server) {
    send_buffer[0].clear();
    for (int i = 0; i < num_of_server - 1; i++) {
        size_t seg = nn[i];
        send_buffer[0].insert(send_buffer[0].end(),
                              recv_buffer[i].begin(),
                              recv_buffer[i].begin() + seg);
    }
}

void Pagerank::gather_pagerank(string opcode) {
    if (pagerank_g.my_ip == pagerank_g.server_ip) {
        myrdma1.recv_t("send");
        fill_send_buffer(pagerank_g.num_of_server);
        if (pagerank_g.diff < 1e-5)
            send_buffer[0][0] += 1.0;
        fill(&send_buffer[1],
             &send_buffer[pagerank_g.num_of_server - 1],
             send_buffer[0]);
    }
}

void Pagerank::scatter_pagerank() {
    if (pagerank_g.my_ip == pagerank_g.server_ip) {
        for (int i = 0; i < pagerank_g.num_of_server - 1; i++)
            myrdma1.rdma_write_pagerank(i);
    } else {
        myrdma1.rdma_recv_pagerank(0, sizeof(double) * pagerank_g.num_of_vertex);
    }
}

// ── Utility ───────────────────────────────────────────────────────────────

string Pagerank::max_pr() {
    int    best_idx = 0;
    double best_val = recv_buffer[0][0];
    for (int i = 1; i < pagerank_g.num_of_vertex; i++) {
        if (recv_buffer[0][i] > best_val) {
            best_idx = i;
            best_val = recv_buffer[0][i];
        }
    }
    ostringstream ss;
    ss << "Most important page: vertex " << best_idx << " (PR=" << best_val << ")";
    return ss.str();
}

void Pagerank::print_pr() {
    double total = accumulate(recv_buffer[0].begin(), recv_buffer[0].end(), 0.0) - 1.0;
    cout.precision(numeric_limits<double>::digits10);
    int nv = pagerank_g.num_of_vertex;
    for (int i = max(0, nv - 200); i < nv; i++)
        cout << "pr[" << i << "]: " << recv_buffer[0][i] << "\n";
    cerr << "[INFO] PR sum = " << round(total) << endl;
}

int Pagerank::get_num_of_vertex() {
    return pagerank_g.num_of_vertex;
}

void Pagerank::init_connection(const char* ip, string server[],
                               int number_of_server, int Port,
                               int num_of_vertex) {
    myrdma1.initialize_rdma_connection_vector(ip, server, number_of_server,
                                              Port, send_buffer, recv_buffer,
                                              num_of_vertex);
    myrdma1.create_rdma_info(send_buffer, recv_buffer);
    myrdma1.send_info_change_qp();

    pagerank_g.my_ip         = string(ip);
    pagerank_g.num_of_server = number_of_server;
    pagerank_g.diff          = 1.0;
    pagerank_g.node          = server;
    pagerank_g.server_ip     = server[0];
}
