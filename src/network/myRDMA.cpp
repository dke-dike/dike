/**
 * @file myRDMA.cpp
 * @brief High-level RDMA communication wrappers for PageRank distribution
 *
 * Provides one-to-many / many-to-one RDMA send/receive helpers built on top
 * of the low-level RDMA and TCP primitives. All vector-based transfers use
 * RDMA write_with_imm for zero-copy semantics.
 */

#include "tcp.hpp"
#include "RDMA.hpp"
#include "myRDMA.hpp"
#include <omp.h>

static std::mutex mutx;
myRDMA myrdma;
RDMA   rdma;
int    partition;
int    partition1;

// Per-connection RDMA resource bundle
struct RdmaInfo {
    struct ibv_context* context;
    struct ibv_pd*      pd;
    int                 cq_size;
    struct ibv_cq*      cq;
    struct ibv_qp*      qp;
    struct ibv_mr*      mr;
    uint16_t            lid;
    uint32_t            qp_num;
};

// Index 0 = send QPs, index 1 = recv QPs
std::array<std::vector<RdmaInfo>, 2> rdma_info1;

vector<double*> send_adrs;
vector<double*> recv_adrs;

// Helper: convert std::string to a static C-string (single-threaded use only)
static char* to_cstr(const string& s) {
    static char buf[buf_size];
    strncpy(buf, s.c_str(), sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    return buf;
}

// ── PageRank-specific RDMA transfers ──────────────────────────────────────

void myRDMA::rdma_send_pagerank(vector<double> /*msg*/, int i) {
    size_t size = sizeof(double) * myrdma.num_of_vertex;
    rdma.post_rdma_send(rdma_info1[0][i].qp, rdma_info1[0][i].mr,
                        send_adrs[i], size,
                        myrdma.qp_key[i].first, myrdma.qp_key[i].second);
    rdma.pollCompletion(rdma_info1[0][i].cq);
}

void myRDMA::rdma_recv_pagerank(int i, size_t size) {
    rdma.post_rdma_recv(rdma_info1[1][i].qp, rdma_info1[1][i].mr,
                        rdma_info1[1][i].cq, recv_adrs[i], size);
    rdma.pollCompletion(rdma_info1[1][i].cq);
}

void myRDMA::rdma_write_pagerank(int i) {
    size_t size = sizeof(double) * myrdma.send[i].size();
    rdma.post_rdma_write_with_imm(rdma_info1[0][i].qp, rdma_info1[0][i].mr,
                                  send_adrs[i], size,
                                  myrdma.qp_key[i].first, myrdma.qp_key[i].second);
    rdma.pollCompletion(rdma_info1[0][i].cq);
}

void myRDMA::rdma_wrecv_pagerank(int i) {
    struct ibv_wc wc;
    size_t size = sizeof(double) * myrdma.num_of_vertex;
    rdma.post_rdma_recv(rdma_info1[1][i].qp, rdma_info1[1][i].mr,
                        rdma_info1[1][i].cq, recv_adrs[i], size);
    while (ibv_poll_cq(rdma_info1[1][i].cq, 1, &wc) == 0);
}

// ── Vector RDMA transfers ─────────────────────────────────────────────────

void myRDMA::rdma_send_vector(vector<double> msg, int i) {
    size_t size = sizeof(double) * msg.size();
    rdma.post_rdma_send(rdma_info1[0][i].qp, rdma_info1[0][i].mr,
                        send_adrs[i], size,
                        myrdma.qp_key[i].first, myrdma.qp_key[i].second);
    rdma.pollCompletion(rdma_info1[0][i].cq);
}

void myRDMA::rdma_write_vector(int i, size_t size) {
    rdma.post_rdma_write_with_imm(rdma_info1[0][i].qp, rdma_info1[0][i].mr,
                                  send_adrs[i], size,
                                  myrdma.qp_key[i].first, myrdma.qp_key[i].second);
    rdma.pollCompletion(rdma_info1[0][i].cq);
}

// ── String RDMA transfers (control-path messages) ─────────────────────────

void myRDMA::rdma_send(string msg, int i) {
    RDMA rdma_local;
    if (msg.size() > 67108863)
        msg.resize(67108863);
    strcpy(myrdma.send_buffer[i], msg.c_str());
    rdma_local.post_rdma_send(get<4>(myrdma.rdma_info[0][i]),
                              get<5>(myrdma.rdma_info[0][i]),
                              myrdma.send_buffer[i],
                              sizeof(myrdma.send_buffer[i]),
                              myrdma.qp_key[i].first, myrdma.qp_key[i].second);
    if (!rdma_local.pollCompletion(get<3>(myrdma.rdma_info[0][i])))
        cerr << "[ERROR] rdma_send failed on connection " << i << endl;
}

void myRDMA::rdma_send_with_imm(string msg, int i) {
    RDMA rdma_local;
    if (msg.size() > 67108863) msg.resize(67108863);
    strcpy(myrdma.send_buffer[i], msg.c_str());
    rdma_local.post_rdma_send_with_imm(get<4>(myrdma.rdma_info[0][i]),
                                       get<5>(myrdma.rdma_info[0][i]),
                                       myrdma.send_buffer[i],
                                       sizeof(myrdma.send_buffer[i]),
                                       myrdma.qp_key[i].first,
                                       myrdma.qp_key[i].second);
    if (!rdma_local.pollCompletion(get<3>(myrdma.rdma_info[0][i])))
        cerr << "[ERROR] rdma_send_with_imm failed on connection " << i << endl;
}

void myRDMA::rdma_write(string /*msg*/, int i) {
    RDMA rdma_local;
    TCP  tcp_local;
    rdma_local.post_rdma_write(get<4>(myrdma.rdma_info[0][i]),
                               get<5>(myrdma.rdma_info[0][i]),
                               myrdma.send_buffer[i],
                               sizeof(myrdma.send_buffer[i]),
                               myrdma.qp_key[i].first, myrdma.qp_key[i].second);
    if (rdma_local.pollCompletion(get<3>(myrdma.rdma_info[0][i])))
        tcp_local.send_msg("1", myrdma.sock_idx[i]);
    else
        cerr << "[ERROR] rdma_write failed on connection " << i << endl;
}

void myRDMA::rdma_write_with_imm(string msg, int i) {
    RDMA rdma_local;
    if (msg.size() > 67108863) msg.resize(67108863);
    strcpy(myrdma.send_buffer[i], msg.c_str());
    rdma_local.post_rdma_write_with_imm(get<4>(myrdma.rdma_info[0][i]),
                                        get<5>(myrdma.rdma_info[0][i]),
                                        myrdma.send_buffer[i],
                                        sizeof(myrdma.send_buffer[i]),
                                        myrdma.qp_key[i].first,
                                        myrdma.qp_key[i].second);
    if (!rdma_local.pollCompletion(get<3>(myrdma.rdma_info[0][i])))
        cerr << "[ERROR] rdma_write_with_imm failed on connection " << i << endl;
}

// ── Receive-side primitives ───────────────────────────────────────────────

void myRDMA::rdma_send_recv(int i) {
    size_t size = sizeof(double) * myrdma.num_of_vertex;
    rdma.post_rdma_recv(rdma_info1[1][i].qp, rdma_info1[1][i].mr,
                        rdma_info1[1][i].cq, recv_adrs[i], size);
    rdma.pollCompletion(rdma_info1[1][i].cq);
}

void myRDMA::rdma_send_rcv(int i, int* /*nn*/, int /*num_of_node*/,
                            vector<double>* /*send*/, vector<double>* /*recv1*/) {
    size_t size = sizeof(double) * myrdma.num_of_vertex;
    rdma.post_rdma_recv(rdma_info1[1][i].qp, rdma_info1[1][i].mr,
                        rdma_info1[1][i].cq, recv_adrs[i], size);
    rdma.pollCompletion(rdma_info1[1][i].cq);
}

void myRDMA::rdma_write_recv(int i) {
    TCP tcp_local;
    while (tcp_local.recv_msg(myrdma.sock_idx[i]) <= 0);
}

// ── Collective communication helpers ──────────────────────────────────────

void myRDMA::rdma_send_msg(string opcode, string msg) {
    if (opcode == "send_with_imm") {
        for (int i = 0; i < myrdma.connect_num; i++)
            rdma_send_with_imm(msg, i);
    } else if (opcode == "write") {
        for (int i = 0; i < myrdma.connect_num; i++)
            rdma_write(msg, i);
    } else if (opcode == "write_with_imm") {
        for (int i = 0; i < myrdma.connect_num; i++)
            rdma_write_vector(i, sizeof(double) * 1);
    } else if (opcode == "send") {
        for (int i = 0; i < myrdma.connect_num; i++)
            rdma_send(msg, i);
    } else {
        cerr << "[ERROR] rdma_send_msg: unknown opcode '" << opcode << "'" << endl;
        exit(1);
    }
}

void myRDMA::rdma_recv_msg(string opcode, int i) {
    if (opcode == "send_with_imm" || opcode == "write_with_imm" || opcode == "send")
        rdma_send_recv(i);
    else if (opcode == "write")
        rdma_write_recv(i);
    else {
        cerr << "[ERROR] rdma_recv_msg: unknown opcode '" << opcode << "'" << endl;
        exit(1);
    }
}

void myRDMA::recv_t(string opcode) {
    vector<thread> workers;
    if (opcode == "send_with_imm" || opcode == "write_with_imm" || opcode == "send") {
        for (int i = 0; i < myrdma.connect_num; i++)
            workers.emplace_back(&myRDMA::rdma_send_recv, myRDMA(), i);
    } else if (opcode == "write") {
        for (int i = 0; i < myrdma.connect_num; i++)
            workers.emplace_back(&myRDMA::rdma_write_recv, myRDMA(), i);
    } else {
        cerr << "[ERROR] recv_t: unknown opcode '" << opcode << "'" << endl;
        exit(1);
    }
    for (auto& w : workers) w.join();
}

void myRDMA::t_recv(string opcode, int* nn, int /*num_of_node*/,
                    vector<double>* send, vector<double>* recv1) {
    vector<thread> workers;
    workers.reserve(myrdma.connect_num);

    if (opcode == "send_with_imm" || opcode == "write_with_imm" || opcode == "send") {
        for (int i = 0; i < myrdma.connect_num; i++)
            workers.emplace_back(&myRDMA::rdma_send_rcv, myRDMA(),
                                 i, nn, myrdma.connect_num + 1, send, recv1);
    }
    for (int i = 0; i < myrdma.connect_num; i++) {
        size_t seg = nn[i];
        send[0].insert(send[0].end(),
                       make_move_iterator(recv1[i].begin()),
                       make_move_iterator(recv1[i].begin() + seg));
        workers[i].join();
    }
}

void myRDMA::send_t(string /*opcode*/) {
    vector<thread> workers;
    workers.reserve(myrdma.connect_num);
    for (int i = 0; i < myrdma.connect_num; i++)
        workers.emplace_back(&myRDMA::rdma_send_pagerank, myRDMA(), myrdma.send[0], i);
    for (auto& w : workers) w.join();
}

void myRDMA::rdma_comm(string opcode, string msg) {
    thread sender(&myRDMA::rdma_send_msg, myRDMA(), opcode, msg);
    recv_t(opcode);
    sender.join();
}

// ── One-to-many / many-to-one wrappers ───────────────────────────────────

void myRDMA::rdma_one_to_many_send_msg(string opcode, string msg) {
    rdma_send_msg(opcode, msg);
}

void myRDMA::rdma_one_to_many_recv_msg(string opcode) {
    rdma_recv_msg(opcode);
}

void myRDMA::rdma_many_to_one_send_msg(string opcode, string msg, vector<double> msg1) {
    if (opcode == "send_with_imm")
        rdma_send_with_imm(msg, 0);
    else if (opcode == "write")
        rdma_write(msg, 0);
    else if (opcode == "write_with_imm")
        rdma_write_with_imm(msg, 0);
    else if (opcode == "send")
        rdma_send_vector(msg1, 0);
    else {
        cerr << "[ERROR] rdma_many_to_one_send_msg: unknown opcode '" << opcode << "'" << endl;
        exit(1);
    }
}

void myRDMA::rdma_many_to_one_recv_msg(string opcode) {
    recv_t(opcode);
}

// ── Connection setup ──────────────────────────────────────────────────────

void myRDMA::send_info_change_qp() {
    TCP  tcp_local;
    RDMA rdma_local;

    for (int k = 0; k < 2; k++) {
        int* clnt_socks = tcp_local.client_sock();
        cout << "[INFO] SEND RDMA INFO[" << k << "] ";

        if (k == 0) {
            for (int idx = 0; idx <= myrdma.connect_num; idx++) {
                if (clnt_socks[idx] != 0)
                    myrdma.sock_idx.push_back(idx);
            }
        }

        for (int j = 0; j < myrdma.connect_num; j++) {
            ostringstream oss;
            if (k == 0)      oss << myrdma.send[j].data();
            else if (k == 1) oss << myrdma.recv[j].data();

            tcp_local.send_msg(to_cstr(oss.str() + "\n"),                                myrdma.sock_idx[j]);
            tcp_local.send_msg(to_cstr(to_string(rdma_info1[k][j].mr->length) + "\n"),  myrdma.sock_idx[j]);
            tcp_local.send_msg(to_cstr(to_string(rdma_info1[k][j].mr->lkey)   + "\n"),  myrdma.sock_idx[j]);
            tcp_local.send_msg(to_cstr(to_string(rdma_info1[k][j].mr->rkey)   + "\n"),  myrdma.sock_idx[j]);
            tcp_local.send_msg(to_cstr(to_string(rdma_info1[k][j].lid)        + "\n"),  myrdma.sock_idx[j]);
            tcp_local.send_msg(to_cstr(to_string(rdma_info1[k][j].qp_num)     + "\n"),  myrdma.sock_idx[j]);
        }
        cout << "- SUCCESS" << endl;

        cout << "[INFO] CHANGE QUEUE PAIR STATE ";
        for (int i = 0; i < myrdma.connect_num; i++) {
            auto info = tcp_local.read_rdma_info(myrdma.sock_idx[i]);
            rdma_local.changeQueuePairStateToInit(rdma_info1[k ^ 1][i].qp);
            rdma_local.changeQueuePairStateToRTR(rdma_info1[k ^ 1][i].qp, PORT,
                                                 stoi(info.find("qp_num")->second),
                                                 stoi(info.find("lid")->second));
            if ((k ^ 1) == 0) {
                rdma_local.changeQueuePairStateToRTS(rdma_info1[k ^ 1][i].qp);
                myrdma.qp_key.emplace_back(info.find("addr")->second,
                                           info.find("rkey")->second);
            }
        }
        cout << "- SUCCESS" << endl;
    }
}

void myRDMA::create_rdma_info(vector<double>* send, vector<double>* recv) {
    RDMA rdma_local;
    cout << "[INFO] CREATE RDMA INFO ";

    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < myrdma.connect_num; i++) {
            struct ibv_context* ctx = rdma_local.createContext();
            struct ibv_pd*      pd  = ibv_alloc_pd(ctx);
            int                 cq_size = 0x10;
            struct ibv_cq*      cq  = ibv_create_cq(ctx, cq_size, nullptr, nullptr, 0);
            struct ibv_qp*      qp  = rdma_local.createQueuePair(pd, cq);

            double* buf  = (j == 0) ? send[i].data() : recv[i].data();
            size_t  bsz  = sizeof(double) * ((j == 0) ? send[i].size() : recv[i].size());
            struct ibv_mr* mr = rdma_local.registerMemoryRegion(pd, buf, bsz);

            uint16_t lid    = rdma_local.getLocalId(ctx, PORT);
            uint32_t qp_num = rdma_local.getQueuePairNumber(qp);

            rdma_info1[j].emplace_back(RdmaInfo{ctx, pd, cq_size, cq, qp, mr, lid, qp_num});
        }
    }
    cout << "- SUCCESS" << endl;
}

void myRDMA::set_buffer(char send[][buf_size], char recv[][buf_size], int num_of_server) {
    myrdma.send_buffer = &send[0];
    myrdma.recv_buffer = &recv[0];
    myrdma.connect_num = num_of_server - 1;
}

void myRDMA::initialize_rdma_connection(const char* ip, string server[],
                                        int number_of_server, int Port,
                                        char /*send*/[][buf_size], char /*recv*/[][buf_size]) {
    TCP tcp_local;
    tcp_local.connect_tcp(ip, server, number_of_server, Port);
    myrdma.connect_num = number_of_server - 1;
}

void myRDMA::initialize_rdma_connection_vector(const char* ip, string server[],
                                               int number_of_server, int Port,
                                               vector<double>* send,
                                               vector<double>* recv,
                                               int num_of_vertex) {
    TCP tcp_local;
    tcp_local.connect_tcp(ip, server, number_of_server, Port);

    myrdma.send         = &send[0];
    myrdma.recv         = &recv[0];
    myrdma.num_of_vertex = num_of_vertex;
    myrdma.connect_num  = number_of_server - 1;

    const int n  = num_of_vertex / (number_of_server - 1);
    partition  = n;
    partition1 = num_of_vertex - n * (number_of_server - 2);

    for (int i = 0; i < number_of_server - 1; i++) {
        send_adrs.push_back(send[i].data());
        recv_adrs.push_back(recv[i].data());
    }

    rdma_info1[0].reserve(100000);
    rdma_info1[1].reserve(100000);
}

void myRDMA::exit_rdma() {
    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < myrdma.connect_num; i++) {
            ibv_destroy_qp(rdma_info1[j][i].qp);
            ibv_dereg_mr(rdma_info1[j][i].mr);
            ibv_destroy_cq(rdma_info1[j][i].cq);
            ibv_dealloc_pd(rdma_info1[j][i].pd);
            ibv_close_device(rdma_info1[j][i].context);
        }
    }
}
