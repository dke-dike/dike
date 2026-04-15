/**
 * @file tcp.cpp
 * @brief TCP utility functions for RDMA connection bootstrapping
 *
 * TCP is used only during the control-path (QP info exchange / IP detection).
 * Data-path communication uses RDMA exclusively.
 *
 * Supported NIC names for IP detection (check_my_ip):
 *   ibp2s0, ibs6, enp5s0, enp6s0
 * Add additional interface names to check_my_ip() as needed.
 */

#include "tcp.hpp"

static std::mutex mutx;
static TCP* tcp_instance = new TCP();

// ── Network utilities ─────────────────────────────────────────────────────

string TCP::domain_to_ip(string domain) {
    char domain_ch[256];
    strncpy(domain_ch, domain.c_str(), sizeof(domain_ch) - 1);

    struct hostent* host;
    char* result_str = nullptr;
    int   retries    = 0;

    while (true) {
        host = gethostbyname(domain_ch);
        if (host) {
            for (int i = 0; host->h_addr_list[i]; i++)
                result_str = inet_ntoa(*(struct in_addr*)host->h_addr_list[i]);
            return string(result_str);
        }
        if (retries == 0) printf("[INFO] Resolving %s...\n", domain_ch);
        retries = 1;
    }
}

/**
 * Returns the IP address of the InfiniBand or Ethernet interface used
 * for RDMA communication. Extend the if-chain below for additional NICs.
 */
string TCP::check_my_ip() {
    // Supported RDMA-capable NIC names
    const char* rdma_nics[] = { "ibp2s0", "ibs6", "enp5s0", "enp6s0", nullptr };

    struct ifaddrs* ifaddr = nullptr;
    getifaddrs(&ifaddr);

    for (struct ifaddrs* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (!ifa->ifa_addr || ifa->ifa_addr->sa_family != AF_INET) continue;

        for (int i = 0; rdma_nics[i] != nullptr; i++) {
            if (strcmp(ifa->ifa_name, rdma_nics[i]) == 0) {
                char buf[INET_ADDRSTRLEN];
                void* addr = &((struct sockaddr_in*)ifa->ifa_addr)->sin_addr;
                inet_ntop(AF_INET, addr, buf, INET_ADDRSTRLEN);
                if (ifaddr) freeifaddrs(ifaddr);
                return string(buf);
            }
        }
    }
    if (ifaddr) freeifaddrs(ifaddr);
    return "error";
}

// ── Message I/O ───────────────────────────────────────────────────────────

void TCP::send_msg(const char* msg, int peer_idx) {
    lock_guard<mutex> lock(mutx);
    write(tcp_instance->clnt_socks[peer_idx], msg, strlen(msg));
}

string TCP::recv_message(int peer_idx) {
    char buf[104857] = {};
    read(tcp_instance->new_sock[peer_idx], buf, sizeof(buf) - 1);
    return string(buf);
}

int TCP::recv_msg(int peer_idx) {
    char buf[100];
    return read(tcp_instance->new_sock[peer_idx], buf, sizeof(buf));
}

// ── Server thread ─────────────────────────────────────────────────────────

void TCP::server(string server[]) {
    int opt = 1;
    serv_sock = socket(PF_INET, SOCK_STREAM, 0);
    setsockopt(serv_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    memset(&serv_adr, 0, sizeof(serv_adr));
    serv_adr.sin_family      = AF_INET;
    serv_adr.sin_addr.s_addr = htonl(INADDR_ANY);
    serv_adr.sin_port        = htons(tcp_instance->Port);

    if (bind(serv_sock, (struct sockaddr*)&serv_adr, sizeof(serv_adr)) == -1) {
        cerr << "[ERROR] TCP bind() failed" << endl;
        exit(1);
    }
    if (listen(serv_sock, 5) == -1) {
        cerr << "[ERROR] TCP listen() failed" << endl;
        exit(1);
    }

    while (tcp_instance->clnt_cnt < tcp_instance->num_of_server - 1) {
        socklen_t adr_sz = sizeof(clnt_adr);
        clnt_sock = accept(serv_sock, (struct sockaddr*)&clnt_adr, &adr_sz);
        if (clnt_sock == -1) {
            fprintf(stderr, "[ERROR] accept() failed for %s\n",
                    inet_ntoa(clnt_adr.sin_addr));
            continue;
        }
        lock_guard<mutex> lock(mutx);
        for (int i = 0; i < tcp_instance->num_of_server; i++) {
            if (string(inet_ntoa(clnt_adr.sin_addr)) == server[i])
                tcp_instance->clnt_socks[i] = clnt_sock;
        }
        tcp_instance->clnt_cnt++;
        printf("[INFO] %s: CONNECTION SUCCESS\n", inet_ntoa(clnt_adr.sin_addr));
    }
}

// ── Client thread ─────────────────────────────────────────────────────────

void TCP::client(string ip, int idx) {
    sock = socket(PF_INET, SOCK_STREAM, 0);
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family      = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(ip.c_str());
    serv_addr.sin_port        = htons(tcp_instance->Port);
    while (connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) == -1);
    tcp_instance->new_sock[idx] = sock;
}

void TCP::client_t(const char* ip, string server[]) {
    for (int i = 0; i < tcp_instance->num_of_server; i++) {
        if (string(ip) != server[i])
            workers.emplace_back(&TCP::client, TCP(), server[i], i);
    }
    for (auto& w : workers) w.join();
}

void TCP::connect_tcp(const char* ip, string server[], int number_of_server, int Port) {
    tcp_instance->num_of_server = number_of_server;
    tcp_instance->Port          = Port;

    thread srv(&TCP::server, TCP(), server);
    client_t(ip, server);
    srv.join();
}

// ── RDMA info exchange ────────────────────────────────────────────────────

map<string, string> TCP::read_rdma_info(int peer_idx) {
    static const string keys[] = { "addr", "len", "lkey", "rkey", "lid", "qp_num" };
    map<string, string> info;

    for (const auto& key : keys) {
        string value;
        char ch[2] = {};
        while (value.empty() || value.back() != '\n') {
            if (read(tcp_instance->new_sock[peer_idx], ch, 1) > 0)
                value += ch[0];
        }
        info[key] = value;
    }
    return info;
}

int* TCP::client_sock() {
    return tcp_instance->clnt_socks;
}
