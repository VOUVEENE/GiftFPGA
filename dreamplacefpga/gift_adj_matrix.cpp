#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <iostream>

namespace py = pybind11;

std::tuple<py::array_t<float>, py::array_t<int>, py::array_t<int>> 
adj_matrix_forward_optimized(
    py::array_t<int> flat_netpin,
    py::array_t<int> netpin_start,
    py::array_t<int> pin2node_map,
    py::array_t<float> net_weights,
    py::array_t<int> net_mask,
    int num_nodes,
    int max_net_size = 1000,     // 超过此大小的网络跳过
    bool use_star_model = true,  // 是否对大网络使用星形模型
    int star_threshold = 100) {  // 星形模型的阈值
    
    auto netpin_ptr = flat_netpin.unchecked<1>();
    auto start_ptr = netpin_start.unchecked<1>();
    auto pin2node_ptr = pin2node_map.unchecked<1>();
    auto weights_ptr = net_weights.unchecked<1>();
    auto mask_ptr = net_mask.unchecked<1>();
    
    int num_nets = netpin_start.size() - 1;
    
    // 使用map存储边权重（只存储上三角部分，避免重复）
    std::unordered_map<long long, float> edge_weights;
    
    int clique_nets = 0, star_nets = 0, skipped_nets = 0;
    
    for (int net_id = 0; net_id < num_nets; ++net_id) {
        if (!mask_ptr(net_id)) continue;
        
        int pin_start = start_ptr(net_id);
        int pin_end = start_ptr(net_id + 1);
        int num_pins = pin_end - pin_start;
        
        if (num_pins < 2) continue;
        
        // 收集有效节点
        std::vector<int> nodes;
        nodes.reserve(num_pins);
        for (int j = pin_start; j < pin_end; ++j) {
            int flat_pin_id = netpin_ptr(j);
            int node_id = pin2node_ptr(flat_pin_id);
            if (node_id < num_nodes) {
                nodes.push_back(node_id);
            }
        }
        
        if (nodes.size() < 2) continue;
        
        // 跳过超大网络
        if (nodes.size() > max_net_size) {
            skipped_nets++;
            continue;
        }
        
        float net_weight = weights_ptr(net_id);
        
        if (use_star_model && nodes.size() > star_threshold) {
            // 星形模型：只连接到第一个节点（中心节点）
            star_nets++;
            float star_weight = 2.0f * net_weight / nodes.size();
            int center_node = nodes[0];
            
            for (size_t i = 1; i < nodes.size(); ++i) {
                int leaf_node = nodes[i];
                int n1 = std::min(center_node, leaf_node);
                int n2 = std::max(center_node, leaf_node);
                
                long long key = ((long long)n1 << 32) | n2;
                edge_weights[key] += star_weight;
            }
        } else {
            // clique模型：全连接
            clique_nets++;
            float edge_weight = 2.0f * net_weight / nodes.size();
            
            for (size_t i = 0; i < nodes.size(); ++i) {
                for (size_t j = i + 1; j < nodes.size(); ++j) {
                    int n1 = nodes[i], n2 = nodes[j];
                    if (n1 > n2) std::swap(n1, n2);
                    
                    long long key = ((long long)n1 << 32) | n2;
                    edge_weights[key] += edge_weight;
                }
            }
        }
    }
    
    std::cout << "网络处理统计: CLIQUE=" << clique_nets 
              << ", STAR=" << star_nets 
              << ", SKIPPED=" << skipped_nets 
              << ", 总边数=" << edge_weights.size() << std::endl;
    
    if (edge_weights.empty()) {
        // 返回空矩阵
        auto data = py::array_t<float>(0);
        auto rows = py::array_t<int>(0);
        auto cols = py::array_t<int>(0);
        return std::make_tuple(data, rows, cols);
    }
    
    // 转换为COO格式（对称矩阵）
    size_t nnz = edge_weights.size() * 2;
    auto data = py::array_t<float>(nnz);
    auto rows = py::array_t<int>(nnz);
    auto cols = py::array_t<int>(nnz);
    
    auto data_ptr = data.mutable_unchecked<1>();
    auto rows_ptr = rows.mutable_unchecked<1>();
    auto cols_ptr = cols.mutable_unchecked<1>();
    
    size_t idx = 0;
    for (const auto& edge : edge_weights) {
        long long key = edge.first;
        int i = (int)(key >> 32);
        int j = (int)(key & 0xFFFFFFFF);
        float w = edge.second;
        
        // 添加(i,j)和(j,i)
        data_ptr(idx) = w; rows_ptr(idx) = i; cols_ptr(idx) = j; idx++;
        data_ptr(idx) = w; rows_ptr(idx) = j; cols_ptr(idx) = i; idx++;
    }
    
    return std::make_tuple(data, rows, cols);
}

PYBIND11_MODULE(gift_adj_cpp, m) {
    m.doc() = "Optimized adjacency matrix builder for GiFt with large net handling";
    
    m.def("adj_matrix_forward", &adj_matrix_forward_optimized,
          "Build adjacency matrix with star model for large nets",
          py::arg("flat_netpin"), py::arg("netpin_start"), py::arg("pin2node_map"),
          py::arg("net_weights"), py::arg("net_mask"), py::arg("num_nodes"),
          py::arg("max_net_size") = 1000, py::arg("use_star_model") = true, 
          py::arg("star_threshold") = 100);
}