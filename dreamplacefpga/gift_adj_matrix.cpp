#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <numeric>

namespace py = pybind11;

// 估算总边数，用于预分配vector容量
size_t estimate_edges(
    py::array_t<int> netpin_start,
    py::array_t<int> net_mask,
    int num_nets,
    bool use_star_model,
    int star_threshold) {
    
    auto start_ptr = netpin_start.unchecked<1>();
    auto mask_ptr = net_mask.unchecked<1>();
    
    size_t estimated = 0;
    for (int net_id = 0; net_id < num_nets; ++net_id) {
        if (!mask_ptr(net_id)) continue;
        
        int degree = start_ptr(net_id + 1) - start_ptr(net_id);
        if (degree < 2) continue;
        
        if (use_star_model && degree > star_threshold) {
            estimated += degree - 1;  // 星形模型：n-1条边
        } else {
            estimated += (size_t)degree * (degree - 1) / 2;  // clique模型：n*(n-1)/2条边
        }
    }
    return estimated;
}

std::tuple<py::array_t<float>, py::array_t<int>, py::array_t<int>> 
adj_matrix_forward_optimized(
    py::array_t<int> flat_netpin,
    py::array_t<int> netpin_start,
    py::array_t<int> pin2node_map,
    py::array_t<float> net_weights,
    py::array_t<int> net_mask,
    int num_nodes,
    int max_net_size = 1000,
    bool use_star_model = true,
    int star_threshold = 100) {
    
    auto netpin_ptr = flat_netpin.unchecked<1>();
    auto start_ptr = netpin_start.unchecked<1>();
    auto pin2node_ptr = pin2node_map.unchecked<1>();
    auto weights_ptr = net_weights.unchecked<1>();
    auto mask_ptr = net_mask.unchecked<1>();
    
    int num_nets = netpin_start.size() - 1;
    
    // 预分配vector容量
    size_t estimated_edges = estimate_edges(netpin_start, net_mask, num_nets, use_star_model, star_threshold);
    estimated_edges *= 2; // 对称矩阵需要双倍容量
    
    std::vector<float> data;
    std::vector<int> rows, cols;
    data.reserve(estimated_edges);
    rows.reserve(estimated_edges);
    cols.reserve(estimated_edges);
    
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
            // 星形模型：连接到第一个节点（中心节点）
            star_nets++;
            float star_weight = 2.0f * net_weight / nodes.size();
            int center_node = nodes[0];
            
            for (size_t i = 1; i < nodes.size(); ++i) {
                int leaf_node = nodes[i];
                
                // 添加对称边 (center -> leaf) 和 (leaf -> center)
                data.push_back(star_weight);
                rows.push_back(center_node);
                cols.push_back(leaf_node);
                
                data.push_back(star_weight);
                rows.push_back(leaf_node);
                cols.push_back(center_node);
            }
        } else {
            // clique模型：全连接
            clique_nets++;
            float edge_weight = 2.0f * net_weight / nodes.size();
            
            for (size_t i = 0; i < nodes.size(); ++i) {
                for (size_t j = i + 1; j < nodes.size(); ++j) {
                    // 添加对称边 (i -> j) 和 (j -> i)
                    data.push_back(edge_weight);
                    rows.push_back(nodes[i]);
                    cols.push_back(nodes[j]);
                    
                    data.push_back(edge_weight);
                    rows.push_back(nodes[j]);
                    cols.push_back(nodes[i]);
                }
            }
        }
    }
    
    std::cout << "网络处理统计: CLIQUE=" << clique_nets 
              << ", STAR=" << star_nets 
              << ", SKIPPED=" << skipped_nets 
              << ", 总边数=" << data.size() / 2 << std::endl;
    
    if (data.empty()) {
        // 返回空矩阵
        auto empty_data = py::array_t<float>(0);
        auto empty_rows = py::array_t<int>(0);
        auto empty_cols = py::array_t<int>(0);
        return std::make_tuple(empty_data, empty_rows, empty_cols);
    }
    
    // 创建numpy数组
    auto data_array = py::array_t<float>(data.size());
    auto rows_array = py::array_t<int>(rows.size());
    auto cols_array = py::array_t<int>(cols.size());
    
    auto data_ptr = data_array.mutable_unchecked<1>();
    auto rows_ptr = rows_array.mutable_unchecked<1>();
    auto cols_ptr = cols_array.mutable_unchecked<1>();
    
    // 复制数据
    for (size_t i = 0; i < data.size(); ++i) {
        data_ptr(i) = data[i];
        rows_ptr(i) = rows[i];
        cols_ptr(i) = cols[i];
    }
    
    return std::make_tuple(data_array, rows_array, cols_array);
}

PYBIND11_MODULE(gift_adj_cpp, m) {
    m.doc() = "Optimized adjacency matrix builder for GiFt with improved performance";
    
    m.def("adj_matrix_forward", &adj_matrix_forward_optimized,
          "Build adjacency matrix with star model for large nets - optimized version",
          py::arg("flat_netpin"), py::arg("netpin_start"), py::arg("pin2node_map"),
          py::arg("net_weights"), py::arg("net_mask"), py::arg("num_nodes"),
          py::arg("max_net_size") = 1000, py::arg("use_star_model") = true, 
          py::arg("star_threshold") = 100);
}