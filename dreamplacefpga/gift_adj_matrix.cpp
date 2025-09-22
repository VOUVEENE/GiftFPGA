#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <unordered_map>

namespace py = pybind11;

std::tuple<py::array_t<float>, py::array_t<int>, py::array_t<int>> 
adj_matrix_forward(
    py::array_t<int> flat_netpin,
    py::array_t<int> netpin_start,
    py::array_t<int> pin2node_map,
    py::array_t<float> net_weights,
    py::array_t<int> net_mask,
    int num_nodes) {
    
    auto netpin_ptr = flat_netpin.unchecked<1>();
    auto start_ptr = netpin_start.unchecked<1>();
    auto pin2node_ptr = pin2node_map.unchecked<1>();
    auto weights_ptr = net_weights.unchecked<1>();
    auto mask_ptr = net_mask.unchecked<1>();
    
    int num_nets = netpin_start.size() - 1;
    std::unordered_map<long long, float> edge_weights;
    
    for (int net_id = 0; net_id < num_nets; ++net_id) {
        if (!mask_ptr(net_id)) continue;
        
        int pin_start = start_ptr(net_id);
        int pin_end = start_ptr(net_id + 1);
        int num_pins = pin_end - pin_start;
        
        if (num_pins < 2) continue;
        
        float net_weight = weights_ptr(net_id);
        float edge_weight = 2.0f * net_weight / num_pins;
        
        std::vector<int> nodes;
        for (int j = pin_start; j < pin_end; ++j) {
            int flat_pin_id = netpin_ptr(j);
            int node_id = pin2node_ptr(flat_pin_id);
            if (node_id < num_nodes) {
                nodes.push_back(node_id);
            }
        }
        
        for (size_t i = 0; i < nodes.size(); ++i) {
            for (size_t j = i + 1; j < nodes.size(); ++j) {
                int n1 = nodes[i], n2 = nodes[j];
                if (n1 > n2) std::swap(n1, n2);
                
                long long key = ((long long)n1 << 32) | n2;
                edge_weights[key] += edge_weight;
            }
        }
    }
    
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
        
        data_ptr(idx) = w; rows_ptr(idx) = i; cols_ptr(idx) = j; idx++;
        data_ptr(idx) = w; rows_ptr(idx) = j; cols_ptr(idx) = i; idx++;
    }
    
    return std::make_tuple(data, rows, cols);
}

PYBIND11_MODULE(gift_adj_cpp, m) {
    m.doc() = "GiFt adjacency matrix builder";
    m.def("adj_matrix_forward", &adj_matrix_forward, "Build adjacency matrix");
}
