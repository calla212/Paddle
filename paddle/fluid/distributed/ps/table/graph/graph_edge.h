// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <set>

namespace paddle {
namespace distributed {

class GraphEdgeBlob {
 public:
  GraphEdgeBlob() {}
  virtual ~GraphEdgeBlob() {}
  size_t size() { return id_arr.size(); }
  virtual void add_edge(int64_t id, float weight);
  int64_t get_id(int idx) { return id_arr[idx]; }
  virtual float get_weight(int idx) { return 1; }
  std::vector<int64_t>& export_id_array() { return id_arr; }
  // virtual void set_unique() {
  //   std::sort(id_arr.begin(), id_arr.end());
  //   id_arr.erase(std::unique(id_arr.begin(), id_arr.end()), id_arr.end());
  // }

 protected:
  std::vector<int64_t> id_arr;
};

class WeightedGraphEdgeBlob : public GraphEdgeBlob {
 public:
  WeightedGraphEdgeBlob() {}
  virtual ~WeightedGraphEdgeBlob() {}
  virtual void add_edge(int64_t id, float weight);
  virtual float get_weight(int idx) { return weight_arr[idx]; }
  // virtual void set_unique() {
  //   if (id_arr.empty()) return;

  //   std::set<int64_t> id_set;
  //   decltype(id_arr) _id_arr(id_arr);
  //   decltype(weight_arr) _weight_arr(weight_arr);
  //   id_arr.clear();
  //   weight_arr.clear();

  //   for (size_t i = 0; i < _id_arr.size(); ++i) {
  //     if (id_set.find(_id_arr[i]) == id_set.end()) {
  //       id_set.insert(_id_arr[i]);
  //       id_arr.push_back(_id_arr[i]);
  //       weight_arr.push_back(_weight_arr[i]);
  //     }
  //   }
  // }

 protected:
  std::vector<float> weight_arr;
};
}  // namespace distributed
}  // namespace paddle
