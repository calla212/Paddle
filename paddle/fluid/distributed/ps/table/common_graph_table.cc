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

#include "paddle/fluid/distributed/ps/table/common_graph_table.h"

#include <time.h>

#include <algorithm>
#include <chrono>
#include <set>
#include <sstream>
#include <metis.h>

#include "gflags/gflags.h"
#include "paddle/fluid/distributed/common/utils.h"
#include "paddle/fluid/distributed/ps/table/graph/graph_node.h"
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/io/fs.h"
#include "paddle/fluid/platform/timer.h"
#include "paddle/fluid/string/printf.h"
#include "paddle/fluid/string/string_helper.h"

DECLARE_bool(graph_load_in_parallel);
DECLARE_bool(graph_get_neighbor_id);

namespace paddle {
namespace distributed {

#ifdef PADDLE_WITH_HETERPS
int32_t GraphTable::Load_to_ssd(const std::string &path,
                                const std::string &param) {
  bool load_edge = (param[0] == 'e');
  bool load_node = (param[0] == 'n');
  if (load_edge) {
    bool reverse_edge = (param[1] == '<');
    std::string edge_type = param.substr(2);
    return this->load_edges_to_ssd(path, reverse_edge, edge_type);
  }
  if (load_node) {
    std::string node_type = param.substr(1);
    return this->load_nodes(path, node_type);
  }
  return 0;
}

paddle::framework::GpuPsCommGraphFea GraphTable::make_gpu_ps_graph_fea(
    std::vector<uint64_t> &node_ids, int slot_num) {
  std::vector<std::vector<uint64_t>> bags(task_pool_size_);
  for (int i = 0; i < task_pool_size_; i++) {
    auto predsize = node_ids.size() / task_pool_size_;
    bags[i].reserve(predsize * 1.2);
  }

  for (auto x : node_ids) {
    int location = x % shard_num % task_pool_size_;
    bags[location].push_back(x);
  }

  std::vector<std::future<int>> tasks;
  std::vector<uint64_t> feature_array[task_pool_size_];
  std::vector<uint8_t> slot_id_array[task_pool_size_];
  std::vector<uint64_t> node_id_array[task_pool_size_];
  std::vector<paddle::framework::GpuPsFeaInfo>
      node_fea_info_array[task_pool_size_];
  slot_feature_num_map_.resize(slot_num);
  for (int k = 0; k < slot_num; ++k) {
    slot_feature_num_map_[k] = 0;
  }

  for (size_t i = 0; i < bags.size(); i++) {
    if (bags[i].size() > 0) {
      tasks.push_back(_shards_task_pool[i]->enqueue([&, i, this]() -> int {
        uint64_t node_id;
        paddle::framework::GpuPsFeaInfo x;
        std::vector<uint64_t> feature_ids;
        for (size_t j = 0; j < bags[i].size(); j++) {
          // TODO use FEATURE_TABLE instead
          Node *v = find_node(1, bags[i][j]);
          node_id = bags[i][j];
          if (v == NULL) {
            x.feature_size = 0;
            x.feature_offset = 0;
            node_fea_info_array[i].push_back(x);
          } else {
            // x <- v
            x.feature_offset = feature_array[i].size();
            int total_feature_size = 0;
            for (int k = 0; k < slot_num; ++k) {
              v->get_feature_ids(k, &feature_ids);
              int feature_ids_size = feature_ids.size();
              if (slot_feature_num_map_[k] < feature_ids_size) {
                slot_feature_num_map_[k] = feature_ids_size;
              }
              total_feature_size += feature_ids_size;
              if (!feature_ids.empty()) {
                feature_array[i].insert(feature_array[i].end(),
                                        feature_ids.begin(),
                                        feature_ids.end());
                slot_id_array[i].insert(
                    slot_id_array[i].end(), feature_ids_size, k);
              }
            }
            x.feature_size = total_feature_size;
            node_fea_info_array[i].push_back(x);
          }
          node_id_array[i].push_back(node_id);
        }
        return 0;
      }));
    }
  }
  for (int i = 0; i < (int)tasks.size(); i++) tasks[i].get();

  std::stringstream ss;
  for (int k = 0; k < slot_num; ++k) {
    ss << slot_feature_num_map_[k] << " ";
  }
  VLOG(0) << "slot_feature_num_map: " << ss.str();

  paddle::framework::GpuPsCommGraphFea res;
  uint64_t tot_len = 0;
  for (int i = 0; i < task_pool_size_; i++) {
    tot_len += feature_array[i].size();
  }
  VLOG(0) << "Loaded feature table on cpu, feature_list_size[" << tot_len
          << "] node_ids_size[" << node_ids.size() << "]";
  res.init_on_cpu(tot_len, (unsigned int)node_ids.size(), slot_num);
  unsigned int offset = 0, ind = 0;
  for (int i = 0; i < task_pool_size_; i++) {
    for (int j = 0; j < (int)node_id_array[i].size(); j++) {
      res.node_list[ind] = node_id_array[i][j];
      res.fea_info_list[ind] = node_fea_info_array[i][j];
      res.fea_info_list[ind++].feature_offset += offset;
    }
    for (size_t j = 0; j < feature_array[i].size(); j++) {
      res.feature_list[offset + j] = feature_array[i][j];
      res.slot_id_list[offset + j] = slot_id_array[i][j];
    }
    offset += feature_array[i].size();
  }
  return res;
}

paddle::framework::GpuPsCommGraph GraphTable::make_gpu_ps_graph(
    int idx, std::vector<uint64_t> ids) {
  std::vector<std::vector<uint64_t>> bags(task_pool_size_);
  for (int i = 0; i < task_pool_size_; i++) {
    auto predsize = ids.size() / task_pool_size_;
    bags[i].reserve(predsize * 1.2);
  }
  for (auto x : ids) {
    int location = x % shard_num % task_pool_size_;
    bags[location].push_back(x);
  }

  std::vector<std::future<int>> tasks;
  std::vector<uint64_t> node_array[task_pool_size_];  // node id list
  std::vector<paddle::framework::GpuPsNodeInfo> info_array[task_pool_size_];
  std::vector<uint64_t> edge_array[task_pool_size_];  // edge id list

  for (size_t i = 0; i < bags.size(); i++) {
    if (bags[i].size() > 0) {
      tasks.push_back(_shards_task_pool[i]->enqueue([&, i, this]() -> int {
        node_array[i].resize(bags[i].size());
        info_array[i].resize(bags[i].size());
        edge_array[i].reserve(bags[i].size());

        for (size_t j = 0; j < bags[i].size(); j++) {
          auto node_id = bags[i][j];
          node_array[i][j] = node_id;
          Node *v = find_node(0, idx, node_id);
          if (v != nullptr) {
            info_array[i][j].neighbor_offset = edge_array[i].size();
            info_array[i][j].neighbor_size = v->get_neighbor_size();
            for (size_t k = 0; k < v->get_neighbor_size(); k++) {
              edge_array[i].push_back(v->get_neighbor_id(k));
            }
          } else {
            info_array[i][j].neighbor_offset = 0;
            info_array[i][j].neighbor_size = 0;
          }
        }
        return 0;
      }));
    }
  }
  for (int i = 0; i < (int)tasks.size(); i++) tasks[i].get();

  int64_t tot_len = 0;
  for (int i = 0; i < task_pool_size_; i++) {
    tot_len += edge_array[i].size();
  }

  paddle::framework::GpuPsCommGraph res;
  res.init_on_cpu(tot_len, ids.size());
  int64_t offset = 0, ind = 0;
  for (int i = 0; i < task_pool_size_; i++) {
    for (int j = 0; j < (int)node_array[i].size(); j++) {
      res.node_list[ind] = node_array[i][j];
      res.node_info_list[ind] = info_array[i][j];
      res.node_info_list[ind++].neighbor_offset += offset;
    }
    for (size_t j = 0; j < edge_array[i].size(); j++) {
      res.neighbor_list[offset + j] = edge_array[i][j];
    }
    offset += edge_array[i].size();
  }
  return res;
}

int32_t GraphTable::add_node_to_ssd(
    int type_id, int idx, uint64_t src_id, char *data, int len) {
  if (_db != NULL) {
    char ch[sizeof(int) * 2 + sizeof(uint64_t)];
    memcpy(ch, &type_id, sizeof(int));
    memcpy(ch + sizeof(int), &idx, sizeof(int));
    memcpy(ch + sizeof(int) * 2, &src_id, sizeof(uint64_t));
    std::string str;
    if (_db->get(src_id % shard_num % task_pool_size_,
                 ch,
                 sizeof(int) * 2 + sizeof(uint64_t),
                 str) == 0) {
      uint64_t *stored_data = ((uint64_t *)str.c_str());
      int n = str.size() / sizeof(uint64_t);
      char *new_data = new char[n * sizeof(uint64_t) + len];
      memcpy(new_data, stored_data, n * sizeof(uint64_t));
      memcpy(new_data + n * sizeof(uint64_t), data, len);
      _db->put(src_id % shard_num % task_pool_size_,
               ch,
               sizeof(int) * 2 + sizeof(uint64_t),
               (char *)new_data,
               n * sizeof(uint64_t) + len);
      delete[] new_data;
    } else {
      _db->put(src_id % shard_num % task_pool_size_,
               ch,
               sizeof(int) * 2 + sizeof(uint64_t),
               (char *)data,
               len);
    }
  }
  return 0;
}
char *GraphTable::random_sample_neighbor_from_ssd(
    int idx,
    uint64_t id,
    int sample_size,
    const std::shared_ptr<std::mt19937_64> rng,
    int &actual_size) {
  if (_db == NULL) {
    actual_size = 0;
    return NULL;
  }
  std::string str;
  VLOG(2) << "sample ssd for key " << id;
  char ch[sizeof(int) * 2 + sizeof(uint64_t)];
  memset(ch, 0, sizeof(int));
  memcpy(ch + sizeof(int), &idx, sizeof(int));
  memcpy(ch + sizeof(int) * 2, &id, sizeof(uint64_t));
  if (_db->get(id % shard_num % task_pool_size_,
               ch,
               sizeof(int) * 2 + sizeof(uint64_t),
               str) == 0) {
    uint64_t *data = ((uint64_t *)str.c_str());
    int n = str.size() / sizeof(uint64_t);
    std::unordered_map<int, int> m;
    // std::vector<uint64_t> res;
    int sm_size = std::min(n, sample_size);
    actual_size = sm_size * Node::id_size;
    char *buff = new char[actual_size];
    for (int i = 0; i < sm_size; i++) {
      std::uniform_int_distribution<int> distrib(0, n - i - 1);
      int t = distrib(*rng);
      // int t = rand() % (n-i);
      int pos = 0;
      auto iter = m.find(t);
      if (iter != m.end()) {
        pos = iter->second;
      } else {
        pos = t;
      }
      auto iter2 = m.find(n - i - 1);

      int key2 = iter2 == m.end() ? n - i - 1 : iter2->second;
      m[t] = key2;
      m.erase(n - i - 1);
      memcpy(buff + i * Node::id_size, &data[pos], Node::id_size);
      // res.push_back(data[pos]);
    }
    for (int i = 0; i < actual_size; i += 8) {
      VLOG(2) << "sampled an neighbor " << *(uint64_t *)&buff[i];
    }
    return buff;
  }
  actual_size = 0;
  return NULL;
}

int64_t GraphTable::load_graph_to_memory_from_ssd(int idx,
                                                  std::vector<uint64_t> &ids) {
  std::vector<std::vector<uint64_t>> bags(task_pool_size_);
  for (auto x : ids) {
    int location = x % shard_num % task_pool_size_;
    bags[location].push_back(x);
  }
  std::vector<std::future<int>> tasks;
  std::vector<int64_t> count(task_pool_size_, 0);
  for (size_t i = 0; i < bags.size(); i++) {
    if (bags[i].size() > 0) {
      tasks.push_back(_shards_task_pool[i]->enqueue([&, i, idx, this]() -> int {
        char ch[sizeof(int) * 2 + sizeof(uint64_t)];
        memset(ch, 0, sizeof(int));
        memcpy(ch + sizeof(int), &idx, sizeof(int));
        for (size_t k = 0; k < bags[i].size(); k++) {
          auto v = bags[i][k];
          memcpy(ch + sizeof(int) * 2, &v, sizeof(uint64_t));
          std::string str;
          if (_db->get(i, ch, sizeof(int) * 2 + sizeof(uint64_t), str) == 0) {
            count[i] += (int64_t)str.size();
            for (size_t j = 0; j < (int)str.size(); j += sizeof(uint64_t)) {
              uint64_t id = *(uint64_t *)(str.c_str() + j);
              add_comm_edge(idx, v, id);
            }
          }
        }
        return 0;
      }));
    }
  }

  for (int i = 0; i < (int)tasks.size(); i++) tasks[i].get();
  int64_t tot = 0;
  for (auto x : count) tot += x;
  return tot;
}

void GraphTable::make_partitions(int idx, int64_t byte_size, int device_len) {
  VLOG(2) << "start to make graph partitions , byte_size = " << byte_size
          << " total memory cost = " << total_memory_cost;
  if (total_memory_cost == 0) {
    VLOG(0) << "no edges are detected,make partitions exits";
    return;
  }
  auto &weight_map = node_weight[0][idx];
  const double a = 2.0, y = 1.25, weight_param = 1.0;
  int64_t gb_size_by_discount = byte_size * 0.8 * device_len;
  if (gb_size_by_discount <= 0) gb_size_by_discount = 1;
  int part_len = total_memory_cost / gb_size_by_discount;
  if (part_len == 0) part_len = 1;

  VLOG(2) << "part_len = " << part_len
          << " byte size = " << gb_size_by_discount;
  partitions[idx].clear();
  partitions[idx].resize(part_len);
  std::vector<double> weight_cost(part_len, 0);
  std::vector<int64_t> memory_remaining(part_len, gb_size_by_discount);
  std::vector<double> score(part_len, 0);
  std::unordered_map<uint64_t, int> id_map;
  std::vector<rocksdb::Iterator *> iters;
  for (int i = 0; i < task_pool_size_; i++) {
    iters.push_back(_db->get_iterator(i));
    iters[i]->SeekToFirst();
  }
  int next = 0;
  while (iters.size()) {
    if (next >= (int)iters.size()) {
      next = 0;
    }
    if (!iters[next]->Valid()) {
      iters.erase(iters.begin() + next);
      continue;
    }
    std::string key = iters[next]->key().ToString();
    int type_idx = *(int *)key.c_str();
    int temp_idx = *(int *)(key.c_str() + sizeof(int));
    if (type_idx != 0 || temp_idx != idx) {
      iters[next]->Next();
      next++;
      continue;
    }
    std::string value = iters[next]->value().ToString();
    std::uint64_t i_key = *(uint64_t *)(key.c_str() + sizeof(int) * 2);
    for (int i = 0; i < part_len; i++) {
      if (memory_remaining[i] < (int64_t)value.size()) {
        score[i] = -100000.0;
      } else {
        score[i] = 0;
      }
    }
    for (size_t j = 0; j < (int)value.size(); j += sizeof(uint64_t)) {
      uint64_t v = *((uint64_t *)(value.c_str() + j));
      int index = -1;
      if (id_map.find(v) != id_map.end()) {
        index = id_map[v];
        score[index]++;
      }
    }
    double base, weight_base = 0;
    double w = 0;
    bool has_weight = false;
    if (weight_map.find(i_key) != weight_map.end()) {
      w = weight_map[i_key];
      has_weight = true;
    }
    int index = 0;
    for (int i = 0; i < part_len; i++) {
      base = gb_size_by_discount - memory_remaining[i] + value.size();
      if (has_weight)
        weight_base = weight_cost[i] + w * weight_param;
      else {
        weight_base = 0;
      }
      score[i] -= a * y * std::pow(1.0 * base, y - 1) + weight_base;
      if (score[i] > score[index]) index = i;
      VLOG(2) << "score" << i << " = " << score[i] << " memory left "
              << memory_remaining[i];
    }
    id_map[i_key] = index;
    partitions[idx][index].push_back(i_key);
    memory_remaining[index] -= (int64_t)value.size();
    if (has_weight) weight_cost[index] += w;
    iters[next]->Next();
    next++;
  }
  for (int i = 0; i < part_len; i++) {
    if (partitions[idx][i].size() == 0) {
      partitions[idx].erase(partitions[idx].begin() + i);
      i--;
      part_len--;
      continue;
    }
    VLOG(2) << " partition " << i << " size = " << partitions[idx][i].size();
    for (auto x : partitions[idx][i]) {
      VLOG(2) << "find a id " << x;
    }
  }
  next_partition = 0;
}

void GraphTable::export_partition_files(int idx, std::string file_path) {
  int part_len = partitions[idx].size();
  if (part_len == 0) return;
  if (file_path == "") file_path = ".";
  if (file_path[(int)file_path.size() - 1] != '/') {
    file_path += "/";
  }
  std::vector<std::future<int>> tasks;
  for (int i = 0; i < part_len; i++) {
    tasks.push_back(_shards_task_pool[i % task_pool_size_]->enqueue(
        [&, i, idx, this]() -> int {
          std::string output_path =
              file_path + "partition_" + std::to_string(i);

          std::ofstream ofs(output_path);
          if (ofs.fail()) {
            VLOG(0) << "creating " << output_path << " failed";
            return 0;
          }
          for (auto x : partitions[idx][i]) {
            auto str = std::to_string(x);
            ofs.write(str.c_str(), str.size());
            ofs.write("\n", 1);
          }
          ofs.close();
          return 0;
        }));
  }

  for (int i = 0; i < (int)tasks.size(); i++) tasks[i].get();
}
void GraphTable::clear_graph(int idx) {
  for (auto p : edge_shards[idx]) {
    delete p;
  }

  edge_shards[idx].clear();
  for (size_t i = 0; i < shard_num_per_server; i++) {
    edge_shards[idx].push_back(new GraphShard());
  }
}
int32_t GraphTable::load_next_partition(int idx) {
  if (next_partition >= (int)partitions[idx].size()) {
    VLOG(0) << "partition iteration is done";
    return -1;
  }
  clear_graph(idx);
  load_graph_to_memory_from_ssd(idx, partitions[idx][next_partition]);
  next_partition++;
  return 0;
}
int32_t GraphTable::load_edges_to_ssd(const std::string &path,
                                      bool reverse_edge,
                                      const std::string &edge_type) {
  int idx = 0;
  if (edge_type == "") {
    VLOG(0) << "edge_type not specified, loading edges to " << id_to_edge[0]
            << " part";
  } else {
    if (edge_to_id.find(edge_type) == edge_to_id.end()) {
      VLOG(0) << "edge_type " << edge_type
              << " is not defined, nothing will be loaded";
      return 0;
    }
    idx = edge_to_id[edge_type];
  }
  total_memory_cost = 0;
  auto paths = paddle::string::split_string<std::string>(path, ";");
  int64_t count = 0;
  std::string sample_type = "random";
  for (auto path : paths) {
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
      VLOG(0) << "get a line from file " << line;
      auto values = paddle::string::split_string<std::string>(line, "\t");
      count++;
      if (values.size() < 2) continue;
      auto src_id = std::stoll(values[0]);
      auto dist_ids = paddle::string::split_string<std::string>(values[1], ";");
      std::vector<uint64_t> dist_data;
      for (auto x : dist_ids) {
        dist_data.push_back(std::stoll(x));
        total_memory_cost += sizeof(uint64_t);
      }
      add_node_to_ssd(0,
                      idx,
                      src_id,
                      (char *)dist_data.data(),
                      (int)(dist_data.size() * sizeof(uint64_t)));
    }
  }
  VLOG(0) << "total memory cost = " << total_memory_cost << " bytes";
  return 0;
}

int32_t GraphTable::dump_edges_to_ssd(int idx) {
  VLOG(2) << "calling dump edges to ssd";
  std::vector<std::future<int64_t>> tasks;
  auto &shards = edge_shards[idx];
  for (size_t i = 0; i < shards.size(); ++i) {
    tasks.push_back(_shards_task_pool[i % task_pool_size_]->enqueue(
        [&, i, this]() -> int64_t {
          int64_t cost = 0;
          std::vector<Node *> &v = shards[i]->get_bucket();
          for (size_t j = 0; j < v.size(); j++) {
            std::vector<uint64_t> s;
            for (size_t k = 0; k < (int)v[j]->get_neighbor_size(); k++) {
              s.push_back(v[j]->get_neighbor_id(k));
            }
            cost += v[j]->get_neighbor_size() * sizeof(uint64_t);
            add_node_to_ssd(0,
                            idx,
                            v[j]->get_id(),
                            (char *)s.data(),
                            s.size() * sizeof(uint64_t));
          }
          return cost;
        }));
  }
  for (size_t i = 0; i < tasks.size(); i++) total_memory_cost += tasks[i].get();
  return 0;
}
int32_t GraphTable::make_complementary_graph(int idx, int64_t byte_size) {
  VLOG(0) << "make_complementary_graph";
  const int64_t fixed_size = byte_size / 8;
  // std::vector<int64_t> edge_array[task_pool_size_];
  std::vector<std::unordered_map<uint64_t, int>> count(task_pool_size_);
  std::vector<std::future<int>> tasks;
  auto &shards = edge_shards[idx];
  for (size_t i = 0; i < shards.size(); ++i) {
    tasks.push_back(
        _shards_task_pool[i % task_pool_size_]->enqueue([&, i, this]() -> int {
          std::vector<Node *> &v = shards[i]->get_bucket();
          size_t ind = i % this->task_pool_size_;
          for (size_t j = 0; j < v.size(); j++) {
            // size_t location = v[j]->get_id();
            for (size_t k = 0; k < v[j]->get_neighbor_size(); k++) {
              count[ind][v[j]->get_neighbor_id(k)]++;
            }
          }
          return 0;
        }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  std::unordered_map<uint64_t, int> final_count;
  std::map<int, std::vector<uint64_t>> count_to_id;
  std::vector<uint64_t> buffer;
  clear_graph(idx);

  for (int i = 0; i < task_pool_size_; i++) {
    for (auto &p : count[i]) {
      final_count[p.first] = final_count[p.first] + p.second;
    }
    count[i].clear();
  }
  for (auto &p : final_count) {
    count_to_id[p.second].push_back(p.first);
    VLOG(2) << p.first << " appear " << p.second << " times";
  }
  auto iter = count_to_id.rbegin();
  while (iter != count_to_id.rend() && byte_size > 0) {
    for (auto x : iter->second) {
      buffer.push_back(x);
      if (buffer.size() >= fixed_size) {
        int64_t res = load_graph_to_memory_from_ssd(idx, buffer);
        buffer.clear();
        byte_size -= res;
      }
      if (byte_size <= 0) break;
    }
    iter++;
  }
  if (byte_size > 0 && buffer.size() > 0) {
    int64_t res = load_graph_to_memory_from_ssd(idx, buffer);
    byte_size -= res;
  }
  std::string sample_type = "random";
  for (auto &shard : edge_shards[idx]) {
    auto bucket = shard->get_bucket();
    for (size_t i = 0; i < bucket.size(); i++) {
      bucket[i]->build_sampler(sample_type);
    }
  }

  return 0;
}
#endif

/*
int CompleteGraphSampler::run_graph_sampling() {
  pthread_rwlock_t *rw_lock = graph_table->rw_lock.get();
  pthread_rwlock_rdlock(rw_lock);
  std::cout << "in graph sampling" << std::endl;
  sample_nodes.clear();
  sample_neighbors.clear();
  sample_res.clear();
  sample_nodes.resize(gpu_num);
  sample_neighbors.resize(gpu_num);
  sample_res.resize(gpu_num);
  std::vector<std::vector<std::vector<paddle::framework::GpuPsGraphNode>>>
      sample_nodes_ex(graph_table->task_pool_size_);
  std::vector<std::vector<std::vector<int64_t>>> sample_neighbors_ex(
      graph_table->task_pool_size_);
  for (int i = 0; i < graph_table->task_pool_size_; i++) {
    sample_nodes_ex[i].resize(gpu_num);
    sample_neighbors_ex[i].resize(gpu_num);
  }
  std::vector<std::future<int>> tasks;
  for (size_t i = 0; i < graph_table->shards.size(); ++i) {
    tasks.push_back(
        graph_table->_shards_task_pool[i % graph_table->task_pool_size_]
            ->enqueue([&, i, this]() -> int {
              if (this->status == GraphSamplerStatus::terminating) return 0;
              paddle::framework::GpuPsGraphNode node;
              std::vector<Node *> &v =
                  this->graph_table->shards[i]->get_bucket();
              size_t ind = i % this->graph_table->task_pool_size_;
              for (size_t j = 0; j < v.size(); j++) {
                size_t location = v[j]->get_id() % this->gpu_num;
                node.node_id = v[j]->get_id();
                node.neighbor_size = v[j]->get_neighbor_size();
                node.neighbor_offset =
                    (int)sample_neighbors_ex[ind][location].size();
                sample_nodes_ex[ind][location].emplace_back(node);
                for (int k = 0; k < node.neighbor_size; k++)
                  sample_neighbors_ex[ind][location].push_back(
                      v[j]->get_neighbor_id(k));
              }
              return 0;
            }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  tasks.clear();
  for (int i = 0; i < gpu_num; i++) {
    tasks.push_back(
        graph_table->_shards_task_pool[i % graph_table->task_pool_size_]
            ->enqueue([&, i, this]() -> int {
              if (this->status == GraphSamplerStatus::terminating) return 0;
              int total_offset = 0;
              size_t ind = i % this->graph_table->task_pool_size_;
              for (int j = 0; j < this->graph_table->task_pool_size_; j++) {
                for (size_t k = 0; k < sample_nodes_ex[j][ind].size(); k++) {
                  sample_nodes[ind].push_back(sample_nodes_ex[j][ind][k]);
                  sample_nodes[ind].back().neighbor_offset += total_offset;
                }
                size_t neighbor_size = sample_neighbors_ex[j][ind].size();
                total_offset += neighbor_size;
                for (size_t k = 0; k < neighbor_size; k++) {
                  sample_neighbors[ind].push_back(
                      sample_neighbors_ex[j][ind][k]);
                }
              }
              return 0;
            }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();

  if (this->status == GraphSamplerStatus::terminating) {
    pthread_rwlock_unlock(rw_lock);
    return 0;
  }
  for (int i = 0; i < gpu_num; i++) {
    sample_res[i].node_list = sample_nodes[i].data();
    sample_res[i].neighbor_list = sample_neighbors[i].data();
    sample_res[i].node_size = sample_nodes[i].size();
    sample_res[i].neighbor_size = sample_neighbors[i].size();
  }
  pthread_rwlock_unlock(rw_lock);
  if (this->status == GraphSamplerStatus::terminating) {
    return 0;
  }
  callback(sample_res);
  return 0;
}
void CompleteGraphSampler::init(size_t gpu_num, GraphTable *graph_table,
                                std::vector<std::string> args) {
  this->gpu_num = gpu_num;
  this->graph_table = graph_table;
}

int BasicBfsGraphSampler::run_graph_sampling() {
  pthread_rwlock_t *rw_lock = graph_table->rw_lock.get();
  pthread_rwlock_rdlock(rw_lock);
  while (rounds > 0 && status == GraphSamplerStatus::running) {
    for (size_t i = 0; i < sample_neighbors_map.size(); i++) {
      sample_neighbors_map[i].clear();
    }
    sample_neighbors_map.clear();
    std::vector<int> nodes_left(graph_table->shards.size(),
                                node_num_for_each_shard);
    std::promise<int> prom;
    std::future<int> fut = prom.get_future();
    sample_neighbors_map.resize(graph_table->task_pool_size_);
    int task_size = 0;
    std::vector<std::future<int>> tasks;
    int init_size = 0;
    //__sync_fetch_and_add
    std::function<int(int, int64_t)> bfs = [&, this](int i, int id) -> int {
      if (this->status == GraphSamplerStatus::terminating) {
        int task_left = __sync_sub_and_fetch(&task_size, 1);
        if (task_left == 0) {
          prom.set_value(0);
        }
        return 0;
      }
      size_t ind = i % this->graph_table->task_pool_size_;
      if (nodes_left[i] > 0) {
        auto iter = sample_neighbors_map[ind].find(id);
        if (iter == sample_neighbors_map[ind].end()) {
          Node *node = graph_table->shards[i]->find_node(id);
          if (node != NULL) {
            nodes_left[i]--;
            sample_neighbors_map[ind][id] = std::vector<int64_t>();
            iter = sample_neighbors_map[ind].find(id);
            size_t edge_fetch_size =
                std::min((size_t) this->edge_num_for_each_node,
                         node->get_neighbor_size());
            for (size_t k = 0; k < edge_fetch_size; k++) {
              int64_t neighbor_id = node->get_neighbor_id(k);
              int node_location = neighbor_id % this->graph_table->shard_num %
                                  this->graph_table->task_pool_size_;
              __sync_add_and_fetch(&task_size, 1);
              graph_table->_shards_task_pool[node_location]->enqueue(
                  bfs, neighbor_id % this->graph_table->shard_num, neighbor_id);
              iter->second.push_back(neighbor_id);
            }
          }
        }
      }
      int task_left = __sync_sub_and_fetch(&task_size, 1);
      if (task_left == 0) {
        prom.set_value(0);
      }
      return 0;
    };
    for (size_t i = 0; i < graph_table->shards.size(); ++i) {
      std::vector<Node *> &v = graph_table->shards[i]->get_bucket();
      if (v.size() > 0) {
        int search_size = std::min(init_search_size, (int)v.size());
        for (int k = 0; k < search_size; k++) {
          init_size++;
          __sync_add_and_fetch(&task_size, 1);
          int64_t id = v[k]->get_id();
          graph_table->_shards_task_pool[i % graph_table->task_pool_size_]
              ->enqueue(bfs, i, id);
        }
      }  // if
    }
    if (init_size == 0) {
      prom.set_value(0);
    }
    fut.get();
    if (this->status == GraphSamplerStatus::terminating) {
      pthread_rwlock_unlock(rw_lock);
      return 0;
    }
    VLOG(0) << "BasicBfsGraphSampler finishes the graph searching task";
    sample_nodes.clear();
    sample_neighbors.clear();
    sample_res.clear();
    sample_nodes.resize(gpu_num);
    sample_neighbors.resize(gpu_num);
    sample_res.resize(gpu_num);
    std::vector<std::vector<std::vector<paddle::framework::GpuPsGraphNode>>>
        sample_nodes_ex(graph_table->task_pool_size_);
    std::vector<std::vector<std::vector<int64_t>>> sample_neighbors_ex(
        graph_table->task_pool_size_);
    for (int i = 0; i < graph_table->task_pool_size_; i++) {
      sample_nodes_ex[i].resize(gpu_num);
      sample_neighbors_ex[i].resize(gpu_num);
    }
    tasks.clear();
    for (size_t i = 0; i < (size_t)graph_table->task_pool_size_; ++i) {
      tasks.push_back(
          graph_table->_shards_task_pool[i]->enqueue([&, i, this]() -> int {
            if (this->status == GraphSamplerStatus::terminating) {
              return 0;
            }
            paddle::framework::GpuPsGraphNode node;
            auto iter = sample_neighbors_map[i].begin();
            size_t ind = i;
            for (; iter != sample_neighbors_map[i].end(); iter++) {
              size_t location = iter->first % this->gpu_num;
              node.node_id = iter->first;
              node.neighbor_size = iter->second.size();
              node.neighbor_offset =
                  (int)sample_neighbors_ex[ind][location].size();
              sample_nodes_ex[ind][location].emplace_back(node);
              for (auto k : iter->second)
                sample_neighbors_ex[ind][location].push_back(k);
            }
            return 0;
          }));
    }

    for (size_t i = 0; i < tasks.size(); i++) {
      tasks[i].get();
      sample_neighbors_map[i].clear();
    }
    tasks.clear();
    if (this->status == GraphSamplerStatus::terminating) {
      pthread_rwlock_unlock(rw_lock);
      return 0;
    }
    for (size_t i = 0; i < (size_t)gpu_num; i++) {
      tasks.push_back(
          graph_table->_shards_task_pool[i % graph_table->task_pool_size_]
              ->enqueue([&, i, this]() -> int {
                if (this->status == GraphSamplerStatus::terminating) {
                  pthread_rwlock_unlock(rw_lock);
                  return 0;
                }
                int total_offset = 0;
                for (int j = 0; j < this->graph_table->task_pool_size_; j++) {
                  for (size_t k = 0; k < sample_nodes_ex[j][i].size(); k++) {
                    sample_nodes[i].push_back(sample_nodes_ex[j][i][k]);
                    sample_nodes[i].back().neighbor_offset += total_offset;
                  }
                  size_t neighbor_size = sample_neighbors_ex[j][i].size();
                  total_offset += neighbor_size;
                  for (size_t k = 0; k < neighbor_size; k++) {
                    sample_neighbors[i].push_back(sample_neighbors_ex[j][i][k]);
                  }
                }
                return 0;
              }));
    }
    for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
    if (this->status == GraphSamplerStatus::terminating) {
      pthread_rwlock_unlock(rw_lock);
      return 0;
    }
    for (int i = 0; i < gpu_num; i++) {
      sample_res[i].node_list = sample_nodes[i].data();
      sample_res[i].neighbor_list = sample_neighbors[i].data();
      sample_res[i].node_size = sample_nodes[i].size();
      sample_res[i].neighbor_size = sample_neighbors[i].size();
    }
    pthread_rwlock_unlock(rw_lock);
    if (this->status == GraphSamplerStatus::terminating) {
      return 0;
    }
    callback(sample_res);
    rounds--;
    if (rounds > 0) {
      for (int i = 0;
           i < interval && this->status == GraphSamplerStatus::running; i++) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
    }
    VLOG(0)<<"bfs returning";
  }
  return 0;
}
void BasicBfsGraphSampler::init(size_t gpu_num, GraphTable *graph_table,
                                std::vector<std::string> args) {
  this->gpu_num = gpu_num;
  this->graph_table = graph_table;
  init_search_size = args.size() > 0 ? std::stoi(args[0]) : 10;
  node_num_for_each_shard = args.size() > 1 ? std::stoi(args[1]) : 10;
  edge_num_for_each_node = args.size() > 2 ? std::stoi(args[2]) : 10;
  rounds = args.size() > 3 ? std::stoi(args[3]) : 1;
  interval = args.size() > 4 ? std::stoi(args[4]) : 60;
}

#endif
*/
std::vector<Node *> GraphShard::get_batch(int start, int end, int step) {
  if (start < 0) start = 0;
  std::vector<Node *> res;
  for (int pos = start; pos < std::min(end, (int)bucket.size()); pos += step) {
    res.push_back(bucket[pos]);
  }
  return res;
}

size_t GraphShard::get_size() { return bucket.size(); }

int32_t GraphTable::add_comm_edge(int idx, uint64_t src_id, uint64_t dst_id) {
  size_t src_shard_id = src_id % shard_num;

  if (src_shard_id >= shard_end || src_shard_id < shard_start) {
    return -1;
  }
  size_t index = src_shard_id - shard_start;
  edge_shards[idx][index]->add_graph_node(src_id)->build_edges(false);
  edge_shards[idx][index]->add_neighbor(src_id, dst_id, 1.0);
  return 0;
}
int32_t GraphTable::add_graph_node(int idx,
                                   std::vector<uint64_t> &id_list,
                                   std::vector<bool> &is_weight_list) {
  auto &shards = edge_shards[idx];
  size_t node_size = id_list.size();
  std::vector<std::vector<std::pair<uint64_t, bool>>> batch(task_pool_size_);
  for (size_t i = 0; i < node_size; i++) {
    size_t shard_id = id_list[i] % shard_num;
    if (shard_id >= shard_end || shard_id < shard_start) {
      continue;
    }
    batch[get_thread_pool_index(id_list[i])].push_back(
        {id_list[i], i < is_weight_list.size() ? is_weight_list[i] : false});
  }
  std::vector<std::future<int>> tasks;
  for (size_t i = 0; i < batch.size(); ++i) {
    if (!batch[i].size()) continue;
    tasks.push_back(
        _shards_task_pool[i]->enqueue([&shards, &batch, i, this]() -> int {
          for (auto &p : batch[i]) {
            size_t index = p.first % this->shard_num - this->shard_start;
            shards[index]->add_graph_node(p.first)->build_edges(p.second);
          }
          return 0;
        }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  return 0;
}

int32_t GraphTable::remove_graph_node(int idx, std::vector<uint64_t> &id_list) {
  size_t node_size = id_list.size();
  std::vector<std::vector<uint64_t>> batch(task_pool_size_);
  for (size_t i = 0; i < node_size; i++) {
    size_t shard_id = id_list[i] % shard_num;
    if (shard_id >= shard_end || shard_id < shard_start) continue;
    batch[get_thread_pool_index(id_list[i])].push_back(id_list[i]);
  }
  auto &shards = edge_shards[idx];
  std::vector<std::future<int>> tasks;
  for (size_t i = 0; i < batch.size(); ++i) {
    if (!batch[i].size()) continue;
    tasks.push_back(
        _shards_task_pool[i]->enqueue([&shards, &batch, i, this]() -> int {
          for (auto &p : batch[i]) {
            size_t index = p % this->shard_num - this->shard_start;
            shards[index]->delete_node(p);
          }
          return 0;
        }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  return 0;
}

void GraphShard::clear() {
  for (size_t i = 0; i < bucket.size(); i++) {
    delete bucket[i];
  }
  bucket.clear();
  node_location.clear();
}

GraphShard::~GraphShard() { clear(); }

void GraphShard::delete_node(uint64_t id) {
  auto iter = node_location.find(id);
  if (iter == node_location.end()) return;
  int pos = iter->second;
  delete bucket[pos];
  if (pos != (int)bucket.size() - 1) {
    bucket[pos] = bucket.back();
    node_location[bucket.back()->get_id()] = pos;
  }
  node_location.erase(id);
  bucket.pop_back();
}
GraphNode *GraphShard::add_graph_node(uint64_t id) {
  if (node_location.find(id) == node_location.end()) {
    node_location[id] = bucket.size();
    bucket.push_back(new GraphNode(id));
  }
  return (GraphNode *)bucket[node_location[id]];
}

GraphNode *GraphShard::add_graph_node(Node *node) {
  auto id = node->get_id();
  if (node_location.find(id) == node_location.end()) {
    node_location[id] = bucket.size();
    bucket.push_back(node);
  }
  return (GraphNode *)bucket[node_location[id]];
}

FeatureNode *GraphShard::add_feature_node(uint64_t id, bool is_overlap) {
  if (node_location.find(id) == node_location.end()) {
    node_location[id] = bucket.size();
    bucket.push_back(new FeatureNode(id));
    return (FeatureNode *)bucket[node_location[id]];
  }
  if (is_overlap) {
    return (FeatureNode *)bucket[node_location[id]];
  }

  return NULL;
}

void GraphShard::add_neighbor(uint64_t id, uint64_t dst_id, float weight) {
  find_node(id)->add_edge(dst_id, weight);
}

Node *GraphShard::find_node(uint64_t id) {
  auto iter = node_location.find(id);
  return iter == node_location.end() ? nullptr : bucket[iter->second];
}

GraphTable::~GraphTable() {
  for (int i = 0; i < (int)edge_shards.size(); i++) {
    for (auto p : edge_shards[i]) {
      delete p;
    }
    edge_shards[i].clear();
  }

  for (int i = 0; i < (int)feature_shards.size(); i++) {
    for (auto p : feature_shards[i]) {
      delete p;
    }
    feature_shards[i].clear();
  }
}

int32_t GraphTable::Load(const std::string &path, const std::string &param) {
  bool load_edge = (param[0] == 'e');
  bool load_node = (param[0] == 'n');
  if (load_edge) {
    bool reverse_edge = (param[1] == '<');
    std::string edge_type = param.substr(2);
    return this->load_edges(path, reverse_edge, edge_type);
  }
  if (load_node) {
    std::string node_type = param.substr(1);
    return this->load_nodes(path, node_type);
  }
  return 0;
}

std::string GraphTable::get_inverse_etype(std::string &etype) {
  auto etype_split = paddle::string::split_string<std::string>(etype, "2");
  std::string res;
  if ((int)etype_split.size() == 3) {
    res = etype_split[2] + "2" + etype_split[1] + "2" + etype_split[0];
  } else {
    res = etype_split[1] + "2" + etype_split[0];
  }
  return res;
}

int32_t GraphTable::parse_type_to_typepath(std::string &type2files,
                                           std::string graph_data_local_path,
                                           std::vector<std::string> &res_type,
                                           std::unordered_map<std::string, std::string> &res_type2path) {
  auto type2files_split = paddle::string::split_string<std::string>(type2files, ",");
  if (type2files_split.size() == 0) {
    return -1;
  }
  for (auto one_type2file : type2files_split) {
    auto one_type2file_split = paddle::string::split_string<std::string>(one_type2file, ":");
    auto type = one_type2file_split[0];
    auto type_dir = one_type2file_split[1];
    res_type.push_back(type);
    res_type2path[type] = graph_data_local_path + "/" + type_dir;
  }
  return 0;
}

int32_t GraphTable::load_node_and_edge_file(std::string etype2files,
                                            std::string ntype2files,
                                            std::string graph_data_local_path,
                                            int part_num,
                                            bool reverse) {
  std::vector<std::string> etypes;
  std::unordered_map<std::string, std::string> edge_to_edgedir;
  int res = parse_type_to_typepath(etype2files, graph_data_local_path, etypes, edge_to_edgedir);
  if (res != 0) {
    VLOG(0) << "parse edge type and edgedir failed!";
    return -1;
  }
  std::vector<std::string> ntypes;
  std::unordered_map<std::string, std::string> node_to_nodedir;
  res = parse_type_to_typepath(ntype2files, graph_data_local_path, ntypes, node_to_nodedir);
  if (res != 0) {
    VLOG(0) << "parse node type and nodedir failed!";
    return -1;
  }

  VLOG(0) << "etypes size: " << etypes.size();
  VLOG(0) << "whether reverse: " << reverse;
  is_load_reverse_edge = reverse;
  std::string delim = ";";
  size_t total_len = etypes.size() + 1;  // 1 is for node
  tot_v_num = tot_e_num = 0;
  v_num.resize(id_to_feature.size());
  e_num.resize(id_to_edge.size());

  std::vector<std::future<int>> tasks;
  for (size_t i = 0; i < total_len; i++) {
    tasks.push_back(
        _shards_task_pool[i % task_pool_size_]->enqueue([&, i, this]() -> int {
          if (i < etypes.size()) {
            std::string etype_path = edge_to_edgedir[etypes[i]];
            auto etype_path_list = paddle::framework::localfs_list(etype_path);
            std::string etype_path_str;
            if (part_num > 0 && part_num < (int)etype_path_list.size()) {
              std::vector<std::string> sub_etype_path_list(
                  etype_path_list.begin(), etype_path_list.begin() + part_num);
              etype_path_str =
                  paddle::string::join_strings(sub_etype_path_list, delim);
            } else {
              etype_path_str =
                  paddle::string::join_strings(etype_path_list, delim);
            }
            this->load_edges(etype_path_str, false, etypes[i]);
            if (reverse) {
              std::string r_etype = get_inverse_etype(etypes[i]);
              this->load_edges(etype_path_str, true, r_etype);
            }
          } else {
            std::string npath = node_to_nodedir[ntypes[0]];
            auto npath_list = paddle::framework::localfs_list(npath);
            std::string npath_str;
            if (part_num > 0 && part_num < (int)npath_list.size()) {
              std::vector<std::string> sub_npath_list(
                  npath_list.begin(), npath_list.begin() + part_num);
              npath_str = paddle::string::join_strings(sub_npath_list, delim);
            } else {
              npath_str = paddle::string::join_strings(npath_list, delim);
            }

            if (ntypes.size() == 0) {
              VLOG(0) << "node_type not specified, nothing will be loaded ";
              return 0;
            }

            if (FLAGS_graph_load_in_parallel) {
              this->load_nodes(npath_str, "");
            } else {
              for (size_t j = 0; j < ntypes.size(); j++) {
                this->load_nodes(npath_str, ntypes[j]);
              }
            }
          }
          return 0;
        }));
  }
  for (int i = 0; i < (int)tasks.size(); i++) tasks[i].get();
  return 0;
}

int GraphTable::load_subgraph_info(const std::string& spath, int ntype_size) {
  std::string ginfo_path = spath + "ginfo";
  std::ifstream fin(ginfo_path);
  if (!fin.is_open()) {
    VLOG(0) << "Cannot open " << ginfo_path;
    return -1;
  }
  std::string v_name, feature_name;
  sg_vertex_info.resize(ntype_size);
  while (fin >> v_name) {
    int feature_num = 0;
    fin >> feature_num;
    if (feature_to_id.find(v_name) == feature_to_id.end()) {
      VLOG(0) << "Cannot find " << v_name << ".";
      return -1;
    }
    int idx = feature_to_id[v_name];
    VLOG(0) << v_name << ' ' << idx;
    while (feature_num--) {
      fin >> feature_name;
      sg_vertex_info[idx].push_back(feature_name);
    }
  }
  fin.close();
  return 0;
}

// int32_t GraphTable::build_nodes(int idx) {
//   auto u2v = paddle::string::split_string<std::string>(id_to_edge[idx], "2");
//   std::string vtype = u2v[1];

//   std::vector<int32_t> sg_list;
//   for (int i = 0; i < id_to_edge.size(); ++i) {
//     u2v = paddle::string::split_string<std::string>(id_to_edge[i], "2");
//     if (u2v[0] == vtype) sg_list.push_back(i);
//   }

//   for (int i = 0; i < shard_num; i++) {
//     auto &shards = edge_shards[idx][i]->get_bucket();
//     for (auto ptr : shards) {
//       auto uid = ptr->get_id();
//       for (size_t neighbor_id = 0; neighbor_id < ptr->get_neighbor_size(); ++neighbor_id) {
//         uint64_t vid = ptr->get_neighbor_id(neighbor_id);
//         for (auto sg : sg_list) {
//           if (edge_shards[sg][(vid % shard_num) - shard_start]->add_graph_node(vid) == nullptr)
//             VLOG(0) << "error when building empty nodes.";
//         }
//       }
//     }
//   }
//   return 0;
// }

// int32_t GraphTable::build_inv_subgraph(int idx) {
//   auto u2v = paddle::string::split_string<std::string>(id_to_edge[idx], "2");
//   int inv_idx = edge_to_id[u2v[1] + "2" + u2v[0]];
//   bool is_weighted = false;
//   float weight = 1.0;

//   uint64_t op_sum = 0;

//   for (int i = 0; i < shard_num; i++) {
//     auto &shards = edge_shards[idx][i]->get_bucket();
//     for (auto ptr : shards) {
//       auto uid = ptr->get_id();
//       for (size_t neighbor_id = 0; neighbor_id < ptr->get_neighbor_size(); ++neighbor_id) {
//         uint64_t vid = ptr->get_neighbor_id(neighbor_id);
//         auto node = edge_shards[inv_idx][(vid % shard_num) - shard_start]->add_graph_node(vid);
//         if (node != NULL) {
//           node->build_edges(is_weighted);
//           node->add_edge(uid, weight);
//           op_sum++;
//         }    
//       }
//     }
//   }

//   VLOG(0) << "Build " << op_sum << " edges for " << id_to_edge[inv_idx] << " graph(inv)";
//   return 0;
// }

int32_t GraphTable::prepare_train_subgraph(std::string etype2files,
                                           std::string ntype2files,
                                           std::string subgraph_path,
                                           int load_sg_id,
                                           int part_num,
                                           bool reverse,
                                           SubGraphParameter sgp) {
  std::string subgraph_data_path = subgraph_path + "sg_" + std::to_string(load_sg_id) + "/";
  std::vector<std::string> etypes;
  std::unordered_map<std::string, std::string> edge_to_edgedir;
  int res = parse_type_to_typepath(etype2files, subgraph_data_path, etypes, edge_to_edgedir);
  if (res != 0) {
    VLOG(0) << "parse edge type and edgedir failed!";
    return -1;
  }
  std::vector<std::string> ntypes;
  std::unordered_map<std::string, std::string> node_to_nodedir;
  res = parse_type_to_typepath(ntype2files, subgraph_data_path, ntypes, node_to_nodedir);
  if (res != 0) {
    VLOG(0) << "parse node type and nodedir failed!";
    return -1;
  }
  if (sg_vertex_info.empty()) {
    load_subgraph_info(subgraph_path, ntypes.size());
    VLOG(0) << "whether reverse: " << reverse;
    is_load_reverse_edge = reverse;
  }
  tot_v_num = tot_e_num = 0;
  v_num.resize(id_to_feature.size());
  e_num.resize(id_to_edge.size());

  VLOG(0) << "[Subgraph_ " << load_sg_id << "] etypes size: " << etypes.size() << " ; ntypes size: " << ntypes.size() << " ; shard_vertex size: " << sg_vertex_info.size();
  std::string delim = ";";
  size_t total_len = etypes.size() * 2 + sg_vertex_info.size();

  std::vector<std::future<int>> tasks;
  for (size_t i = 0; i < total_len; i++) {
    tasks.push_back(
      _shards_task_pool[i % task_pool_size_]->enqueue([&, i, this]() -> int {
        if (i < (etypes.size() << 1)) {
          auto etype = etypes[i >> 1];
          if (i & 1) etype = get_inverse_etype(etype);
          std::string etype_path = subgraph_data_path + etype + "/";
          auto etype_path_list = paddle::framework::localfs_list(etype_path);
          std::string etype_path_str =
            paddle::string::join_strings(etype_path_list, delim);
          this->load_edges(etype_path_str, false, etype, 1);
        } else {
          if (ntypes.size() == 0) {
            VLOG(0) << "node_type not specified, nothing will be loaded ";
            return 0;
          }

          int idx = i - etypes.size() * 2;
          // auto npath = spath + "node_" + ntypes[idx];
          std::string npath = node_to_nodedir[ntypes[idx]];
          auto npath_list = paddle::framework::localfs_list(npath);
          std::string npath_str = paddle::string::join_strings(npath_list, delim);

          if (FLAGS_graph_load_in_parallel) {
            this->load_nodes(npath_str, "", idx);
          } else {
            this->load_nodes(npath_str, ntypes[idx], idx);
          }
        }
        return 0;
    }));
  }
  for (int i = 0; i < (int)tasks.size(); i++) tasks[i].get();

  // for (int idx = 0; idx < id_to_edge.size(); idx += 2) {
  //   this->build_inv_subgraph(idx);
  // }

  if (tot_v_num == 0) {
    for (auto i : v_num) tot_v_num += i;
  }
  if (tot_e_num == 0) {
    for (auto i : e_num) tot_e_num += i;
  }
  VLOG(0) << "Load " << tot_v_num << " vertices, " << tot_e_num << " edges.";

  if (sgp.halo_mode) {
    std::string halograph_data_path = subgraph_data_path + "halo/";
    if (sgp.halo_mode == 1)
      prepare_halograph(halograph_data_path, sgp.halo_a, sgp.halo_b, sgp.epoch_id, sgp.epoch_num, sgp.layer_num);
    if (sgp.halo_mode == 2)
      prepare_halograph(halograph_data_path, sgp.budget, sgp.epoch_id, sgp.epoch_num, sgp.layer_num);
  }
  return 0;
}

bool halo_func(double a, double b, double s_val, double dist_val, int epoch_id, double epoch_val) {
  double t = a * 1.0 / epoch_id * dist_val + b * epoch_val * s_val;
  double t_max = a  * 1.0 / epoch_id + b * epoch_val;
  
  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<double> dis(0.0, t_max);
  return dis(gen) < t;
}

int GraphTable::prepare_halograph(std::string path, // Prepare HaloGraph with Node Sampling
                                  double halo_a, double halo_b, int epoch_id, int epoch_num, int layer_num) {
  // v_id[uint64] dist[int] S[double] ngb_num[uint64] node_type[int] ngbs...[uint64] feats...[uint64]
  
  std::vector<std::future<int>> tasks;
  std::vector<std::vector<std::vector<std::map<uint64_t, std::vector<uint64_t>>>>> halo_info(shard_num);
  const double epoch_val = 1.0 * epoch_id / epoch_num;

  std::vector<uint64_t> part_v_sum(shard_num), part_e_sum(shard_num), part_v_cnt(shard_num), part_e_cnt(shard_num);

  VLOG(0) << "Begin : Load&Pick HaloGraph";
  for (int part_id = 0; part_id < shard_num; ++part_id) {
    tasks.push_back(
      _shards_task_pool[part_id % task_pool_size_]->enqueue([&, part_id]() -> int {
        std::string halo_path = path + "hg-" + std::to_string(part_id);
        std::ifstream file(halo_path, std::ios::in | std::ios::binary | std::ios::ate);
        uint64_t local_count = 0;
        uint64_t local_valid_count = 0;
        halo_info[part_id].resize(shard_num);
        for (int i = 0; i < shard_num; ++i)
          halo_info[part_id][i].resize(id_to_edge.size());
  
        if (!file.is_open()) {
          VLOG(0) << "Open " << halo_path << " Failed.";
          return -1;
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        file.read(buffer.data(), size);
        file.close();

        char* ptr = buffer.data();
        char* const beg_ptr = ptr;
        while ((ptr - beg_ptr) < size) {
          uint64_t u_id = *((uint64_t*)ptr);
          ptr += sizeof(uint64_t);
          uint32_t u_idx = *ptr;
          ptr += sizeof(int);
          uint32_t dist = *ptr;
          ptr += sizeof(int);
          double s_val = *((double*)ptr);
          ptr += sizeof(double);

          auto& feature_list = sg_vertex_info[u_idx];
          const int feat_num = feature_list.size();
          
          part_v_sum[part_id]++;

          if (halo_func(halo_a, halo_b, s_val, 1.0 * (layer_num - dist + 1) / layer_num , epoch_id, epoch_val)) {
            part_v_cnt[part_id]++;
            for (auto e_idx : search_graphs[u_idx]) {
              uint64_t ngb_num = *((uint64_t*)ptr);
              ptr += sizeof(uint64_t);
              part_e_sum[part_id] += ngb_num;
              auto e_ptr = edge_shards[e_idx][part_id]->add_graph_node(u_id);
              if (e_ptr != nullptr) {
                int v_idx = get_idx(e_idx, 1);
                for (uint64_t ngb_id = 0; ngb_id < ngb_num; ++ngb_id) {
                  uint64_t v_id = *((uint64_t*)ptr);
                  ptr += sizeof(uint64_t);
                  halo_info[part_id][v_id % shard_num][v_idx][v_id].push_back(u_id);
                  halo_info[part_id][part_id][u_idx][u_id].push_back(v_id);
                }
              }
              else ptr += sizeof(uint64_t) * ngb_num;
            }

            auto f_ptr = feature_shards[u_idx][part_id]->add_feature_node(u_id, false);
            if (f_ptr != NULL) {
              for (int feat_id = 0; feat_id < feat_num; ++feat_id) {
                uint64_t feat_val = *((uint64_t*)ptr);
                ptr += sizeof(uint64_t);
                parse_shard_feature(u_idx, feature_list[feat_id], feat_val, f_ptr);
              }
            }
            else ptr += sizeof(uint64_t) * feat_num;
          }
          else {
            for (auto e_idx : search_graphs[u_idx]) {
              uint64_t ngb_num = *((uint64_t*)ptr);
              ptr += sizeof(uint64_t) * (ngb_num + 1);
              part_e_sum[part_id] += ngb_num;
            }
            ptr += sizeof(uint64_t) * feat_num;
          }
        }
        return 0;
    }));
  }
  for (int i = 0; i < (int)tasks.size(); i++) tasks[i].get();
  VLOG(0) << "Finish : Load&Pick HaloGraph";

  bool is_weighted = false;
  float weight = 1.0;
  tasks.clear();
  VLOG(0) << "Begin : Pick Trained HaloGraph";
  for (int part_id = 0; part_id < shard_num; ++part_id) {
    tasks.push_back(
      _shards_task_pool[part_id % task_pool_size_]->enqueue([&, part_id]() -> int {
        for (int _part_id = 0; _part_id < shard_num; ++_part_id) {
          for (int e_idx = 0; e_idx < id_to_edge.size(); ++e_idx) {
            int v_idx = get_idx(e_idx, 1);
            for (auto halo_edge : halo_info[_part_id][part_id][e_idx]) {
              uint64_t u_id = halo_edge.first;
              auto u_ptr = find_node(0, e_idx, u_id);
              if (u_ptr != nullptr) {
                for (auto v_id : halo_edge.second) {
                  if (find_node(0, v_idx, v_id) != nullptr) {
                    u_ptr->build_edges(is_weighted);
                    u_ptr->add_edge(v_id, weight);
                    part_e_cnt[part_id]++;
                  }
                }
              }
            }
          }
        }
        return 0;
    }));
  }
  for (int i = 0; i < (int)tasks.size(); i++) tasks[i].get();

  uint64_t v_sum = 0, v_cnt = 0, e_sum = 0, e_cnt = 0;
  for (int i = 0; i < shard_num; ++i) {
    v_sum += part_v_sum[i];
    v_cnt += part_v_cnt[i];
    e_sum += part_e_sum[i];
    e_cnt += part_e_cnt[i];
  }
  e_sum <<= 1;
  VLOG(0) << "[HaloGraph] Read " << v_sum << " vertices. Pick " << v_cnt << " vertices " << 1.0 * v_cnt / v_sum;
  VLOG(0) << "[HaloGraph] Read " << e_sum << " edges.    Pick " << e_cnt << " edges    " << 1.0 * e_cnt / e_sum;

  VLOG(0) << "Finish : Build HaloGraph";
  return 0;
}

int GraphTable::prepare_halograph(std::string path, // Prepare HaloGraph with Random Walk
                                  const double budget, const int epoch_id, const int epoch_num, const int layer_num) {
  // v_id[uint64] dist[int] S[double] ngb_num[uint64] node_type[int] ngbs...[uint64] feats...[uint64]
  
  std::vector<std::future<int>> tasks;
  struct HaloInfo {
    std::vector<uint64_t> ngb;
    std::vector<uint64_t> ngb_num;
    std::vector<double> feat;
    int idx;
    double score;
    uint64_t id;

    double cal_score(double weight, double rand_num) {
      return -logl(rand_num) / weight;
    }
  };
  std::vector<std::vector<std::vector<std::map<uint64_t, HaloInfo*>>>> halo_info(shard_num);

  const int epoch_dist = 1.0 * epoch_id / epoch_num * layer_num + 0.5;
  const uint64_t seed_num = budget * tot_v_num / epoch_dist;
  VLOG(0) << "Budget = " << budget << ", Epoch_dist = " << epoch_dist << ", Seed_num = " << seed_num << ' ' << tot_v_num;

  std::vector<uint64_t> part_v_sum(shard_num), part_e_sum(shard_num), part_v_cnt(shard_num), part_e_cnt(shard_num);
  std::vector<std::priority_queue<std::pair<double, HaloInfo*>>> seed_vertices(shard_num);

  VLOG(0) << "Begin : Load HaloGraph";
  for (int part_id = 0; part_id < shard_num; ++part_id) {
    tasks.push_back(
      _shards_task_pool[part_id % task_pool_size_]->enqueue([&, part_id]() -> int {
        std::string halo_path = path + "hg-" + std::to_string(part_id);
        std::ifstream file(halo_path, std::ios::in | std::ios::binary | std::ios::ate);
        uint64_t local_count = 0;
        uint64_t local_valid_count = 0;
        halo_info[part_id].resize(epoch_dist);
        for (int i = 0; i < epoch_dist; ++i)
          halo_info[part_id][i].resize(id_to_edge.size());
  
        if (!file.is_open()) {
          VLOG(0) << "Open " << halo_path << " Failed.";
          return -1;
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        file.read(buffer.data(), size);
        file.close();

        char* ptr = buffer.data();
        char* const beg_ptr = ptr;

        std::random_device rd;  // Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<double> dis(0.0, 1.0);

        while ((ptr - beg_ptr) < size) {
          HaloInfo *halo_ptr;
          uint64_t u_id = *((uint64_t*)ptr);
          ptr += sizeof(uint64_t);
          uint32_t u_idx = *ptr;
          ptr += sizeof(int);
          uint32_t dist = *ptr;
          ptr += sizeof(int);
          double u_weight = *((double*)ptr);
          ptr += sizeof(double);
          
          halo_ptr->idx = u_idx;
          for (auto e_idx : search_graphs[u_idx]) {
            uint64_t ngb_num = *((uint64_t*)ptr);
            ptr += sizeof(uint64_t);
            halo_ptr->ngb_num.push_back(ngb_num);
            part_e_sum[part_id] += ngb_num;
            int v_idx = get_idx(e_idx, 1);
            for (uint64_t ngb_id = 0; ngb_id < ngb_num; ++ngb_id) {
              uint64_t v_id = *((uint64_t*)ptr);
              ptr += sizeof(uint64_t);
              halo_ptr->ngb.push_back(v_id);
            }
          }
          const int feat_num = sg_vertex_info[u_idx].size();
          for (int feat_id = 0; feat_id < feat_num; ++feat_id) {
            uint64_t feat_val = *((uint64_t*)ptr);
            ptr += sizeof(uint64_t);
            halo_ptr->feat.push_back(feat_val);
          }

          auto &hinfo = halo_info[part_id][dist][u_idx];
          if (hinfo.find(u_id) == hinfo.end()) {
            halo_ptr->score = halo_ptr->cal_score(u_weight, dis(gen));
            halo_ptr->id = u_id;
            part_v_sum[part_id]++;
            if (dist == epoch_dist) {
              seed_vertices[part_id].push(std::make_pair(halo_ptr->score, halo_ptr));
              if (seed_vertices[part_id].size() > seed_num) seed_vertices[part_id].pop();
            }
          }
          else {
            VLOG(0) << "Deplicated Halo Vertex " << u_id << " (" << id_to_feature[u_idx] << ")"; 
            delete halo_ptr;
          }
        }
        return 0;
    }));
  }
  for (int i = 0; i < (int)tasks.size(); i++) tasks[i].get();
  VLOG(0) << "Finish : Load HaloGraph";

  VLOG(0) << "Begin : Pick Trained HaloGraph";
  tasks.clear();
  std::priority_queue<std::pair<double, int>> top_part_val;
  uint64_t remove_cnt = 0, remove_sum = 0;
  for (int part_id = 0; part_id < shard_num; ++part_id) {
    top_part_val.push(std::make_pair(seed_vertices[part_id].top().first, part_id));
    remove_sum += top_part_val.size();
  }
  remove_sum -= seed_num;
  while (remove_cnt++ < remove_sum) {
    auto val_pair = top_part_val.top();
    top_part_val.pop();
    seed_vertices[val_pair.second].pop();
    top_part_val.push(std::make_pair(seed_vertices[val_pair.second].top().first, val_pair.second));
  }

  std::vector<std::vector<std::vector<HaloInfo*>>> spl_halo;
  spl_halo.resize(shard_num);
  tasks.clear();
  for (int part_id = 0; part_id < shard_num; ++part_id) {
    tasks.push_back(
      _shards_task_pool[part_id % task_pool_size_]->enqueue([&, part_id]() -> int {
        spl_halo[part_id].resize(shard_num);
        while (!seed_vertices[part_id].empty()) {
          auto u_pair = seed_vertices[part_id].top();
          seed_vertices[part_id].pop();
          auto u_ptr = u_pair.second;
          auto u_id = u_ptr->id;
          spl_halo[part_id][part_id].push_back(u_ptr);
          for (int dist = epoch_dist - 1; dist >= 1; --dist) {
            uint64_t ngb_offset = 0;
            double val_min = -1.0;
            HaloInfo* spl_ptr;
            // auto spl_ptr = std::make_shared<HaloInfo>();
            int u_idx = u_ptr->idx;
            for (int i = 0; i < search_graphs[u_idx].size(); ++i) {
              int v_idx = get_idx(search_graphs[u_idx][i], 1);
              for (uint64_t ngb_id = 0; ngb_id < u_ptr->ngb_num[i]; ++ngb_id) {
                auto v_id = u_ptr->ngb[ngb_offset + ngb_id];
                auto &hinfo = halo_info[v_id % shard_num][dist][v_idx];
                if (hinfo.find(v_id) != hinfo.end()) {
                  if (hinfo[v_id]->score < val_min || val_min < 0) {
                    val_min = hinfo[v_id]->score;
                  }
                }
                else VLOG(0) << "Missing HaloVertex " << v_id << " (" << v_idx << ")";
              }
              ngb_offset += u_ptr->ngb_num[i];
            }
            spl_halo[part_id][(spl_ptr->id) % shard_num].push_back(spl_ptr);
            u_ptr = spl_ptr;
          }
        }
        return 0;
    }));
  }
  for (int i = 0; i < (int)tasks.size(); i++) tasks[i].get();

  
  std::vector<std::vector<std::vector<std::map<uint64_t, std::vector<uint64_t>>>>> halo_edges(shard_num);
  tasks.clear();
  for (int part_id = 0; part_id < shard_num; ++part_id) {
    tasks.push_back(
      _shards_task_pool[part_id % task_pool_size_]->enqueue([&, part_id]() -> int {
        halo_edges[part_id].resize(shard_num);
        for (int i = 0; i < shard_num; ++i)
          halo_edges[part_id][i].resize(id_to_edge.size());

        for (int _part_id = 0; _part_id < shard_num; ++_part_id) {
          for (auto u_ptr : spl_halo[_part_id][part_id]) {
            int u_idx = u_ptr->idx;
            auto u_id = u_ptr->id;
            uint64_t ngb_offset = 0;

            for (int i = 0; i < search_graphs[u_idx].size(); ++i) {
              auto e_ptr = edge_shards[search_graphs[u_idx][i]][part_id]->add_graph_node(u_id);
              if (e_ptr != nullptr) {
                int v_idx = get_idx(search_graphs[u_idx][i], 1);
                for (uint64_t ngb_id = 0; ngb_id < u_ptr->ngb_num[i]; ++ngb_id) {
                  auto v_id = u_ptr->ngb[ngb_offset + ngb_id];
                  if (find_node(0, v_idx, v_id) != nullptr) {
                    halo_edges[part_id][v_id % shard_num][v_idx][v_id].push_back(u_id);
                    halo_edges[part_id][part_id][u_idx][u_id].push_back(v_id);
                  }
                }
                ngb_offset += u_ptr->ngb_num[i];
              }
              else VLOG(0) << "Build EdgeVertex for Vertex " << u_id << " with Edge " << id_to_edge[search_graphs[u_idx][i]] << " Failed.";
            }

            auto f_ptr = feature_shards[u_idx][part_id]->add_feature_node(u_id, false);
            if (f_ptr != NULL) {
              auto& feature_list = sg_vertex_info[u_idx];
              const int feat_num = feature_list.size();
              for (int feat_id = 0; feat_id < feat_num; ++feat_id) {
                uint64_t feat_val = u_ptr->feat[feat_id];
                parse_shard_feature(u_idx, feature_list[feat_id], feat_val, f_ptr);
              }
            }
            else VLOG(0) << "Build FeatureVertex for Vertex " << u_id << " (" << id_to_feature[u_idx] << ") Failed.";
          }
        }
        return 0;
    }));
  }
  for (int i = 0; i < (int)tasks.size(); i++) tasks[i].get();

  bool is_weighted = false;
  float weight = 1.0;
  tasks.clear();
  for (int part_id = 0; part_id < shard_num; ++part_id) {
    tasks.push_back(
      _shards_task_pool[part_id % task_pool_size_]->enqueue([&, part_id]() -> int {
        for (int _part_id = 0; _part_id < shard_num; ++_part_id) {
          for (int e_idx = 0; e_idx < id_to_edge.size(); ++e_idx) {
            int v_idx = get_idx(e_idx, 1);
            for (auto halo_edge : halo_edges[_part_id][part_id][e_idx]) {
              uint64_t u_id = halo_edge.first;
              auto u_ptr = find_node(0, e_idx, u_id);
              for (auto v_id : halo_edge.second) {
                u_ptr->build_edges(is_weighted);
                u_ptr->add_edge(v_id, weight);
                part_e_cnt[part_id]++;
              }
            }
          }
        }
        return 0;
    }));
  }
  for (int i = 0; i < (int)tasks.size(); i++) tasks[i].get();

  uint64_t v_sum = 0, v_cnt = 0, e_sum = 0, e_cnt = 0;
  for (int i = 0; i < shard_num; ++i) {
    v_sum += part_v_sum[i];
    v_cnt += part_v_cnt[i];
    e_sum += part_e_sum[i];
    e_cnt += part_e_cnt[i];
  }
  e_sum <<= 1;
  VLOG(0) << "[HaloGraph] Read " << v_sum << " vertices. Pick " << v_cnt << " vertices " << 1.0 * v_cnt / v_sum;
  VLOG(0) << "[HaloGraph] Read " << e_sum << " edges.    Pick " << e_cnt << " edges    " << 1.0 * e_cnt / e_sum;
  VLOG(0) << "Finish : Build HaloGraph";
  return 0;
}

uint64_t node_cost(GraphNode* node) {
  assert(node != nullptr);
  uint64_t ret = 0;
  // node->unique_edge();
  ret = node->get_neighbor_size();
  return ret;
}

int GraphTable::metis_partition_coregraph(int subgraph_num,
                                          // std::vector<std::vector<std::vector<std::vector<GraphNode*>>>>& core_nodes,
                                          // std::vector<std::vector<std::vector<std::vector<FeatureNode*>>>>& core_features,
                                          std::vector<std::vector<std::vector<std::vector<uint64_t>>>>& core_vertices,
                                          std::vector<std::vector<std::map<uint64_t, int>>>& vertex_colors) {
  struct MetisCSR {
    // std::map<std::pair<int, uint64_t>, idx_t> g2m;
    // std::vector<std::pair<int, uint64_t>> m2g;
    // std::vector<bool> flag;
    std::vector<idx_t> xadj;
    std::vector<idx_t> adjncy;
    // idx_t node_num = 0;

    std::vector<idx_t> part_res;

    // idx_t get_new_id(uint64_t uid, int utype) {
    //   auto u = std::make_pair(utype, uid);
    //   if (g2m.find(u) == g2m.end()) {
    //     g2m[u] = node_cnt++;
    //     m2g.push_back(u);
    //     flag.push_back(false);
    //     xadj.push_back(0);
    //   }
    //   return g2m[u];
    // }

    // void add_index(uint64_t uid, int utype) {
    //   auto vid = get_new_id(uid, utype);
    //   adjncy.push_back(vid);
    // }

    void part(idx_t part_num) {
      idx_t node_num = xadj.size() - 1;
      part_res.resize(node_num);
      idx_t nWeights = 1;                // 节点权重维数
      idx_t objval;                      // 目标函数值
      int ret = -1;
      idx_t options[METIS_NOPTIONS];

      // For Kway
      // METIS_OPTION_OBJTYPE, METIS_OPTION_CTYPE, METIS_OPTION_IPTYPE,
      // METIS_OPTION_RTYPE, METIS_OPTION_NO2HOP, METIS_OPTION_NCUTS,
      // METIS_OPTION_NITER, METIS_OPTION_UFACTOR, METIS_OPTION_MINCONN,
      // METIS_OPTION_CONTIG, METIS_OPTION_SEED, METIS_OPTION_NUMBERING,
      // METIS_OPTION_DBGLVL
      METIS_SetDefaultOptions(options);
      options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
      options[METIS_OPTION_NUMBERING] = 0;
      options[METIS_OPTION_MINCONN] = 0;


      // ret = METIS_PartGraphRecursive(&node_num, &nWeights, xadj.data(), adjncy.data(),
      //                                 NULL, NULL, NULL, &part_num, NULL,
      //                                 NULL, NULL, &objval, part_res.data());
    
      ret = METIS_PartGraphKway(&node_num, &nWeights, xadj.data(), adjncy.data(),
                                NULL, NULL, NULL, &part_num, NULL,
                                NULL, NULL, &objval, part_res.data());

      if (ret == METIS_OK) {
        VLOG(0) << "METIS Completed for " << part_num << " subgraphs, objval = " << objval << ", status = " << ret;
      }
      if (ret == METIS_ERROR_INPUT) VLOG(0) << "METIS_ERROR_INPUT";
      if (ret == METIS_ERROR_MEMORY) VLOG(0) << "METIS_ERROR_MEMORY";
      if (ret == METIS_ERROR) VLOG(0) << "METIS_ERROR";
    }

    void print_csr(void) {
      // VLOG(0) << "node num : " << node_num;
      // VLOG(0) << "g2m size : " << g2m.size();
      // VLOG(0) << "m2g size : " << m2g.size();
      // VLOG(0) << "flag size : " << flag.size();
      VLOG(0) << "xadj size : " << xadj.size();
      VLOG(0) << "adjncy size : " << adjncy.size();
    }

    void init(idx_t _node_num, idx_t _edge_num) {
      xadj.resize(_node_num);
      adjncy.resize(_edge_num);
    }

    void check_csr() {
      for (auto i : xadj) {
        if (i < 0 || i >= adjncy.size()) {
          VLOG(0) << "adj error.";
        }
      }
      for (auto i : adjncy) {
        if (i < 0 || i >= xadj.size()) {
          VLOG(0) << "adjncy error.";
        }
      }
      VLOG(0) << "csr check completed.";
      return;
    }
  };

  MetisCSR m_csr;

  std::vector<std::map<std::pair<int, uint64_t>, std::pair<idx_t, idx_t>>> part_g2m;
  std::vector<std::pair<int, uint64_t>> m2g;
  std::vector<std::future<std::pair<idx_t, idx_t>>> tasks;
  part_g2m.resize(shard_num);

  // for (int part_id = 0; part_id < shard_num; ++part_id) {
  //   tasks.push_back(_shards_task_pool[part_id % task_pool_size_]->enqueue(
  //     [&, part_id]() mutable -> std::pair<idx_t, idx_t> {
  //       auto &pg2m = part_g2m[part_id];
  //       idx_t part_node_cnt = 0;
  //       idx_t part_neighbor_cnt = 0;
  //       for (int idx = 0; idx < id_to_edge.size(); ++idx) {
  //         auto node_type = paddle::string::split_string<std::string>(id_to_edge[idx], "2");
  //         int src_idx = feature_to_id[node_type[0]];
  //         auto& shards = edge_shards[idx][part_id]->get_bucket();
  //         for (auto node : shards) {
  //           auto uid = node->get_id();
  //           auto u = std::make_pair(src_idx, uid);
  //           // if (find_node(1, src_idx, node->get_id()) == nullptr)
  //           //   VLOG(0) << "Error Vertex " << u.first << '-' << src_idx << ' ' << node->get_id() << '-' << u.second;
  //           if (pg2m.find(u) == pg2m.end()) {
  //             pg2m[u] = {part_node_cnt++, part_neighbor_cnt};
  //             for (auto idx : search_graphs[src_idx]) {
  //               auto _node = find_node(0, idx, uid);
  //               if (_node != nullptr)
  //                 part_neighbor_cnt += _node->get_neighbor_size();
  //             }
  //           }
  //         }
  //       }
  //       return {part_node_cnt, part_neighbor_cnt};
  //     }));
  // }

  for (int part_id = 0; part_id < shard_num; ++part_id) {
    tasks.push_back(_shards_task_pool[part_id % task_pool_size_]->enqueue(
      [&, part_id]() mutable -> std::pair<idx_t, idx_t> {
        auto &pg2m = part_g2m[part_id];
        idx_t part_v_cnt = 0;
        idx_t part_e_cnt = 0;
        for (int u_idx = 0; u_idx < id_to_feature.size(); ++u_idx) {
          auto& shards = feature_shards[u_idx][part_id]->get_bucket();
          for (auto feat_ptr : shards) {
            auto u_id = feat_ptr->get_id();
            auto u = std::make_pair(u_idx, u_id);
            if (pg2m.find(u) == pg2m.end()) {
              pg2m[u] = {part_v_cnt++, part_e_cnt};
              for (auto e_idx : search_graphs[u_idx]) {
                auto u_ptr = find_node(0, e_idx, u_id);
                if (u_ptr != nullptr)
                  part_e_cnt += u_ptr->get_neighbor_size();
              }
            }
          }
        }
        return {part_v_cnt, part_e_cnt};
      }));
  }

  std::vector<idx_t> suffix_cnt(shard_num + 1);
  std::vector<idx_t> suffix_neighbor_cnt(shard_num + 1);
  for (size_t i = 0; i < tasks.size(); i++) {
    auto part_cnt = tasks[i].get();
    // VLOG(0) << i << ' ' << part_cnt.first << ' ' << part_cnt.second;
    suffix_cnt[i + 1] = suffix_cnt[i] + part_cnt.first;
    suffix_neighbor_cnt[i + 1] = suffix_neighbor_cnt[i] + part_cnt.second;
  }
  // VLOG(0) << suffix_cnt[shard_num] << ' ' << suffix_neighbor_cnt[shard_num];
  m_csr.init(suffix_cnt[shard_num], suffix_neighbor_cnt[shard_num]);

  tasks.clear();

  for (int part_id = 0; part_id < shard_num; ++part_id) {
    tasks.push_back(_shards_task_pool[part_id % task_pool_size_]->enqueue(
      [&, part_id]() mutable -> std::pair<idx_t, idx_t> {
        auto xadj_offset = suffix_cnt[part_id];
        auto adjncy_offset = suffix_neighbor_cnt[part_id];
        for (auto u_pair : part_g2m[part_id]) {
          size_t neighbor_pos = adjncy_offset + u_pair.second.second;
          m_csr.xadj[xadj_offset + u_pair.second.first] = neighbor_pos;
          
          for (auto idx : search_graphs[u_pair.first.first]) {
            int dst_idx = get_idx(idx, 1);
            auto node = find_node(0, idx, u_pair.first.second);
            for (size_t i = 0; i < node->get_neighbor_size(); ++i) {
              auto v_id = node->get_neighbor_id(i);
              auto& vpart = part_g2m[v_id % shard_num];
              auto v = std::make_pair(dst_idx, v_id);
              if (vpart.find(v) != vpart.end()) {
                m_csr.adjncy[neighbor_pos++] = vpart[v].first + suffix_cnt[v_id % shard_num];
              }
              else VLOG(0) << "Wrong in Building CSR for METIS.";
            }
          }
        }
        return {0, 0};
      }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();

  m_csr.check_csr();

  m_csr.xadj.push_back(m_csr.adjncy.size());
  m_csr.print_csr();

  // uint64_t _vid = 8109952151798241000;
  // std::pair<int, uint64_t> _u = {0, 8109952151798241000};
  // auto _nuid = 250;
  // VLOG(0) << part_g2m[0][_u].first << ' ' << _nuid << ' ' << m_csr.xadj[_nuid] << "===" << m_csr.xadj[_nuid + 1];
  // for (int64_t i = m_csr.xadj[_nuid]; i < m_csr.xadj[_nuid + 1]; ++i) {
  //   VLOG(0) << "---->" << m_csr.adjncy[i];
  // }
  // VLOG(0) << "-=-=-=-===>" << part_g2m[792][{1, 18360610248344993792ul}].first;
  // VLOG(0) << "-=-=-=-===>" << part_g2m[840][{1, 13093045838639267840ul}].first;
  // VLOG(0) << "-=-=-=-===>" << part_g2m[638][{1, 14681717957525896638ul}].first;
  // return 0;

  m_csr.part(subgraph_num);

  // for (size_t muid = 0; muid < m_csr.node_cnt; ++muid) {
  //   auto u = m_csr.m2g[muid];
  //   auto uid = u.first;
  //   auto utype = u.second;
  //   for (auto idx : search_graphs[utype]) {
  //     auto g_ptr = dynamic_cast<GraphNode*>(find_node(0, idx, uid));
  //     if (g_ptr != nullptr)
  //       core_nodes[idx][m_csr.part_res[muid]][uid % shard_num].push_back(g_ptr);
  //   }
  //   auto f_ptr = dynamic_cast<FeatureNode*>(find_node(1, utype, uid));
  //   if (f_ptr != nullptr)
  //     core_features[utype][m_csr.part_res[muid]][uid % shard_num].push_back(f_ptr);
  // }

  tasks.clear();
  for (int part_id = 0; part_id < shard_num; ++part_id) {
    tasks.push_back(_shards_task_pool[part_id % task_pool_size_]->enqueue(
      [&, part_id]() mutable -> std::pair<idx_t, idx_t> {
        auto xadj_offset = suffix_cnt[part_id];
        for (auto u_pair : part_g2m[part_id]) {
          auto src_idx = u_pair.first.first;
          // if ((src_idx & 1) == 0) {
          auto u_id = u_pair.first.second;
          auto sg_id = m_csr.part_res[xadj_offset + u_pair.second.first];
          if (sg_id >= subgraph_num || sg_id < 0) {
            VLOG(0) << "Wrong ShardGraph ID.";
            continue;
          }

          if (find_node(1, src_idx, u_id) != nullptr) {
            core_vertices[part_id][src_idx][sg_id].push_back(u_id);
            vertex_colors[src_idx][part_id][u_id] = sg_id;
          }
          else VLOG(0) << "ERROR : nullptr with " << u_id;

          // for (auto idx : search_graphs[src_idx]) {
          //   // auto g_ptr = dynamic_cast<GraphNode*>(find_node(0, idx, u_id));
          //   // if (g_ptr != nullptr) {
          //   if (find_node(0, idx, u_id) != nullptr) {
          //     // core_nodes[idx][sg_id][part_id].push_back(g_ptr);
          //     core_vertices[part_id][idx][sg_id].push_back(u_id);
          //     vertex_colors[idx][part_id][u_id] = sg_id;
          //     // auto f_ptr = dynamic_cast<FeatureNode*>(find_node(1, src_idx, uid));
          //     // if (f_ptr != nullptr)
          //       // core_features[src_idx][sg_id][part_id].push_back(f_ptr);
              
          //     // auto node_type = paddle::string::split_string<std::string>(id_to_edge[idx], "2");
          //     // int dst_idx = feature_to_id[node_type[1]];
          //     // for (size_t i = 0; i < g_ptr->get_neighbor_size(); ++i) {
          //     //   auto vid = g_ptr->get_neighbor_id(i);
          //     //   auto f_ptr = dynamic_cast<FeatureNode*>(find_node(1, dst_idx, vid));
          //     //   if (f_ptr != nullptr)
          //     //     core_features[dst_idx][sg_id][part_id].push_back(f_ptr);
          //     // }
          //   }
          //   else VLOG(0) << "ERROR : nullptr with " << u_id;
          // }
          // }
        }
        return {0, 0};
      }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  VLOG(0) << "METIS Partition Completed.";
  return 0;
}

int GraphTable::quick_partition_coregraph(int subgraph_num,
                                          // std::vector<std::vector<std::vector<std::vector<GraphNode*>>>>& core_nodes,
                                          // std::vector<std::vector<std::vector<std::vector<FeatureNode*>>>>& core_features,
                                          std::vector<std::vector<std::vector<std::vector<uint64_t>>>>& core_vertices,
                                          std::vector<std::vector<std::map<uint64_t, int>>>& vertex_colors) {
  std::vector<std::future<int64_t>> tasks;
  // for (int part_id = 0; part_id < shard_num; ++part_id) {
  //   tasks.push_back(_shards_task_pool[part_id % task_pool_size_]->enqueue(
  //     [&, part_id]() mutable -> int64_t {
  //       for (int idx = 0; idx < id_to_edge.size(); idx += 2) {
  //         auto node_type = paddle::string::split_string<std::string>(id_to_edge[idx], "2");
  //         int src_idx = feature_to_id[node_type[0]];

  //         auto& shards = edge_shards[idx][part_id]->get_bucket();
  //         size_t v_num = shards.size();
  //         size_t v_shard_num = (v_num + subgraph_num - 1) / subgraph_num;
  //         for (int shard_id = 0; shard_id < subgraph_num; ++shard_id) {
  //           size_t v_start = shard_id * v_shard_num, v_end = std::min((shard_id + 1) * v_shard_num, v_num);
  //           for (; v_start < v_end; ++v_start) {
  //             auto node = shards[v_start];
  //             core_nodes[idx >> 1][shard_id][part_id].push_back(dynamic_cast<GraphNode*>(shards[v_start]));
  //             vertex_colors[idx >> 1][node->get_id()] = shard_id;
  //             auto feature = dynamic_cast<FeatureNode*>(find_node(1, src_idx, node->get_id()));
  //             if (feature != nullptr) // To Check: Ignore the vertices which can not be find in the feature_shards
  //               core_features[src_idx][shard_id][part_id].push_back(feature);
  //           }
  //         }
  //       }
  //       return 0;
  //     }));
  // }
  // for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();

  // for (int part_id = 0; part_id < shard_num; ++part_id) {
  //   tasks.push_back(_shards_task_pool[part_id % task_pool_size_]->enqueue(
  //     [&, part_id]() -> int64_t {
  //       for (int e_idx = 0; e_idx < id_to_edge.size(); ++e_idx) {
  //         int u_idx = get_idx(e_idx, 0);
  //         auto& shards = edge_shards[e_idx][part_id]->get_bucket();
  //         size_t u_num = shards.size();
  //         size_t u_sg_num = (u_num + subgraph_num - 1) / subgraph_num;
  //         for (int sg_id = 0; sg_id < subgraph_num; ++sg_id) {
  //           size_t u_start = sg_id * u_sg_num, u_end = std::min((sg_id + 1) * u_sg_num, u_num);
  //           for (; u_start < u_end; ++u_start) {
  //             auto u_id = shards[u_start]->get_id();
  //             core_vertices[part_id][u_idx][sg_id].push_back(u_id);
  //             vertex_colors[u_idx][part_id][u_id] = sg_id;
  //           }
  //         }
  //       }
  //       return 0;
  //     }));
  // }

  for (int part_id = 0; part_id < shard_num; ++part_id) {
    tasks.push_back(_shards_task_pool[part_id % task_pool_size_]->enqueue(
      [&, part_id]() -> int64_t {
        for (int u_idx = 0; u_idx < id_to_feature.size(); ++u_idx) {
          auto& shards = feature_shards[u_idx][part_id]->get_bucket();
          size_t u_num = shards.size();
          size_t u_sg_num = (u_num + subgraph_num - 1) / subgraph_num;
          for (int sg_id = 0; sg_id < subgraph_num; ++sg_id) {
            size_t u_start = sg_id * u_sg_num, u_end = std::min((sg_id + 1) * u_sg_num, u_num);
            for (; u_start < u_end; ++u_start) {
              auto u_id = shards[u_start]->get_id();
              core_vertices[part_id][u_idx][sg_id].push_back(u_id);
              vertex_colors[u_idx][part_id][u_id] = sg_id;
            }
          }
        }
        return 0;
      }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  VLOG(0) << "Quick Partition Completed.";

  // tasks.clear();
  // for (int i = 0; i < core_features.size() * subgraph_num; ++i) {
  //   tasks.push_back(_shards_task_pool[i % task_pool_size_]->enqueue(
  //     [&, i]() mutable -> int64_t {
  //       int dst_idx = i / subgraph_num;
  //       int shard_id = i % subgraph_num;
  //       auto dst_type = id_to_feature[dst_idx];

  //       for (int idx = 0; idx < core_nodes.size(); ++idx) {
  //         auto node_type = paddle::string::split_string<std::string>(id_to_edge[idx << 1], "2");
  //         if (node_type[node_type.size() - 1] == dst_type) {
  //           for (int part_id = 0; part_id < shard_num; ++part_id) {
  //             for (auto node : core_nodes[idx][shard_id][part_id]) {
  //               for (size_t j = 0; j < node->get_neighbor_size(); ++j) {
  //                 auto vid = node->get_neighbor_id(j);
  //                 auto vfeature = dynamic_cast<FeatureNode*>(find_node(1, dst_idx, vid));
  //                 if (vfeature != nullptr)
  //                   core_features[dst_idx][shard_id][vid % shard_num].push_back(vfeature);
  //               }
  //             }
  //           }
  //         }
  //       }
  //       return 0;
  //     }));
  // }
  // for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  return 0;
}

int GraphTable::build_coregraph(int subgraph_num,
                                // std::vector<std::vector<std::vector<std::vector<GraphNode*>>>>& core_nodes,
                                // std::vector<std::vector<std::vector<std::vector<FeatureNode*>>>>& core_features,
                                std::vector<std::vector<std::vector<std::vector<uint64_t>>>>& core_vertices,
                                std::vector<std::vector<std::map<uint64_t, int>>>& vertex_colors,
                                std::string part_method,
                                const std::string& subgraph_path) {
  if (search_graphs.empty()) {
    search_graphs.resize(id_to_feature.size());
    for (int idx = 0; idx < id_to_edge.size(); ++idx) {
      auto node_type = paddle::string::split_string<std::string>(id_to_edge[idx], "2");
      int utype_id = feature_to_id[node_type[0]];
      search_graphs[utype_id].push_back(idx);
    }
  }

  std::vector<std::future<int64_t>> tasks;
  for (int part_id = 0; part_id < shard_num; ++part_id) {
  tasks.push_back(_shards_task_pool[part_id % task_pool_size_]->enqueue(
    [&, part_id]() -> int64_t {
      for (int e_idx = 0; e_idx < id_to_edge.size(); ++e_idx) {
        int u_idx = get_idx(e_idx, 0);
        auto& shards = edge_shards[e_idx][part_id]->get_bucket();
        for (auto e_ptr : shards) {
          auto u_id = e_ptr->get_id();
          if (find_node(1, u_idx, u_id) == nullptr) {
            // VLOG(0) << u_id << " does not have features.";
            feature_shards[u_idx][part_id]->add_feature_node(u_id, false);
          }
        }
      }
      return 0;
    }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();

  if (part_method == "metis") {
    // metis_partition_coregraph(subgraph_num, core_nodes, core_features, vertex_colors);
    metis_partition_coregraph(subgraph_num, core_vertices, vertex_colors);
  }
  else {
    if (part_method == "quick") {
      // quick_partition_coregraph(subgraph_num, core_nodes, core_features, vertex_colors);
      quick_partition_coregraph(subgraph_num, core_vertices, vertex_colors);
    }
    else {
      random_partition_coregraph(subgraph_num, core_vertices, vertex_colors);
    }
  }
  // write_coregraph(subgraph_num, core_nodes, core_features, vertex_colors, subgraph_path);
  write_coregraph(subgraph_num, core_vertices, vertex_colors, subgraph_path);
  return 0;
}

int GraphTable::normal_partition_coregraph(int subgraph_num,
                                           std::vector<std::vector<std::vector<std::vector<GraphNode*>>>>& core_nodes,
                                           std::vector<std::vector<std::vector<std::vector<FeatureNode*>>>>& core_features) {
  std::vector<uint64_t> shard_graph_size;
  shard_graph_size.resize(subgraph_num);

  for (int idx = 0; idx < id_to_edge.size(); idx += 2) {
    auto node_type = paddle::string::split_string<std::string>(id_to_edge[idx], "2");
    int src_idx = feature_to_id[node_type[0]];
    int dst_idx = feature_to_id[node_type[1]];
    
    std::priority_queue<std::pair<uint64_t, int>, std::vector<std::pair<uint64_t, int>>, std::greater<std::pair<uint64_t, int>>> pri_q;

    for (int i = 0; i < subgraph_num; ++i) {
      pri_q.push(std::make_pair(shard_graph_size[i], i));
    }

    for (auto edge_shard : edge_shards[idx]) {
      auto& shards = edge_shard->get_bucket();
      for (auto node : shards) {
        auto shard_info = pri_q.top();
        pri_q.pop();
        shard_info.first += node_cost(dynamic_cast<GraphNode*>(node));
        pri_q.push(shard_info);

        uint64_t id = node->get_id();
        core_nodes[idx >> 1][shard_info.second][id % shard_num].push_back(dynamic_cast<GraphNode*>(node));

        auto feature = dynamic_cast<FeatureNode*>(find_node(1, src_idx, id));
        if (feature != nullptr) // To Check: Ignore the vertices which can not be find in the feature_shards
          core_features[src_idx][shard_info.second][id % shard_num].push_back(feature);
        for (size_t i = 0; i < node->get_neighbor_size(); ++i) {
          auto vid = node->get_neighbor_id(i);
          auto vfeature = dynamic_cast<FeatureNode*>(find_node(1, dst_idx, vid));
          if (vfeature != nullptr)
            core_features[dst_idx][shard_info.second][vid % shard_num].push_back(vfeature);
        }
      }
    }

    while(!pri_q.empty()) {
      auto shard_info = pri_q.top();
      pri_q.pop();
      shard_graph_size[shard_info.second] = shard_info.first;
    }

    VLOG(0) << "Normal Partitioning for " << id_to_edge[idx] << " Completed.";
  }

  // output result
  return 0;
}

int GraphTable::random_partition_coregraph(int subgraph_num,
                                           std::vector<std::vector<std::vector<std::vector<u_int64_t>>>>& core_vertices,
                                           std::vector<std::vector<std::map<uint64_t, int>>>& vertex_colors) {
  std::vector<std::future<int64_t>> tasks;
  for (int part_id = 0; part_id < shard_num; ++part_id) {
    tasks.push_back(_shards_task_pool[part_id % task_pool_size_]->enqueue(
      [&, part_id]() -> int64_t {
        std::random_device rd;
        // std::default_random_engine gen(rd());
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> sg_id_random(0, subgraph_num - 1);

        for (int u_idx = 0; u_idx < id_to_feature.size(); ++u_idx) {
          auto& shards = feature_shards[u_idx][part_id]->get_bucket();
          for (auto u_ptr : shards) {
            auto u_id = u_ptr->get_id();
            int sg_id = sg_id_random(gen);
            core_vertices[part_id][u_idx][sg_id].push_back(u_id);
            vertex_colors[u_idx][part_id][u_id] = sg_id;
          }
        }
        return 0;
      }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  VLOG(0) << "Random Partition Completed.";
  return 0;
}

std::string get_subgraph_filepath(const std::string& part_path, int sg_id, const std::string& graph_name, int part_id) {
  std::string pstr = std::to_string(part_id);
  for (int i = pstr.size(); i < 5; ++i) pstr = "0" + pstr;
  auto ret = part_path + "sg_" + std::to_string(sg_id) + "/" + graph_name + "/part-" + pstr;
  return ret;
}

int GraphTable::write_coregraph(int subgraph_num,
                                // std::vector<std::vector<std::vector<std::vector<GraphNode*>>>>& core_nodes,
                                // std::vector<std::vector<std::vector<std::vector<FeatureNode*>>>>& core_features,
                                std::vector<std::vector<std::vector<std::vector<uint64_t>>>>& core_vertices,
                                std::vector<std::vector<std::map<uint64_t, int>>>& vertex_colors,
                                const std::string& subgraph_path) {
  std::vector<std::future<int64_t>> tasks;
  VLOG(0) << "Begin : Write CoreGraph into Disk.";

  /*for (int shard_id = 0; shard_id < subgraph_num * id_to_feature.size(); ++shard_id) {
    tasks.push_back(_shards_task_pool[shard_id % task_pool_size_]->enqueue(
      [&, shard_id]() mutable -> int64_t {
          std::ofstream fout;
          int idx = shard_id / subgraph_num;
          shard_id %= subgraph_num;
          const auto& utype = id_to_feature[idx];
          uint64_t op_sum = 0;

          for (int part_id = 0; part_id < shard_num; ++part_id) {
            auto& shard_node = core_features[idx][shard_id][part_id];

            std::sort(shard_node.begin(), shard_node.end());
            shard_node.erase(std::unique(shard_node.begin(), shard_node.end()), shard_node.end());

            auto shard_file_path = get_subgraph_filepath(subgraph_path, shard_id, "node_" + utype, part_id);
            fout.open(shard_file_path, std::ios::binary);
            if (!fout.is_open()) {
              VLOG(0) << "open output file " << shard_file_path << " failed!";
            }

            for (auto node : shard_node) {
              auto uid = node->get_id();
        
              std::vector<uint64_t> feature_ids;
              node->get_feature_ids(&feature_ids);

              fout.write((char*)(&uid), sizeof(uint64_t));
              for (auto feature_id : feature_ids)
                fout.write((char*)(&feature_id), sizeof(uint64_t));
              ++op_sum;
            }
            fout.close();
          }
          VLOG(0) << "shard_" << shard_id << " " << utype << " vertex_sum = " << op_sum;
          return 0;
      }));
  }

  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  tasks.clear();

  for (int shard_id = 0; shard_id < subgraph_num * core_nodes.size(); ++shard_id) {
    tasks.push_back(_shards_task_pool[shard_id % task_pool_size_]->enqueue(
      [&, shard_id]() mutable -> int64_t {
        int idx = shard_id / subgraph_num;
        shard_id %= subgraph_num;
        uint64_t v_sum = 0, e_sum = 0; // inv_v_sum = 0, inv_e_sum = 0;
        // std::map<uint64_t, std::vector<uint64_t>> inv_csr;

        for (int part_id = 0; part_id < shard_num; ++part_id) {
          std::ofstream fout;
          auto shard_file_path = get_subgraph_filepath(subgraph_path, shard_id, id_to_edge[idx << 1], part_id);
          fout.open(shard_file_path, std::ios::binary);
          if (!fout.is_open()) {
            VLOG(0) << "open output file " << shard_file_path << " failed!";
          }

          for (auto node : core_nodes[idx][shard_id][part_id]) {
            auto u = node->get_id();
            ++v_sum;
            fout.write((char*)(&u), sizeof(uint64_t));
            auto deg = node->get_neighbor_size();
            uint64_t sg_deg = 0;
            for (size_t k = 0; k < deg; ++k) {
              auto v_id = node->get_neighbor_id(k);
              if (vertex_colors[idx][v_id % shard_num][v_id] == shard_id) ++sg_deg;
            }
            fout.write((char*)(&sg_deg), sizeof(uint64_t));
            for (size_t k = 0; k < deg; ++k) {
              auto v_id = node->get_neighbor_id(k);
              if (vertex_colors[idx][v_id % shard_num][v_id] == shard_id) {
                // float w = node->get_neighbor_weight(k);
                fout.write((char*)(&v_id), sizeof(uint64_t));
                // fout.write((char*)(&w), sizeof(float)); // TODO : Optimize the Weight     
                // inv_csr[v].push_back(u);
                ++e_sum;
              }
            }
          }
          fout.close();
        }

        // std::ofstream *fouts = new std::ofstream[shard_num];
        // for (int part_id = 0; part_id < shard_num; ++part_id) {
        //   auto shard_file_path = get_subgraph_filepath(part_path, shard_id, id_to_edge[(idx << 1) | 1], part_id);
        //   fouts[part_id].open(shard_file_path, std::ios::binary);
        //   if (!fouts[part_id].is_open()) {
        //     VLOG(0) << "open output file " << shard_file_path << " failed!";
        //   }
        // }
        // for (auto v_csr : inv_csr) {
        //   auto v = v_csr.first;
        //   auto& fout = fouts[v % shard_num];
        //   fout.write((char*)(&v), sizeof(uint64_t));
        //   auto deg = v_csr.second.size();
        //   fout.write((char*)(&deg), sizeof(uint64_t));
        //   for (auto u : v_csr.second) {
        //     fout.write((char*)(&u), sizeof(uint64_t));
        //     ++inv_e_sum;
        //   }
        //   ++inv_v_sum;
        // }
        // for (int part_id = 0; part_id < shard_num; ++part_id)
        //   fouts[part_id].close();
        // VLOG(0) << "shard_" << shard_id << " " << id_to_edge[idx << 1] << " sum_vertex = " << v_sum << "/" << inv_v_sum << ", sum_edge = " << e_sum << "/" << inv_e_sum;
        VLOG(0) << "shard_" << shard_id << " " << id_to_edge[idx << 1] << " sum_vertex = " << v_sum << ", sum_edge = " << e_sum;
      return 0;
    }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  tasks.clear();
  */
  
  
  std::vector<std::vector<std::vector<uint64_t>>> part_v_cnt(shard_num);
  std::vector<std::vector<std::vector<uint64_t>>> part_e_cnt(shard_num);
  std::vector<std::vector<std::vector<uint64_t>>> part_e_sum(shard_num);

  for (int part_id = 0; part_id < shard_num; ++part_id) {
    tasks.push_back(_shards_task_pool[part_id % task_pool_size_]->enqueue(
      [&, part_id]() mutable -> int64_t {
        part_v_cnt[part_id].resize(id_to_feature.size());
        part_e_cnt[part_id].resize(id_to_edge.size());
        part_e_sum[part_id].resize(id_to_edge.size());

        for (int i = 0; i < id_to_feature.size(); ++i)
          part_v_cnt[part_id][i].resize(subgraph_num);
        
        for (int i = 0; i < id_to_edge.size(); ++i) {
          part_e_cnt[part_id][i].resize(subgraph_num);
          part_e_sum[part_id][i].resize(subgraph_num);
        }
        
        for (int u_idx = 0; u_idx < id_to_feature.size(); ++u_idx) {
          for (int sg_id = 0; sg_id < subgraph_num; ++sg_id) {
            std::ofstream *fout_e = new std::ofstream[id_to_edge.size()];
            for (int e_idx = 0; e_idx < id_to_edge.size(); ++e_idx) {
              std::string sg_e_path = get_subgraph_filepath(subgraph_path, sg_id, id_to_edge[e_idx], part_id);
              fout_e[e_idx].open(sg_e_path, std::ios::binary | std::ios::out | std::ios::app);
              if (!fout_e[e_idx].is_open()) {
                VLOG(0) << "open output file " << sg_e_path << " failed!";
              }
            }

            std::string sg_v_path = get_subgraph_filepath(subgraph_path, sg_id, "node_" + id_to_feature[u_idx], part_id);
            std::ofstream fout_v;
            fout_v.open(sg_v_path, std::ios::binary | std::ios::out);
            if (!fout_v.is_open()) {
              VLOG(0) << "open output file " << sg_v_path << " failed!";
            }
              
            for (auto u_id : core_vertices[part_id][u_idx][sg_id]) {
              auto f_ptr = find_node(1, u_idx, u_id);
              
              if (f_ptr != nullptr) {
                std::vector<uint64_t> feature_ids;
                f_ptr->get_feature_ids(&feature_ids);
                fout_v.write((char*)(&u_id), sizeof(uint64_t));
                for (auto feature_id : feature_ids)
                  fout_v.write((char*)(&feature_id), sizeof(uint64_t));
                part_v_cnt[part_id][u_idx][sg_id]++;
              }

              for (auto e_idx : search_graphs[u_idx]) {
                auto e_ptr = find_node(0, e_idx, u_id);
                if (e_ptr != nullptr) {
                  int v_idx = get_idx(e_idx, 1);
                  auto deg = e_ptr->get_neighbor_size();
                  std::vector<uint64_t> sg_ngb;
                  for (uint64_t ngb_id = 0; ngb_id < deg; ++ngb_id) {
                    auto v_id = e_ptr->get_neighbor_id(ngb_id);
                    if (sg_id == vertex_colors[v_idx][v_id % shard_num][v_id])
                      sg_ngb.push_back(v_id);
                  }
                  part_e_sum[part_id][e_idx][sg_id] += deg;

                  uint64_t ngb_num = sg_ngb.size();
                  fout_e[e_idx].write((char*)(&u_id), sizeof(uint64_t));
                  fout_e[e_idx].write((char*)(&ngb_num), sizeof(uint64_t));
                  for (auto v_id : sg_ngb)
                    fout_e[e_idx].write((char*)(&v_id), sizeof(uint64_t));
                  
                  part_e_cnt[part_id][e_idx][sg_id] += ngb_num;
                }
              }
            }

            for (int e_idx = 0; e_idx < id_to_edge.size(); ++e_idx) {
              if (fout_e[e_idx].is_open())
                fout_e[e_idx].close();
            }
            delete[] fout_e;
          }
        }
        return 0;
      }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  
  for (int idx = 0; idx < id_to_feature.size(); ++idx) {
    for (int sg_id = 0; sg_id < subgraph_num; ++sg_id) {
      uint64_t v_sum = 0;
      for (int part_id = 0; part_id < shard_num; ++part_id) {
        v_sum += part_v_cnt[part_id][idx][sg_id];
      }
      VLOG(0) << id_to_feature[idx] << " in coregraph_" << sg_id << " : " << v_sum << " vertices";
    }
  }

  uint64_t graph_e_cnt = 0, graph_e_sum = 0;
  for (int idx = 0; idx < id_to_edge.size(); ++idx) {
    uint64_t sg_e_cnt = 0, sg_e_sum = 0;
    for (int sg_id = 0; sg_id < subgraph_num; ++sg_id) {
      uint64_t e_cnt = 0, e_sum = 0;
      for (int part_id = 0; part_id < shard_num; ++part_id) {
        e_cnt += part_e_cnt[part_id][idx][sg_id];
        e_sum += part_e_sum[part_id][idx][sg_id];
      }
      VLOG(0) << id_to_edge[idx] << " in coregraph_" << sg_id << " : " << e_cnt << " / " << e_sum << " edges\t" << 1.0 * e_cnt / e_sum;
      sg_e_cnt += e_cnt;
      sg_e_sum += e_sum;
    }
    graph_e_cnt += sg_e_cnt;
    graph_e_sum += sg_e_sum;
  }
  VLOG(0) << "E_CNT = " << graph_e_cnt << ", E_SUM = " << graph_e_sum << "\t" << 1.0 * graph_e_cnt / graph_e_sum;

  std::ofstream fout(subgraph_path + "ginfo");
  for (auto node_info : feature_to_id) {
    auto u_name = node_info.first;
    auto u_idx = node_info.second;
    for (auto shard : feature_shards[u_idx]) {
      auto tmp_bucket = shard->get_bucket();
      if (!tmp_bucket.empty()) { 
        auto tmp_node = tmp_bucket[0];
        std::vector<uint64_t> feature_ids;
        std::vector<std::string> valid_feature_name;
        for (int i = 0; i < tmp_node->get_feature_size(); ++i) {
          feature_ids.clear();
          tmp_node->get_feature_ids(i, &feature_ids);
          if (!feature_ids.empty()) {
            for (auto it : feat_id_map[u_idx]) {
              if (it.second == i) {
                valid_feature_name.push_back(it.first);
                break;
              }
            }
          }
        }
        fout << u_name << '\t' << valid_feature_name.size();
        for (auto vfn : valid_feature_name) fout << '\t' << vfn;
        fout << std::endl;
        break;
      }
      else VLOG(0) << "Nodetype " << u_name << " does not exist.";
    }
  }
  fout.close();
  VLOG(0) << "Finish : Write CoreGraph into Disk.";
  return 0;
}

int GraphTable::get_idx(int e_idx, int uv_pos) {
  auto u2v = paddle::string::split_string<std::string>(id_to_edge[e_idx], "2");
  return feature_to_id[u2v[uv_pos]];
}

int GraphTable::build_halograph(const int subgraph_num,
                                const int layer_num,
                                std::vector<std::vector<std::vector<std::vector<uint64_t>>>>& core_vertices,
                                std::vector<std::vector<std::map<uint64_t, int>>>& vertex_colors, 
                                const std::string& subgraph_path) {
  std::vector<std::future<int64_t>> tasks;
  struct HaloVertex {
    std::vector<uint64_t> color_deg;
    std::vector<int> color_dist;

    HaloVertex(int subgraph_num) {
      color_deg.clear();
      color_deg.resize(subgraph_num);
      color_dist.clear();
      color_dist.resize(subgraph_num);
    }

    void update_halodeg(int color, int dist) {
      auto& c_dist = color_dist[color];
      if (c_dist == 0) {
        c_dist = dist;
        color_deg[color] = 1;
      }
      else {
        if (c_dist > dist) {
          c_dist = dist;
          color_deg[color] = 1;
        }
        else {
          color_deg[color] += (c_dist == dist);
        }
      }
    }

    double get_norm_weight(int color, uint64_t max_num) {
      if (color_dist[color]) {
        return 1.0 * color_deg[color] / max_num;
      }
      return 0.0;
    }
  };

  std::vector<std::vector<std::map<uint64_t, HaloVertex*>>> halo_vertices;
  halo_vertices.resize(shard_num);
  std::vector<std::vector<uint64_t>> part_max_deg;
  part_max_deg.resize(shard_num);

  VLOG(0) << "Begin : Build HaloGraph";
  for (int part_id = 0; part_id < shard_num; ++part_id) {
    tasks.push_back(_shards_task_pool[part_id % task_pool_size_]->enqueue(
      [&, part_id]() -> int64_t {
        halo_vertices[part_id].resize(id_to_feature.size());
        part_max_deg[part_id].resize(subgraph_num);
        for (int u_idx = 0; u_idx < id_to_edge.size(); ++u_idx) {
          for (int sg_id = 0; sg_id < subgraph_num; ++sg_id) {
            HaloVertex* hv_ptr = new HaloVertex(subgraph_num);
            for (auto u_id : core_vertices[part_id][u_idx][sg_id]) {
              std::set<std::pair<int, uint64_t>> visited;
              std::deque<std::pair<int, uint64_t>> frontier;
              auto u = std::make_pair(u_idx, u_id);
              visited.insert(u);
              frontier.push_back(u);
              uint64_t frontier_size = 1;
              int u_color = vertex_colors[u_idx][part_id][u_id];
              
              for (int layer_dist = 1; layer_dist <= layer_num; ++layer_dist) {
                while (frontier_size--) {
                  auto v = frontier.front();
                  frontier.pop_front();

                  int v_idx = v.first;
                  uint64_t v_id  = v.second;
                  for (auto e_idx : search_graphs[v_idx]) {
                    auto v_ptr = find_node(0, e_idx, v_id);  
                    auto deg = v_ptr->get_neighbor_size();
                    int x_idx = get_idx(e_idx, 1);

                    for (uint64_t ngb_id = 0; ngb_id < deg; ++ngb_id) {
                      auto x_id = v_ptr->get_neighbor_id(ngb_id);
                      auto x = std::make_pair(x_idx, x_id);
                      if (visited.find(x) == visited.end()) {
                        visited.insert(x);
                        frontier.push_back(x);
                        int x_color = vertex_colors[x_idx][x_id % shard_num][x_id];
                        if (x_color != sg_id) {
                          hv_ptr->update_halodeg(x_color, layer_dist);
                        }
                      }
                    }
                  }
                }
                frontier_size = frontier.size();
              }
              for (int sg_id = 0; sg_id < subgraph_num; ++sg_id) {
                if (hv_ptr->color_dist[sg_id]) {
                  part_max_deg[part_id][sg_id] = std::max(part_max_deg[part_id][sg_id], hv_ptr->color_deg[sg_id]);
                }
              }
              halo_vertices[part_id][u_idx][u_id] = hv_ptr;
            }
          }
        }
        return 0;
      }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();

  std::vector<uint64_t> halodeg_max(subgraph_num, 0);
  for (int part_id = 0; part_id < shard_num; ++part_id) {
    for (int sg_id = 0; sg_id < subgraph_num; ++sg_id) {
      if (part_max_deg[part_id][sg_id] > halodeg_max[sg_id])
        halodeg_max[sg_id] = part_max_deg[part_id][sg_id];
    }
  }

  VLOG(0) << "Finish : Build HaloGraph";

  std::vector<std::vector<uint64_t>> part_v_cnt(shard_num);
  std::vector<std::vector<uint64_t>> part_e_cnt(shard_num);
  tasks.clear();
  VLOG(0) << "Begin : Write HaloGraph into Disk.";
  for (int part_id = 0; part_id < shard_num; ++part_id) {
    tasks.push_back(_shards_task_pool[part_id % task_pool_size_]->enqueue(
      [&, part_id]() mutable -> int64_t {
        part_v_cnt[part_id].resize(subgraph_num);
        part_e_cnt[part_id].resize(subgraph_num);
        std::ofstream *fouts = new std::ofstream[subgraph_num];
        for (int sg_id = 0; sg_id < subgraph_num; ++sg_id) {
          std::string halo_path = subgraph_path + "sg_" + std::to_string(sg_id) + "/halo/hg-" + std::to_string(part_id);
          fouts[sg_id].open(halo_path, std::ios::out | std::ios::binary);
          if (!fouts[sg_id].is_open()) {
            VLOG(0) << "open file " << halo_path << "failed";
          }
        }

        for (int u_idx = 0; u_idx < id_to_feature.size(); ++u_idx) {
          for (auto halo_pair : halo_vertices[part_id][u_idx]) {
            auto u_id = halo_pair.first;
            auto halo_ptr = halo_pair.second;
            for (int sg_id = 0; sg_id < subgraph_num; ++sg_id) {
              int u_dist = halo_ptr->color_dist[sg_id];
              if (u_dist != 0) {
                part_v_cnt[part_id][sg_id]++;
                fouts[sg_id].write((char*)(&u_id), sizeof(uint64_t));
                fouts[sg_id].write((char*)(&u_idx), sizeof(int));
                double op_w = halo_ptr->get_norm_weight(sg_id, halodeg_max[sg_id]);
                fouts[sg_id].write((char*)(&u_dist), sizeof(int));
                fouts[sg_id].write((char*)(&op_w), sizeof(double));

                for (auto e_idx : search_graphs[u_idx]) {
                  auto u_ptr = find_node(0, e_idx, u_id);
                  if (u_ptr != nullptr) {
                    int v_idx = get_idx(e_idx, 1);
                    auto deg = u_ptr->get_neighbor_size();
                    std::vector<uint64_t> sg_ngb;
                    for (uint64_t ngb_id = 0; ngb_id < deg; ++ngb_id) {
                      auto v_id = u_ptr->get_neighbor_id(ngb_id);
                      if (sg_id == vertex_colors[v_idx][v_id % shard_num][v_id] || halo_vertices[v_id % shard_num][v_idx][v_id]->color_dist[sg_id] == u_dist - 1 )
                        sg_ngb.push_back(v_id);
                    }
                    uint64_t ngb_num = sg_ngb.size();
                    fouts[sg_id].write((char*)(&ngb_num), sizeof(uint64_t));
                    for (auto v_id : sg_ngb)
                      fouts[sg_id].write((char*)(&v_id), sizeof(uint64_t));
                    part_e_cnt[part_id][sg_id] += ngb_num;
                  }
                }

                auto f_ptr = find_node(1, u_idx, u_id);
                std::vector<uint64_t> feature_ids;
                f_ptr->get_feature_ids(&feature_ids);
                for (auto feature_id : feature_ids)
                  fouts[sg_id].write((char*)(&feature_id), sizeof(uint64_t));
              }
            }
          }
        }

        for (int sg_id = 0; sg_id < subgraph_num; ++sg_id) {
          if (fouts[sg_id].is_open()) {
            fouts[sg_id].close();
          }
        }
        delete[] fouts;
        return 0;
      }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();

  for (int sg_id = 0; sg_id < subgraph_num; ++sg_id) {
    uint64_t v_sum = 0;
    for (int part_id = 0; part_id < shard_num; ++part_id) {
      v_sum += part_v_cnt[part_id][sg_id];
    }
    VLOG(0) << "halograph_" << sg_id << " : " << v_sum << " vertices";
  }

  for (int sg_id = 0; sg_id < subgraph_num; ++sg_id) {
    uint64_t e_sum = 0;
    for (int part_id = 0; part_id < shard_num; ++part_id) {
      e_sum += part_e_cnt[part_id][sg_id];
    }
    VLOG(0) << "halograph_" << sg_id << " : " << e_sum << " edges";
  }

  VLOG(0) << "Finish : Write HaloGraph into Disk.";
  return 0;
}

int32_t GraphTable::build_subgraph_file(int subgraph_num,
                                        int layer_num,
                                        const std::string& subgraph_path, 
                                        std::string part_method,
                                        bool build_halo) {
  std::vector<std::future<int64_t>> tasks;

  // std::vector<std::vector<std::vector<std::vector<GraphNode*>>>> core_vertices;
  // std::vector<std::vector<std::vector<std::vector<FeatureNode*>>>> core_features;
  std::vector<std::vector<std::vector<std::vector<uint64_t>>>> core_vertices;
  std::vector<std::vector<std::map<uint64_t, int>>> vertex_colors;

  // core_vertices.resize(id_to_edge.size());
  // for (int i = 0; i < (id_to_edge.size()); ++i) {
  //   core_vertices[i].resize(subgraph_num);
  //   for (int j = 0; j < subgraph_num; ++j)
  //     core_vertices[i][j].resize(shard_num);
  // }

  // core_features.resize(id_to_feature.size());
  // for (int i = 0; i < id_to_feature.size(); ++i) {
  //   core_features[i].resize(subgraph_num);
  //   for (int j = 0; j < subgraph_num; ++j)
  //     core_features[i][j].resize(shard_num);
  // }

  core_vertices.resize(shard_num);
  for (int i = 0; i < shard_num; ++i) {
    core_vertices[i].resize(id_to_feature.size());
    for (int j = 0; j < id_to_feature.size(); ++j)
      core_vertices[i][j].resize(subgraph_num);
  }

  vertex_colors.resize(id_to_feature.size());
  for (int i = 0; i < (id_to_feature.size()); ++i) {
    vertex_colors[i].resize(shard_num);
  }

  build_coregraph(subgraph_num, core_vertices, vertex_colors, part_method, subgraph_path);
  if (build_halo)
    build_halograph(subgraph_num, layer_num, core_vertices, vertex_colors, subgraph_path);
  // VLOG(0) << "count start.";
  // for (int idx = 0; idx < id_to_edge.size(); idx += 2) {
  //   for (int shard_id = 0; shard_id < subgraph_num; ++shard_id) {
  //     uint64_t v_sum = 0, e_sum = 0;
  //     std::map<uint64_t, bool> flag;
  //     for (int part_id = 0; part_id < shard_num; ++part_id) {
  //       for (auto node : core_vertices[idx >> 1][shard_id][part_id]) {
  //         ++v_sum;
  //         auto deg = node->get_neighbor_size();
  //         for (size_t k = 0; k < deg; ++k) {
  //           auto v = node->get_neighbor_id(k);
  //           flag[v] = 1;
  //           ++e_sum;
  //         }
  //       }
  //     }
  //     VLOG(0) << idx << "-" << shard_id << "    " << v_sum << ' ' << e_sum << ' ' << flag.size();
  //   }
  // }
  return 0;
}

void GraphTable::clear_subgraph_table(void) {
  for (int i = 0; i < (int)edge_shards.size(); i++) {
    for (size_t j = 0; j < edge_shards[i].size(); ++j) {
      edge_shards[i][j]->clear();
    }
  }

  for (int i = 0; i < (int)feature_shards.size(); i++) {
    for (size_t j = 0; j < feature_shards[i].size(); ++j) {
      feature_shards[i][j]->clear();
    }
  }
}

int32_t GraphTable::get_nodes_ids_by_ranges(
    int type_id,
    int idx,
    std::vector<std::pair<int, int>> ranges,
    std::vector<uint64_t> &res) {
  std::mutex mutex;
  int start = 0, end, index = 0, total_size = 0;
  res.clear();
  auto &shards = type_id == 0 ? edge_shards[idx] : feature_shards[idx];
  std::vector<std::future<size_t>> tasks;
  for (size_t i = 0; i < shards.size() && index < (int)ranges.size(); i++) {
    end = total_size + shards[i]->get_size();
    start = total_size;
    while (start < end && index < (int)ranges.size()) {
      if (ranges[index].second <= start)
        index++;
      else if (ranges[index].first >= end) {
        break;
      } else {
        int first = std::max(ranges[index].first, start);
        int second = std::min(ranges[index].second, end);
        start = second;
        first -= total_size;
        second -= total_size;
        tasks.push_back(_shards_task_pool[i % task_pool_size_]->enqueue(
            [&shards, this, first, second, i, &res, &mutex]() -> size_t {
              std::vector<uint64_t> keys;
              shards[i]->get_ids_by_range(first, second, &keys);

              size_t num = keys.size();
              mutex.lock();
              res.reserve(res.size() + num);
              for (auto &id : keys) {
                res.push_back(id);
                std::swap(res[rand() % res.size()], res[(int)res.size() - 1]);
              }
              mutex.unlock();

              return num;
            }));
      }
    }
    total_size += shards[i]->get_size();
  }
  for (size_t i = 0; i < tasks.size(); i++) {
    tasks[i].get();
  }
  return 0;
}

std::pair<uint64_t, uint64_t> GraphTable::parse_node_file(
    const std::string &path, const std::string &node_type, int idx) {
  std::ifstream file(path);
  std::string line;
  uint64_t local_count = 0;
  uint64_t local_valid_count = 0;

  int num = 0;
  std::vector<paddle::string::str_ptr> vals;
  size_t n = node_type.length();
  while (std::getline(file, line)) {
    if (strncmp(line.c_str(), node_type.c_str(), n) != 0) {
      continue;
    }
    vals.clear();
    num = paddle::string::split_string_ptr(
        line.c_str() + n + 1, line.length() - n - 1, '\t', &vals);
    if (num == 0) {
      continue;
    }
    uint64_t id = std::strtoul(vals[0].ptr, NULL, 10);
    size_t shard_id = id % shard_num;
    if (shard_id >= shard_end || shard_id < shard_start) {
      VLOG(4) << "will not load " << id << " from " << path
              << ", please check id distribution";
      continue;
    }
    local_count++;

    size_t index = shard_id - shard_start;
    auto node = feature_shards[idx][index]->add_feature_node(id, false);
    if (node != NULL) {
      node->set_feature_size(feat_name[idx].size());
      for (int i = 1; i < num; ++i) {
        auto &v = vals[i];
        parse_feature(idx, v.ptr, v.len, node);
      }
    }
    local_valid_count++;
  }
  VLOG(2) << "node_type[" << node_type << "] loads " << local_count
          << " nodes from filepath->" << path;
  return {local_count, local_valid_count};
}

std::pair<uint64_t, uint64_t> GraphTable::parse_node_file(
    const std::string &path) {
  std::ifstream file(path);
  std::string line;
  uint64_t local_count = 0;
  uint64_t local_valid_count = 0;
  int idx = 0;

  auto path_split = paddle::string::split_string<std::string>(path, "/");
  auto path_name = path_split[path_split.size() - 1];

  int num = 0;
  std::vector<paddle::string::str_ptr> vals;

  while (std::getline(file, line)) {
    vals.clear();
    num = paddle::string::split_string_ptr(
        line.c_str(), line.length(), '\t', &vals);
    if (vals.empty()) {
      continue;
    }
    std::string parse_node_type = vals[0].to_string();
    auto it = feature_to_id.find(parse_node_type);
    if (it == feature_to_id.end()) {
      // VLOG(0) << parse_node_type << "type error, please check";
      continue;
    }
    idx = it->second;
    uint64_t id = std::strtoul(vals[1].ptr, NULL, 10);
    size_t shard_id = id % shard_num;
    if (shard_id >= shard_end || shard_id < shard_start) {
      VLOG(4) << "will not load " << id << " from " << path
              << ", please check id distribution";
      continue;
    }
    local_count++;

    size_t index = shard_id - shard_start;
    auto node = feature_shards[idx][index]->add_feature_node(id, false);
    if (node != NULL) {
      for (int i = 2; i < num; ++i) {
        auto &v = vals[i];
        parse_feature(idx, v.ptr, v.len, node);
      }
    }
    local_valid_count++;
  }
  VLOG(2) << local_valid_count << "/" << local_count << " nodes from filepath->"
          << path;
  return {local_count, local_valid_count};
}

std::pair<uint64_t, uint64_t> GraphTable::parse_core_node_file(
    const std::string &path, int idx) {
  std::ifstream file(path, std::ios::in | std::ios::binary | std::ios::ate);
  uint64_t local_count = 0;
  uint64_t local_valid_count = 0;

  // std::vector<paddle::string::str_ptr> vals;
  auto& feature_list = sg_vertex_info[idx];

  if (!file.is_open()) {
    VLOG(0) << "Open " << path << " Failed.";
    return {0, 0};
  }

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(size);
  file.read(buffer.data(), size);
  file.close();

  const int feature_cnt = feature_list.size();
  const int line_size = sizeof(uint64_t) * (feature_cnt + 1);

  for (size_t offset = 0; offset < size; offset += line_size) {
    uint64_t *v_ptr = (uint64_t*)(buffer.data() + offset);
    uint64_t v_id = *v_ptr;
    size_t shard_id = v_id % shard_num;
    if (shard_id >= shard_end || shard_id < shard_start) {
      VLOG(4) << "will not load " << v_id << " from " << path
              << ", please check id distribution";
      continue;
    }
    local_count++;

    size_t index = shard_id - shard_start;
    auto node = feature_shards[idx][index]->add_feature_node(v_id, false);
    if (node != NULL) {
      for (int feature_id = 0; feature_id < feature_cnt; ++feature_id) {
        uint64_t *feature_ptr = (uint64_t*)(buffer.data() + offset + feature_id * sizeof(uint64_t) + sizeof(uint64_t));
        uint64_t feature_value = *feature_ptr;
        parse_shard_feature(idx, feature_list[feature_id], feature_value, node);
      }
      local_valid_count++;
    }
  }
  VLOG(2) << local_valid_count << "/" << local_count << " nodes from filepath->"
          << path;
  return {local_count, local_valid_count};
}

// TODO opt load all node_types in once reading
int32_t GraphTable::load_nodes(const std::string &path, std::string node_type, int load_idx) {
  auto paths = paddle::string::split_string<std::string>(path, ";");
  uint64_t count = 0;
  uint64_t valid_count = 0;
  int idx = 0;
  if (FLAGS_graph_load_in_parallel) {
    if (load_idx < 0 && node_type == "") {
      VLOG(0) << "Begin GraphTable::load_nodes(), will load all node_type once";
    }
    if (load_idx >= 0) {
      VLOG(0) << "Begin GraphTable::load_nodes(), will load " << id_to_feature[load_idx];
    }
    std::vector<std::future<std::pair<uint64_t, uint64_t>>> tasks;
    for (size_t i = 0; i < paths.size(); i++) {
      tasks.push_back(load_node_edge_task_pool->enqueue(
          [&, i, this]() -> std::pair<uint64_t, uint64_t> {
            if (load_idx >= 0) return parse_core_node_file(paths[i], load_idx);
            return parse_node_file(paths[i]);
          }));
    }
    for (int i = 0; i < (int)tasks.size(); i++) {
      auto res = tasks[i].get();
      count += res.first;
      valid_count += res.second;
    }
  } else {
    VLOG(0) << "Begin GraphTable::load_nodes() node_type[" << node_type << "]";
    if (load_idx < 0) {
      if (node_type == "") {
        VLOG(0) << "node_type not specified, loading edges to "
                << id_to_feature[0] << " part";
      } else {
        if (feature_to_id.find(node_type) == feature_to_id.end()) {
          VLOG(0) << "node_type " << node_type
                  << " is not defined, nothing will be loaded";
          return 0;
        }
        idx = feature_to_id[node_type];
      }
    }
    else VLOG(0) << "Begin GraphTable::load_nodes(), will load all node_type in shard_graph serially";

    for (auto path : paths) {
      VLOG(2) << "Begin GraphTable::load_nodes(), path[" << path << "]";
      std::pair<uint64_t, uint64_t> res;
      if (load_idx >= 0) {
        res = parse_core_node_file(path, load_idx);
      }
      else res = parse_node_file(path, node_type, idx);
      count += res.first;
      valid_count += res.second;
    }
  }

  if (load_idx >= 0) {
    VLOG(0) << valid_count << "/" << count << " nodes in node_type[" << id_to_feature[load_idx] << "] are loaded successfully!";
    v_num[load_idx] = valid_count;
  }
  else {
    VLOG(0) << valid_count << "/" << count << " nodes in node_type[" << node_type << "] are loaded successfully!";
    v_num[feature_to_id[node_type]] = valid_count;
  }
  return 0;
}

int32_t GraphTable::build_sampler(int idx, std::string sample_type) {
  for (auto &shard : edge_shards[idx]) {
    auto bucket = shard->get_bucket();
    for (size_t i = 0; i < bucket.size(); i++) {
      bucket[i]->build_sampler(sample_type);
    }
  }
  return 0;
}

std::pair<uint64_t, uint64_t> GraphTable::parse_edge_file(
    const std::string &path, int idx, bool reverse) {
  std::string sample_type = "random";
  bool is_weighted = false;
  std::ifstream file(path);
  std::string line;
  uint64_t local_count = 0;
  uint64_t local_valid_count = 0;
  uint64_t part_num = 0;
  if (FLAGS_graph_load_in_parallel) {
    auto path_split = paddle::string::split_string<std::string>(path, "/");
    auto part_name_split = paddle::string::split_string<std::string>(
        path_split[path_split.size() - 1], "-");
    part_num = std::stoull(part_name_split[part_name_split.size() - 1]);
  }

  while (std::getline(file, line)) {
    size_t start = line.find_first_of('\t');
    if (start == std::string::npos) continue;
    local_count++;
    uint64_t src_id = std::stoull(&line[0]);
    uint64_t dst_id = std::stoull(&line[start + 1]);
    if (reverse) {
      std::swap(src_id, dst_id);
    }
    size_t src_shard_id = src_id % shard_num;
    if (FLAGS_graph_load_in_parallel) {
      if (src_shard_id != (part_num % shard_num)) {
        continue;
      }
    }

    float weight = 1;
    size_t last = line.find_last_of('\t');
    if (start != last) {
      weight = std::stof(&line[last + 1]);
      sample_type = "weighted";
      is_weighted = true;
    }

    if (src_shard_id >= shard_end || src_shard_id < shard_start) {
      VLOG(4) << "will not load " << src_id << " from " << path
              << ", please check id distribution";
      continue;
    }
    size_t index = src_shard_id - shard_start;
    auto node = edge_shards[idx][index]->add_graph_node(src_id);
    if (node != NULL) {
      node->build_edges(is_weighted);
      node->add_edge(dst_id, weight);
    }

    local_valid_count++;
  }
  VLOG(2) << local_count << " edges are loaded from filepath->" << path;
  return {local_count, local_valid_count};
}

std::pair<uint64_t, uint64_t> GraphTable::parse_core_edge_file(
    const std::string &path, int idx) {
  std::string sample_type = "random";
  bool is_weighted = false;
  std::ifstream file(path, std::ios::in | std::ios::binary | std::ios::ate);
  uint64_t local_count = 0;
  uint64_t local_valid_count = 0;
  // uint64_t part_num = 0;
  // if (FLAGS_graph_load_in_parallel) {
  //   auto path_split = paddle::string::split_string<std::string>(path, "/");
  //   auto part_name_split = paddle::string::split_string<std::string>(
  //       path_split[path_split.size() - 1], "-");
  //   part_num = std::stoull(part_name_split[part_name_split.size() - 1]);
  // }

  if (!file.is_open()) {
    VLOG(0) << "Open " << path << " Failed.";
    return {0, 0};
  }

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(size);
  file.read(buffer.data(), size);
  file.close();

  size_t offset = 0;
  while (offset < size) {
    uint64_t *ptr = (uint64_t*)(buffer.data() + offset);
    uint64_t src_id = *ptr;

    ptr = (uint64_t*)(buffer.data() + offset + 8);
    uint64_t deg = *ptr;
    size_t _offset = offset + 16;
    offset += (deg << 3) + 16;

    // float *weight_ptr = (float*)(buffer.data() + offset + 16);
    float weight = 1.0;
    // float weight = *weight_ptr;

    size_t src_shard_id = src_id % shard_num;

    // if (weight >= 0) { // TODO : Update for weighted graph in info file
    //   sample_type = "weighted";
    //   is_weighted = true;
    // }

    // if (FLAGS_graph_load_in_parallel) {
    //   if (src_shard_id != (part_num % shard_num)) {
    //     VLOG(0) << "Vertex Location Error";
    //     continue;
    //   }
    // }

    if (src_shard_id >= shard_end || src_shard_id < shard_start) {
      VLOG(0) << "will not load " << src_id << " from " << path
              << ", please check id distribution";
      continue;
    }
    size_t src_index = src_shard_id - shard_start;
   
    local_count++;
    auto node = edge_shards[idx][src_index]->add_graph_node(src_id);

    if (node != nullptr) {
      for (; _offset < offset; _offset += 8) {  
        ptr = (uint64_t*)(buffer.data() + _offset);
        uint64_t dst_id = *ptr;
        node->build_edges(is_weighted);
        node->add_edge(dst_id, weight);
        local_valid_count++;
      }
    }
  }

  VLOG(2) << local_count << " edges are loaded from filepath->" << path;
  return {local_count, local_valid_count};
}

int32_t GraphTable::load_edges(const std::string &path,
                               bool reverse_edge,
                               const std::string &edge_type,
                               int load_mode) { // 0 : normal; 1 : load subgraph
#ifdef PADDLE_WITH_HETERPS
  if (search_level == 2) total_memory_cost = 0;
  const uint64_t fixed_load_edges = 1000000;
#endif
  int idx = 0;
  if (edge_type == "") {
    VLOG(0) << "edge_type not specified, loading edges to " << id_to_edge[0]
            << " part";
  } else {
    if (edge_to_id.find(edge_type) == edge_to_id.end()) {
      VLOG(0) << "edge_type " << edge_type
              << " is not defined, nothing will be loaded";
      return 0;
    }
    idx = edge_to_id[edge_type];
  }

  auto paths = paddle::string::split_string<std::string>(path, ";");
  uint64_t count = 0;
  uint64_t valid_count = 0;

  VLOG(0) << "Begin GraphTable::load_edges() edge_type[" << edge_type << "]";
  if (FLAGS_graph_load_in_parallel) {
    std::vector<std::future<std::pair<uint64_t, uint64_t>>> tasks;
    for (int i = 0; i < paths.size(); i++) {
      tasks.push_back(load_node_edge_task_pool->enqueue(
          [&, i, idx, this]() -> std::pair<uint64_t, uint64_t> {
            if (load_mode == 1) return parse_core_edge_file(paths[i], idx);
            return parse_edge_file(paths[i], idx, reverse_edge);
          }));
    }
    for (int j = 0; j < (int)tasks.size(); j++) {
      auto res = tasks[j].get();
      count += res.first;
      valid_count += res.second;
    }
  } else {
    for (auto path : paths) {
      std::pair<uint64_t, uint64_t> res = {0, 0};
      if (load_mode == 1) res = parse_core_edge_file(path, idx);
      else {
        res = parse_edge_file(path, idx, reverse_edge);
      }
      count += res.first;
      valid_count += res.second;
    }
  }
  VLOG(0) << valid_count << "/" << count << " edge_type[" << edge_type
          << "] edges are loaded successfully";
  e_num[edge_to_id[edge_type]] = valid_count;

#ifdef PADDLE_WITH_HETERPS
  if (search_level == 2) {
    if (count > 0) {
      dump_edges_to_ssd(idx);
      VLOG(0) << "dumping edges to ssd, edge count is reset to 0";
      clear_graph(idx);
      count = 0;
    }
    return 0;
  }
#endif

  if (!build_sampler_on_cpu) {
    // To reduce memory overhead, CPU samplers won't be created in gpugraph.
    // In order not to affect the sampler function of other scenario,
    // this optimization is only performed in load_edges function.
    VLOG(0) << "run in gpugraph mode!";
  } else {
    std::string sample_type = "random";
    VLOG(0) << "build sampler ... ";
    for (auto &shard : edge_shards[idx]) {
      auto bucket = shard->get_bucket();
      for (size_t i = 0; i < bucket.size(); i++) {
        bucket[i]->build_sampler(sample_type);
      }
    }
  }

  return 0;
}

Node *GraphTable::find_node(int type_id, uint64_t id) {
  size_t shard_id = id % shard_num;
  if (shard_id >= shard_end || shard_id < shard_start) {
    return nullptr;
  }
  Node *node = nullptr;
  size_t index = shard_id - shard_start;
  auto &search_shards = type_id == 0 ? edge_shards : feature_shards;
  for (auto &search_shard : search_shards) {
    PADDLE_ENFORCE_NOT_NULL(search_shard[index],
                            paddle::platform::errors::InvalidArgument(
                                "search_shard[%d] should not be null.", index));
    node = search_shard[index]->find_node(id);
    if (node != nullptr) {
      break;
    }
  }
  return node;
}

Node *GraphTable::find_node(int type_id, int idx, uint64_t id) {
  size_t shard_id = id % shard_num;
  if (shard_id >= shard_end || shard_id < shard_start) {
    return nullptr;
  }
  size_t index = shard_id - shard_start;
  auto &search_shards = type_id == 0 ? edge_shards[idx] : feature_shards[idx];
  PADDLE_ENFORCE_NOT_NULL(search_shards[index],
                          paddle::platform::errors::InvalidArgument(
                              "search_shard[%d] should not be null.", index));
  Node *node = search_shards[index]->find_node(id);
  return node;
}
uint32_t GraphTable::get_thread_pool_index(uint64_t node_id) {
  return node_id % shard_num % shard_num_per_server % task_pool_size_;
}

uint32_t GraphTable::get_thread_pool_index_by_shard_index(
    uint64_t shard_index) {
  return shard_index % shard_num_per_server % task_pool_size_;
}

int32_t GraphTable::clear_nodes(int type_id, int idx) {
  auto &search_shards = type_id == 0 ? edge_shards[idx] : feature_shards[idx];
  for (size_t i = 0; i < search_shards.size(); i++) {
    search_shards[i]->clear();
  }
  return 0;
}

int32_t GraphTable::random_sample_nodes(int type_id,
                                        int idx,
                                        int sample_size,
                                        std::unique_ptr<char[]> &buffer,
                                        int &actual_size) {
  int total_size = 0;
  auto &shards = type_id == 0 ? edge_shards[idx] : feature_shards[idx];
  for (int i = 0; i < (int)shards.size(); i++) {
    total_size += shards[i]->get_size();
  }
  if (sample_size > total_size) sample_size = total_size;
  int range_num = random_sample_nodes_ranges;
  if (range_num > sample_size) range_num = sample_size;
  if (sample_size == 0 || range_num == 0) return 0;
  std::vector<int> ranges_len, ranges_pos;
  int remain = sample_size, last_pos = -1, num;
  std::set<int> separator_set;
  for (int i = 0; i < range_num - 1; i++) {
    while (separator_set.find(num = rand() % (sample_size - 1)) !=
           separator_set.end())
      ;
    separator_set.insert(num);
  }
  for (auto p : separator_set) {
    ranges_len.push_back(p - last_pos);
    last_pos = p;
  }
  ranges_len.push_back(sample_size - 1 - last_pos);
  remain = total_size - sample_size + range_num;
  separator_set.clear();
  for (int i = 0; i < range_num; i++) {
    while (separator_set.find(num = rand() % remain) != separator_set.end())
      ;
    separator_set.insert(num);
  }
  int used = 0, index = 0;
  last_pos = -1;
  for (auto p : separator_set) {
    used += p - last_pos - 1;
    last_pos = p;
    ranges_pos.push_back(used);
    used += ranges_len[index++];
  }
  std::vector<std::pair<int, int>> first_half, second_half;
  int start_index = rand() % total_size;
  for (size_t i = 0; i < ranges_len.size() && i < ranges_pos.size(); i++) {
    if (ranges_pos[i] + ranges_len[i] - 1 + start_index < total_size)
      first_half.push_back({ranges_pos[i] + start_index,
                            ranges_pos[i] + ranges_len[i] + start_index});
    else if (ranges_pos[i] + start_index >= total_size) {
      second_half.push_back(
          {ranges_pos[i] + start_index - total_size,
           ranges_pos[i] + ranges_len[i] + start_index - total_size});
    } else {
      first_half.push_back({ranges_pos[i] + start_index, total_size});
      second_half.push_back(
          {0, ranges_pos[i] + ranges_len[i] + start_index - total_size});
    }
  }
  for (auto &pair : first_half) second_half.push_back(pair);
  std::vector<uint64_t> res;
  get_nodes_ids_by_ranges(type_id, idx, second_half, res);
  actual_size = res.size() * sizeof(uint64_t);
  buffer.reset(new char[actual_size]);
  char *pointer = buffer.get();
  memcpy(pointer, res.data(), actual_size);
  return 0;
}
int32_t GraphTable::random_sample_neighbors(
    int idx,
    uint64_t *node_ids,
    int sample_size,
    std::vector<std::shared_ptr<char>> &buffers,
    std::vector<int> &actual_sizes,
    bool need_weight) {
  size_t node_num = buffers.size();
  std::function<void(char *)> char_del = [](char *c) { delete[] c; };
  std::vector<std::future<int>> tasks;
  std::vector<std::vector<uint32_t>> seq_id(task_pool_size_);
  std::vector<std::vector<SampleKey>> id_list(task_pool_size_);
  size_t index;
  for (size_t idy = 0; idy < node_num; ++idy) {
    index = get_thread_pool_index(node_ids[idy]);
    seq_id[index].emplace_back(idy);
    id_list[index].emplace_back(idx, node_ids[idy], sample_size, need_weight);
  }

  for (int i = 0; i < (int)seq_id.size(); i++) {
    if (seq_id[i].size() == 0) continue;
    tasks.push_back(_shards_task_pool[i]->enqueue([&, i, this]() -> int {
      uint64_t node_id;
      std::vector<std::pair<SampleKey, SampleResult>> r;
      LRUResponse response = LRUResponse::blocked;
      if (use_cache) {
        response =
            scaled_lru->query(i, id_list[i].data(), id_list[i].size(), r);
      }
      int index = 0;
      std::vector<SampleResult> sample_res;
      std::vector<SampleKey> sample_keys;
      auto &rng = _shards_task_rng_pool[i];
      for (size_t k = 0; k < id_list[i].size(); k++) {
        if (index < (int)r.size() &&
            r[index].first.node_key == id_list[i][k].node_key) {
          int idy = seq_id[i][k];
          actual_sizes[idy] = r[index].second.actual_size;
          buffers[idy] = r[index].second.buffer;
          index++;
        } else {
          node_id = id_list[i][k].node_key;
          Node *node = find_node(0, idx, node_id);
          int idy = seq_id[i][k];
          int &actual_size = actual_sizes[idy];
          if (node == nullptr) {
#ifdef PADDLE_WITH_HETERPS
            if (search_level == 2) {
              VLOG(2) << "enter sample from ssd for node_id " << node_id;
              char *buffer_addr = random_sample_neighbor_from_ssd(
                  idx, node_id, sample_size, rng, actual_size);
              if (actual_size != 0) {
                std::shared_ptr<char> &buffer = buffers[idy];
                buffer.reset(buffer_addr, char_del);
              }
              VLOG(2) << "actual sampled size from ssd = " << actual_sizes[idy];
              continue;
            }
#endif
            actual_size = 0;
            continue;
          }
          std::shared_ptr<char> &buffer = buffers[idy];
          std::vector<int> res = node->sample_k(sample_size, rng);
          actual_size =
              res.size() * (need_weight ? (Node::id_size + Node::weight_size)
                                        : Node::id_size);
          int offset = 0;
          uint64_t id;
          float weight;
          char *buffer_addr = new char[actual_size];
          if (response == LRUResponse::ok) {
            sample_keys.emplace_back(idx, node_id, sample_size, need_weight);
            sample_res.emplace_back(actual_size, buffer_addr);
            buffer = sample_res.back().buffer;
          } else {
            buffer.reset(buffer_addr, char_del);
          }
          for (int &x : res) {
            id = node->get_neighbor_id(x);
            memcpy(buffer_addr + offset, &id, Node::id_size);
            offset += Node::id_size;
            if (need_weight) {
              weight = node->get_neighbor_weight(x);
              memcpy(buffer_addr + offset, &weight, Node::weight_size);
              offset += Node::weight_size;
            }
          }
        }
      }
      if (sample_res.size()) {
        scaled_lru->insert(
            i, sample_keys.data(), sample_res.data(), sample_keys.size());
      }
      return 0;
    }));
  }
  for (auto &t : tasks) {
    t.get();
  }
  return 0;
}

int32_t GraphTable::get_node_feat(int idx,
                                  const std::vector<uint64_t> &node_ids,
                                  const std::vector<std::string> &feature_names,
                                  std::vector<std::vector<std::string>> &res) {
  size_t node_num = node_ids.size();
  std::vector<std::future<int>> tasks;
  for (size_t idy = 0; idy < node_num; ++idy) {
    uint64_t node_id = node_ids[idy];
    tasks.push_back(_shards_task_pool[get_thread_pool_index(node_id)]->enqueue(
        [&, idx, idy, node_id]() -> int {
          Node *node = find_node(1, idx, node_id);

          if (node == nullptr) {
            return 0;
          }
          for (int feat_idx = 0; feat_idx < (int)feature_names.size();
               ++feat_idx) {
            const std::string &feature_name = feature_names[feat_idx];
            if (feat_id_map[idx].find(feature_name) != feat_id_map[idx].end()) {
              // res[feat_idx][idx] =
              // node->get_feature(feat_id_map[feature_name]);
              auto feat = node->get_feature(feat_id_map[idx][feature_name]);
              res[feat_idx][idy] = feat;
            }
          }
          return 0;
        }));
  }
  for (size_t idy = 0; idy < node_num; ++idy) {
    tasks[idy].get();
  }
  return 0;
}

int32_t GraphTable::set_node_feat(
    int idx,
    const std::vector<uint64_t> &node_ids,
    const std::vector<std::string> &feature_names,
    const std::vector<std::vector<std::string>> &res) {
  size_t node_num = node_ids.size();
  std::vector<std::future<int>> tasks;
  for (size_t idy = 0; idy < node_num; ++idy) {
    uint64_t node_id = node_ids[idy];
    tasks.push_back(_shards_task_pool[get_thread_pool_index(node_id)]->enqueue(
        [&, idx, idy, node_id]() -> int {
          size_t index = node_id % this->shard_num - this->shard_start;
          auto node = feature_shards[idx][index]->add_feature_node(node_id);
          node->set_feature_size(this->feat_name[idx].size());
          for (int feat_idx = 0; feat_idx < (int)feature_names.size();
               ++feat_idx) {
            const std::string &feature_name = feature_names[feat_idx];
            if (feat_id_map[idx].find(feature_name) != feat_id_map[idx].end()) {
              node->set_feature(feat_id_map[idx][feature_name],
                                res[feat_idx][idy]);
            }
          }
          return 0;
        }));
  }
  for (size_t idy = 0; idy < node_num; ++idy) {
    tasks[idy].get();
  }
  return 0;
}

void string_vector_2_string(std::vector<std::string>::iterator strs_begin,
                            std::vector<std::string>::iterator strs_end,
                            char delim,
                            std::string *output) {
  size_t i = 0;
  for (std::vector<std::string>::iterator iter = strs_begin; iter != strs_end;
       ++iter) {
    if (i > 0) {
      *output += delim;
    }

    *output += *iter;
    ++i;
  }
}

void string_vector_2_string(
    std::vector<paddle::string::str_ptr>::iterator strs_begin,
    std::vector<paddle::string::str_ptr>::iterator strs_end,
    char delim,
    std::string *output) {
  size_t i = 0;
  for (auto iter = strs_begin; iter != strs_end; ++iter) {
    if (i > 0) {
      output->append(&delim, 1);
    }
    output->append((*iter).ptr, (*iter).len);
    ++i;
  }
}

void string_vector_2_string(
    std::string feat_str,
    char delim,
    std::string *output) {
  *output += feat_str;
}

int GraphTable::parse_feature(int idx,
                              const char *feat_str,
                              size_t len,
                              FeatureNode *node) {
  // Return (feat_id, btyes) if name are in this->feat_name, else return (-1,
  // "")
  thread_local std::vector<paddle::string::str_ptr> fields;
  fields.clear();
  char c = slot_feature_separator_.at(0);
  paddle::string::split_string_ptr(feat_str, len, c, &fields);

  thread_local std::vector<paddle::string::str_ptr> fea_fields;
  fea_fields.clear();
  c = feature_separator_.at(0);
  paddle::string::split_string_ptr(fields[1].ptr, fields[1].len, c, &fea_fields);

  std::string name = fields[0].to_string();
  auto it = feat_id_map[idx].find(name);
  if (it != feat_id_map[idx].end()) {
    int32_t id = it->second;
    std::string *fea_ptr = node->mutable_feature(id);
    std::string dtype = this->feat_dtype[idx][id];
    if (dtype == "feasign") {
      //      string_vector_2_string(fields.begin() + 1, fields.end(), ' ',
      //      fea_ptr);
      FeatureNode::parse_value_to_bytes<uint64_t>(
          fea_fields.begin(), fea_fields.end(), fea_ptr);
      return 0;
    } else if (dtype == "string") {
      string_vector_2_string(fea_fields.begin(), fea_fields.end(), ' ', fea_ptr);
      return 0;
    } else if (dtype == "float32") {
      FeatureNode::parse_value_to_bytes<float>(
          fea_fields.begin(), fea_fields.end(), fea_ptr);
      return 0;
    } else if (dtype == "float64") {
      FeatureNode::parse_value_to_bytes<double>(
          fea_fields.begin(), fea_fields.end(), fea_ptr);
      return 0;
    } else if (dtype == "int32") {
      FeatureNode::parse_value_to_bytes<int32_t>(
          fea_fields.begin(), fea_fields.end(), fea_ptr);
      return 0;
    } else if (dtype == "int64") {
      FeatureNode::parse_value_to_bytes<uint64_t>(
          fea_fields.begin(), fea_fields.end(), fea_ptr);
      return 0;
    }
  } else {
    VLOG(2) << "feature_name[" << name << "] is not in feat_id_map, ntype_id["
            << idx << "] feat_id_map_size[" << feat_id_map.size() << "]";
  }

  return -1;
}

int GraphTable::parse_shard_feature(int idx,
                                    const std::string& name,
                                    uint64_t value,
                                    FeatureNode *node) {
  auto it = feat_id_map[idx].find(name);
  if (it != feat_id_map[idx].end()) {
    int32_t id = it->second;
    std::string *fea_ptr = node->mutable_feature(id);
    std::string dtype = this->feat_dtype[idx][id];
    std::string value_str = std::to_string(value);
    paddle::string::str_ptr feat_str(value_str.c_str(), value_str.size());

    if (dtype == "feasign") {
      FeatureNode::parse_one_value_to_bytes<uint64_t>(
        feat_str, fea_ptr);
      return 0;
    } else if (dtype == "string") {
      string_vector_2_string(value_str, ' ', fea_ptr);
      return 0;
    } else if (dtype == "float32") {
      FeatureNode::parse_one_value_to_bytes<float>(
        feat_str, fea_ptr);
      return 0;
    } else if (dtype == "float64") {
      FeatureNode::parse_one_value_to_bytes<double>(
        feat_str, fea_ptr);
      return 0;
    } else if (dtype == "int32") {
      FeatureNode::parse_one_value_to_bytes<int32_t>(
        feat_str, fea_ptr);
      return 0;
    } else if (dtype == "int64") {
      FeatureNode::parse_one_value_to_bytes<uint64_t>(
        feat_str, fea_ptr);
      return 0;
    }
  } else {
    VLOG(0) << "feature_name[" << name << "] is not in feat_id_map, ntype_id["
            << idx << "] feat_id_map_size[" << feat_id_map.size() << "]";
  }
  return -1;
}

// thread safe shard vector merge
class MergeShardVector {
 public:
  MergeShardVector(std::vector<std::vector<uint64_t>> *output, int slice_num) {
    _slice_num = slice_num;
    _shard_keys = output;
    _shard_keys->resize(slice_num);
    _mutexs = new std::mutex[slice_num];
  }
  ~MergeShardVector() {
    if (_mutexs != nullptr) {
      delete[] _mutexs;
      _mutexs = nullptr;
    }
  }
  // merge shard keys
  void merge(const std::vector<std::vector<uint64_t>> &shard_keys) {
    // add to shard
    for (int shard_id = 0; shard_id < _slice_num; ++shard_id) {
      auto &dest = (*_shard_keys)[shard_id];
      auto &src = shard_keys[shard_id];

      _mutexs[shard_id].lock();
      dest.insert(dest.end(), src.begin(), src.end());
      _mutexs[shard_id].unlock();
    }
  }

 private:
  int _slice_num = 0;
  std::mutex *_mutexs = nullptr;
  std::vector<std::vector<uint64_t>> *_shard_keys;
};

int GraphTable::get_all_id(int type_id,
                           int slice_num,
                           std::vector<std::vector<uint64_t>> *output) {
  MergeShardVector shard_merge(output, slice_num);
  auto &search_shards = type_id == 0 ? edge_shards : feature_shards;
  std::vector<std::future<size_t>> tasks;
  for (int idx = 0; idx < search_shards.size(); idx++) {
    for (int j = 0; j < search_shards[idx].size(); j++) {
      tasks.push_back(_shards_task_pool[j % task_pool_size_]->enqueue(
          [&search_shards, idx, j, slice_num, &shard_merge]() -> size_t {
            std::vector<std::vector<uint64_t>> shard_keys;
            size_t num =
                search_shards[idx][j]->get_all_id(&shard_keys, slice_num);
            // add to shard
            shard_merge.merge(shard_keys);
            return num;
          }));
    }
  }
  for (size_t i = 0; i < tasks.size(); ++i) {
    tasks[i].wait();
  }
  return 0;
}

int GraphTable::get_all_neighbor_id(
    int type_id, int slice_num, std::vector<std::vector<uint64_t>> *output) {
  MergeShardVector shard_merge(output, slice_num);
  auto &search_shards = type_id == 0 ? edge_shards : feature_shards;
  std::vector<std::future<size_t>> tasks;
  for (int idx = 0; idx < search_shards.size(); idx++) {
    for (int j = 0; j < search_shards[idx].size(); j++) {
      tasks.push_back(_shards_task_pool[j % task_pool_size_]->enqueue(
          [&search_shards, idx, j, slice_num, &shard_merge]() -> size_t {
            std::vector<std::vector<uint64_t>> shard_keys;
            size_t num = search_shards[idx][j]->get_all_neighbor_id(&shard_keys,
                                                                    slice_num);
            // add to shard
            shard_merge.merge(shard_keys);
            return num;
          }));
    }
  }
  for (size_t i = 0; i < tasks.size(); ++i) {
    tasks[i].wait();
  }
  return 0;
}

int GraphTable::get_all_id(int type_id,
                           int idx,
                           int slice_num,
                           std::vector<std::vector<uint64_t>> *output) {
  MergeShardVector shard_merge(output, slice_num);
  auto &search_shards = type_id == 0 ? edge_shards[idx] : feature_shards[idx];
  std::vector<std::future<size_t>> tasks;
  VLOG(3) << "begin task, task_pool_size_[" << task_pool_size_ << "]";
  for (size_t i = 0; i < search_shards.size(); i++) {
    tasks.push_back(_shards_task_pool[i % task_pool_size_]->enqueue(
        [&search_shards, i, slice_num, &shard_merge]() -> size_t {
          std::vector<std::vector<uint64_t>> shard_keys;
          size_t num = search_shards[i]->get_all_id(&shard_keys, slice_num);
          // add to shard
          shard_merge.merge(shard_keys);
          return num;
        }));
  }
  for (size_t i = 0; i < tasks.size(); ++i) {
    tasks[i].wait();
  }
  VLOG(3) << "end task, task_pool_size_[" << task_pool_size_ << "]";
  return 0;
}

int GraphTable::get_all_neighbor_id(
    int type_id,
    int idx,
    int slice_num,
    std::vector<std::vector<uint64_t>> *output) {
  MergeShardVector shard_merge(output, slice_num);
  auto &search_shards = type_id == 0 ? edge_shards[idx] : feature_shards[idx];
  std::vector<std::future<size_t>> tasks;
  VLOG(3) << "begin task, task_pool_size_[" << task_pool_size_ << "]";
  for (int i = 0; i < search_shards.size(); i++) {
    tasks.push_back(_shards_task_pool[i % task_pool_size_]->enqueue(
        [&search_shards, i, slice_num, &shard_merge]() -> size_t {
          std::vector<std::vector<uint64_t>> shard_keys;
          size_t num =
              search_shards[i]->get_all_neighbor_id(&shard_keys, slice_num);
          // add to shard
          shard_merge.merge(shard_keys);
          return num;
        }));
  }
  for (size_t i = 0; i < tasks.size(); ++i) {
    tasks[i].wait();
  }
  VLOG(3) << "end task, task_pool_size_[" << task_pool_size_ << "]";
  return 0;
}

int GraphTable::get_all_feature_ids(
    int type_id,
    int idx,
    int slice_num,
    std::vector<std::vector<uint64_t>> *output) {
  MergeShardVector shard_merge(output, slice_num);
  auto &search_shards = type_id == 0 ? edge_shards[idx] : feature_shards[idx];
  std::vector<std::future<size_t>> tasks;
  for (int i = 0; i < search_shards.size(); i++) {
    tasks.push_back(_shards_task_pool[i % task_pool_size_]->enqueue(
        [&search_shards, i, slice_num, &shard_merge]() -> size_t {
          std::vector<std::vector<uint64_t>> shard_keys;
          size_t num =
              search_shards[i]->get_all_feature_ids(&shard_keys, slice_num);
          // add to shard
          shard_merge.merge(shard_keys);
          return num;
        }));
  }
  for (size_t i = 0; i < tasks.size(); ++i) {
    tasks[i].wait();
  }
  return 0;
}

int GraphTable::get_node_embedding_ids(
    int slice_num, std::vector<std::vector<uint64_t>> *output) {
  if (is_load_reverse_edge and !FLAGS_graph_get_neighbor_id) {
    return get_all_id(0, slice_num, output);
  } else {
    get_all_id(0, slice_num, output);
    return get_all_neighbor_id(0, slice_num, output);
  }
}

int32_t GraphTable::pull_graph_list(int type_id,
                                    int idx,
                                    int start,
                                    int total_size,
                                    std::unique_ptr<char[]> &buffer,
                                    int &actual_size,
                                    bool need_feature,
                                    int step) {
  if (start < 0) start = 0;
  int size = 0, cur_size;
  auto &search_shards = type_id == 0 ? edge_shards[idx] : feature_shards[idx];
  std::vector<std::future<std::vector<Node *>>> tasks;
  for (size_t i = 0; i < search_shards.size() && total_size > 0; i++) {
    cur_size = search_shards[i]->get_size();
    if (size + cur_size <= start) {
      size += cur_size;
      continue;
    }
    int count = std::min(1 + (size + cur_size - start - 1) / step, total_size);
    int end = start + (count - 1) * step + 1;
    tasks.push_back(_shards_task_pool[i % task_pool_size_]->enqueue(
        [&search_shards, this, i, start, end, step, size]()
            -> std::vector<Node *> {
          return search_shards[i]->get_batch(start - size, end - size, step);
        }));
    start += count * step;
    total_size -= count;
    size += cur_size;
  }
  for (size_t i = 0; i < tasks.size(); ++i) {
    tasks[i].wait();
  }
  size = 0;
  std::vector<std::vector<Node *>> res;
  for (size_t i = 0; i < tasks.size(); i++) {
    res.push_back(tasks[i].get());
    for (size_t j = 0; j < res.back().size(); j++) {
      size += res.back()[j]->get_size(need_feature);
    }
  }
  char *buffer_addr = new char[size];
  buffer.reset(buffer_addr);
  int index = 0;
  for (size_t i = 0; i < res.size(); i++) {
    for (size_t j = 0; j < res[i].size(); j++) {
      res[i][j]->to_buffer(buffer_addr + index, need_feature);
      index += res[i][j]->get_size(need_feature);
    }
  }
  actual_size = size;
  return 0;
}

void GraphTable::set_feature_separator(const std::string &ch) {
  feature_separator_ = ch;
}

void GraphTable::set_slot_feature_separator(const std::string &ch) {
  slot_feature_separator_ = ch;
}

int32_t GraphTable::get_server_index_by_id(uint64_t id) {
  return id % shard_num / shard_num_per_server;
}
int32_t GraphTable::Initialize(const TableParameter &config,
                               const FsClientParameter &fs_config) {
  LOG(INFO) << "in graphTable initialize";
  _config = config;
  if (InitializeAccessor() != 0) {
    LOG(WARNING) << "Table accessor initialize failed";
    return -1;
  }

  if (_afs_client.initialize(fs_config) != 0) {
    LOG(WARNING) << "Table fs_client initialize failed";
    // return -1;
  }
  auto graph = config.graph_parameter();
  shard_num = _config.shard_num();
  LOG(INFO) << "in graphTable initialize over";
  return Initialize(graph);
}

void GraphTable::load_node_weight(int type_id, int idx, std::string path) {
  auto paths = paddle::string::split_string<std::string>(path, ";");
  int64_t count = 0;
  auto &weight_map = node_weight[type_id][idx];
  for (auto path : paths) {
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
      auto values = paddle::string::split_string<std::string>(line, "\t");
      count++;
      if (values.size() < 2) continue;
      auto src_id = std::stoull(values[0]);
      double weight = std::stod(values[1]);
      weight_map[src_id] = weight;
    }
  }
}
int32_t GraphTable::Initialize(const GraphParameter &graph) {
  task_pool_size_ = graph.task_pool_size();
  build_sampler_on_cpu = graph.build_sampler_on_cpu();

#ifdef PADDLE_WITH_HETERPS
  _db = NULL;
  search_level = graph.search_level();
  if (search_level >= 2) {
    _db = paddle::distributed::RocksDBHandler::GetInstance();
    _db->initialize("./temp_gpups_db", task_pool_size_);
  }
// gpups_mode = true;
// auto *sampler =
//     CREATE_PSCORE_CLASS(GraphSampler, graph.gpups_graph_sample_class());
// auto slices =
//     string::split_string<std::string>(graph.gpups_graph_sample_args(), ",");
// std::cout << "slices" << std::endl;
// for (auto x : slices) std::cout << x << std::endl;
// sampler->init(graph.gpu_num(), this, slices);
// graph_sampler.reset(sampler);
#endif
  if (shard_num == 0) {
    server_num = 1;
    _shard_idx = 0;
    shard_num = graph.shard_num();
  }
  use_cache = graph.use_cache();
  if (use_cache) {
    cache_size_limit = graph.cache_size_limit();
    cache_ttl = graph.cache_ttl();
    make_neighbor_sample_cache((size_t)cache_size_limit, (size_t)cache_ttl);
  }
  _shards_task_pool.resize(task_pool_size_);
  for (size_t i = 0; i < _shards_task_pool.size(); ++i) {
    _shards_task_pool[i].reset(new ::ThreadPool(1));
    _shards_task_rng_pool.push_back(paddle::framework::GetCPURandomEngine(0));
  }
  load_node_edge_task_pool.reset(new ::ThreadPool(load_thread_num));

  auto graph_feature = graph.graph_feature();
  auto node_types = graph.node_types();
  auto edge_types = graph.edge_types();
  VLOG(0) << "got " << edge_types.size() << "edge types in total";
  feat_id_map.resize(node_types.size());
  for (int k = 0; k < edge_types.size(); k++) {
    VLOG(0) << "in initialize: get a edge_type " << edge_types[k];
    edge_to_id[edge_types[k]] = k;
    id_to_edge.push_back(edge_types[k]);
  }
  feat_name.resize(node_types.size());
  feat_shape.resize(node_types.size());
  feat_dtype.resize(node_types.size());
  VLOG(0) << "got " << node_types.size() << "node types in total";
  for (int k = 0; k < node_types.size(); k++) {
    feature_to_id[node_types[k]] = k;
    auto node_type = node_types[k];
    auto feature = graph_feature[k];
    id_to_feature.push_back(node_type);
    int feat_conf_size = static_cast<int>(feature.name().size());

    for (int i = 0; i < feat_conf_size; i++) {
      // auto &f_name = common.attributes()[i];
      // auto &f_shape = common.dims()[i];
      // auto &f_dtype = common.params()[i];
      auto &f_name = feature.name()[i];
      auto &f_shape = feature.shape()[i];
      auto &f_dtype = feature.dtype()[i];
      feat_name[k].push_back(f_name);
      feat_shape[k].push_back(f_shape);
      feat_dtype[k].push_back(f_dtype);
      feat_id_map[k][f_name] = i;
      VLOG(0) << "init graph table feat conf name:" << f_name
              << " shape:" << f_shape << " dtype:" << f_dtype;
    }
  }
  // this->table_name = common.table_name();
  // this->table_type = common.name();
  this->table_name = graph.table_name();
  this->table_type = graph.table_type();
  VLOG(0) << " init graph table type " << this->table_type << " table name "
          << this->table_name;
  // int feat_conf_size = static_cast<int>(common.attributes().size());
  // int feat_conf_size = static_cast<int>(graph_feature.name().size());
  VLOG(0) << "in init graph table shard num = " << shard_num << " shard_idx"
          << _shard_idx;
  shard_num_per_server = sparse_local_shard_num(shard_num, server_num);
  shard_start = _shard_idx * shard_num_per_server;
  shard_end = shard_start + shard_num_per_server;
  VLOG(0) << "in init graph table shard idx = " << _shard_idx << " shard_start "
          << shard_start << " shard_end " << shard_end;
  edge_shards.resize(id_to_edge.size());
  node_weight.resize(2);
  node_weight[0].resize(id_to_edge.size());
#ifdef PADDLE_WITH_HETERPS
  partitions.resize(id_to_edge.size());
#endif
  for (int k = 0; k < (int)edge_shards.size(); k++) {
    for (size_t i = 0; i < shard_num_per_server; i++) {
      edge_shards[k].push_back(new GraphShard());
    }
  }
  node_weight[1].resize(id_to_feature.size());
  feature_shards.resize(id_to_feature.size());
  for (int k = 0; k < (int)feature_shards.size(); k++) {
    for (size_t i = 0; i < shard_num_per_server; i++) {
      feature_shards[k].push_back(new GraphShard());
    }
  }

  return 0;
}

}  // namespace distributed
};  // namespace paddle
