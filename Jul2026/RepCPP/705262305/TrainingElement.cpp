#include <algorithm>
#include <cblas.h>
#include <omp.h>
#include <random>
#include <stack>
#include <tuple>

#include "MAE.hpp"
#include "TrainingElement.hpp"

/**************************/
/*                        */
/* TRAINING ELEMENT CLASS */
/*                        */
/**************************/

//
TrainingElement::TrainingElement() {
  this->depth = 0;
  this->node = nullptr;
  this->index = std::vector<size_t>();
}

//
TrainingElement::TrainingElement(TreeNode *node,
                                 const std::vector<size_t> &index,
                                 uint16_t depth) {
  this->node = node;
  this->depth = depth;
  this->index = std::move(index);
}

//
TrainingElement::TrainingElement(const TrainingElement &TE) {
  this->node = TE.node;
  this->depth = TE.depth;
  this->index = TE.index;
}

//
TrainingElement::TrainingElement(TrainingElement &&TE) {
  this->node = std::move(TE.node);
  this->depth = TE.depth;
  this->index = std::move(TE.index);
}

TrainingElement &TrainingElement::operator=(TrainingElement &&TE) {
  this->node = std::move(TE.node);
  this->depth = TE.depth;
  this->index = std::move(TE.index);
  return *this;
}

//
TrainingElement &TrainingElement::operator=(const TrainingElement &TE) {
  this->node = TE.node;
  this->depth = TE.depth;
  this->index = TE.index;
  return *this;
}

//
TrainingElement::~TrainingElement(){};

//
const std::vector<size_t> &TrainingElement::get_Index() const {
  return this->index;
}

//
void TrainingElement::set_Node(TreeNode *node) { this->node = node; }

//
void TrainingElement::set_Index(const std::vector<size_t> &index) {
  this->index = index;
}

//
void TrainingElement::set_depth(uint16_t depth) { this->depth = depth; }

//
void TrainingElement::set_Root(size_t dataset_Size, TreeNode *node) {
  this->depth = 0;
  bootstrap_Index(dataset_Size);
  this->node = node;
}

//
void TrainingElement::bootstrap_Index(size_t dataset_Size) {

  // Generate a unique seed using hardware entropy
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> dist(0, dataset_Size - 1);
  std::vector<size_t> idx(dataset_Size);

  for (size_t i = 0; i < dataset_Size; ++i) {
    idx[i] = dist(gen);
  }
  this->index = std::move(idx);
}

//
double
TrainingElement::mean_Vector_At_Index(const std::vector<double> &vector,
                                      const std::vector<size_t> &index) const {
  if (index.empty()) {
    return 0.0;
  }

  double mean = 0.0;
  const double length = index.size();

  for (const auto &idx : index) {
    mean += vector[idx];
  }

  mean *= (1.0 / length);
  return (mean);
}

//
std::tuple<std::optional<std::vector<size_t>>,
           std::optional<std::vector<size_t>>>
TrainingElement::split_Index(const std::vector<double> &column,
                             const std::vector<size_t> &index,
                             double criterion) {

  // Check column in bounds
  if (index.empty()) {
    return {std::nullopt, std::nullopt};
  }

  std::vector<size_t> sub_Index_Right;
  std::vector<size_t> sub_Index_Left;

  sub_Index_Right.reserve(index.size());
  sub_Index_Left.reserve(index.size());

  for (const auto &row : index) {
    if (column[row] < criterion) {
      sub_Index_Left.push_back(row);
    } else {
      sub_Index_Right.push_back(row);
    }
  }

  return std::make_pair(std::move(sub_Index_Left), std::move(sub_Index_Right));
}

//
std::tuple<std::optional<std::vector<double>>,
           std::optional<std::vector<double>>>
TrainingElement::split_Labels(const std::vector<double> &column,
                              const std::vector<double> &labels,
                              double criterion,
                              const std::vector<size_t> &idx) const {
  // Check column in bounds
  if (idx.empty()) {
    return {std::nullopt, std::nullopt};
  }

  auto [left_Index, right_Index] = split_Index(column, idx, criterion);

  std::vector<double> sub_Labels_Left(left_Index.value().size());
  std::vector<double> sub_Labels_Right(right_Index.value().size());

  size_t i;
  for (i = 0; i < sub_Labels_Left.size(); ++i) {
    const auto &row = left_Index.value()[i];
    sub_Labels_Left[i] = labels[row];
  }

  for (i = 0; i < sub_Labels_Right.size(); ++i) {
    const auto &row = right_Index.value()[i];
    sub_Labels_Right[i] = labels[row];
  }

  return std::make_pair(std::move(sub_Labels_Left),
                        std::move(sub_Labels_Right));
}

//
double TrainingElement::compute_Split_Value(const std::vector<size_t> &index,
                                            const DataSet &data, size_t feature,
                                            double criteria,
                                            const IOperator *op) const {

  auto [left_Labels, right_Labels] = split_Labels(
      data.get_Column(feature), data.get_Labels(), criteria, index);

  if (!left_Labels && !right_Labels) {
    return -1.0;
  }

  const size_t base_Population = index.size();
  const size_t left_Population = left_Labels.value().size();
  const size_t right_Population = right_Labels.value().size();

  double metric_Result_Left = 0.0;
  double metric_Result_Right = 0.0;

  double left_Prediction =
      cblas_dasum(left_Population, left_Labels.value().data(), 1.0) * 1.0 /
      left_Population;

  double right_Prediction =
      cblas_dasum(right_Population, right_Labels.value().data(), 1.0) * 1.0 /
      right_Population;

  metric_Result_Left = op->compute(left_Labels.value(), left_Prediction);
  metric_Result_Right = op->compute(right_Labels.value(), right_Prediction);

  // Compute the result of the metric for the split at position
  double res = ((metric_Result_Left * left_Population) +
                (metric_Result_Right * right_Population));

  res *= (1.0 / base_Population);

  return res;
}

//
std::tuple<std::optional<TrainingElement>, std::optional<TrainingElement>>
TrainingElement::split_Node(const DataSet &data,
                            const IOperator *splitting_Operator,
                            const ICriteria *splitting_Criteria) {
  // Left node
  std::optional<TrainingElement> train_Left = std::nullopt;

  // Right node
  std::optional<TrainingElement> train_Right = std::nullopt;

  // Compute split attributes
  auto [column, criterion] =
      find_Best_Split(data, splitting_Operator, splitting_Criteria);

  // Compute new indexes
  auto [left_index, right_index] =
      split_Index(data.get_Column(column), this->get_Index(), criterion);

  if (!left_index && !right_index) {
    return {train_Left, train_Right};
  }

  uint16_t next_Depth = this->depth + 1;

  // Set the datas for the current node
  this->node->set_Split_Column(column);
  this->node->set_Split_Criterion(criterion);

  // Case 1 : Build Left Node (if information gained)
  if (left_index.has_value()) {
    double predic_Left = mean_Vector_At_Index(data.get_Labels(), *left_index);
    TreeNode left{};
    left.set_Predicted_Value(predic_Left);
    this->node->add_Left(std::make_unique<TreeNode>(std::move(left)));
    train_Left = std::move(TrainingElement(this->node->get_Left_Node(),
                                           std::move(*left_index), next_Depth));
  }

  // Case 2 : Build Right Node (if information gained)
  if (right_index.has_value()) {
    double predic_Right = mean_Vector_At_Index(data.get_Labels(), *right_index);
    TreeNode right{};
    right.set_Predicted_Value(predic_Right);
    this->node->add_Right(std::make_unique<TreeNode>(std::move(right)));
    train_Right = std::move(TrainingElement(
        this->node->get_Right_Node(), std::move(*right_index), next_Depth));
  }

  return {train_Left, train_Right};
}

//
std::tuple<size_t, double>
TrainingElement::find_Best_Split(const DataSet &data,
                                 const IOperator *splitting_Operator,
                                 const ICriteria *splitting_Criteria) const {
  std::vector<std::tuple<double, double, size_t>> candidates;

  std::vector<std::vector<double>> splitting_Thresholds;
  splitting_Thresholds.resize(data.features_Number());

#pragma omp parallel
  {
    size_t column;
    int thread_Id = omp_get_thread_num();

#pragma omp single
    {
      candidates.resize(omp_get_num_threads(),
                        {std::numeric_limits<double>::max(), 0.0, 0});
    }

#pragma omp for schedule(static)
    for (column = 0; column < data.features_Number(); ++column) {
      splitting_Thresholds[column] =
          splitting_Criteria->compute(data.get_Column(column), this->index);
    }

// No wait just not to have two barriers that follows each other
#pragma omp for schedule(static) nowait
    for (column = 0; column < data.features_Number(); ++column) {
      double split_Score;
      for (const auto spliting_Threshold : splitting_Thresholds[column]) {
        split_Score = compute_Split_Value(
            this->index, data, column, spliting_Threshold, splitting_Operator);

        if (split_Score < std::get<0>(candidates[thread_Id])) {
          candidates[thread_Id] = {split_Score, spliting_Threshold, column};
        }
      }
    }
  } // End of pragma omp parallel

  std::tuple<double, double, size_t> best_Split = {-1.0, -1.0, 0};
  for (const auto &candidate : candidates) {
    const auto &best = std::get<0>(best_Split);
    if (best > std::get<0>(candidate) || best == -1.0) {
      best_Split = std::move(candidate);
    }
  }

  return {std::get<2>(best_Split), std::get<1>(best_Split)};
}

//
void TrainingElement::train(const DataSet &data, TreeNode *Node,
                            const IOperator *splitting_Operator,
                            const ICriteria *splitting_Criteria,
                            uint16_t max_Depth, size_t threshold) {
  TrainingElement base_Node{};

  // Initialize the root Node
  base_Node.set_Root(data.labels_Number(), Node);
  double base_Prediction =
      base_Node.mean_Vector_At_Index(data.get_Labels(), base_Node.index);
  base_Node.node->set_Predicted_Value(base_Prediction);

  // Initialize a stack of Nodes that will be splitted
  std::stack<TrainingElement> remaining;
  remaining.push(base_Node);

  // Build iteratively the tree frame
  while (not remaining.empty()) {
    auto elem = remaining.top();

    remaining.pop();

    if (elem.depth >= max_Depth) {
      continue;
    }

    auto [left, right] =
        elem.split_Node(data, splitting_Operator, splitting_Criteria);

    if (left) {
      // Verify we gained information
      if (left.value().index.size() != elem.index.size() &&
          left.value().index.size() > threshold) {
        remaining.push(std::move(*left));
      }
    }

    if (right) {
      // Verify we gained information
      if (right.value().index.size() != elem.index.size() &&
          right.value().index.size() > threshold) {
        remaining.push(std::move(*right));
      }
    }
  }
}