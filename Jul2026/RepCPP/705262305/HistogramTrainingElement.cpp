#include <algorithm>
#include <cblas.h>
#include <omp.h>
#include <random>
#include <stack>
#include <tuple>

#include "HistogramTrainingElement.hpp"

/**************************/
/*                        */
/* TRAINING ELEMENT CLASS */
/*                        */
/**************************/

//
HistogramTrainingElement::HistogramTrainingElement() {
  this->depth = 0;
  this->node = nullptr;
  this->bins = 256;
}

//
HistogramTrainingElement::HistogramTrainingElement(
    TreeNode *node, const std::vector<size_t> &index, uint16_t depth,
    uint64_t bins) {

  this->node = node;
  this->depth = depth;
  this->index = index;
  if (bins == 0) {
    std::cerr << " < Number of bins is incorrect, shall be > 0\n";
    exit(1);
  } else {
    this->bins = bins;
  }
}

//
HistogramTrainingElement::HistogramTrainingElement(
    const HistogramTrainingElement &TE) {
  this->node = TE.node;
  this->depth = TE.depth;
  this->Histograms = TE.Histograms;
  this->index = TE.index;
  this->bins = TE.bins;
}

//
HistogramTrainingElement::HistogramTrainingElement(
    HistogramTrainingElement &&TE) {
  this->node = std::move(TE.node);
  this->depth = TE.depth;
  this->Histograms = std::move(TE.Histograms);
  this->index = std::move(TE.index);
  this->bins = TE.bins;
}

HistogramTrainingElement &
HistogramTrainingElement::operator=(HistogramTrainingElement &&TE) {
  this->node = std::move(TE.node);
  this->depth = TE.depth;
  this->Histograms = std::move(TE.Histograms);
  this->index = std::move(TE.index);
  this->bins = TE.bins;
  return *this;
}

//
HistogramTrainingElement &
HistogramTrainingElement::operator=(const HistogramTrainingElement &TE) {
  this->node = TE.node;
  this->depth = TE.depth;
  this->Histograms = TE.Histograms;
  this->index = TE.index;
  this->bins = TE.bins;
  return *this;
}

//
HistogramTrainingElement::~HistogramTrainingElement(){};

//
void HistogramTrainingElement::set_Node(TreeNode *node) { this->node = node; }

//
void HistogramTrainingElement::set_depth(uint16_t depth) {
  this->depth = depth;
}

//
void HistogramTrainingElement::set_Root(TreeNode *node) {
  this->depth = 0;
  this->node = node;
}

//
void HistogramTrainingElement::set_Bins(uint64_t new_Bins) {
  if (new_Bins == 0) {
    std::cerr << " < Number of bins is incorrect, shall be > 0\n";
    exit(1);
  } else {
    this->bins = new_Bins;
  }
}

//
void HistogramTrainingElement::set_Histogram(size_t feature,
                                             Histogram &&histogram) {
  if (this->Histograms.size() < feature) {
    std::cerr << " < Histogram to set is out of range\n";
    exit(1);
  }
  this->Histograms.at(feature) = std::move(histogram);
}

//
void HistogramTrainingElement::set_Index(std::vector<size_t> &&new_Index) {
  this->index = std::move(new_Index);
}

//
std::vector<size_t> HistogramTrainingElement::bootstrap(size_t dataset_Size) {

  // Generate a unique seed using hardware entropy
  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<size_t> dist(0, dataset_Size - 1);
  std::vector<size_t> index(dataset_Size);

  for (size_t i = 0; i < dataset_Size; ++i) {
    index[i] = dist(gen);
  }

  return std::move(index);
}

//
double HistogramTrainingElement::compute_Residual(double actual,
                                                  double predicted) const {
  return actual - predicted;
}

//
double HistogramTrainingElement::mean_Vector_At_Index(
    const std::vector<double> &vector, const std::vector<size_t> &index) const {
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
void HistogramTrainingElement::fill_Histograms(
    const DataSet &data, const std::vector<size_t> &index) {

  size_t number_Of_Features = data.features_Number();

  const std::vector<double> &labels = data.get_Labels();

  double label_Mean = mean_Vector_At_Index(labels, index);
  this->node->set_Predicted_Value(label_Mean);

#pragma omp for schedule(static)
  for (size_t feature = 0; feature < number_Of_Features; ++feature) {
    const std::vector<double> &values = data.get_Column(feature);

    for (auto &sample_Index : index) {

      double sample_Value = values[sample_Index];

      double target = labels[sample_Index];

      double residual = -compute_Residual(target, label_Mean);

      this->Histograms[feature].add_Point(sample_Value, residual);
    }
  }
}

//
void HistogramTrainingElement::init_Histograms(
    const DataSet &data, const std::vector<size_t> &index) {
  uint64_t bins = this->bins;
#pragma omp parallel
  {
#pragma omp for schedule(static)
    for (size_t feature = 0; feature < data.features_Number(); ++feature) {
      Histogram h(bins, data.get_Column(feature), index);
#pragma omp critical
      { this->Histograms[feature] = std::move(h); }
    }
    this->fill_Histograms(data, index);
  }
}

//
std::tuple<std::optional<std::vector<size_t>>,
           std::optional<std::vector<size_t>>>
HistogramTrainingElement::split_Index(const DataSet &data, size_t bin,
                                      size_t feature) const {

  if (this->Histograms.at(feature).get_Number_Of_Bins() <= 0) {
    return {std::nullopt, std::nullopt};
  }

  double split_Value = this->Histograms.at(feature).get_Bins()[bin].get_Max();

  std::vector<size_t> left_Idx;
  std::vector<size_t> right_Idx;

  for (const size_t &idx : this->index) {
    if (data.get_Column(feature)[idx] < split_Value) {
      left_Idx.push_back(idx);
    } else {
      right_Idx.push_back(idx);
    }
  }
  return {std::move(left_Idx), std::move(right_Idx)};
}

//
double HistogramTrainingElement::compute_Split_Value(size_t bin_Index,
                                                     size_t feature) const {

  const Histogram &histogram = this->Histograms.at(feature);
  size_t number_Of_Bins = histogram.get_Number_Of_Bins();
  const std::vector<Bin> &bins = histogram.get_Bins();

  double residual_Sums_Left = 0;
  double left_Population = 0;

  double residual_Sums_Right = 0;
  double right_Population = 0;

  // Not sure tho
  double hessian_Value = 1.0;

  // Two for just to avoid branching
  size_t bin;
  for (bin = 0; bin < bin_Index; ++bin) {
    const Bin &curr_Bin = bins[bin];

    left_Population += curr_Bin.get_Count();

    residual_Sums_Left += curr_Bin.get_Residual_Sum();
  }

  for (bin = bin_Index; bin < number_Of_Bins; ++bin) {
    const Bin &curr_Bin = bins[bin];

    right_Population += curr_Bin.get_Count();

    residual_Sums_Right += curr_Bin.get_Residual_Sum();
  }

  double score_Left =
      pow(residual_Sums_Left, 2) * (1.0 / (hessian_Value * left_Population));

  double score_Right =
      pow(residual_Sums_Right, 2) * (1.0 / (hessian_Value * right_Population));

  double res = std::abs(score_Left + score_Right);

  return res;
}

//
std::tuple<double, size_t, size_t>
HistogramTrainingElement::best_Histogram_Split(size_t feature) const {

  const Histogram &histogram = this->Histograms.at(feature);

  double best_Score = 0.0;
  std::tuple<double, size_t, size_t> candidate;
  // Might start at one
  for (size_t bin = 0; bin < histogram.get_Number_Of_Bins(); ++bin) {
    double split_Score = compute_Split_Value(bin, feature);

    if (split_Score > best_Score) {
      best_Score = split_Score;
      candidate = {split_Score, bin, feature};
    }
  }
  return candidate;
}

//
std::tuple<size_t, size_t> HistogramTrainingElement::find_Best_Split() const {

  std::vector<std::tuple<double, size_t, size_t>> candidates;

  candidates.resize(this->Histograms.size(), {0.0, 0, 0});
#pragma omp parallel
  {
#pragma omp for schedule(static) nowait
    for (size_t idx_H = 0; idx_H < this->Histograms.size(); ++idx_H) {
      candidates[idx_H] = std::move(best_Histogram_Split(idx_H));
    }
  } // End of pragma omp parallel

  std::tuple<double, size_t, size_t> best_Split = {-1.0, 0, 0};
  for (const auto &candidate : candidates) {
    const auto &best = std::get<0>(best_Split);
    if (best < std::get<0>(candidate) || best == -1.0) {
      best_Split = std::move(candidate);
    }
  }
  return {std::get<1>(best_Split), std::get<2>(best_Split)};
}

//
std::tuple<std::optional<HistogramTrainingElement>,
           std::optional<HistogramTrainingElement>>
HistogramTrainingElement::split_Node(const DataSet &data) {

  // Left node
  std::optional<HistogramTrainingElement> train_Left = std::nullopt;

  // Right node
  std::optional<HistogramTrainingElement> train_Right = std::nullopt;

  // Compute split attributes
  auto [bin, feature] = find_Best_Split();

  // Set new indexes (based on the)
  auto [left_Index, right_Index] = split_Index(data, bin, feature);

  if (!left_Index && !right_Index) {
    return {train_Left, train_Right};
  }

  uint16_t next_Depth = this->depth + 1;

  double split_Criterion = this->Histograms[feature].get_Bins()[bin].get_Max();

  // Set the datas for the current node
  this->node->set_Split_Column(feature);
  this->node->set_Split_Criterion(split_Criterion);
  uint64_t curr_Bins = this->bins;
  // Case 1 : Build Left Node(if information gained)
  if (left_Index) {

    TreeNode left{};

    this->node->add_Left(std::make_unique<TreeNode>(std::move(left)));

    HistogramTrainingElement train_Left_Tmp(this->node->get_Left_Node(),
                                            std::move(*left_Index), next_Depth,
                                            curr_Bins);

    train_Left_Tmp.init_Histograms(data, train_Left_Tmp.index);

    train_Left = std::move(train_Left_Tmp);
  }

  // Case 2 : Build Right Node (if information gained)
  if (right_Index) {

    TreeNode right{};

    this->node->add_Right(std::make_unique<TreeNode>(std::move(right)));

    HistogramTrainingElement train_Right_Tmp(this->node->get_Right_Node(),
                                             std::move(*right_Index),
                                             next_Depth, curr_Bins);

    train_Right_Tmp.init_Histograms(data, train_Right_Tmp.index);

    train_Right = std::move(train_Right_Tmp);
  }

  return {train_Left, train_Right};
}

//
void HistogramTrainingElement::train(const DataSet &data, TreeNode *node,
                                     uint16_t max_Depth, size_t threshold,
                                     uint64_t bins) {

  HistogramTrainingElement base_Node;
  // Compute bootstrapped indexes for samples
  std::vector<size_t> base_Index =
      std::move(base_Node.bootstrap(data.samples_Number()));

  // Set Base Node
  base_Node.set_Root(node);
  base_Node.set_Index(std::move(base_Index));
  base_Node.set_Bins(bins);

  base_Node.init_Histograms(data, base_Node.index);

  // Initialize a stack of Nodes that will be splitted
  std::stack<HistogramTrainingElement> remaining;
  remaining.push(base_Node);

  // Build iteratively the tree frame
  while (not remaining.empty()) {
    auto elem = remaining.top();

    remaining.pop();

    if (elem.depth >= max_Depth) {
      continue;
    }

    auto [left, right] = elem.split_Node(data);

    if (left) {
      if (left.value().index.size() != elem.index.size() &&
          left.value().index.size() > threshold) {
        remaining.push(std::move(*left));
      }
    }

    if (right) {
      if (right.value().index.size() != elem.index.size() &&
          right.value().index.size() > threshold) {
        remaining.push(std::move(*right));
      }
    }
  }
}