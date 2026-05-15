#pragma once

#include <cstdint>

#include "portfolio.hpp"
#include "solution.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hh {

// ---------------------------------------------------------------------------
// ContextState — compact description of the current construction situation.
// All features are normalized to [0, 1].
// ---------------------------------------------------------------------------
struct ContextState {
	double coverage_gap = 0.0;
	double exclusivity_pressure = 0.0;
	double window_urgency = 0.0;
	double ammo_depletion = 0.0;
};

// ---------------------------------------------------------------------------
// ContextSpec — binning thresholds for a binary contextual UCB table.
// The default is a 4-feature, 16-bin table (low/high per feature).
// ---------------------------------------------------------------------------
struct ContextSpec {
	std::array<double, 4> thresholds = {0.50, 0.10, 0.25, 0.50};

	static constexpr std::size_t feature_count() { return 4; }
	static constexpr std::size_t context_count() { return 1u << feature_count(); }
};

// ---------------------------------------------------------------------------
// PendingCredit — delayed reward assignment for contextual UCB.
// ---------------------------------------------------------------------------
struct PendingCredit {
	std::size_t context_bin = 0;
	std::size_t arm_idx = 0;
	int age = 0;
	double acc = 0.0;
	double next_discount = 1.0;
};

// ---------------------------------------------------------------------------
// UCBTable — per-context counts and reward sums.
// ---------------------------------------------------------------------------
struct UCBTable {
	std::size_t context_count = 0;
	std::size_t arm_count = 0;
	double exploration = 1.0;
	std::vector<std::vector<int>> uses;
	std::vector<std::vector<double>> reward_sum;
	std::vector<int> total_uses;

	UCBTable() = default;
	UCBTable(std::size_t context_count_, std::size_t arm_count_, double exploration_ = 1.0)
		{ reset(context_count_, arm_count_, exploration_); }

	void reset(std::size_t context_count_, std::size_t arm_count_, double exploration_ = 1.0) {
		context_count = context_count_;
		arm_count = arm_count_;
		exploration = exploration_;
		uses.assign(context_count, std::vector<int>(arm_count, 0));
		reward_sum.assign(context_count, std::vector<double>(arm_count, 0.0));
		total_uses.assign(context_count, 0);
	}

	bool valid(std::size_t context_bin, std::size_t arm_idx) const {
		return context_bin < context_count && arm_idx < arm_count;
	}

	int pulls(std::size_t context_bin, std::size_t arm_idx) const {
		return valid(context_bin, arm_idx) ? uses[context_bin][arm_idx] : 0;
	}

	double mean(std::size_t context_bin, std::size_t arm_idx) const {
		int n = pulls(context_bin, arm_idx);
		return n > 0 ? reward_sum[context_bin][arm_idx] / static_cast<double>(n) : 0.0;
	}

	double ucb(std::size_t context_bin, std::size_t arm_idx) const {
		if (!valid(context_bin, arm_idx)) {
			return -std::numeric_limits<double>::infinity();
		}
		int n = uses[context_bin][arm_idx];
		if (n <= 0) {
			return std::numeric_limits<double>::infinity();
		}
		double logN = std::log(static_cast<double>(std::max(1, total_uses[context_bin])));
		double bonus = exploration * std::sqrt(logN / static_cast<double>(n));
		return mean(context_bin, arm_idx) + bonus;
	}

	std::size_t select_arm(std::size_t context_bin) const {
		if (context_bin >= context_count || arm_count == 0) return 0;
		constexpr double eps = 1e-12;
		std::size_t best_arm = 0;
		double best_score = -std::numeric_limits<double>::infinity();
		for (std::size_t arm = 0; arm < arm_count; ++arm) {
			double score = ucb(context_bin, arm);
			if (score > best_score + eps) {
				best_score = score;
				best_arm = arm;
			}
		}
		return best_arm;
	}

	void record(std::size_t context_bin, std::size_t arm_idx, double reward) {
		if (!valid(context_bin, arm_idx)) return;
		uses[context_bin][arm_idx] += 1;
		reward_sum[context_bin][arm_idx] += reward;
		total_uses[context_bin] += 1;
	}

	void apply_decay(double factor) {
		if (factor <= 0.0 || factor > 1.0) return;
		for (std::size_t ctx = 0; ctx < context_count; ++ctx) {
			total_uses[ctx] = static_cast<int>(std::round(total_uses[ctx] * factor));
			for (std::size_t arm = 0; arm < arm_count; ++arm) {
				uses[ctx][arm] = static_cast<int>(std::round(uses[ctx][arm] * factor));
				reward_sum[ctx][arm] *= factor;
			}
		}
	}
};

// ---------------------------------------------------------------------------
// Feature extraction helpers.
// ---------------------------------------------------------------------------
inline double clamp_unit(double value) {
	return std::clamp(value, 0.0, 1.0);
}

inline double coverage_gap_from_survival(const Solution& sol) {
	double total_threat = 0.0;
	double weighted_survival = 0.0;
	for (const auto& [tid, threat] : sol.threat_score) {
		auto it = sol.survival_rate.find(tid);
		if (it == sol.survival_rate.end()) continue;
		total_threat += threat;
		weighted_survival += threat * it->second;
	}
	if (total_threat <= 0.0) return 0.0;
	return clamp_unit(weighted_survival / total_threat);
}

inline std::unordered_map<int, int>
count_target_feasible_weapons(const std::vector<portfolio::Candidate>& candidates) {
	std::unordered_map<int, int> counts;
	for (const auto& c : candidates) counts[c.tid] += 1;
	return counts;
}

inline std::unordered_map<int, int>
count_weapon_feasible_actions(const std::vector<portfolio::Candidate>& candidates) {
	std::unordered_map<int, int> counts;
	for (const auto& c : candidates) counts[c.wid] += 1;
	return counts;
}

inline double exclusivity_pressure(const std::vector<portfolio::Candidate>& candidates) {
	if (candidates.empty()) return 0.0;
	auto target_counts = count_target_feasible_weapons(candidates);
	int exclusive_targets = 0;
	for (const auto& [tid, count] : target_counts) {
		(void)tid;
		if (count == 1) exclusive_targets += 1;
	}
	return clamp_unit(static_cast<double>(exclusive_targets) / static_cast<double>(target_counts.size()));
}

inline double ammo_depletion(const Solution& sol, int initial_ammo_total) {
	if (initial_ammo_total <= 0) return 0.0;
	int remaining = 0;
	for (const auto& [wid, ammo] : sol.remaining_ammo) {
		(void)wid;
		remaining += ammo;
	}
	double used = static_cast<double>(std::max(0, initial_ammo_total - remaining));
	return clamp_unit(used / static_cast<double>(initial_ammo_total));
}

inline double window_urgency(
	const Solution& sol,
	const std::vector<portfolio::Candidate>& candidates,
	double slack_factor = 2.0
) {
	if (candidates.empty()) return 0.0;
	int urgent = 0;
	for (const auto& c : candidates) {
		auto it = sol.windows->find(pair_key(c.wid, c.tid));
		if (it == sol.windows->end()) continue;
		double slack = it->second.second - c.t;
		double burst = sol.burst_dur->at(c.wid);
		if (slack <= slack_factor * burst) urgent += 1;
	}
	return clamp_unit(static_cast<double>(urgent) / static_cast<double>(candidates.size()));
}

inline ContextState build_context_state(
	const Solution& sol,
	const std::vector<portfolio::Candidate>& candidates,
	int initial_ammo_total
) {
	ContextState state;
	state.coverage_gap = coverage_gap_from_survival(sol);
	state.exclusivity_pressure = exclusivity_pressure(candidates);
	state.window_urgency = window_urgency(sol, candidates);
	state.ammo_depletion = ammo_depletion(sol, initial_ammo_total);
	return state;
}

// ---------------------------------------------------------------------------
// Context binning.
// ---------------------------------------------------------------------------
inline std::size_t bin_feature(double value, double threshold) {
	return value >= threshold ? 1u : 0u;
}

inline std::size_t context_bin_index(const ContextState& state, const ContextSpec& spec = {}) {
	std::size_t bin = 0;
	bin |= bin_feature(state.coverage_gap, spec.thresholds[0]) << 3;
	bin |= bin_feature(state.exclusivity_pressure, spec.thresholds[1]) << 2;
	bin |= bin_feature(state.window_urgency, spec.thresholds[2]) << 1;
	bin |= bin_feature(state.ammo_depletion, spec.thresholds[3]) << 0;
	return bin;
}

} // namespace hh
