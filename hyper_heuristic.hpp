#pragma once
#include "portfolio.hpp"
#include "solution.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <random>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// slots_overlap — number of d-wide slots fitting in [a,b] ∩ [s,e]
// ---------------------------------------------------------------------------
static inline int slots_overlap(double a, double b, double s, double e, double d) {
    double lo     = std::max(a, s);
    double hi     = std::min(b, e);
    double length = hi - lo;
    return length >= d ? static_cast<int>(length / d) : 0;
}

// ---------------------------------------------------------------------------
// score — pair score used only to rank targets for a chosen weapon
// ---------------------------------------------------------------------------
static double score(const Solution& sol, int wid, int tid, double t,
                    int exclusive_cnt) {
    (void)t;
    (void)exclusive_cnt;
    double p = sol.p_ij->count(pair_key(wid, tid)) ? sol.p_ij->at(pair_key(wid, tid)) : 0.0;
    return sol.survival(tid) * (1.0 - p);
}

// ---------------------------------------------------------------------------
// grasp_construction — incremental-scoring GRASP construction (mutates sol)
//
// Scoring invariant:
//   After committing (wid*, tid*), two things change:
//     1. free[wid*] and cap[(wid*,·)] change  → all pairs of wid* need rescoring
//     2. survival[tid*] decreases             → gain/opp terms for every weapon
//        that targets tid* need rescoring
//   Dirty set = {wid*} ∪ {w : tid* ∈ weapon_targets[w]}
//   target_weapons (reverse lookup) is built once; stale entries are harmless —
//   try_score returns immediately when a pair is no longer live.
// ---------------------------------------------------------------------------
Solution& grasp_construction(Solution& sol, double alpha, std::mt19937& rng) {
    (void)rng;
    // Reverse lookup: tid -> weapons that initially target it
    std::unordered_map<int, std::vector<int>> target_weapons;
    for (auto& [wid, tgts] : sol.weapon_targets)
        for (int tid : tgts)
            target_weapons[tid].push_back(wid);

    // Score cache: wid -> { tid -> {fire_time, score} }
    struct CS { double t, sc; };
    std::unordered_map<int, std::unordered_map<int, CS>> cache;

    // Score one pair and insert into cache if feasible
    auto try_score = [&](int wid, int tid) {
        uint64_t key    = pair_key(wid, tid);
        auto     cap_it = sol.cap.find(key);
        if (cap_it == sol.cap.end()) return;
        int ammo = sol.remaining_ammo.at(wid);
        if (ammo <= 0) return;
        int k_val = 0;
        if (auto ki = sol.k.find(key); ki != sol.k.end()) k_val = ki->second;
        if (k_val >= sol.max_shots->at(wid)) return;
        double t = sol.first_slot(wid, tid);
        if (std::isnan(t)) return;

        // exclusive(wid): targets covered exclusively by this weapon
        int excl = 0;
        auto wt_it = sol.weapon_targets.find(wid);
        if (wt_it != sol.weapon_targets.end()) {
            for (int j : wt_it->second) {
                auto tw_it = target_weapons.find(j);
                if (tw_it != target_weapons.end() && tw_it->second.size() == 1)
                    ++excl;
            }
        }

        cache[wid][tid] = {t, score(sol, wid, tid, t, excl)};
    };

    // Initial full scoring
    for (auto& [wid, tgts] : sol.weapon_targets)
        for (int tid : tgts)
            try_score(wid, tid);

    // UCB1 selector state over a fixed heuristic portfolio.
    const std::vector<portfolio::HeuristicId> active_heuristics = {
        portfolio::HeuristicId::H_SURV,
        portfolio::HeuristicId::H_EXCLUSIVE_GUARD,
        portfolio::HeuristicId::H_WINDOW_CLOSURE,
        portfolio::HeuristicId::H_COVER_FIRST,
        portfolio::HeuristicId::H_BACKLOG_RELIEF,
        portfolio::HeuristicId::H_FINISH_STARTED,
        portfolio::HeuristicId::H_SPREAD_THEN_FOCUS,
        portfolio::HeuristicId::H_ANTI_BOTTLENECK,
        portfolio::HeuristicId::H_OPPORTUNITY_LOCK,
    };
    constexpr int       phase_bins = 3;
    std::array<std::vector<int>, phase_bins> hh_uses;
    std::array<std::vector<double>, phase_bins> hh_reward_sum;
    std::array<int, phase_bins> hh_total_uses = {0, 0, 0};
    constexpr double    ucb_c = 1.0;
    constexpr int       reward_horizon_k = 7;
    constexpr double    reward_gamma = 0.9;

    for (int b = 0; b < phase_bins; ++b) {
        hh_uses[b] = std::vector<int>(active_heuristics.size(), 0);
        hh_reward_sum[b] = std::vector<double>(active_heuristics.size(), 0.0);
    }

    struct PendingCredit {
        int phase_bin = 0;
        int heuristic_idx = -1;
        int age = 0;
        double acc = 0.0;
        double next_discount = 1.0;
    };
    std::vector<PendingCredit> pending_credits;
    pending_credits.reserve(64);

    int initial_ammo_total = 0;
    for (auto& [wid, ammo] : sol.remaining_ammo) {
        (void)wid;
        initial_ammo_total += ammo;
    }
    int step = 0;
    std::unordered_map<int, int> prev_target_feasible_weapons;

    while (!cache.empty()) {
        // Find best score — O(live pairs), fast linear scan in C++
        double best_sc = -std::numeric_limits<double>::infinity();
        for (auto& [wid, inner] : cache)
            for (auto& [tid, cs] : inner)
                best_sc = std::max(best_sc, cs.sc);
        if (best_sc <= 0.0) break;
        constexpr double eps = 1e-12;
        double threshold = (1.0 - alpha) * best_sc;

        // Build candidate set above GRASP threshold.
        std::vector<portfolio::Candidate> candidates;
        candidates.reserve(64);
        for (auto& [wid, inner] : cache) {
            for (auto& [tid, cs] : inner) {
                if (cs.sc + eps < threshold) continue;
                candidates.push_back({wid, tid, cs.t});
            }
        }
        if (candidates.empty()) break;

        // Coverage guard: if there are feasible untouched targets, prioritize them.
        std::unordered_map<int, int> target_engaged_count;
        for (const auto& [key, shots] : sol.k) {
            if (shots <= 0) continue;
            int tid = static_cast<int>(key & 0xFFFFFFFF);
            target_engaged_count[tid] += shots;
        }
        std::vector<portfolio::Candidate> untouched_candidates;
        untouched_candidates.reserve(candidates.size());
        for (const auto& c : candidates) {
            if (!target_engaged_count.count(c.tid))
                untouched_candidates.push_back(c);
        }
        if (!untouched_candidates.empty())
            candidates.swap(untouched_candidates);

        // Track feasible-weapon counts for H_OPPORTUNITY_LOCK in next step.
        std::unordered_map<int, int> cur_target_feasible_weapons;
        for (const auto& c : candidates) cur_target_feasible_weapons[c.tid]++;

        double phase = 0.0;
        if (initial_ammo_total > 0)
            phase = std::min(1.0, static_cast<double>(step) / static_cast<double>(initial_ammo_total));

        // UCB1 per phase-bin: try each heuristic once in the bin, then maximize mean + c*sqrt(log(N)/n).
        int phase_bin = 0;
        if (phase >= (2.0 / 3.0)) phase_bin = 2;
        else if (phase >= (1.0 / 3.0)) phase_bin = 1;

        int chosen_h_idx = -1;
        for (int i = 0; i < static_cast<int>(hh_uses[phase_bin].size()); ++i) {
            if (hh_uses[phase_bin][i] == 0) {
                chosen_h_idx = i;
                break;
            }
        }
        if (chosen_h_idx < 0) {
            double logN = std::log(static_cast<double>(std::max(1, hh_total_uses[phase_bin])));
            double best_ucb = -std::numeric_limits<double>::infinity();
            for (int i = 0; i < static_cast<int>(hh_uses[phase_bin].size()); ++i) {
                double mean = hh_reward_sum[phase_bin][i] / static_cast<double>(hh_uses[phase_bin][i]);
                double bonus = ucb_c * std::sqrt(logN / static_cast<double>(hh_uses[phase_bin][i]));
                double ucb = mean + bonus;
                if (ucb > best_ucb + eps) {
                    best_ucb = ucb;
                    chosen_h_idx = i;
                }
            }
        }

        portfolio::SelectContext ctx;
        ctx.phase = phase;
        ctx.prev_target_feasible_weapons = prev_target_feasible_weapons.empty()
            ? nullptr
            : &prev_target_feasible_weapons;

        portfolio::SelectResult selected = portfolio::select_candidate(
            active_heuristics[chosen_h_idx], sol, candidates, ctx);
        if (!selected.found) {
            // Robust fallback when a restrictive heuristic yields no option.
            selected = portfolio::select_candidate(portfolio::HeuristicId::H_SURV, sol, candidates, ctx);
        }

        if (!selected.found) break;

        int chosen_wid = selected.cand.wid;
        int chosen_tid = selected.cand.tid;
        double chosen_t = selected.cand.t;

        double obj_before = sol.objective();

        // Dirty weapons before state mutates
        std::vector<int> dirty = {chosen_wid};
        if (auto it = target_weapons.find(chosen_tid); it != target_weapons.end())
            for (int w : it->second)
                if (w != chosen_wid) dirty.push_back(w);

        // Invalidate cache for dirty weapons
        for (int w : dirty) cache.erase(w);

        sol.commit(chosen_wid, chosen_tid, chosen_t);

        double obj_after = sol.objective();
        double raw_reward = obj_before - obj_after;
        double reward_denom = std::max(std::fabs(obj_before), 1e-9);
        double reward = raw_reward / reward_denom;
        reward = std::clamp(reward, -1.0, 1.0);

        for (auto& pc : pending_credits) {
            pc.acc += pc.next_discount * reward;
            pc.next_discount *= reward_gamma;
            pc.age += 1;
        }

        std::vector<PendingCredit> still_pending;
        still_pending.reserve(pending_credits.size() + 1);
        for (auto& pc : pending_credits) {
            if (pc.age >= reward_horizon_k) {
                hh_uses[pc.phase_bin][pc.heuristic_idx] += 1;
                hh_reward_sum[pc.phase_bin][pc.heuristic_idx] += pc.acc;
                hh_total_uses[pc.phase_bin] += 1;
            } else {
                still_pending.push_back(pc);
            }
        }
        pending_credits.swap(still_pending);

        PendingCredit cur;
        cur.phase_bin = phase_bin;
        cur.heuristic_idx = chosen_h_idx;
        cur.age = 1;
        cur.acc = reward;
        cur.next_discount = reward_gamma;
        pending_credits.push_back(cur);

        prev_target_feasible_weapons = std::move(cur_target_feasible_weapons);
        step += 1;

        // Rescore surviving pairs of dirty weapons
        for (int w : dirty) {
            auto tgts_it = sol.weapon_targets.find(w);
            if (tgts_it == sol.weapon_targets.end()) continue;
            for (int j : tgts_it->second)
                try_score(w, j);
            if (cache.count(w) && cache[w].empty()) cache.erase(w);
        }
    }

    // Flush residual delayed credits at the end of construction.
    for (const auto& pc : pending_credits) {
        hh_uses[pc.phase_bin][pc.heuristic_idx] += 1;
        hh_reward_sum[pc.phase_bin][pc.heuristic_idx] += pc.acc;
        hh_total_uses[pc.phase_bin] += 1;
    }

    return sol;
}

// ---------------------------------------------------------------------------
// grasp — run `restarts` constructions, return best solution
// ---------------------------------------------------------------------------
Solution grasp(
    const std::vector<Weapon>&  weapons,
    const std::vector<Target>&  targets,
    const std::unordered_map<uint64_t, double>&                   p_ij,
    const std::unordered_map<uint64_t, std::pair<double,double>>& windows,
    const std::unordered_map<int, double>&  burst_dur,
    const std::unordered_map<int, int>&     max_shots,
    const std::unordered_map<int, int>&     vessel_id_map,
    double horizon,
    double alpha    = 0.15,
    int    restarts = 1,
    uint32_t seed   = 42)
{
    std::mt19937 rng(seed);
    Solution     best;
    bool         have_best = false;

    for (int r = 0; r < restarts; ++r) {
        Solution sol = Solution::empty(
            weapons, targets, p_ij, windows,
            burst_dur, max_shots, vessel_id_map, horizon);
        grasp_construction(sol, alpha, rng);
        if (!have_best || sol.objective() < best.objective()) {
            best      = sol;
            have_best = true;
        }
    }
    assert(have_best);
    return best;
}

// Compile: g++ -std=c++17 -O3 -march=native -I/opt/conda/include -o wta_solver main.cpp
// Run:     ./wta_solver [scenario.json] [--restarts N] [--alpha A] [--seed S]