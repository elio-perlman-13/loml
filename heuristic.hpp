#pragma once
#include "solution.hpp"
#include <algorithm>
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
    (void)t; (void)wid; (void)exclusive_cnt;
    return sol.survival(tid);
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

    while (!cache.empty()) {
        // Find best score — O(live pairs), fast linear scan in C++
        double best_sc = -std::numeric_limits<double>::infinity();
        for (auto& [wid, inner] : cache)
            for (auto& [tid, cs] : inner)
                best_sc = std::max(best_sc, cs.sc);
        if (best_sc <= 0.0) break;

        // Regret-based weapon choice:
        //   1) For each weapon, compute best and second-best eligible actions.
        //   2) Pick weapon by priority = (best-second) + rho*best.
        //   3) Execute that weapon's best action by main score.
        constexpr double eps = 1e-12;
        constexpr double rho = 0.05;
        double threshold = (1.0 - alpha) * best_sc;

        bool   have_choice = false;
        int    chosen_wid  = -1;
        int    chosen_tid  = -1;
        double chosen_t    = 0.0;
        double chosen_best = -std::numeric_limits<double>::infinity();
        double chosen_pri  = -std::numeric_limits<double>::infinity();

        for (auto& [wid, inner] : cache) {
            int    best_tid  = -1;
            double best_t    = 0.0;
            double best_sc_w = -std::numeric_limits<double>::infinity();
            double second_sc = -std::numeric_limits<double>::infinity();

            for (auto& [tid, cs] : inner) {
                if (cs.sc + eps < threshold) continue;

                if (cs.sc > best_sc_w + eps ||
                    (std::fabs(cs.sc - best_sc_w) <= eps &&
                     (sol.threat_score.at(tid) > sol.threat_score.at(best_tid) + eps ||
                      (std::fabs(sol.threat_score.at(tid) - sol.threat_score.at(best_tid)) <= eps &&
                       (tid < best_tid || (tid == best_tid && cs.t < best_t)))))) {
                    second_sc = best_sc_w;
                    best_sc_w = cs.sc;
                    best_tid  = tid;
                    best_t    = cs.t;
                } else if (cs.sc > second_sc + eps) {
                    second_sc = cs.sc;
                }
            }

            if (best_tid < 0) continue;
            if (!std::isfinite(second_sc)) second_sc = 0.0;

            double regret = best_sc_w - second_sc;
            double priority = regret + rho * best_sc_w;

            if (!have_choice ||
                priority > chosen_pri + eps ||
                (std::fabs(priority - chosen_pri) <= eps &&
                 (best_sc_w > chosen_best + eps ||
                  (std::fabs(best_sc_w - chosen_best) <= eps &&
                    (sol.threat_score.at(best_tid) > sol.threat_score.at(chosen_tid) + eps ||
                    (std::fabs(sol.threat_score.at(best_tid) - sol.threat_score.at(chosen_tid)) <= eps &&
                     (wid < chosen_wid ||
                      (wid == chosen_wid && (best_tid < chosen_tid ||
                       (best_tid == chosen_tid && best_t < chosen_t)))))))))) {
                have_choice = true;
                chosen_wid  = wid;
                chosen_tid  = best_tid;
                chosen_t    = best_t;
                chosen_best = best_sc_w;
                chosen_pri  = priority;
            }
        }

        if (!have_choice) break;

        // Dirty weapons before state mutates
        std::vector<int> dirty = {chosen_wid};
        if (auto it = target_weapons.find(chosen_tid); it != target_weapons.end())
            for (int w : it->second)
                if (w != chosen_wid) dirty.push_back(w);

        // Invalidate cache for dirty weapons
        for (int w : dirty) cache.erase(w);

        sol.commit(chosen_wid, chosen_tid, chosen_t);

        // Rescore surviving pairs of dirty weapons
        for (int w : dirty) {
            auto tgts_it = sol.weapon_targets.find(w);
            if (tgts_it == sol.weapon_targets.end()) continue;
            for (int j : tgts_it->second)
                try_score(w, j);
            if (cache.count(w) && cache[w].empty()) cache.erase(w);
        }
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