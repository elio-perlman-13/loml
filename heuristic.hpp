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
// score — target score used in stage-1 target selection
// ---------------------------------------------------------------------------
static double score(const Solution& sol, int wid, int tid, double t,
                    int exclusive_cnt) {
    (void)t;
    (void)wid;
    (void)exclusive_cnt;
    return sol.survival(tid) * sol.threat_score.at(tid);
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

        // Two-stage lexicographic choice:
        //   1) Pick a target from RCL by target score, then threat, then earliest close.
        //   2) For that target, pick weapon by scarcity, then exclusivity, then earliest fire time.
        constexpr double eps = 1e-12;
        double threshold = (1.0 - alpha) * best_sc;

        bool   have_choice = false;
        int    chosen_wid  = -1;
        int    chosen_tid  = -1;
        double chosen_t    = 0.0;
        int    selected_tid = -1;
        double selected_sc  = -std::numeric_limits<double>::infinity();
        double selected_th  = -std::numeric_limits<double>::infinity();
        double selected_we  = std::numeric_limits<double>::infinity();

        // Stage 1: choose target using target score and target-side tie-breaks.
        for (auto& [wid, inner] : cache) {
            for (auto& [tid, cs] : inner) {
                if (cs.sc + eps < threshold) continue;

                uint64_t key = pair_key(wid, tid);
                auto w_it = sol.windows->find(key);
                double window_end = (w_it == sol.windows->end())
                    ? std::numeric_limits<double>::infinity()
                    : w_it->second.second;
                double th = sol.threat_score.at(tid);

                if (selected_tid < 0 ||
                    cs.sc > selected_sc + eps ||
                    (std::fabs(cs.sc - selected_sc) <= eps &&
                     (th > selected_th + eps ||
                      (std::fabs(th - selected_th) <= eps &&
                       (window_end < selected_we - eps ||
                        (std::fabs(window_end - selected_we) <= eps && tid < selected_tid)))))) {
                    selected_tid = tid;
                    selected_sc  = cs.sc;
                    selected_th  = th;
                    selected_we  = window_end;
                }
            }
        }

        // Stage 2: choose weapon for selected target by weapon-side rules.
        if (selected_tid >= 0) {
            int best_scarcity = std::numeric_limits<int>::max();
            int best_excl     = -1;

            for (auto& [wid, inner] : cache) {
                auto it = inner.find(selected_tid);
                if (it == inner.end()) continue;
                const CS& cs = it->second;
                if (cs.sc + eps < threshold) continue;

                int scarcity = static_cast<int>(inner.size());

                int excl_live = 0;
                for (auto& [j, cs2] : inner) {
                    (void)cs2;
                    auto tw_it = target_weapons.find(j);
                    if (tw_it != target_weapons.end() && tw_it->second.size() == 1)
                        ++excl_live;
                }

                if (!have_choice ||
                    scarcity < best_scarcity ||
                    (scarcity == best_scarcity &&
                     (excl_live > best_excl ||
                      (excl_live == best_excl &&
                       (cs.t < chosen_t - eps ||
                        (std::fabs(cs.t - chosen_t) <= eps && wid < chosen_wid)))))) {
                    have_choice  = true;
                    chosen_wid   = wid;
                    chosen_tid   = selected_tid;
                    chosen_t     = cs.t;
                    best_scarcity = scarcity;
                    best_excl     = excl_live;
                }
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