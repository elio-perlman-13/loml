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
// opp_cost — opportunity cost (≤ 0) of committing (wid, tid) at time t
//
// Only the covering interval (fs, fe) changes.  For each other live target j':
//   before = slots in [a_j', b_j'] ∩ (fs, fe)
//   after  = slots in [a_j', b_j'] ∩ (fs, t) + slots in [a_j', b_j'] ∩ (t+d, fe)
//   delta  = after - before  (≤ 0)
//   cap_after = cap_before + delta   (clamped to effective ammo/max_shots limits)
// ---------------------------------------------------------------------------
static double opp_cost(const Solution& sol, int wid, int tid, double t) {
    double         d    = sol.burst_dur->at(wid);
    const auto&    fv   = sol.free.at(wid);
    double         end  = t + d;
    int            ammo = sol.remaining_ammo.at(wid);
    int            M    = sol.max_shots->at(wid);

    // find covering interval via upper_bound then step back
    auto it = std::upper_bound(fv.begin(), fv.end(),
        Interval{t, std::numeric_limits<double>::infinity()},
        [](const Interval& val, const Interval& iv){ return val.s < iv.s; });
    if (it != fv.begin()) --it;
    double fs = it->s, fe = it->e;

    double opp = 0.0;
    auto   tgts_it = sol.weapon_targets.find(wid);
    if (tgts_it == sol.weapon_targets.end()) return 0.0;

    for (int jp : tgts_it->second) {
        if (jp == tid) continue;
        uint64_t key = pair_key(wid, jp);

        auto cap_it = sol.cap.find(key);
        if (cap_it == sol.cap.end() || cap_it->second <= 0) continue;
        if (ammo <= 1) continue;
        int k_jp = 0;
        if (auto ki = sol.k.find(key); ki != sol.k.end()) k_jp = ki->second;
        if (k_jp >= M) continue;

        auto [a, b] = sol.windows->at(key);
        int before = slots_overlap(a, b, fs, fe, d);
        if (before == 0) continue;

        int after = slots_overlap(a, b, fs,  t,  d)
                  + slots_overlap(a, b, end, fe,  d);
        int delta = after - before;
        if (delta == 0) continue;

        int cap_before = std::min({cap_it->second, ammo,     M - k_jp});
        if (cap_before <= 0) continue;
        int cap_after  = std::min({cap_it->second + delta, ammo - 1, M - k_jp - 1});
        cap_after      = std::max(cap_after, 0);
        if (cap_after >= cap_before) continue;

        double p       = sol.p_ij->count(key) ? sol.p_ij->at(key) : 0.0;
        double surv_jp = sol.survival(jp);
        double threat_jp = sol.threat_score.at(jp);
        opp += 1e-2 * surv_jp * threat_jp * (std::pow(1.0 - p, cap_after) - std::pow(1.0 - p, cap_before));
    }
    return opp;  // ≤ 0
}

struct TailRiskState {
    double lambda = 0.0;
    double cutoff = 0.0;
};

static TailRiskState compute_tail_risk_state(const Solution& sol) {
    constexpr std::size_t top_k = 5;

    TailRiskState state;
    if (sol.threat_score.empty()) return state;

    std::vector<double> residuals;
    residuals.reserve(sol.threat_score.size());
    double mean_residual = 0.0;
    for (auto& [tid, _] : sol.threat_score) {
        double residual = sol.residual_threat(tid);
        residuals.push_back(residual);
        mean_residual += residual;
    }
    mean_residual /= static_cast<double>(residuals.size());

    std::sort(residuals.begin(), residuals.end(), std::greater<double>());
    std::size_t use_k = std::min(top_k, residuals.size());
    double top_sum = 0.0;
    for (std::size_t i = 0; i < use_k; ++i) top_sum += residuals[i];

    double avg_top = top_sum / static_cast<double>(use_k);
    state.cutoff = residuals[use_k - 1];
    if (mean_residual > 0.0) {
        state.lambda = std::min(1.0, std::max(0.0, avg_top / mean_residual - 1.0));
    }
    return state;
}

// ---------------------------------------------------------------------------
// score — approximate rollout score for candidate (wid, tid, t)
//   score = gain + opp_cost
//   gain     = survival[j] * p_ij          (direct threat reduction, > 0)
//   opp_cost = Σ_{j'≠j} survival[j'] * ((1-p)^cap_after - (1-p)^cap_before)  (≥ 0 when capacity lost)
// ---------------------------------------------------------------------------
static double score(const Solution& sol, const TailRiskState& tail_state, int wid, int tid, double t) {
    uint64_t key = pair_key(wid, tid);
    double   p   = sol.p_ij->count(key) ? sol.p_ij->at(key) : 0.0;
    double residual_before = sol.residual_threat(tid);
    double residual_after  = residual_before * (1.0 - p);
    double delta_mean      = residual_before - residual_after;
    double delta_tail      = std::max(residual_before - tail_state.cutoff, 0.0)
                           - std::max(residual_after  - tail_state.cutoff, 0.0);
//    return delta_mean + tail_state.lambda * delta_tail + opp_cost(sol, wid, tid, t);
    return delta_mean + tail_state.lambda * delta_tail;
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
        if (sol.remaining_ammo.at(wid) <= 0) return;
        int k_val = 0;
        if (auto ki = sol.k.find(key); ki != sol.k.end()) k_val = ki->second;
        if (k_val >= sol.max_shots->at(wid)) return;
        double t = sol.first_slot(wid, tid);
        if (std::isnan(t)) return;
        cache[wid][tid] = {t, 0.0};
    };

    auto refresh_scores = [&](const TailRiskState& tail_state) {
        for (auto& [wid, inner] : cache)
            for (auto& [tid, cs] : inner)
                cs.sc = score(sol, tail_state, wid, tid, cs.t);
    };

    // Initial full scoring
    for (auto& [wid, tgts] : sol.weapon_targets)
        for (int tid : tgts)
            try_score(wid, tid);

    refresh_scores(compute_tail_risk_state(sol));

    while (!cache.empty()) {
        refresh_scores(compute_tail_risk_state(sol));

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

            int    best_tid = -1;
            double best_t   = 0.0;
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