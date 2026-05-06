/* ─────────────────────────────────────────────
HELPER FUNCTIONS
─────────────────────────────────────────────

contrib(s, e, a, b, d):
    lo ← max(s, a)
    hi ← min(e, b)
    return floor((hi - lo) / d)  if hi > lo  else 0

count_slots(free[i], a, b, d):
    return Σ contrib(s, e, a, b, d)  for each (s,e) in free[i]

first_slot(free[i], a, b, d):
    for each (s, e) in free[i] sorted by s:
        t ← max(s, a)
        if t + d ≤ min(e, b):
            return t
    return None   // infeasible

─────────────────────────────────────────────
INITIALIZE
─────────────────────────────────────────────

for each i in W:
    free[i]  ← [(0, +∞)]
    ammo[i]  ← μ[i]

for each j in T:
    survival[j] ← w[j]

for each (i, j):
    k[i][j]     ← 0
    d[i]        ← b_i + R_i
    sched[i][j] ← count_slots(free[i], a[i][j], b[i][j], d[i])
    f[i][j]     ← first_slot(free[i], a[i][j], b[i][j], d[i])
    cap[i][j]   ← min(sched[i][j], ammo[i], M[i])

─────────────────────────────────────────────
MAIN GREEDY LOOP
─────────────────────────────────────────────

while exists (i,j) with cap[i][j] > 0:

    best_score ← -∞,  best_i ← None,  best_j ← None

    for each (i, j) with cap[i][j] > 0:

        f     ← f[i][j]
        (s,e) ← interval in free[i] that contains f   // the split interval

        gain ← survival[j] * p[i][j]

        opp_cost ← 0
        for each j' ≠ j with cap[i][j'] > 0:
            old ← contrib(s, e,        a[i][j'], b[i][j'], d[i])
            lft ← contrib(s, f,        a[i][j'], b[i][j'], d[i])
            rgt ← contrib(f+d[i], e,   a[i][j'], b[i][j'], d[i])
            lost_sched ← old - lft - rgt                // ∈ {0, 1, 2}

            new_cap ← min(sched[i][j'] - lost_sched,
                          ammo[i] - 1,
                          M[i] - k[i][j'])
            lost_total ← cap[i][j'] - new_cap           // ≥ 0

            opp_cost += lost_total * survival[j'] * p[i][j']

        score ← gain - opp_cost

        if score > best_score:
            best_score ← score
            best_i ← i,  best_j ← j

    ─────────────────────────────────────────
    COMMIT (best_i, best_j)
    ─────────────────────────────────────────

    i*, j* ← best_i, best_j
    f*      ← f[i*][j*]
    (s, e)  ← interval in free[i*] containing f*

    // 1. Update free interval list
    remove (s, e) from free[i*]
    if f* > s:          insert (s, f*)        into free[i*]
    if f* + d[i*] < e:  insert (f*+d[i*], e)  into free[i*]

    // 2. Update assignment state
    k[i*][j*]  += 1
    ammo[i*]   -= 1

    // 3. Update survival
    survival[j*] *= (1 - p[i*][j*])

    // 4. Update sched, f, cap for all j' under weapon i*
    for each j' in T:

        if j' == j*:
            sched[i*][j*] ← count_slots(free[i*], a[i*][j*], b[i*][j*], d[i*])
            if f[i*][j*] == f*:
                f[i*][j*] ← first_slot(free[i*], a[i*][j*], b[i*][j*], d[i*])
            cap[i*][j*] ← min(sched[i*][j*], ammo[i*], M[i*] - k[i*][j*])

        else:
            // sched: O(1) delta on split interval only
            old ← contrib(s, e,       a[i*][j'], b[i*][j'], d[i*])
            lft ← contrib(s, f*,      a[i*][j'], b[i*][j'], d[i*])
            rgt ← contrib(f*+d[i*],e, a[i*][j'], b[i*][j'], d[i*])
            sched[i*][j'] -= (old - lft - rgt)

            // f: advance only if it was pointing at the committed slot
            if f[i*][j'] == f*:
                f[i*][j'] ← first_slot(free[i*], a[i*][j'], b[i*][j'], d[i*])

            // cap: recheck all three constraints
            cap[i*][j'] ← min(sched[i*][j'], ammo[i*], M[i*] - k[i*][j'])

─────────────────────────────────────────────
OUTPUT
─────────────────────────────────────────────

return { (i, j, f[i][j], f[i][j] + k[i][j]*d[i]) : k[i][j] > 0 }
// maps to WTAAssignment { WeaponID, TargetID, AmmoUsed=k[i][j],
//                         FireTime, EndTime, PKill=1-survival[j]/w[j] } */