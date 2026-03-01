// =============================================================================
// Cargo.toml:
//
// [package]
// name    = "heroes"
// version = "0.1.0"
// edition = "2021"
//
// [dependencies]
// csv       = "1.3"
// ndarray   = "0.16"
// rand      = "0.8"
// rayon     = "1.10"
// serde     = { version = "1", features = ["derive"] }
// indicatif = { version = "0.17", features = ["rayon"] }
//
// [profile.release]
// opt-level     = 3
// lto           = true
// codegen-units = 1
// =============================================================================

#![allow(clippy::needless_range_loop)]

use std::cmp::{max, min, Reverse};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use csv::ReaderBuilder;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use ndarray::Array2;
use rand::seq::SliceRandom;
use rand::Rng;
use rayon::prelude::*;
use serde::Deserialize;

// ─── Domain constants ────────────────────────────────────────────────────────

const VISIT_COST: u32 = 100;
const HERO_COST:  u32 = 2500;
const MAX_DAY:    u32 = 7;
const MAX_HEROES: u32 = 100;

// ─── Progress-bar styles ─────────────────────────────────────────────────────

fn spinner_style() -> ProgressStyle {
    ProgressStyle::with_template("{spinner:.cyan} [{elapsed_precise}] {msg}")
        .unwrap()
        .tick_strings(&["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"])
}

fn bar_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{msg}\n  [{wide_bar:.cyan/blue}] {pos}/{len} ({percent}%)  eta {eta}",
    )
    .unwrap()
    .progress_chars("█▉▊▋▌▍▎▏  ")
}

fn gen_bar_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "GA   [{elapsed_precise}] [{wide_bar:.yellow/white}] \
         gen {pos}/{len} ({percent}%)  best={msg}  eta {eta}",
    )
    .unwrap()
    .progress_chars("█▉▊▋▌▍▎▏  ")
}

fn post_bar_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{msg}\n  [{wide_bar:.magenta/white}] {pos}/{len} ({percent}%)  eta {eta}",
    )
    .unwrap()
    .progress_chars("█▉▊▋▌▍▎▏  ")
}

// ─── CSV deserialisation structs ──────────────────────────────────────────────

#[derive(Deserialize)] struct ObjectRecord  { object_id: u32, day_open: u32, reward: u32 }
#[derive(Deserialize)] struct HeroRecord    { hero_id: u32, move_points: u32 }
#[derive(Deserialize)] struct DistStartRecord { object_id: u32, dist_start: u32 }

// ─── Domain structs ───────────────────────────────────────────────────────────

#[derive(Clone)] struct Mill { day_open: u32, reward: u32 }
#[derive(Clone)] struct Hero { move_points: u32 }

/// GA individual.
///
/// `hero_rewards` caches `simulate_hero()` output per hero so local-search
/// moves that touch only one or two heroes update fitness in O(1) instead of
/// re-simulating the entire solution.
#[derive(Clone)]
struct Solution {
    max_id:       u32,
    routes:       HashMap<u32, Vec<u32>>, // hero_id → ordered object_ids
    hero_rewards: HashMap<u32, u32>,      // hero_id → cached on-time reward
    fitness:      Option<i64>,            // Σ rewards  −  max_id × HERO_COST
}

// ─── Problem data (shared, read-only) ────────────────────────────────────────

struct ProblemData {
    heroes:      HashMap<u32, Hero>,
    mills:       HashMap<u32, Mill>,
    /// Flat distance matrix: (n_mills+1) × (n_mills+1), index 0 = castle.
    dist:        Array2<u32>,
    /// Mills grouped by day_open for fast greedy / crossover access.
    mills_by_day: [Vec<u32>; 8], // index 0 unused, [1..=7] hold mill ids
    /// All mill ids sorted for deterministic iteration.
    mill_ids:    Vec<u32>,
    n_mills:     usize,
}

// Array2<u32> and HashMap are Send; we only ever read after construction.
unsafe impl Sync for ProblemData {}

// ─── Data loading ─────────────────────────────────────────────────────────────

fn load_data(mp: &MultiProgress) -> ProblemData {
    let sp = mp.add(ProgressBar::new_spinner());
    sp.set_style(spinner_style());
    sp.enable_steady_tick(Duration::from_millis(80));

    sp.set_message("Loading data_objects.csv …");
    let mut mills: HashMap<u32, Mill> = HashMap::new();
    for r in ReaderBuilder::new().has_headers(true)
        .from_path("data_objects.csv").expect("data_objects.csv missing").deserialize()
    {
        let rec: ObjectRecord = r.unwrap();
        mills.insert(rec.object_id, Mill { day_open: rec.day_open, reward: rec.reward });
    }
    let n_mills = mills.len();

    sp.set_message("Loading data_heroes.csv …");
    let mut heroes: HashMap<u32, Hero> = HashMap::new();
    for r in ReaderBuilder::new().has_headers(true)
        .from_path("data_heroes.csv").expect("data_heroes.csv missing").deserialize()
    {
        let rec: HeroRecord = r.unwrap();
        heroes.insert(rec.hero_id, Hero { move_points: rec.move_points });
    }

    sp.set_message("Loading dist_start.csv …");
    let mut dist_start: HashMap<u32, u32> = HashMap::new();
    for r in ReaderBuilder::new().has_headers(true)
        .from_path("dist_start.csv").expect("dist_start.csv missing").deserialize()
    {
        let rec: DistStartRecord = r.unwrap();
        dist_start.insert(rec.object_id, rec.dist_start);
    }

    sp.set_message("Loading dist_objects.csv …");
    let file = File::open("dist_objects.csv").expect("dist_objects.csv missing");
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);
    let headers = rdr.headers().unwrap().clone();
    let col_idx: Vec<usize> = headers.iter()
        .map(|h| h.split('_').last().unwrap_or(h).parse::<usize>().unwrap())
        .collect();

    let mut tmp = vec![vec![0u32; n_mills]; n_mills];
    for (i, r) in rdr.deserialize::<Vec<u32>>().enumerate() {
        let row = r.unwrap();
        for (j, &v) in row.iter().enumerate() { tmp[i][col_idx[j] - 1] = v; }
    }

    let mut dist = Array2::<u32>::zeros((n_mills + 1, n_mills + 1));
    for i in 1..=n_mills { for j in 1..=n_mills { dist[[i, j]] = tmp[i-1][j-1]; } }
    for i in 1..=n_mills {
        let d = *dist_start.get(&(i as u32)).unwrap();
        dist[[0, i]] = d; dist[[i, 0]] = d;
    }

    let mut mill_ids: Vec<u32> = mills.keys().copied().collect();
    mill_ids.sort_unstable();

    // Pre-group mills by day_open
    let mut mills_by_day: [Vec<u32>; 8] = Default::default();
    for &id in &mill_ids { mills_by_day[mills[&id].day_open as usize].push(id); }

    sp.finish_with_message(format!("✓ Loaded  {} mills  {} heroes", n_mills, heroes.len()));

    ProblemData { heroes, mills, dist, mills_by_day, mill_ids, n_mills }
}

// ─── Simulation ───────────────────────────────────────────────────────────────

/// Simulate hero travelling `route`. Returns the total on-time reward.
#[inline]
fn simulate_hero(hero_id: u32, route: &[u32], d: &ProblemData) -> u32 {
    if route.is_empty() { return 0; }
    let mp_full   = d.heroes[&hero_id].move_points;
    let mut day   = 1u32;
    let mut rem   = mp_full;
    let mut pos   = 0u32;
    let mut total = 0u32;

    for &mid in route {
        // ── Travel (may span multiple days) ───────────────────────────────
        let mut dist = d.dist[[pos as usize, mid as usize]];
        loop {
            if rem >= dist { rem -= dist; break; }
            dist -= rem;
            day  += 1;
            if day > MAX_DAY { return total; }
            rem = mp_full;
        }

        // ── Wait at early arrival ─────────────────────────────────────────
        let mill = &d.mills[&mid];
        if day < mill.day_open { day = mill.day_open; rem = mp_full; }
        if day > MAX_DAY { return total; }

        // ── Visit (last-move rule: 1 MP suffices) ────────────────────────
        if rem == 0 { pos = mid; continue; }
        rem = rem.saturating_sub(VISIT_COST);
        if day == mill.day_open { total += mill.reward; }
        pos = mid;
    }
    total
}

// ─── Cached-fitness helpers ───────────────────────────────────────────────────

/// Recompute reward for a single hero and update the fitness delta.
/// O(route_len) — much cheaper than full recompute when only one route changed.
fn refresh_hero(sol: &mut Solution, hid: u32, d: &ProblemData) {
    let route   = sol.routes.get(&hid).map_or(&[] as &[u32], |v| v.as_slice());
    let new_rew = if hid <= sol.max_id { simulate_hero(hid, route, d) } else { 0 };
    let old_rew = sol.hero_rewards.insert(hid, new_rew).unwrap_or(0);
    if let Some(f) = sol.fitness.as_mut() { *f += new_rew as i64 - old_rew as i64; }
}

/// Full recompute from scratch — used after structural changes (max_id change, crossover).
fn recompute(sol: &mut Solution, d: &ProblemData) {
    sol.hero_rewards.clear();
    let mut total = 0i64;
    for hid in 1..=sol.max_id {
        let r   = sol.routes.get(&hid).map_or(&[] as &[u32], |v| v.as_slice());
        let rew = simulate_hero(hid, r, d);
        sol.hero_rewards.insert(hid, rew);
        total += rew as i64;
    }
    sol.fitness = Some(total - sol.max_id as i64 * HERO_COST as i64);
}

// ─── Solution construction helpers ───────────────────────────────────────────

fn empty_routes(max_id: u32) -> HashMap<u32, Vec<u32>> {
    (1..=max_id).map(|i| (i, Vec::new())).collect()
}

fn make_solution(max_id: u32, routes: HashMap<u32, Vec<u32>>, d: &ProblemData) -> Solution {
    let mut sol = Solution { max_id, routes, hero_rewards: HashMap::new(), fitness: None };
    recompute(&mut sol, d);
    sol
}

// ─── Greedy initialisation ────────────────────────────────────────────────────
//
// Strategy: for each hero, repeatedly pick the unassigned mill with the lowest
// combined cost:  distance_from_current  +  wait_days × move_points.
// This naturally groups same-day mills on the same hero and avoids late arrivals.

fn greedy_solution(d: &ProblemData, rng: &mut impl Rng) -> Solution {
    let max_id     = rng.gen_range(3u32..=25);
    let mut routes = empty_routes(max_id);
    let mut done   = HashSet::<u32>::new();

    for hid in 1..=max_id {
        let mp         = d.heroes[&hid].move_points;
        let mut pos    = 0u32;
        let mut day    = 1u32;
        let mut rem    = mp;

        loop {
            // Nearest feasible unassigned mill
            let best = d.mill_ids.iter()
                .filter(|&&id| !done.contains(&id))
                .filter_map(|&id| {
                    let mill       = &d.mills[&id];
                    let dist       = d.dist[[pos as usize, id as usize]];
                    let days_move  = if dist <= rem { 0 } else { 1 + (dist - rem + mp - 1) / mp };
                    let arrive_day = day + days_move;
                    let visit_day  = arrive_day.max(mill.day_open);
                    if visit_day > MAX_DAY { return None; }
                    let wait = mill.day_open.saturating_sub(arrive_day);
                    Some((dist + wait * mp, id, visit_day))
                })
                .min_by_key(|&(score, _, _)| score);

            let (_, mid, visit_day) = match best { None => break, Some(v) => v };

            done.insert(mid);
            routes.get_mut(&hid).unwrap().push(mid);

            // Advance simulation state
            let mut dl = d.dist[[pos as usize, mid as usize]];
            loop {
                if rem >= dl { rem -= dl; break; }
                dl -= rem; day += 1; rem = mp;
            }
            if day < d.mills[&mid].day_open { day = d.mills[&mid].day_open; rem = mp; }
            rem = rem.saturating_sub(VISIT_COST);
            pos = mid;
            day = visit_day;
        }
    }

    make_solution(max_id, routes, d)
}

// ─── Randomised solution ──────────────────────────────────────────────────────
//
// Assign a random 60-100% subset of mills to random heroes.
// Keeping a subset means some mills are left for other individuals —
// genetic diversity benefits from incomplete solutions early on.

fn random_solution(d: &ProblemData, rng: &mut impl Rng) -> Solution {
    let max_id     = rng.gen_range(1u32..=MAX_HEROES);
    let mut routes = empty_routes(max_id);
    let mut mills  = d.mill_ids.clone();
    mills.shuffle(rng);
    let take = rng.gen_range(mills.len() * 6 / 10..=mills.len());
    for &m in &mills[..take] {
        let h = rng.gen_range(1..=max_id);
        routes.get_mut(&h).unwrap().push(m);
    }
    make_solution(max_id, routes, d)
}

// ─── Population init (parallel) ───────────────────────────────────────────────

fn init_population(pop_size: usize, d: &ProblemData, mp: &MultiProgress) -> Vec<Solution> {
    let pb = mp.add(ProgressBar::new(pop_size as u64));
    pb.set_style(bar_style());
    pb.set_message("Initialising population");

    let greedy_count = (pop_size * 30 / 100).max(1); // 30% greedy seeds

    let mut pop: Vec<Solution> = (0..pop_size)
        .into_par_iter()
        .map(|i| {
            let mut rng = rand::thread_rng();
            let sol = if i < greedy_count { greedy_solution(d, &mut rng) }
                      else                { random_solution(d, &mut rng)  };
            pb.inc(1);
            sol
        })
        .collect();

    // Seed the very best greedy solution deterministically on a single thread
    // so we always start with at least one high-quality individual.
    let mut seed_rng = rand::thread_rng();
    let seed = greedy_solution(d, &mut seed_rng);
    pop.push(seed);

    pb.finish_with_message(format!(
        "✓ Pop {pop_size}  ({greedy_count} greedy + {} random + 1 seed)",
        pop_size - greedy_count
    ));
    pop
}

// ─── Tournament selection ────────────────────────────────────────────────────

fn tournament<'a>(pop: &'a [Solution], k: usize, rng: &mut impl Rng) -> &'a Solution {
    pop.choose_multiple(rng, k)
        .max_by_key(|s| s.fitness.unwrap_or(i64::MIN))
        .unwrap()
}

// ─── Crossover ────────────────────────────────────────────────────────────────
//
// Two strategies used in a 60/40 split:
//
// 1. Day-aware crossover (60%):
//    For each day 1..7, flip a coin to pick the "donor" parent.
//    Inherit all mills of that day from the donor, preserving hero assignment.
//    Sort each hero's route by day_open afterwards.
//    Result: time-compatible mills naturally cluster on the same heroes.
//
// 2. Hero-route crossover (40%):
//    For each hero slot i ≤ max_id, flip a coin to take the full route from
//    parent 1 or parent 2.  Deduplicate globally (first occurrence wins).
//    More disruptive — good for escaping local optima.

fn day_crossover(p1: &Solution, p2: &Solution, d: &ProblemData, rng: &mut impl Rng) -> Solution {
    let max_id     = rng.gen_range(min(p1.max_id, p2.max_id)..=max(p1.max_id, p2.max_id));
    let mut routes = empty_routes(max_id);
    let mut seen   = HashSet::<u32>::new();

    for day in 1..=MAX_DAY {
        let donor = if rng.gen_bool(0.5) { p1 } else { p2 };
        for hid in 1..=donor.max_id.min(max_id) {
            if let Some(route) = donor.routes.get(&hid) {
                for &mid in route {
                    if d.mills[&mid].day_open == day && seen.insert(mid) {
                        routes.get_mut(&hid).unwrap().push(mid);
                    }
                }
            }
        }
    }
    // Sort each route chronologically so the simulation sees mills in time order
    for r in routes.values_mut() { r.sort_by_key(|&m| d.mills[&m].day_open); }
    make_solution(max_id, routes, d)
}

fn hero_crossover(p1: &Solution, p2: &Solution, d: &ProblemData, rng: &mut impl Rng) -> Solution {
    let max_id     = rng.gen_range(min(p1.max_id, p2.max_id)..=max(p1.max_id, p2.max_id));
    let mut routes = empty_routes(max_id);
    let mut seen   = HashSet::<u32>::new();

    for i in 1..=max_id {
        let src = match (i <= p1.max_id, i <= p2.max_id) {
            (true, true)  => if rng.gen_bool(0.5) { &p1.routes[&i] } else { &p2.routes[&i] },
            (true, false) => &p1.routes[&i],
            _             => &p2.routes[&i],
        };
        routes.insert(i, src.iter().copied().filter(|&m| seen.insert(m)).collect());
    }
    make_solution(max_id, routes, d)
}

// ─── Mutation ─────────────────────────────────────────────────────────────────
//
// Five operators chosen uniformly:
//
// 0  swap      — swap two mills within one hero's route
// 1  relocate  — move a mill to a different position (possibly different hero)
// 2  add       — insert an unvisited mill in sorted day-order
// 3  delete    — remove a random mill from a route
// 4  resize    — hire or dismiss one hero

fn mutate(mut sol: Solution, d: &ProblemData) -> Solution {
    let mut rng = rand::thread_rng();

    match rng.gen_range(0u8..5) {
        // ── 0: swap ──────────────────────────────────────────────────────
        0 => {
            let eligible: Vec<u32> =
                (1..=sol.max_id).filter(|&h| sol.routes[&h].len() >= 2).collect();
            if let Some(&h) = eligible.choose(&mut rng) {
                let r = sol.routes.get_mut(&h).unwrap();
                let n = r.len();
                let i = rng.gen_range(0..n);
                let mut j = rng.gen_range(0..n);
                while j == i { j = rng.gen_range(0..n); }
                r.swap(i, j);
                refresh_hero(&mut sol, h, d);
            }
        }
        // ── 1: relocate ──────────────────────────────────────────────────
        1 => {
            let positions: Vec<(u32, usize)> = sol.routes.iter()
                .flat_map(|(&h, r)| (0..r.len()).map(move |i| (h, i)))
                .collect();
            if let Some(&(src, idx)) = positions.choose(&mut rng) {
                let mill = sol.routes.get_mut(&src).unwrap().remove(idx);
                let dst  = rng.gen_range(1..=sol.max_id);
                let r    = sol.routes.get_mut(&dst).unwrap();
                // Insert maintaining day-order
                let pos = r.partition_point(|&m| d.mills[&m].day_open <= d.mills[&mill].day_open);
                r.insert(pos, mill);
                refresh_hero(&mut sol, src, d);
                if dst != src { refresh_hero(&mut sol, dst, d); }
            }
        }
        // ── 2: add unvisited mill ─────────────────────────────────────────
        2 => {
            let visited: HashSet<u32> = sol.routes.values().flatten().copied().collect();
            let avail: Vec<u32> =
                d.mill_ids.iter().copied().filter(|k| !visited.contains(k)).collect();
            if let Some(&mill) = avail.choose(&mut rng) {
                let h = rng.gen_range(1..=sol.max_id);
                let r = sol.routes.get_mut(&h).unwrap();
                let p = r.partition_point(|&m| d.mills[&m].day_open <= d.mills[&mill].day_open);
                r.insert(p, mill);
                refresh_hero(&mut sol, h, d);
            }
        }
        // ── 3: delete random mill ─────────────────────────────────────────
        3 => {
            let positions: Vec<(u32, usize)> = sol.routes.iter()
                .flat_map(|(&h, r)| (0..r.len()).map(move |i| (h, i)))
                .collect();
            if let Some(&(h, idx)) = positions.choose(&mut rng) {
                sol.routes.get_mut(&h).unwrap().remove(idx);
                refresh_hero(&mut sol, h, d);
            }
        }
        // ── 4: hire / dismiss hero ────────────────────────────────────────
        _ => {
            if rng.gen_bool(0.5) && sol.max_id < MAX_HEROES {
                sol.max_id += 1;
                let new_id = sol.max_id; // copy before any borrow of sol
                sol.routes.insert(new_id, Vec::new());
                refresh_hero(&mut sol, new_id, d);
                // Adjust fitness for the new hire cost
                if let Some(f) = sol.fitness.as_mut() { *f -= HERO_COST as i64; }
            } else if sol.max_id > 1 {
                let old = sol.max_id;
                sol.max_id -= 1;
                for h in (sol.max_id + 1)..=old {
                    if let Some(r) = sol.routes.remove(&h) {
                        sol.hero_rewards.remove(&h);
                        // Redistribute dismissed hero's mills in day order
                        for mill in r {
                            let dst = rng.gen_range(1..=sol.max_id);
                            let route = sol.routes.get_mut(&dst).unwrap();
                            let p = route.partition_point(|&m| {
                                d.mills[&m].day_open <= d.mills[&mill].day_open
                            });
                            route.insert(p, mill);
                        }
                    }
                }
                // Multiple routes changed → full recompute
                recompute(&mut sol, d);
            }
        }
    }
    sol
}

// ─── Or-opt local search ─────────────────────────────────────────────────────
//
// Or-opt moves segments of 1, 2, or 3 consecutive mills to a better position,
// both within the same hero's route (intra) and across heroes (inter).
//
// Using the per-hero reward cache means each accepted move updates fitness with
// two simulate_hero calls instead of a full-solution recompute.
//
// Three phases each iteration:
//   A. Intra-route Or-opt   (segments within one hero)
//   B. Inter-route Or-opt   (move a segment from hero h1 to hero h2)
//   C. Inter-route swap     (exchange one mill between two heroes)

fn or_opt(mut sol: Solution, max_iter: u32, d: &ProblemData) -> Solution {
    if sol.fitness.is_none() { recompute(&mut sol, d); }

    // ── Late-mill priority: sort hero scan order so heroes carrying day-5/6/7
    //    mills come first.  Those routes have the tightest time windows and
    //    benefit most from positional improvements.
    let hero_order: Vec<u32> = {
        let mut order: Vec<u32> = (1..=sol.max_id).collect();
        order.sort_by_key(|&h| {
            // Primary key: max day_open in route (descending → Reverse)
            let max_day = sol.routes.get(&h)
                .and_then(|r| r.iter().map(|&m| d.mills[&m].day_open).max())
                .unwrap_or(0);
            Reverse(max_day)
        });
        order
    };

    for _ in 0..max_iter {
        let mut improved = false;

        // ── A: Intra-route Or-opt (late-mill heroes first) ────────────────
        'intra: for &h in &hero_order {
            let n = sol.routes.get(&h).map_or(0, |r| r.len());
            if n < 2 { continue; }

            // Within the route, try late-day segments before early-day ones.
            // Build a scan order for segment start indices sorted by the
            // day_open of the first mill in that segment (descending).
            let mut seg_starts: Vec<usize> = (0..n).collect();
            seg_starts.sort_by_key(|&i| {
                Reverse(d.mills[&sol.routes[&h][i]].day_open)
            });

            for seg in 1..=3usize {
                if seg >= n { continue; }
                for &i in &seg_starts {
                    if i + seg > n { continue; }
                    for j in 0..=(n - seg) {
                        if j >= i.saturating_sub(1) && j <= i + seg { continue; }

                        let old = sol.routes[&h].clone();
                        let mut new_r = old.clone();
                        let chunk: Vec<u32> = new_r.drain(i..i + seg).collect();
                        let ins = if j > i { (j - seg).min(new_r.len()) } else { j };
                        for (k, &m) in chunk.iter().enumerate() { new_r.insert(ins + k, m); }

                        let old_rew = sol.hero_rewards[&h];
                        let new_rew = simulate_hero(h, &new_r, d);
                        if new_rew > old_rew {
                            sol.routes.insert(h, new_r);
                            *sol.hero_rewards.get_mut(&h).unwrap()  = new_rew;
                            *sol.fitness.as_mut().unwrap() += new_rew as i64 - old_rew as i64;
                            improved = true;
                            break 'intra;
                        }
                    }
                }
            }
        }

        // ── B: Inter-route Or-opt (late-mill heroes as source first) ──────
        'inter: for &h1 in &hero_order {
            for &h2 in &hero_order {
                if h1 == h2 { continue; }
                let n1 = sol.routes.get(&h1).map_or(0, |r| r.len());
                let n2 = sol.routes.get(&h2).map_or(0, |r| r.len());
                if n1 == 0 { continue; }

                // Prioritise moving late-day segments out of h1
                let mut seg_starts: Vec<usize> = (0..n1).collect();
                seg_starts.sort_by_key(|&i| {
                    Reverse(d.mills[&sol.routes[&h1][i]].day_open)
                });

                for seg in 1..=3usize {
                    if seg > n1 { continue; }
                    for &i in &seg_starts {
                        if i + seg > n1 { continue; }
                        for ins in 0..=n2 {
                            let old1 = sol.routes[&h1].clone();
                            let old2 = sol.routes[&h2].clone();

                            let mut new1 = old1.clone();
                            let chunk: Vec<u32> = new1.drain(i..i + seg).collect();
                            let mut new2 = old2.clone();
                            for (k, &m) in chunk.iter().enumerate() { new2.insert(ins + k, m); }

                            let or1 = sol.hero_rewards[&h1];
                            let or2 = sol.hero_rewards[&h2];
                            let nr1 = simulate_hero(h1, &new1, d);
                            let nr2 = simulate_hero(h2, &new2, d);

                            if nr1 + nr2 > or1 + or2 {
                                sol.routes.insert(h1, new1);
                                sol.routes.insert(h2, new2);
                                *sol.hero_rewards.get_mut(&h1).unwrap() = nr1;
                                *sol.hero_rewards.get_mut(&h2).unwrap() = nr2;
                                *sol.fitness.as_mut().unwrap() +=
                                    (nr1 + nr2) as i64 - (or1 + or2) as i64;
                                improved = true;
                                break 'inter;
                            }
                        }
                    }
                }
            }
        }

        // ── C: Inter-route swap (late-mill heroes first) ──────────────────
        'swap: for &h1 in &hero_order {
            for &h2 in &hero_order {
                if h2 <= h1 { continue; } // avoid duplicate pairs
                let n1 = sol.routes.get(&h1).map_or(0, |r| r.len());
                let n2 = sol.routes.get(&h2).map_or(0, |r| r.len());
                if n1 == 0 || n2 == 0 { continue; }

                for i in 0..n1 {
                    for j in 0..n2 {
                        let mut new1 = sol.routes[&h1].clone();
                        let mut new2 = sol.routes[&h2].clone();
                        std::mem::swap(&mut new1[i], &mut new2[j]);

                        let or1 = sol.hero_rewards[&h1];
                        let or2 = sol.hero_rewards[&h2];
                        let nr1 = simulate_hero(h1, &new1, d);
                        let nr2 = simulate_hero(h2, &new2, d);

                        if nr1 + nr2 > or1 + or2 {
                            sol.routes.insert(h1, new1);
                            sol.routes.insert(h2, new2);
                            *sol.hero_rewards.get_mut(&h1).unwrap() = nr1;
                            *sol.hero_rewards.get_mut(&h2).unwrap() = nr2;
                            *sol.fitness.as_mut().unwrap() +=
                                (nr1 + nr2) as i64 - (or1 + or2) as i64;
                            improved = true;
                            break 'swap;
                        }
                    }
                }
            }
        }

        if !improved { break; }
    }
    sol
}

// ─── Day-order repair ────────────────────────────────────────────────────────
//
// After crossover and mutation routes can be out of time order.
// Sorting by day_open is a cheap, always-safe repair that never hurts fitness
// (the simulation itself is order-dependent, so sorted = more reward).

fn repair_order(mut sol: Solution, d: &ProblemData) -> Solution {
    let mut changed = false;
    for r in sol.routes.values_mut() {
        if r.windows(2).any(|w| d.mills[&w[0]].day_open > d.mills[&w[1]].day_open) {
            r.sort_by_key(|&m| d.mills[&m].day_open);
            changed = true;
        }
    }
    if changed { recompute(&mut sol, d); }
    sol
}


// ─── Double-bridge perturbation ──────────────────────────────────────────────
//
// Cuts one hero's route into 4 segments and reconnects them in a new order
// (seg0 + seg2 + seg1 + seg3).  This is the classic "Lin-Kernighan" escape
// move: it cannot be undone by any sequence of 2-opt moves, so it genuinely
// breaks out of local-optima basins.
//
// Applied per-hero independently when stagnation is high.

fn double_bridge(mut sol: Solution, d: &ProblemData, rng: &mut impl Rng) -> Solution {
    let heroes: Vec<u32> = (1..=sol.max_id)
        .filter(|&h| sol.routes.get(&h).map_or(false, |r| r.len() >= 8))
        .collect();

    if heroes.is_empty() { return sol; }

    for &h in &heroes {
        let n = sol.routes[&h].len();
        // Draw 3 unique cut-points in 1..n-1 and sort them → a < b < c
        let mut cuts: Vec<usize> = (1..n).collect();
        cuts.partial_shuffle(rng, 3);
        let mut c = [cuts[0], cuts[1], cuts[2]];
        c.sort_unstable();
        let (a, b, cc) = (c[0], c[1], c[2]);

        let route = sol.routes[&h].clone();
        // Classic double-bridge reconnection order: seg0 + seg2 + seg1 + seg3
        let new_route: Vec<u32> = route[..a].iter()
            .chain(&route[b..cc])
            .chain(&route[a..b])
            .chain(&route[cc..])
            .copied()
            .collect();

        sol.routes.insert(h, new_route);
        refresh_hero(&mut sol, h, d);
    }
    sol
}

// ─── Insert-missing-mills pass ────────────────────────────────────────────────
//
// Scans every mill not yet in any route.  For each missing mill, tries every
// (hero, position) insertion and keeps the one with the largest reward gain.
// Only inserts if the gain is strictly positive (i.e. the mill actually pays
// off given travel cost and time window).
//
// Parallelised: each missing mill's best insertion is found independently by a
// rayon worker; the results are then applied sequentially in day-open order so
// earlier insertions don't invalidate later ones (route lengths stay consistent).

fn insert_missing_mills(
    mut sol: Solution,
    d: &ProblemData,
    pb: &ProgressBar,
) -> Solution {
    if sol.fitness.is_none() { recompute(&mut sol, d); }

    let visited: HashSet<u32> = sol.routes.values().flatten().copied().collect();
    let mut missing: Vec<u32> = d.mill_ids.iter()
        .copied()
        .filter(|m| !visited.contains(m))
        .collect();

    // Prioritise earlier-opening mills — they have tighter time constraints and
    // constrain routing options most; inserting them first leaves the widest
    // flexibility for later-day mills.
    missing.sort_by_key(|&m| d.mills[&m].day_open);

    pb.set_length(missing.len() as u64);
    pb.set_message(format!("insert_missing  {} mills to try", missing.len()));
    pb.reset();

    // For each missing mill: find (hero, pos, gain) in parallel.
    // We snapshot routes & rewards before the loop so parallel workers all
    // read the same consistent state.  Insertions are then applied one-by-one
    // on the main thread so the cache stays coherent.
    let snapshot_routes:  Vec<(u32, Vec<u32>)> = sol.routes.iter()
        .map(|(&h, r)| (h, r.clone()))
        .collect();
    let snapshot_rewards: HashMap<u32, u32>    = sol.hero_rewards.clone();

    // For each missing mill find the best (hero_id, insert_position, gain)
    let candidates: Vec<Option<(u32, usize, u32)>> = missing
        .par_iter()
        .map(|&mill_id| {
            let mut best_gain: u32  = 0;
            let mut best_hero: u32  = 0;
            let mut best_pos: usize = 0;

            for &(hid, ref route) in &snapshot_routes {
                if hid > sol.max_id { continue; }
                let old_rew = *snapshot_rewards.get(&hid).unwrap_or(&0);

                for pos in 0..=route.len() {
                    let mut trial = route.clone();
                    trial.insert(pos, mill_id);
                    let new_rew = simulate_hero(hid, &trial, d);
                    if new_rew > old_rew + best_gain {
                        best_gain = new_rew - old_rew;
                        best_hero = hid;
                        best_pos  = pos;
                    }
                }
            }

            pb.inc(1);
            if best_hero > 0 { Some((best_hero, best_pos, best_gain)) } else { None }
        })
        .collect();

    // Apply insertions sequentially (route lengths shift after each insert,
    // so positions from the parallel scan are stale — recompute via partition_point).
    for (mill_id, maybe) in missing.iter().zip(candidates.iter()) {
        if let Some(&(hero, _pos, _gain)) = maybe.as_ref() {
            // Read pos first (immutable borrow), then drop before mut borrow.
            let pos = {
                let route = sol.routes.get(&hero).unwrap();
                route.partition_point(|&m| {
                    d.mills[&m].day_open <= d.mills[mill_id].day_open
                })
            }; // immutable borrow ends here
            sol.routes.get_mut(&hero).unwrap().insert(pos, *mill_id);
            refresh_hero(&mut sol, hero, d);
        }
    }

    pb.finish_with_message(format!(
        "✓ insert_missing  inserted {}/{}",
        candidates.iter().filter(|c| c.is_some()).count(),
        missing.len()
    ));
    sol
}


// ─── Trim unprofitable heroes ────────────────────────────────────────────────
//
// A hero whose total collected reward is less than their hire cost (HERO_COST)
// is a net drag on fitness.  We remove the worst-offending hero, redistribute
// their mills to the remaining heroes (inserting in day-open order to preserve
// temporal feasibility), and recompute fitness.  Repeat until all remaining
// heroes are profitable or only one hero is left.
//
// "Redistribute" uses the same day-aware insert used by mutation: mills are
// placed at the partition_point that keeps each route sorted by day_open.
// A small random jitter is applied to the destination hero so the same hero
// doesn't absorb all orphaned mills.

fn trim_unprofitable_heroes(mut sol: Solution, d: &ProblemData, rng: &mut impl Rng) -> Solution {
    loop {
        if sol.max_id <= 1 { break; }

        // Find the hero with the lowest reward-to-cost ratio
        let worst = (1..=sol.max_id).min_by_key(|&h| {
            sol.hero_rewards.get(&h).copied().unwrap_or(0)
        });

        let worst_hid = match worst { Some(h) => h, None => break };
        let worst_rew = sol.hero_rewards.get(&worst_hid).copied().unwrap_or(0);

        // Only remove if the hero genuinely costs more than they earn
        if worst_rew as i64 >= HERO_COST as i64 { break; }

        // Remove hero and collect their mills
        let orphans = sol.routes.remove(&worst_hid).unwrap_or_default();
        sol.hero_rewards.remove(&worst_hid);

        // Renumber: shift all heroes above worst_hid down by one so the
        // id space stays contiguous (required by the rest of the algorithm).
        for h in worst_hid..sol.max_id {
            let route   = sol.routes.remove(&(h + 1)).unwrap_or_default();
            let reward  = sol.hero_rewards.remove(&(h + 1)).unwrap_or(0);
            sol.routes.insert(h, route);
            sol.hero_rewards.insert(h, reward);
        }
        sol.max_id -= 1;

        // Redistribute orphaned mills to random remaining heroes in day order
        for mill in orphans {
            let dst   = rng.gen_range(1..=sol.max_id);
            let route = sol.routes.get_mut(&dst).unwrap();
            let pos   = route.partition_point(|&m| {
                d.mills[&m].day_open <= d.mills[&mill].day_open
            });
            route.insert(pos, mill);
        }

        // Full recompute because multiple routes changed
        recompute(&mut sol, d);
    }
    sol
}

// ─── Population restart ───────────────────────────────────────────────────────
//
// When the GA has stagnated for `threshold` generations, the bottom half of the
// population is replaced with fresh greedy individuals.  The top half (elites)
// is kept so all accumulated learning is preserved.  After restart, stagnation
// counter is reset and the new individuals go through or-opt before joining the
// next generation.

fn restart_population(
    mut pop: Vec<Solution>,
    d: &ProblemData,
    ls_iter: u32,
    pb: &ProgressBar,
) -> Vec<Solution> {
    let keep = pop.len() / 2;
    pop.sort_by_key(|s| Reverse(s.fitness.unwrap_or(i64::MIN)));
    pop.truncate(keep);

    let fresh_count = pop.len(); // refill back to original size
    pb.reset();
    pb.set_length(fresh_count as u64);
    pb.set_message(format!("Population restart — generating {} new individuals", fresh_count));

    let fresh: Vec<Solution> = (0..fresh_count)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng();
            let sol = greedy_solution(d, &mut rng);
            let sol = or_opt(sol, ls_iter, d);
            pb.inc(1);
            sol
        })
        .collect();

    pop.extend(fresh);
    pb.finish_with_message(format!("✓ Restart complete — pop size {}", pop.len()));
    pop
}

// ─── CSV output ───────────────────────────────────────────────────────────────

fn save_csv(sol: &Solution, path: &str) {
    let mut f = File::create(path).expect("Cannot create output file");
    writeln!(f, "hero_id,object_id").unwrap();
    for hid in 1..=sol.max_id {
        if let Some(r) = sol.routes.get(&hid) {
            for &obj in r { writeln!(f, "{},{}", hid, obj).unwrap(); }
        }
    }
}

// ─── Genetic algorithm ────────────────────────────────────────────────────────
//
// Key design decisions:
//
// • 30% greedy + 70% random initial population → good diversity from start
// • Two crossover strategies (60% day-aware / 40% hero-route) alternating
// • day-order repair after every crossover
// • Five mutation operators, applied once (+ 30% chance of a second pass)
// • Adaptive mutation: rate scales up linearly with stagnation depth
// • Or-opt local search (3 phases: intra, inter-move, inter-swap)
// • Per-hero reward cache: local-search moves re-evaluate in O(route_len)
// • Elite preservation: top `elite_ratio` of population survives unchanged
// • Child generation is fully parallel via rayon; selection stays on main thread
// • AtomicU64 counter drives per-generation progress bar with no lock contention

fn genetic_algorithm(
    pop_size:          usize,
    generations:       usize,
    elite_ratio:       f32,
    base_mut_rate:     f64,
    ls_iter:           u32,   // or-opt iterations per individual
    db_stag_threshold: usize, // stagnation gens before double-bridge kicks in
    im_stag_threshold: usize, // stagnation gens before insert-missing kicks in
    restart_threshold: usize, // stagnation gens before population restart
    trim_threshold:    usize, // stagnation gens before hero trimming
    best_file:         &str,
    d:                 &ProblemData,
    mp:                &MultiProgress,
) -> Solution {
    // ── Init ─────────────────────────────────────────────────────────────
    let mut pop        = init_population(pop_size, d, mp);
    let elite_sz       = ((pop_size as f32 * elite_ratio) as usize).max(1);
    let children_need  = pop_size - elite_sz;

    pop.sort_by_key(|s| Reverse(s.fitness.unwrap_or(i64::MIN)));
    let mut best = pop[0].clone();
    save_csv(&best, best_file);

    // ── Progress bars ─────────────────────────────────────────────────────
    let gen_pb = mp.add(ProgressBar::new(generations as u64));
    gen_pb.set_style(gen_bar_style());
    gen_pb.set_message(best.fitness.unwrap_or(0).to_string());

    let child_pb = mp.add(ProgressBar::new(children_need as u64));
    child_pb.set_style(bar_style());

    let counter = Arc::new(AtomicU64::new(0));

    // Progress bars for the three post-processing phases
    let post_pb = mp.add(ProgressBar::new(1));
    post_pb.set_style(post_bar_style());
    post_pb.set_message("waiting…");

    // ── Adaptive mutation + phase timing state ───────────────────────────────
    let mut stagnation       = 0usize;
    let mut last_best        = best.fitness.unwrap_or(i64::MIN);
    let mut rng              = rand::thread_rng();
    let run_start            = Instant::now();

    // Profiling accumulators (nanoseconds)
    let mut t_selection:  u128 = 0;
    let mut t_crossover:  u128 = 0;
    let mut t_or_opt:     u128 = 0;
    let mut t_db:         u128 = 0;
    let mut t_im:         u128 = 0;
    let mut t_restart:    u128 = 0;
    let mut db_triggers:   usize = 0;
    let mut im_triggers:   usize = 0;
    let mut restart_count: usize = 0;
    let mut t_trim:        u128  = 0;
    let mut trim_count:    usize = 0;

    for generation in 0..generations {
        // Mutation rate grows with stagnation (caps at 0.95)
        let mut_rate = (base_mut_rate * (1.0 + stagnation as f64 * 0.1)).min(0.95);

        // ── Reset child bar ────────────────────────────────────────────────
        child_pb.reset();
        child_pb.set_length(children_need as u64);
        child_pb.set_message(format!(
            "Gen {:>4}/{}  mut={:.2}  stag={}", generation + 1, generations, mut_rate, stagnation
        ));
        counter.store(0, Ordering::Relaxed);

        // ── Selection on main thread (needs &mut rng) ─────────────────────
        let t0 = Instant::now();
        let pairs: Vec<(Solution, Solution)> = (0..children_need)
            .map(|_| {
                let p1 = tournament(&pop, 5, &mut rng).clone();
                let p2 = tournament(&pop, 5, &mut rng).clone();
                (p1, p2)
            })
            .collect();
        t_selection += t0.elapsed().as_nanos();

        // ── Parallel child generation ──────────────────────────────────────
        let cnt_ref = Arc::clone(&counter);
        let cpb_ref = &child_pb;

        // Per-thread timing accumulators — folded into outer counters after collect
        // AtomicU64 is stable and holds up to ~584 years of nanoseconds.
        let par_t_xover  = Arc::new(AtomicU64::new(0));
        let par_t_oropt  = Arc::new(AtomicU64::new(0));
        let xover_ref    = Arc::clone(&par_t_xover);
        let oropt_ref    = Arc::clone(&par_t_oropt);

        let mut children: Vec<Solution> = pairs
            .into_par_iter()
            .map(|(p1, p2)| {
                let mut rng = rand::thread_rng();

                let tc = Instant::now();
                let mut child = if rng.gen_bool(0.6) {
                    day_crossover(&p1, &p2, d, &mut rng)
                } else {
                    hero_crossover(&p1, &p2, d, &mut rng)
                };
                child = repair_order(child, d);
                if rng.gen_bool(mut_rate)       { child = mutate(child, d); }
                if rng.gen_bool(mut_rate * 0.3) { child = mutate(child, d); }
                xover_ref.fetch_add(tc.elapsed().as_nanos() as u64, Ordering::Relaxed);

                let to = Instant::now();
                child = or_opt(child, ls_iter, d);
                oropt_ref.fetch_add(to.elapsed().as_nanos() as u64, Ordering::Relaxed);

                let prev = cnt_ref.fetch_add(1, Ordering::Relaxed);
                cpb_ref.set_position(prev + 1);
                child
            })
            .collect();
        t_crossover += par_t_xover.load(Ordering::Relaxed) as u128;
        t_or_opt    += par_t_oropt.load(Ordering::Relaxed) as u128;

        // ── Assemble next generation ───────────────────────────────────────
        pop.sort_by_key(|s| Reverse(s.fitness.unwrap_or(i64::MIN)));
        let mut next: Vec<Solution> = pop[..elite_sz].to_vec(); // elites unchanged
        next.append(&mut children);
        pop = next;

        // ── Double-bridge perturbation (applied to every child when stagnating) ─
        if stagnation >= db_stag_threshold {
            let tdb = Instant::now();
            db_triggers += 1;
            let cnt2 = Arc::clone(&counter);
            let pb2  = &post_pb;
            pb2.reset();
            pb2.set_length(pop.len() as u64);
            pb2.set_message(format!("double-bridge  stag={stagnation}"));

            pop = pop
                .into_par_iter()
                .map(|sol| {
                    let mut rng = rand::thread_rng();
                    let sol = double_bridge(sol, d, &mut rng);
                    let sol = or_opt(sol, ls_iter, d);
                    cnt2.fetch_add(1, Ordering::Relaxed);
                    pb2.set_position(cnt2.load(Ordering::Relaxed));
                    sol
                })
                .collect();
            pb2.finish_with_message(format!("✓ double-bridge #{db_triggers} done"));
            t_db += tdb.elapsed().as_nanos();
        }

        // ── Insert-missing-mills pass (triggered on its own threshold) ─────
        if stagnation >= im_stag_threshold {
            let tim = Instant::now();
            im_triggers += 1;
            // Apply to the current best individual only (most worth polishing)
            pop.sort_by_key(|s| Reverse(s.fitness.unwrap_or(i64::MIN)));
            let top = pop.remove(0);
            let top = insert_missing_mills(top, d, &post_pb);
            let top = or_opt(top, ls_iter * 2, d); // extra polish after insertion
            pop.insert(0, top);
            t_im += tim.elapsed().as_nanos();
        }

        // ── Trim unprofitable heroes ──────────────────────────────────────
        // A hero who earns less gold than their hire cost (2500) is a net
        // negative.  Removing them and redistributing their mills to the
        // remaining heroes often both raises fitness and frees up a hero slot
        // for the mutation operator to try a more useful configuration.
        if stagnation >= trim_threshold {
            let tt = Instant::now();
            trim_count += 1;
            post_pb.reset();
            post_pb.set_message(format!("trim-heroes  stag={stagnation}"));
            pop = pop
                .into_par_iter()
                .map(|sol| {
                    let mut rng = rand::thread_rng();
                    trim_unprofitable_heroes(sol, d, &mut rng)
                })
                .collect();
            post_pb.finish_with_message(format!("✓ trim-heroes #{trim_count} done"));
            t_trim += tt.elapsed().as_nanos();
        }

        // ── Population restart (hardest reset — triggered last) ───────────
        if stagnation >= restart_threshold {
            let tr = Instant::now();
            restart_count += 1;
            pop = restart_population(pop, d, ls_iter, &post_pb);
            stagnation = 0; // explicit reset so we don't immediately retrigger
            t_restart += tr.elapsed().as_nanos();
        }

        // ── Track global best ──────────────────────────────────────────────
        if let Some(cur) = pop.iter().max_by_key(|s| s.fitness.unwrap_or(i64::MIN)) {
            if cur.fitness > best.fitness {
                best = cur.clone();
                save_csv(&best, best_file);
            }
        }
        let cur_best = best.fitness.unwrap_or(i64::MIN);
        if cur_best > last_best { last_best = cur_best; stagnation = 0; }
        else                    { stagnation += 1; }

        // ── Update generation bar ──────────────────────────────────────────
        let e = run_start.elapsed().as_secs();
        gen_pb.set_message(format!(
            "{}  [{:02}:{:02}:{:02}]  heroes={}",
            cur_best, e/3600, (e%3600)/60, e%60, best.max_id
        ));
        gen_pb.inc(1);
    }

    child_pb.finish_and_clear();
    post_pb.finish_and_clear();
    gen_pb.finish_with_message(format!(
        "✓ Done  fitness={}  heroes={}",
        best.fitness.unwrap_or(0), best.max_id
    ));

    // ── Phase timing report ──────────────────────────────────────────────
    let total_ns = (t_selection + t_crossover + t_or_opt + t_db + t_im + t_trim + t_restart).max(1);
    eprintln!("\n─── Phase timing ────────────────────────────────");
    eprintln!("  selection   {:>8.2}s  ({:>5.1}%)",
        t_selection  as f64 / 1e9, t_selection  as f64 / total_ns as f64 * 100.0);
    eprintln!("  crossover   {:>8.2}s  ({:>5.1}%)",
        t_crossover  as f64 / 1e9, t_crossover  as f64 / total_ns as f64 * 100.0);
    eprintln!("  or-opt      {:>8.2}s  ({:>5.1}%)",
        t_or_opt     as f64 / 1e9, t_or_opt     as f64 / total_ns as f64 * 100.0);
    eprintln!("  dbl-bridge  {:>8.2}s  ({:>5.1}%)  triggered {} times",
        t_db         as f64 / 1e9, t_db         as f64 / total_ns as f64 * 100.0, db_triggers);
    eprintln!("  ins-missing {:>8.2}s  ({:>5.1}%)  triggered {} times",
        t_im         as f64 / 1e9, t_im         as f64 / total_ns as f64 * 100.0, im_triggers);
    eprintln!("  trim-heroes {:>8.2}s  ({:>5.1}%)  triggered {} times",
        t_trim       as f64 / 1e9, t_trim       as f64 / total_ns as f64 * 100.0, trim_count);
    eprintln!("  restart     {:>8.2}s  ({:>5.1}%)  triggered {} times",
        t_restart    as f64 / 1e9, t_restart    as f64 / total_ns as f64 * 100.0, restart_count);
    eprintln!("─────────────────────────────────────────────────\n");

    best
}

// ─── Entry point ──────────────────────────────────────────────────────────────

fn main() {
    let mp   = MultiProgress::new();
    let data = load_data(&mp);

    // Uncomment to pin thread count (default = logical CPU count):
    // rayon::ThreadPoolBuilder::new().num_threads(8).build_global().unwrap();

    let best = genetic_algorithm(
        /* pop_size          */ 200,
        /* generations       */ 500,
        /* elite_ratio       */ 0.10,  // keep top 10%
        /* base_mut_rate     */ 0.40,
        /* ls_iter           */ 8,     // or-opt passes per individual
        /* db_stag_threshold */ 15,    // double-bridge after 15 stagnant gens
        /* im_stag_threshold */ 20,    // insert-missing after 20 stagnant gens
        /* restart_threshold */ 60,    // population restart after 60 stagnant gens
        /* trim_threshold    */ 25,    // trim unprofitable heroes after 25 stagnant gens
        /* best_file         */ "best.csv",
        &data,
        &mp,
    );

    save_csv(&best, "final.csv");
    drop(mp);

    println!("\nFinal fitness  : {}", best.fitness.unwrap_or(0));
    println!("Heroes used    : {}", best.max_id);
    println!("Gold collected : {}",
        best.fitness.unwrap_or(0) + best.max_id as i64 * HERO_COST as i64);
    println!("Saved to       : final.csv");
}