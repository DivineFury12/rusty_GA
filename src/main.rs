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

#[derive(Deserialize)] struct ObjectRecord    { object_id: u32, day_open: u32, reward: u32 }
#[derive(Deserialize)] struct HeroRecord      { hero_id: u32, move_points: u32 }
#[derive(Deserialize)] struct DistStartRecord { object_id: u32, dist_start: u32 }

// ─── Domain structs ───────────────────────────────────────────────────────────

#[derive(Clone)] struct Mill { day_open: u32, reward: u32 }
#[derive(Clone)] struct Hero { move_points: u32 }

// ─── Solution ─────────────────────────────────────────────────────────────────
//
// KEY DESIGN: each Route owns its hero_id so the MP lookup in simulate_hero is
// always stable.  No renumbering is ever performed — removing a route is a plain
// Vec::swap_remove.  Hero identity is never confused with slot index.
//
// fitness = Σ route.reward  −  routes.len() × HERO_COST

#[derive(Clone)]
struct Route {
    hero_id: u32,      // actual hero from d.heroes (determines move_points)
    mills:   Vec<u32>, // ordered mill visits (sorted by day_open)
    reward:  u32,      // cached simulate_hero result
}

#[derive(Clone)]
struct Solution {
    routes:  Vec<Route>,
    fitness: i64,
}

// ─── Problem data ─────────────────────────────────────────────────────────────

struct ProblemData {
    heroes:       HashMap<u32, Hero>,
    mills:        HashMap<u32, Mill>,
    dist:         Array2<u32>,
    mills_by_day: [Vec<u32>; 8],
    mill_ids:     Vec<u32>,
    /// Hero ids sorted by move_points descending — we prefer high-MP heroes.
    hero_ids_by_mp: Vec<u32>,
    n_mills:      usize,
}

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

    let mut mills_by_day: [Vec<u32>; 8] = Default::default();
    for &id in &mill_ids { mills_by_day[mills[&id].day_open as usize].push(id); }

    // Sort hero ids by MP descending so greedy/init always picks the best heroes first
    let mut hero_ids_by_mp: Vec<u32> = heroes.keys().copied().collect();
    hero_ids_by_mp.sort_by(|&a, &b| {
        heroes[&b].move_points.cmp(&heroes[&a].move_points).then(a.cmp(&b))
    });

    sp.finish_with_message(format!("✓ Loaded  {} mills  {} heroes", n_mills, heroes.len()));
    ProblemData { heroes, mills, dist, mills_by_day, mill_ids, hero_ids_by_mp, n_mills }
}

// ─── Simulation ───────────────────────────────────────────────────────────────

#[inline]
fn simulate_hero(hero_id: u32, route: &[u32], d: &ProblemData) -> u32 {
    if route.is_empty() { return 0; }
    let mp_full = d.heroes[&hero_id].move_points;
    let mut day = 1u32;
    let mut rem = mp_full;
    let mut pos = 0u32;
    let mut total = 0u32;

    for &mid in route {
        let mut dist = d.dist[[pos as usize, mid as usize]];
        loop {
            if rem >= dist { rem -= dist; break; }
            dist -= rem;
            day  += 1;
            if day > MAX_DAY { return total; }
            rem = mp_full;
        }
        let mill = &d.mills[&mid];
        if day < mill.day_open { day = mill.day_open; rem = mp_full; }
        if day > MAX_DAY { return total; }
        if rem == 0 { pos = mid; continue; }
        rem = rem.saturating_sub(VISIT_COST);
        if day == mill.day_open { total += mill.reward; }
        pos = mid;
    }
    total
}

// ─── Solution helpers ─────────────────────────────────────────────────────────

fn make_solution(routes: Vec<Route>) -> Solution {
    let gross: i64 = routes.iter().map(|r| r.reward as i64).sum();
    let cost:  i64 = routes.len() as i64 * HERO_COST as i64;
    Solution { fitness: gross - cost, routes }
}

/// Recompute all route rewards and fitness from scratch.
fn recompute(sol: &mut Solution, d: &ProblemData) {
    let mut gross = 0i64;
    for r in &mut sol.routes {
        r.reward = simulate_hero(r.hero_id, &r.mills, d);
        gross += r.reward as i64;
    }
    sol.fitness = gross - sol.routes.len() as i64 * HERO_COST as i64;
}

/// Re-simulate one route in-place and update fitness delta — O(route_len).
#[inline]
fn refresh_route(sol: &mut Solution, idx: usize, d: &ProblemData) {
    let old = sol.routes[idx].reward;
    let new = simulate_hero(sol.routes[idx].hero_id, &sol.routes[idx].mills, d);
    sol.routes[idx].reward = new;
    sol.fitness += new as i64 - old as i64;
}

/// All mills currently assigned in a solution.
fn visited_set(sol: &Solution) -> HashSet<u32> {
    sol.routes.iter().flat_map(|r| r.mills.iter().copied()).collect()
}

/// Heroes in use.
fn used_heroes(sol: &Solution) -> HashSet<u32> {
    sol.routes.iter().map(|r| r.hero_id).collect()
}

// ─── Greedy initialisation ────────────────────────────────────────────────────
//
// Picks `n_heroes` heroes (preferring high-MP) and builds routes greedily:
// nearest unassigned mill by  dist + wait_days × move_points.

fn greedy_solution(n_heroes: usize, d: &ProblemData, rng: &mut impl Rng) -> Solution {
    // Choose n_heroes from the top of hero_ids_by_mp with a small random shuffle
    // among heroes of equal MP so we don't always pick the exact same set.
    let mut candidates = d.hero_ids_by_mp.clone();
    // Shuffle within equal-MP tiers to vary the hero selection
    candidates.chunks_mut(1).for_each(|_| {}); // no-op; real shuffle below
    candidates.shuffle(rng);
    // But re-sort by MP (stable) so high-MP heroes are still preferred
    candidates.sort_by_key(|&h| std::cmp::Reverse(d.heroes[&h].move_points));
    let chosen: Vec<u32> = candidates.into_iter().take(n_heroes).collect();

    let mut done: HashSet<u32> = HashSet::new();
    let mut routes: Vec<Route> = Vec::with_capacity(n_heroes);

    for &hero_id in &chosen {
        let mp = d.heroes[&hero_id].move_points;
        let mut pos = 0u32;
        let mut day = 1u32;
        let mut rem = mp;
        let mut mills: Vec<u32> = Vec::new();

        loop {
            let best = d.mill_ids.iter()
                .filter(|&&id| !done.contains(&id))
                .filter_map(|&id| {
                    let mill      = &d.mills[&id];
                    let dist      = d.dist[[pos as usize, id as usize]];
                    let days_move = if dist <= rem { 0 } else { 1 + (dist - rem + mp - 1) / mp };
                    let arrive    = day + days_move;
                    let visit_day = arrive.max(mill.day_open);
                    if visit_day > MAX_DAY { return None; }
                    let wait = mill.day_open.saturating_sub(arrive);
                    Some((dist + wait * mp, id, visit_day))
                })
                .min_by_key(|&(score, _, _)| score);

            let (_, mid, visit_day) = match best { None => break, Some(v) => v };
            done.insert(mid);
            mills.push(mid);

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

        let reward = simulate_hero(hero_id, &mills, d);
        routes.push(Route { hero_id, mills, reward });
    }

    make_solution(routes)
}

// ─── Random initialisation ────────────────────────────────────────────────────

fn random_solution(n_heroes: usize, d: &ProblemData, rng: &mut impl Rng) -> Solution {
    let mut hids = d.hero_ids_by_mp.clone();
    hids.shuffle(rng);
    let chosen: Vec<u32> = hids.into_iter().take(n_heroes).collect();

    let mut all_mills = d.mill_ids.clone();
    all_mills.shuffle(rng);
    let take = rng.gen_range(all_mills.len() * 6 / 10..=all_mills.len());

    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); n_heroes];
    for &m in &all_mills[..take] {
        buckets[rng.gen_range(0..n_heroes)].push(m);
    }

    let routes: Vec<Route> = chosen.into_iter().zip(buckets).map(|(hero_id, mut mills)| {
        mills.sort_by_key(|&m| d.mills[&m].day_open);
        let reward = simulate_hero(hero_id, &mills, d);
        Route { hero_id, mills, reward }
    }).collect();

    make_solution(routes)
}

// ─── Population init ──────────────────────────────────────────────────────────

fn init_population(
    pop_size:    usize,
    d:           &ProblemData,
    mp:          &MultiProgress,
    checkpoints: &[&str],
) -> Vec<Solution> {
    let pb = mp.add(ProgressBar::new(pop_size as u64));
    pb.set_style(bar_style());
    pb.set_message("Initialising population");

    let greedy_count = (pop_size * 40 / 100).max(1);

    let mut pop: Vec<Solution> = (0..pop_size)
        .into_par_iter()
        .map(|i| {
            let mut rng = rand::thread_rng();
            // Vary hero count around the data-proven sweet spot (20-24)
            let n = rng.gen_range(20usize..=24);
            let sol = if i < greedy_count {
                greedy_solution(n, d, &mut rng)
            } else {
                let n2 = rng.gen_range(19usize..=26);
                random_solution(n2, d, &mut rng)
            };
            pb.inc(1);
            sol
        })
        .collect();

    // Inject checkpoints
    let mut ckpt_count = 0usize;
    for &path in checkpoints {
        if let Some(ckpt) = load_checkpoint(path, d) {
            // Replace worst individual
            if let Some(i) = pop.iter().enumerate()
                .min_by_key(|(_, s)| s.fitness).map(|(i, _)| i)
            {
                pop[i] = ckpt;
                ckpt_count += 1;
            }
        }
    }

    pb.finish_with_message(format!(
        "✓ Pop {}  ({} greedy + {} random + {} checkpoints)",
        pop.len(), greedy_count, pop_size - greedy_count, ckpt_count
    ));
    pop
}

// ─── Checkpoint loading ───────────────────────────────────────────────────────
//
// Reads a hero_id,object_id CSV and reconstructs a Solution.
// Each row's hero_id is used directly as the route's hero_id so the correct
// move_points is always used for simulation.

fn load_checkpoint(path: &str, d: &ProblemData) -> Option<Solution> {
    let file = match File::open(path) {
        Ok(f)  => f,
        Err(_) => { eprintln!("  [ckpt] {path} not found, skipping"); return None; }
    };

    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);
    let mut map: HashMap<u32, Vec<u32>> = HashMap::new();

    for result in rdr.deserialize::<(u32, u32)>() {
        let (hero_id, object_id) = match result {
            Ok(r) => r,
            Err(e) => { eprintln!("  [ckpt] parse error: {e}"); return None; }
        };
        if d.mills.contains_key(&object_id) && d.heroes.contains_key(&hero_id) {
            map.entry(hero_id).or_default().push(object_id);
        }
    }

    if map.is_empty() { return None; }

    let routes: Vec<Route> = map.into_iter().map(|(hero_id, mut mills)| {
        mills.sort_by_key(|&m| d.mills[&m].day_open);
        let reward = simulate_hero(hero_id, &mills, d);
        Route { hero_id, mills, reward }
    }).collect();

    let sol = make_solution(routes);
    eprintln!("  [ckpt] {path}  fitness={}  heroes={}  mills={}",
        sol.fitness, sol.routes.len(),
        sol.routes.iter().map(|r| r.mills.len()).sum::<usize>());
    Some(sol)
}

// ─── Tournament selection ─────────────────────────────────────────────────────

fn tournament<'a>(pop: &'a [Solution], k: usize, rng: &mut impl Rng) -> &'a Solution {
    pop.choose_multiple(rng, k).max_by_key(|s| s.fitness).unwrap()
}

// ─── Crossover ────────────────────────────────────────────────────────────────
//
// Day-aware crossover: for each day 1..=7 flip a coin to pick donor parent.
// Inherit all mills of that day from that parent, preserving the donor's
// hero assignment.  Deduplication ensures no mill appears twice.
//
// The resulting routes are re-evaluated with the correct hero's MP.

fn crossover(p1: &Solution, p2: &Solution, d: &ProblemData, rng: &mut impl Rng) -> Solution {
    let use_day_aware = rng.gen_bool(0.6);

    if use_day_aware {
        // Collect (hero_id, mill) pairs from chosen donor per day
        let mut assignment: HashMap<u32, Vec<u32>> = HashMap::new(); // hero_id → mills
        let mut seen = HashSet::<u32>::new();

        for day in 1..=MAX_DAY {
            let donor = if rng.gen_bool(0.5) { p1 } else { p2 };
            for r in &donor.routes {
                for &m in &r.mills {
                    if d.mills[&m].day_open == day && seen.insert(m) {
                        assignment.entry(r.hero_id).or_default().push(m);
                    }
                }
            }
        }

        // Randomly drop or keep some heroes to vary count around sweet spot
        let all_hids: Vec<u32> = assignment.keys().copied().collect();
        let routes: Vec<Route> = all_hids.into_iter().filter_map(|hero_id| {
            let mills = assignment.remove(&hero_id).unwrap();
            if mills.is_empty() { return None; }
            let mut sorted = mills;
            sorted.sort_by_key(|&m| d.mills[&m].day_open);
            let reward = simulate_hero(hero_id, &sorted, d);
            Some(Route { hero_id, mills: sorted, reward })
        }).collect();

        if routes.is_empty() {
            // Fallback: clone a parent
            return if rng.gen_bool(0.5) { p1.clone() } else { p2.clone() };
        }
        make_solution(routes)
    } else {
        // Hero-route crossover: for each hero slot take route from p1 or p2
        let mut seen = HashSet::<u32>::new();
        let n = p1.routes.len().max(p2.routes.len());
        let mut routes: Vec<Route> = Vec::new();

        for i in 0..n {
            let src = match (i < p1.routes.len(), i < p2.routes.len()) {
                (true,  true)  => if rng.gen_bool(0.5) { &p1.routes[i] } else { &p2.routes[i] },
                (true,  false) => &p1.routes[i],
                (false, true)  => &p2.routes[i],
                _              => break,
            };
            let mills: Vec<u32> = src.mills.iter().copied().filter(|m| seen.insert(*m)).collect();
            if mills.is_empty() { continue; }
            let reward = simulate_hero(src.hero_id, &mills, d);
            routes.push(Route { hero_id: src.hero_id, mills, reward });
        }

        if routes.is_empty() { return p1.clone(); }
        make_solution(routes)
    }
}

// ─── Mutation ─────────────────────────────────────────────────────────────────
//
// Six operators chosen uniformly:
//   0  swap      — swap two mills within one route
//   1  relocate  — move a mill to a different route (or same route, new pos)
//   2  add       — insert an unvisited mill into a route
//   3  delete    — remove a random mill from a route
//   4  hire      — add a new route for an unused hero
//   5  merge     — absorb shortest route into others via best-fit, drop hero

fn mutate(mut sol: Solution, d: &ProblemData, rng: &mut impl Rng) -> Solution {
    let n = sol.routes.len();
    if n == 0 { return sol; }

    match rng.gen_range(0u8..6) {
        // ── 0: swap two mills within one route ───────────────────────────
        0 => {
            let i = rng.gen_range(0..n);
            if sol.routes[i].mills.len() >= 2 {
                let len = sol.routes[i].mills.len();
                let a = rng.gen_range(0..len);
                let mut b = rng.gen_range(0..len);
                while b == a { b = rng.gen_range(0..len); }
                sol.routes[i].mills.swap(a, b);
                refresh_route(&mut sol, i, d);
            }
        }
        // ── 1: relocate a mill to another route ──────────────────────────
        1 => {
            let src = rng.gen_range(0..n);
            if !sol.routes[src].mills.is_empty() {
                let idx = rng.gen_range(0..sol.routes[src].mills.len());
                let mill = sol.routes[src].mills.remove(idx);
                let dst  = rng.gen_range(0..n);
                let pos  = sol.routes[dst].mills.partition_point(
                    |&m| d.mills[&m].day_open <= d.mills[&mill].day_open
                );
                sol.routes[dst].mills.insert(pos, mill);
                refresh_route(&mut sol, src, d);
                if dst != src { refresh_route(&mut sol, dst, d); }
            }
        }
        // ── 2: add an unvisited mill ──────────────────────────────────────
        2 => {
            let visited = visited_set(&sol);
            let avail: Vec<u32> = d.mill_ids.iter().copied()
                .filter(|m| !visited.contains(m)).collect();
            if let Some(&mill) = avail.choose(rng) {
                let i   = rng.gen_range(0..n);
                let pos = sol.routes[i].mills.partition_point(
                    |&m| d.mills[&m].day_open <= d.mills[&mill].day_open
                );
                sol.routes[i].mills.insert(pos, mill);
                refresh_route(&mut sol, i, d);
            }
        }
        // ── 3: delete a random mill ───────────────────────────────────────
        3 => {
            let i = rng.gen_range(0..n);
            if !sol.routes[i].mills.is_empty() {
                let idx = rng.gen_range(0..sol.routes[i].mills.len());
                sol.routes[i].mills.remove(idx);
                refresh_route(&mut sol, i, d);
            }
        }
        // ── 4: hire a new hero (unused, high-MP preferred) ───────────────
        4 => {
            let used = used_heroes(&sol);
            let avail: Vec<u32> = d.hero_ids_by_mp.iter().copied()
                .filter(|h| !used.contains(h)).collect();
            if let Some(&hero_id) = avail.first() {
                sol.routes.push(Route { hero_id, mills: Vec::new(), reward: 0 });
                sol.fitness -= HERO_COST as i64;
            }
        }
        // ── 5: merge smallest route into others, drop hero ────────────────
        _ => {
            if n <= 1 { return sol; }
            // Find route with fewest mills
            let thin_idx = sol.routes.iter().enumerate()
                .min_by_key(|(_, r)| r.mills.len()).map(|(i, _)| i).unwrap();
            if sol.routes[thin_idx].mills.len() > 8 { return sol; }

            let orphans = sol.routes[thin_idx].mills.clone();
            let old_fit = sol.fitness;

            // Try merging via best-fit insertion
            let mut trial = sol.clone();
            // Remove and adjust fitness
            let old_rew = trial.routes[thin_idx].reward as i64;
            trial.fitness -= old_rew + HERO_COST as i64; // lose reward, save hire cost
            trial.routes.swap_remove(thin_idx);          // O(1), no renumbering

            for mill in &orphans {
                let mut best_gain = i64::MIN;
                let mut best_ri   = 0usize;
                let mut best_pos  = 0usize;
                for ri in 0..trial.routes.len() {
                    let old_rew = trial.routes[ri].reward;
                    let lo = trial.routes[ri].mills.partition_point(|&m|
                        d.mills[&m].day_open < d.mills[mill].day_open.saturating_sub(1));
                    let hi = (trial.routes[ri].mills.partition_point(|&m|
                        d.mills[&m].day_open <= d.mills[mill].day_open + 1))
                        .min(trial.routes[ri].mills.len());
                    for pos in lo..=hi {
                        let mut nr = trial.routes[ri].mills.clone();
                        nr.insert(pos, *mill);
                        let new_rew = simulate_hero(trial.routes[ri].hero_id, &nr, d);
                        let gain    = new_rew as i64 - old_rew as i64;
                        if gain > best_gain {
                            best_gain = gain; best_ri = ri; best_pos = pos;
                        }
                    }
                }
                trial.routes[best_ri].mills.insert(best_pos, *mill);
                refresh_route(&mut trial, best_ri, d);
            }

            if trial.fitness > old_fit { return trial; }
        }
    }
    sol
}

// ─── Or-opt local search ─────────────────────────────────────────────────────
//
// Three phases per iteration:
//   A. Intra-route: move a segment of 1–3 mills within the same route.
//   B. Inter-route: move a segment from one route to another.
//   C. Inter-route swap: exchange single mills between two routes.
//
// Heroes with late-day mills are scanned first (most constrained → most gain).

fn or_opt(mut sol: Solution, max_iter: u32, d: &ProblemData) -> Solution {
    // Build scan order: routes with highest max(day_open) first
    let priority_order = |sol: &Solution| -> Vec<usize> {
        let mut order: Vec<usize> = (0..sol.routes.len()).collect();
        order.sort_by_key(|&i| {
            let max_day = sol.routes[i].mills.iter()
                .map(|&m| d.mills[&m].day_open).max().unwrap_or(0);
            std::cmp::Reverse(max_day)
        });
        order
    };

    for _ in 0..max_iter {
        let mut improved = false;
        let order = priority_order(&sol);

        // ── A: Intra-route Or-opt ────────────────────────────────────────
        'intra: for &ri in &order {
            let n = sol.routes[ri].mills.len();
            if n < 2 { continue; }

            // Seg-start priority: late-day segments first
            let mut starts: Vec<usize> = (0..n).collect();
            starts.sort_by_key(|&i| std::cmp::Reverse(d.mills[&sol.routes[ri].mills[i]].day_open));

            for seg in 1..=3usize {
                if seg >= n { continue; }
                for &i in &starts {
                    if i + seg > n { continue; }
                    for j in 0..=(n - seg) {
                        if j >= i.saturating_sub(1) && j <= i + seg { continue; }
                        let mut new_mills = sol.routes[ri].mills.clone();
                        let chunk: Vec<u32> = new_mills.drain(i..i+seg).collect();
                        let ins = if j > i { (j - seg).min(new_mills.len()) } else { j };
                        for (k, &m) in chunk.iter().enumerate() { new_mills.insert(ins+k, m); }
                        let old_rew = sol.routes[ri].reward;
                        let new_rew = simulate_hero(sol.routes[ri].hero_id, &new_mills, d);
                        if new_rew > old_rew {
                            sol.routes[ri].mills  = new_mills;
                            sol.fitness          += new_rew as i64 - old_rew as i64;
                            sol.routes[ri].reward = new_rew;
                            improved = true;
                            break 'intra;
                        }
                    }
                }
            }
        }

        // ── B: Inter-route Or-opt ────────────────────────────────────────
        'inter: for &r1 in &order {
            let n1 = sol.routes[r1].mills.len();
            if n1 == 0 { continue; }
            let mut starts: Vec<usize> = (0..n1).collect();
            starts.sort_by_key(|&i| std::cmp::Reverse(d.mills[&sol.routes[r1].mills[i]].day_open));

            for r2 in 0..sol.routes.len() {
                if r2 == r1 { continue; }
                let n2 = sol.routes[r2].mills.len();

                for seg in 1..=3usize {
                    if seg > n1 { continue; }
                    for &i in &starts {
                        if i + seg > n1 { continue; }
                        for ins in 0..=n2 {
                            let mut new1 = sol.routes[r1].mills.clone();
                            let chunk: Vec<u32> = new1.drain(i..i+seg).collect();
                            let mut new2 = sol.routes[r2].mills.clone();
                            for (k, &m) in chunk.iter().enumerate() { new2.insert(ins+k, m); }

                            let or1 = sol.routes[r1].reward;
                            let or2 = sol.routes[r2].reward;
                            let nr1 = simulate_hero(sol.routes[r1].hero_id, &new1, d);
                            let nr2 = simulate_hero(sol.routes[r2].hero_id, &new2, d);

                            if nr1 + nr2 > or1 + or2 {
                                sol.routes[r1].mills  = new1;
                                sol.routes[r2].mills  = new2;
                                sol.fitness          += (nr1+nr2) as i64 - (or1+or2) as i64;
                                sol.routes[r1].reward = nr1;
                                sol.routes[r2].reward = nr2;
                                improved = true;
                                break 'inter;
                            }
                        }
                    }
                }
            }
        }

        // ── C: Inter-route swap ──────────────────────────────────────────
        'swap: for &r1 in &order {
            let n1 = sol.routes[r1].mills.len();
            if n1 == 0 { continue; }
            for r2 in (r1+1)..sol.routes.len() {
                let n2 = sol.routes[r2].mills.len();
                if n2 == 0 { continue; }
                for i in 0..n1 {
                    for j in 0..n2 {
                        let mut new1 = sol.routes[r1].mills.clone();
                        let mut new2 = sol.routes[r2].mills.clone();
                        std::mem::swap(&mut new1[i], &mut new2[j]);
                        let or1 = sol.routes[r1].reward;
                        let or2 = sol.routes[r2].reward;
                        let nr1 = simulate_hero(sol.routes[r1].hero_id, &new1, d);
                        let nr2 = simulate_hero(sol.routes[r2].hero_id, &new2, d);
                        if nr1 + nr2 > or1 + or2 {
                            sol.routes[r1].mills  = new1;
                            sol.routes[r2].mills  = new2;
                            sol.fitness          += (nr1+nr2) as i64 - (or1+or2) as i64;
                            sol.routes[r1].reward = nr1;
                            sol.routes[r2].reward = nr2;
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

// ─── Double-bridge perturbation ───────────────────────────────────────────────

fn double_bridge(mut sol: Solution, d: &ProblemData, rng: &mut impl Rng) -> Solution {
    for i in 0..sol.routes.len() {
        let n = sol.routes[i].mills.len();
        if n < 8 { continue; }
        let mut cuts: Vec<usize> = (1..n).collect();
        cuts.partial_shuffle(rng, 3);
        let mut c = [cuts[0], cuts[1], cuts[2]];
        c.sort_unstable();
        let (a, b, cc) = (c[0], c[1], c[2]);
        let route = sol.routes[i].mills.clone();
        let new_mills: Vec<u32> = route[..a].iter()
            .chain(&route[b..cc])
            .chain(&route[a..b])
            .chain(&route[cc..])
            .copied().collect();
        sol.routes[i].mills = new_mills;
        refresh_route(&mut sol, i, d);
    }
    sol
}

// ─── Insert missing mills ─────────────────────────────────────────────────────

fn insert_missing_mills(mut sol: Solution, d: &ProblemData, pb: &ProgressBar) -> Solution {
    let visited = visited_set(&sol);
    let mut missing: Vec<u32> = d.mill_ids.iter().copied()
        .filter(|m| !visited.contains(m)).collect();
    missing.sort_by_key(|&m| d.mills[&m].day_open);

    pb.set_length(missing.len() as u64);
    pb.set_message(format!("insert-missing  {} mills", missing.len()));
    pb.reset();

    // Snapshot for parallel candidate search
    let snap: Vec<(u32, Vec<u32>, u32)> = sol.routes.iter()
        .map(|r| (r.hero_id, r.mills.clone(), r.reward)).collect();

    let candidates: Vec<Option<(usize, usize)>> = missing.par_iter()
        .map(|&mill| {
            let mut best_gain = 0i64;
            let mut best_ri   = None;
            let mut best_pos  = 0usize;
            for (ri, (hero_id, mills, old_rew)) in snap.iter().enumerate() {
                for pos in 0..=mills.len() {
                    let mut nr = mills.clone();
                    nr.insert(pos, mill);
                    let new_rew = simulate_hero(*hero_id, &nr, d);
                    let gain    = new_rew as i64 - *old_rew as i64;
                    if gain > best_gain { best_gain = gain; best_ri = Some(ri); best_pos = pos; }
                }
            }
            pb.inc(1);
            best_ri.map(|ri| (ri, best_pos))
        })
        .collect();

    let mut inserted = 0usize;
    for (mill, maybe) in missing.iter().zip(candidates.iter()) {
        if let Some(&(ri, _)) = maybe.as_ref() {
            let pos = {
                let r = &sol.routes[ri];
                r.mills.partition_point(|&m| d.mills[&m].day_open <= d.mills[mill].day_open)
            };
            sol.routes[ri].mills.insert(pos, *mill);
            refresh_route(&mut sol, ri, d);
            inserted += 1;
        }
    }
    pb.finish_with_message(format!("✓ insert-missing  inserted {inserted}/{}",  missing.len()));
    sol
}

// ─── Trim unprofitable heroes ─────────────────────────────────────────────────
//
// Removes routes earning less than HERO_COST, redistributing their mills via
// best-fit insertion.  Uses swap_remove (O(1)) — no renumbering, no MP confusion.

fn trim_unprofitable(mut sol: Solution, d: &ProblemData, rng: &mut impl Rng) -> Solution {
    loop {
        if sol.routes.len() <= 1 { break; }
        let worst_idx = match sol.routes.iter().enumerate()
            .filter(|(_, r)| (r.reward as i64) < HERO_COST as i64)
            .min_by_key(|(_, r)| r.reward)
            .map(|(i, _)| i)
        {
            Some(i) => i,
            None    => break,
        };

        let orphans = sol.routes[worst_idx].mills.clone();
        let old_rew = sol.routes[worst_idx].reward as i64;
        sol.fitness -= old_rew + HERO_COST as i64; // remove reward, recover hire cost
        sol.routes.swap_remove(worst_idx);

        for mill in orphans {
            let dst = rng.gen_range(0..sol.routes.len());
            let pos = sol.routes[dst].mills.partition_point(
                |&m| d.mills[&m].day_open <= d.mills[&mill].day_open
            );
            sol.routes[dst].mills.insert(pos, mill);
            refresh_route(&mut sol, dst, d);
        }
    }
    sol
}

// ─── Aggressive merge ─────────────────────────────────────────────────────────
//
// Tries to merge the thinnest route into others using best-fit insertion.
// Accepts only if net fitness strictly improves (saves HERO_COST, must offset any loss).

fn aggressive_merge(sol: Solution, d: &ProblemData) -> Solution {
    if sol.routes.len() <= 1 { return sol; }

    let thin_idx = sol.routes.iter().enumerate()
        .min_by_key(|(_, r)| r.mills.len()).map(|(i, _)| i).unwrap();

    let old_fit = sol.fitness;
    let orphans = sol.routes[thin_idx].mills.clone();
    let old_rew = sol.routes[thin_idx].reward as i64;

    // Work on a clone so the original is untouched if we reject.
    // This is the critical fix: the old code mutated `sol` in-place then set
    // sol.fitness = old_fit on rejection, returning routes that didn't match
    // the claimed fitness value — causing fitness to drift upward each call.
    let mut trial = sol.clone();
    trial.fitness -= old_rew + HERO_COST as i64;
    trial.routes.swap_remove(thin_idx);

    for mill in &orphans {
        let mut best_gain = i64::MIN;
        let mut best_ri   = 0usize;
        let mut best_pos  = 0usize;
        for ri in 0..trial.routes.len() {
            let or = trial.routes[ri].reward;
            for pos in 0..=trial.routes[ri].mills.len() {
                let mut nr = trial.routes[ri].mills.clone();
                nr.insert(pos, *mill);
                let nr_rew = simulate_hero(trial.routes[ri].hero_id, &nr, d);
                let gain   = nr_rew as i64 - or as i64;
                if gain > best_gain { best_gain = gain; best_ri = ri; best_pos = pos; }
            }
        }
        trial.routes[best_ri].mills.insert(best_pos, *mill);
        refresh_route(&mut trial, best_ri, d);
    }

    // Accept only if strictly better; otherwise return the original untouched.
    if trial.fitness > old_fit { trial } else { sol }
}

// ─── Population restart ───────────────────────────────────────────────────────

fn restart_population(mut pop: Vec<Solution>, d: &ProblemData, ls_iter: u32, pb: &ProgressBar) -> Vec<Solution> {
    let keep = pop.len() / 2;
    pop.sort_by_key(|s| std::cmp::Reverse(s.fitness));
    pop.truncate(keep);

    let fresh_n = pop.len();
    pb.reset();
    pb.set_length(fresh_n as u64);
    pb.set_message(format!("restart — generating {fresh_n} new individuals"));

    let fresh: Vec<Solution> = (0..fresh_n)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng();
            let n   = rng.gen_range(20usize..=24);
            let sol = greedy_solution(n, d, &mut rng);
            let sol = or_opt(sol, ls_iter, d);
            pb.inc(1);
            sol
        })
        .collect();

    pop.extend(fresh);
    pb.finish_with_message(format!("✓ restart — pop {}", pop.len()));
    pop
}

// ─── CSV output ───────────────────────────────────────────────────────────────

fn save_csv(sol: &Solution, path: &str) {
    let mut f = File::create(path).expect("cannot create output");
    writeln!(f, "hero_id,object_id").unwrap();
    for r in &sol.routes {
        for &obj in &r.mills {
            writeln!(f, "{},{}", r.hero_id, obj).unwrap();
        }
    }
}

// ─── Genetic algorithm ────────────────────────────────────────────────────────

fn genetic_algorithm(
    pop_size:          usize,
    generations:       usize,
    elite_ratio:       f64,
    base_mut_rate:     f64,
    ls_iter:           u32,
    db_stag_threshold: usize,
    im_stag_threshold: usize,
    trim_threshold:    usize,
    restart_threshold: usize,
    best_file:         &str,
    checkpoints:       &[&str],
    d:                 &ProblemData,
    mp:                &MultiProgress,
) -> Solution {
    let mut pop = init_population(pop_size, d, mp, checkpoints);
    pop.sort_by_key(|s| std::cmp::Reverse(s.fitness));

    let elite_sz      = ((pop_size as f64 * elite_ratio) as usize).max(1);
    let children_need = pop_size - elite_sz;

    // Guard against best_file being overwritten with a worse solution on startup.
    // Always recompute fitness from simulate_hero so no stale value from a
    // prior buggy run can carry over through the checkpoint file.
    let mut best = {
        let init_best = pop[0].clone();
        let file_best = load_checkpoint(best_file, d); // rewards already fresh-computed inside
        match file_best {
            Some(fb) if fb.fitness > init_best.fitness => { fb }
            _ => { save_csv(&init_best, best_file); init_best }
        }
    };
    // Verify best fitness by full recompute — catches any drift in loaded solution.
    recompute(&mut best, d);

    let gen_pb   = mp.add(ProgressBar::new(generations as u64));
    gen_pb.set_style(gen_bar_style());
    gen_pb.set_message(best.fitness.to_string());

    let child_pb = mp.add(ProgressBar::new(children_need as u64));
    child_pb.set_style(bar_style());

    let post_pb  = mp.add(ProgressBar::new(1));
    post_pb.set_style(post_bar_style());

    let counter   = Arc::new(AtomicU64::new(0));
    let mut stag  = 0usize;
    let mut last  = best.fitness;
    let mut rng   = rand::thread_rng();
    let run_start = Instant::now();

    // Profiling
    let (mut t_xover, mut t_oropt, mut t_db, mut t_im, mut t_trim, mut t_restart): (u128,u128,u128,u128,u128,u128) = Default::default();
    let (mut n_db, mut n_im, mut n_trim, mut n_restart): (usize,usize,usize,usize) = Default::default();

    for generation in 0..generations {
        let mut_rate = (base_mut_rate * (1.0 + stag as f64 * 0.08)).min(0.90);

        child_pb.reset();
        child_pb.set_length(children_need as u64);
        child_pb.set_message(format!(
            "Gen {:>4}/{}  mut={:.2}  stag={}", generation+1, generations, mut_rate, stag
        ));
        counter.store(0, Ordering::Relaxed);

        let pairs: Vec<(Solution, Solution)> = (0..children_need).map(|_| {
            let p1 = tournament(&pop, 5, &mut rng).clone();
            let p2 = tournament(&pop, 5, &mut rng).clone();
            (p1, p2)
        }).collect();

        let cnt_ref  = Arc::clone(&counter);
        let cpb_ref  = &child_pb;
        let t_x_acc  = Arc::new(AtomicU64::new(0));
        let t_o_acc  = Arc::new(AtomicU64::new(0));
        let tx2 = Arc::clone(&t_x_acc);
        let to2 = Arc::clone(&t_o_acc);

        let mut children: Vec<Solution> = pairs
            .into_par_iter()
            .map(|(p1, p2)| {
                let mut rng = rand::thread_rng();

                let tc = Instant::now();
                let mut child = crossover(&p1, &p2, d, &mut rng);
                if rng.gen_bool(mut_rate) { child = mutate(child, d, &mut rng); }
                if rng.gen_bool(mut_rate * 0.3) { child = mutate(child, d, &mut rng); }
                tx2.fetch_add(tc.elapsed().as_nanos() as u64, Ordering::Relaxed);

                let to = Instant::now();
                child = or_opt(child, ls_iter, d);
                to2.fetch_add(to.elapsed().as_nanos() as u64, Ordering::Relaxed);

                cnt_ref.fetch_add(1, Ordering::Relaxed);
                cpb_ref.set_position(cnt_ref.load(Ordering::Relaxed));
                child
            })
            .collect();

        t_xover += t_x_acc.load(Ordering::Relaxed) as u128;
        t_oropt += t_o_acc.load(Ordering::Relaxed) as u128;

        pop.sort_by_key(|s| std::cmp::Reverse(s.fitness));
        let mut next = pop[..elite_sz].to_vec();
        next.append(&mut children);
        pop = next;

        // ── Double-bridge ─────────────────────────────────────────────────
        if stag >= db_stag_threshold {
            let t = Instant::now();
            n_db += 1;
            post_pb.reset();
            post_pb.set_length(pop.len() as u64);
            post_pb.set_message(format!("double-bridge  stag={stag}"));
            let cnt2 = Arc::clone(&counter);
            counter.store(0, Ordering::Relaxed);
            pop = pop.into_par_iter().map(|sol| {
                let mut rng = rand::thread_rng();
                let sol = double_bridge(sol, d, &mut rng);
                let sol = or_opt(sol, ls_iter, d);
                cnt2.fetch_add(1, Ordering::Relaxed);
                post_pb.set_position(cnt2.load(Ordering::Relaxed));
                sol
            }).collect();
            post_pb.finish_with_message(format!("✓ double-bridge #{n_db}"));
            pop.iter_mut().for_each(|s| recompute(s, d));
            t_db += t.elapsed().as_nanos();
        }

        // ── Insert-missing on best individual ─────────────────────────────
        if stag >= im_stag_threshold {
            let t = Instant::now();
            n_im += 1;
            pop.sort_by_key(|s| std::cmp::Reverse(s.fitness));
            let top = pop.remove(0);
            let top = insert_missing_mills(top, d, &post_pb);
            let top = or_opt(top, ls_iter * 2, d);
            pop.insert(0, top);
            t_im += t.elapsed().as_nanos();
        }

        // ── Trim + aggressive merge ───────────────────────────────────────
        if stag >= trim_threshold {
            let t = Instant::now();
            n_trim += 1;
            post_pb.reset();
            post_pb.set_length(pop.len() as u64);
            post_pb.set_message(format!("trim+merge  stag={stag}"));
            pop = pop.into_par_iter().map(|sol| {
                let mut rng = rand::thread_rng();
                let sol = trim_unprofitable(sol, d, &mut rng);
                let sol = aggressive_merge(sol, d);
                let sol = or_opt(sol, ls_iter, d);
                post_pb.inc(1);
                sol
            }).collect();
            post_pb.finish_with_message(format!("✓ trim+merge #{n_trim}"));
            // Safety recompute: rebuild every fitness value from scratch so any
            // residual delta-drift is eliminated before tournament selection.
            pop.iter_mut().for_each(|s| recompute(s, d));
            t_trim += t.elapsed().as_nanos();
        }

        // ── Population restart ────────────────────────────────────────────
        if stag >= restart_threshold {
            let t = Instant::now();
            n_restart += 1;
            pop = restart_population(pop, d, ls_iter, &post_pb);
            stag = 0;
            t_restart += t.elapsed().as_nanos();
        }

        // ── Track global best ─────────────────────────────────────────────
        if let Some(cur) = pop.iter().max_by_key(|s| s.fitness) {
            if cur.fitness > best.fitness {
                best = cur.clone();
                save_csv(&best, best_file);
            }
        }
        if best.fitness > last { last = best.fitness; stag = 0; }
        else                   { stag += 1; }

        let e = run_start.elapsed().as_secs();
        gen_pb.set_message(format!(
            "{}  [{:02}:{:02}:{:02}]  heroes={}  mills={}",
            best.fitness, e/3600, (e%3600)/60, e%60, best.routes.len(),
            best.routes.iter().map(|r| r.mills.len()).sum::<usize>()
        ));
        gen_pb.inc(1);
    }

    child_pb.finish_and_clear();
    post_pb.finish_and_clear();
    gen_pb.finish_with_message(format!(
        "✓ Done  fitness={}  heroes={}  mills={}",
        best.fitness, best.routes.len(),
        best.routes.iter().map(|r| r.mills.len()).sum::<usize>()
    ));

    let total = (t_xover + t_oropt + t_db + t_im + t_trim + t_restart).max(1);
    eprintln!("\n─── Phase timing ─────────────────────────────────");
    for (name, t, n) in [
        ("crossover  ", t_xover,   0usize),
        ("or-opt     ", t_oropt,   0),
        ("dbl-bridge ", t_db,      n_db),
        ("ins-missing", t_im,      n_im),
        ("trim+merge ", t_trim,    n_trim),
        ("restart    ", t_restart, n_restart),
    ] {
        eprintln!("  {name} {:>8.1}s  ({:>5.1}%)  {}",
            t as f64 / 1e9, t as f64 / total as f64 * 100.0,
            if n > 0 { format!("× {n}") } else { String::new() });
    }
    eprintln!("──────────────────────────────────────────────────\n");

    best
}

// ─── Entry point ──────────────────────────────────────────────────────────────

fn main() {
    let mp   = MultiProgress::new();
    let data = load_data(&mp);

    let best = genetic_algorithm(
        /* pop_size          */ 200,
        /* generations       */ 500,
        /* elite_ratio       */ 0.10,
        /* base_mut_rate     */ 0.40,
        /* ls_iter           */ 8,
        /* db_stag_threshold */ 15,
        /* im_stag_threshold */ 20,
        /* trim_threshold    */ 40,
        /* restart_threshold */ 60,
        /* best_file         */ "best.csv",
        /* checkpoints       */ &["best.csv"],
        &data,
        &mp,
    );

    save_csv(&best, "final.csv");
    drop(mp);

    let gross = best.fitness + best.routes.len() as i64 * HERO_COST as i64;
    let mills: usize = best.routes.iter().map(|r| r.mills.len()).sum();
    println!("\nFinal fitness  : {}", best.fitness);
    println!("Heroes used    : {}", best.routes.len());
    println!("Mills visited  : {}/700", mills);
    println!("Gross reward   : {gross}");
    println!("Hero cost      : {}", best.routes.len() as i64 * HERO_COST as i64);
    println!("Saved to       : final.csv");
}