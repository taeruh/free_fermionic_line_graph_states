use std::collections::HashSet;

use rand::seq::SliceRandom;
use rand_pcg::Pcg64Mcg;

use crate::random;

pub struct Hitter {
    rng: Pcg64Mcg,
}

impl Hitter {
    pub fn new(seed: Option<u128>) -> Self {
        Self {
            rng: random::new_pcg64mcg(seed),
        }
    }

    // TODO: replace "unwrap" with unsafe "unwrap_unchecked" eventually if convinced that
    // it never fails
    pub fn hit<UI>(
        &mut self,
        mut sets_to_hit: Vec<Vec<UI>>,
        mut weights: Vec<f64>,
        randomise: bool,
    ) -> HashSet<UI>
    where
        // PERF: UI will be u(8|16|32|64|128|size); I'm trusting here that the compiler
        // will optimize the conversions into simple casts (and in the case of usize,
        // no-op)
        UI: TryFrom<usize>
            + TryInto<usize>
            + Copy
            + Ord
            + std::hash::Hash
            + std::ops::SubAssign,
        <UI as std::convert::TryFrom<usize>>::Error: std::fmt::Debug,
        <UI as std::convert::TryInto<usize>>::Error: std::fmt::Debug,
    {
        let mut solution_vec = Vec::new();
        // let mut to_hit: HashSet<UI> =
        //     HashSet::from_iter((0..sets_to_hit.len()).map(|i| i as UI));
        // TODO: change this later back to the hashset version; at the moment, I want it
        // deterministic for testing purposes
        let mut to_hit: Vec<UI> =
            Vec::from_iter((0..sets_to_hit.len()).map(|i| i.try_into().unwrap()));

        if randomise {
            sets_to_hit.shuffle(&mut self.rng);
        }

        while !to_hit.is_empty() {
            let pricing_set = sets_to_hit
                .get((*to_hit.first().unwrap()).try_into().unwrap())
                .unwrap();
            let mut iter = pricing_set.iter();
            let mut min_element = *iter.next().unwrap();
            let mut min_weight = weights[min_element.try_into().unwrap()];
            for &element in iter {
                let weight = weights[element.try_into().unwrap()];
                if weight < min_weight {
                    min_element = element;
                    min_weight = weight;
                }
            }
            for &element in pricing_set {
                weights[element.try_into().unwrap()] -= min_weight;
            }
            solution_vec.push(min_element);
            to_hit
                .retain(|&i| !sets_to_hit[i.try_into().unwrap()].contains(&min_element));
        }

        let mut solution = HashSet::from_iter(solution_vec.iter().copied());

        for element in solution_vec {
            let mut solution_reduced = solution.clone();
            solution_reduced.remove(&element);
            let mut hit_all = true;
            for set_to_hit in sets_to_hit.iter() {
                if set_to_hit.iter().any(|&e| solution_reduced.contains(&e)) {
                    continue;
                } else {
                    hit_all = false;
                    break;
                }
            }
            if hit_all {
                solution = solution_reduced;
            }
        }

        solution
    }
}
