use rand::{SeedableRng, rngs::SysRng};
use rand_pcg::Pcg64Mcg;

pub fn new_pcg64mcg(seed: Option<u128>) -> Pcg64Mcg {
    if let Some(seed) = seed {
        Pcg64Mcg::new(seed)
    } else {
        Pcg64Mcg::try_from_rng(&mut SysRng)
            .expect("failed to create Pcg64Mcg from SysRng")
    }
}
