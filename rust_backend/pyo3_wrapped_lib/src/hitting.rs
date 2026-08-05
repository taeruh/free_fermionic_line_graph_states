use std::collections::HashSet;

use pyo3::{PyResult, Python, types::PyModuleMethods};

use crate::Module;

type LibHitter = core_lib::hitting::Hitter;

macro_rules! make_hitter {
    ($name:ident, $ty:ty) => {
        #[pyo3::pyclass(subclass)]
        pub struct $name(LibHitter);

        #[pyo3::pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = (seed=None))]
            fn __new__(seed: Option<u128>) -> Self {
                Self(LibHitter::new(seed))
            }

            fn hit(
                &mut self,
                sets_to_hit: Vec<Vec<$ty>>,
                weights: Vec<f64>,
                randomise: bool,
            ) -> HashSet<$ty> {
                self.0.hit::<$ty>(sets_to_hit, weights, randomise)
            }
        }
    };
}

make_hitter!(HitterU8, u8);
make_hitter!(HitterU16, u16);
make_hitter!(HitterU32, u32);
make_hitter!(HitterU64, u64);
make_hitter!(HitterU128, u128);
make_hitter!(HitterUsize, usize);

pub fn add_this_module(py: Python<'_>, parent_module: &Module) -> PyResult<()> {
    let module = Module::new(py, "hitting", parent_module.path.clone())?;
    module.pymodule.add_class::<HitterU8>()?;
    module.pymodule.add_class::<HitterU16>()?;
    module.pymodule.add_class::<HitterU32>()?;
    module.pymodule.add_class::<HitterU64>()?;
    module.pymodule.add_class::<HitterU128>()?;
    module.pymodule.add_class::<HitterUsize>()?;
    parent_module.add_submodule(py, module)?;
    Ok(())
}
