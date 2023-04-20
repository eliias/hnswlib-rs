use std::cell::RefCell;
use wasm_bindgen::describe::WasmDescribe;
use wasm_bindgen::JsObject;
use wasm_bindgen::prelude::*;
use crate::hnsw::Hnsw;
use crate::prelude::DistL2;

#[wasm_bindgen]
pub struct ANN {
    pub(crate) hnsw: RefCell<Hnsw<f32, DistL2>>
}

#[wasm_bindgen]
impl ANN {
    #[wasm_bindgen(constructor)]
    pub fn new() -> ANN {
        // let anndata = AnnBenchmarkData::new(fname).unwrap();
        // let nb_elem = anndata.train_data.len();
        let nb_elem = 314;
        let max_nb_connection = 24;
        let nb_layer = 16.min((nb_elem as f32).ln().trunc() as usize);
        let ef_c = 400;

        ANN { hnsw: RefCell::from(Hnsw::new(max_nb_connection, nb_elem, nb_layer, ef_c, DistL2{})) }
    }

    #[wasm_bindgen]
    pub fn set_extend_candidates(&self, v: bool) {
        self.hnsw.borrow_mut().set_extend_candidates(v);
    }
}
