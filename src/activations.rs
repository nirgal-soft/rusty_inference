use candle_core::{Result, Tensor};

pub fn gelu(x: &Tensor) -> Result<Tensor>{
  x.gelu()
}
