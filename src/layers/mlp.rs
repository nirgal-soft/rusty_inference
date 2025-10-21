use candle_core::{Result, Tensor};
use candle_nn::{VarBuilder, Module, Linear};
use crate::activations::gelu;
use crate::model::{ModelConfig, linear_with_bias};

pub struct MLP{
  c_fc: Linear,
  c_proj: Linear,
}

impl MLP{
  pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self>{
    let n_embd = cfg.n_embd;
    let c_fc = linear_with_bias(n_embd, 4*n_embd, vb.pp("c_fc"))?;
    let c_proj = linear_with_bias(4*n_embd, n_embd, vb.pp("c_proj"))?;

    Ok(Self{c_fc, c_proj})
  }

  pub fn forward(&self, x: &Tensor) -> Result<Tensor>{
    let x = self.c_fc.forward(x)?;
    let x = x.gelu()?;
    let x = self.c_proj.forward(&x)?;
    Ok(x)
  }
}
