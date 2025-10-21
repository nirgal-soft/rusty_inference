use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, Module, Linear, Embedding, LayerNorm};
use crate::activations::gelu;
use crate::model::ModelConfig;

pub struct MLP{
  c_fc: Linear,
  c_proj: Linear,
}

impl MLP{
  pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self>{
    let n_embd = cfg.n_embd;
    let c_fc = candle_nn::linear(n_embd, 4*n_embd, vb.pp("c_fc"))?;
    let c_proj = candle_nn::linear(4*n_embd, n_embd, vb.pp("c_projf"))?;

    Ok(Self{c_fc, c_proj})
  }

  pub fn forward(&self, x: &Tensor) -> Result<Tensor>{
    let x = self.c_fc.forward(x)?;
    let x = gelu(&x)?;
    let x = self.c_proj.forward(&x)?;
    Ok(x)
  }
}
