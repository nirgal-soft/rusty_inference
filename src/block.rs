use candle_core::{Result, Tensor};
use candle_nn::{VarBuilder, Module, LayerNorm};
use crate::model::ModelConfig;
use crate::layers::{attention::Attention, mlp::MLP};

pub struct Block{
  ln1: LayerNorm,
  attn: Attention,
  ln2: LayerNorm,
  mlp: MLP,
}

impl Block{
  pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self>{
    let ln1 = candle_nn::layer_norm(cfg.n_embd, cfg.eps as f64, vb.pp("ln_1"))?;
    let attn = Attention::new(cfg, vb.pp("attn"))?;
    let ln2 = candle_nn::layer_norm(cfg.n_embd, cfg.eps as f64, vb.pp("ln_2"))?;
    let mlp = MLP::new(cfg, vb.pp("mlp"))?;

    Ok(Self{ln1, attn, ln2, mlp})
  }

  pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor>{
    let resid = x.clone();
    let x = self.ln1.forward(x)?;
    let x = self.attn.forward(&x, mask)?;
    let x = (x + resid)?;

    let resid = x.clone();
    let x = self.ln2.forward(&x)?;
    let x = self.mlp.forward(&x)?;
    let x = (x + resid)?;

    Ok(x)
  }
}
