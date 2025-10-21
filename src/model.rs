use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, Module, Linear, Embedding, LayerNorm};

#[derive(Debug, Clone)]
pub struct ModelConfig{
  pub vocab_size: usize,
  pub n_ctx: usize,
  pub n_embd: usize,
  pub n_head: usize,
  pub n_layer: usize,
  pub eps: f32
}

impl ModelConfig{
  pub fn gpt2_small() -> Self{
    Self{
      vocab_size: 50257,
      n_ctx: 1024,
      n_embd: 768,
      n_head: 12,
      n_layer: 12,
      eps: 1e-5
    }
  }

  pub fn head_dim(&self) -> usize{
    self.n_embd / self.n_head
  }
}
