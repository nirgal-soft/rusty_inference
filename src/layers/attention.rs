use candle_core::{Result, Tensor, IndexOp};
use candle_nn::{VarBuilder, Module, Linear,};
use crate::model::{ModelConfig, linear_with_bias};

#[allow(dead_code)]
pub struct Attention{
  c_attn: Linear,
  c_proj: Linear,
  bias: Option<Tensor>,
  n_head: usize,
  n_embd: usize,
}

impl Attention{
  pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self>{
    let n_embd = cfg.n_embd;
    let n_head = cfg.n_head;

    let c_attn = linear_with_bias(n_embd, 3 * n_embd, vb.pp("c_attn"))?;
    let c_proj = linear_with_bias(n_embd, n_embd, vb.pp("c_proj"))?;

    let bias = vb.get((1, 1, cfg.n_ctx, cfg.n_ctx), "bias").ok();

    Ok(Self{c_attn, c_proj, bias, n_head, n_embd})
  }

  pub fn forward(&self, x: &Tensor, _mask: Option<&Tensor>) -> Result<Tensor>{
    let(b, t, c) = x.dims3()?;
    let head_dim = c / self.n_head;

    //compute qkv
    let qkv = self.c_attn.forward(x)?;
    let q = qkv.narrow(2, 0, c)?;
    let k = qkv.narrow(2, c, c)?;
    let v = qkv.narrow(2, 2*c, c)?;

    //reshape to [batch, n_head, seq_len, head_dim]
    let q = q.reshape((b, t, self.n_head, head_dim))?
      .transpose(1, 2)?;
    let k = k.reshape((b, t, self.n_head, head_dim))?
      .transpose(1, 2)?;
    let v = v.reshape((b, t, self.n_head, head_dim))?
      .transpose(1, 2)?;

    //scaled dot product attention
    let scores = q.matmul(&k.t()?)?;
    let scale = (head_dim as f64).sqrt();
    let scores = (scores/scale)?;

    //apply casual mask if provided
    // let scores = if let Some(mask) = mask{
    //   scores.broadcast_add(mask)?
    let scores = if let Some(bias) = &self.bias{
      let mask = bias.i((.., .., ..t, ..t))?;
      let mask = ((mask - 1.0)? * 1e10)?;
      scores.broadcast_add(&mask)?
    }else{
      scores
    };

    //softmax and apply to values
    let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
    let out = attn_weights.matmul(&v)?;

    //reshape back to [batch, seq_len, n_embd]
    let out = out.transpose(1, 2)?
      .reshape((b, t, c))?;

    //output projection
    self.c_proj.forward(&out)
  }
}
