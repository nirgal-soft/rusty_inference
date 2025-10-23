use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, Module, Embedding, LayerNorm, Linear};
use crate::block::Block;

pub fn linear_with_bias(in_features: usize, out_features: usize, vb: VarBuilder) -> Result<Linear> {
  let weight = vb.get((in_features, out_features), "weight")?;
  let weight = weight.t()?;
  let bias = vb.get(out_features, "bias")?;
  Ok(Linear::new(weight, Some(bias)))
}

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

#[allow(dead_code)]
pub struct GPT{
  pub wte: Embedding,
  pub wpe: Embedding,
  pub blocks: Vec<Block>,
  pub ln_f: LayerNorm,
  pub config: ModelConfig,
}

impl GPT{
  pub fn new(cfg: ModelConfig, vb: VarBuilder) -> Result<Self>{
    let wte = candle_nn::embedding(cfg.vocab_size, cfg.n_embd, vb.pp("wte"))?;
    let wpe = candle_nn::embedding(cfg.n_ctx, cfg.n_embd, vb.pp("wpe"))?;

    let mut blocks = Vec::new();
    for i in 0..cfg.n_layer{
      let block_vb = vb.pp(format!("h.{}", i));
      blocks.push(Block::new(&cfg, block_vb)?);
    }

    let ln_f = candle_nn::layer_norm(cfg.n_embd, cfg.eps as f64, vb.pp("ln_f"))?;

    Ok(Self{wte, wpe, blocks, ln_f, config: cfg})
  }

  pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor>{
    let (_b, t) = input_ids.dims2()?;
    let device = input_ids.device();

    let tok_emb = self.wte.forward(input_ids)?;
    let pos_ids = Tensor::arange(0u32, t as u32, device)?;
    let pos_emb = self.wpe.forward(&pos_ids)?.unsqueeze(0)?;
    let mut x = (&tok_emb + &pos_emb)?;

    // Run through ALL blocks
    for block in &self.blocks{
      x = block.forward(&x, None)?;
    }

    let x = self.ln_f.forward(&x)?;

    let (b, t, c) = x.dims3()?;
    let x_flat = x.reshape((b*t, c))?;
    let output_weights = self.wte.embeddings().t()?;
    let logits_flat = x_flat.matmul(&output_weights)?;
    let logits = logits_flat.reshape((b, t, self.config.vocab_size))?;

    Ok(logits)
  }
}

pub fn load_model(path: &str, device: &Device) -> Result<GPT>{
  let config = ModelConfig::gpt2_small();
  let vb = unsafe{
    VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)?
  };
  GPT::new(config, vb)
}
