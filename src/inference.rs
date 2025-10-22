use candle_core::{Device, Result, Tensor, IndexOp};
use tokenizers::Tokenizer;
use std::io::Write;

use crate::model::GPT;
use crate::sampling::SamplingParams;

pub struct InferenceEngine{
  model: GPT,
  tokenizer: Tokenizer,
  device: Device,
}

impl InferenceEngine{
  pub fn new(model: GPT, tokenizer: Tokenizer, device: Device) -> Self{
    Self{model, tokenizer, device,}
  }

  pub fn generate(&self, prompt: &str, params: &SamplingParams) -> Result<String>{
    let encoding = self.tokenizer
      .encode(prompt, false)
      .map_err(|e| candle_core::Error::Msg(format!("tokenization error: {}", e)))?;
    let mut tokens = encoding.get_ids().to_vec();

    println!("starting generation with {} initial tokens", tokens.len());

    for _i in 0..params.max_tokens{
      let input_ids = Tensor::new(&tokens[..], &self.device)?
        .unsqueeze(0)?;

      let logits = self.model.forward(&input_ids)?;

      let last_logits = logits.i((0, tokens.len() - 1))?;


      // let logits_vec = last_logits.to_vec1::<f32>()?;
      // let mut indexed: Vec<(usize, f32)> = logits_vec.iter()
      //   .copied()
      //   .enumerate()
      //   .collect();
      // indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
      // println!("Top 5 logits:");
      // for (idx, logit) in indexed.iter().take(5) {
      //   println!("  Token {}: {:.4}", idx, logit);
      // }

      // Check logit statistics
      let logits_vec = last_logits.to_vec1::<f32>()?;
      let max_logit = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
      let min_logit = logits_vec.iter().cloned().fold(f32::INFINITY, f32::min);
      let mean_logit: f32 = logits_vec.iter().sum::<f32>() / logits_vec.len() as f32;

      println!("Token {}: max={:.2}, min={:.2}, mean={:.2}", 
      tokens.len(), max_logit, min_logit, mean_logit);
      let next_token = self.sample(&last_logits, params)?;

      // if self.is_eos_token(next_token){
      //   break;
      // }

      tokens.push(next_token);

      if let Ok(decoded) = self.tokenizer.decode(&[next_token], true){
        print!("{}", decoded);
        std::io::stdout().flush().unwrap();
      }
    }

    println!();
    let generated = self.tokenizer
      .decode(&tokens, true)
      .map_err(|e| candle_core::Error::Msg(format!("decoding error: {}", e)))?;

    Ok(generated)
  }

  fn sample(&self, logits: &Tensor, params: &SamplingParams) -> Result<u32>{
    let logits = logits.to_vec1::<f32>()?;

    //apply temp
    let logits = if params.temperature != 1.0{
      logits.iter()
        .map(|&x| x / params.temperature as f32)
        .collect::<Vec<_>>()
    }else{
      logits
    };

    //apply top-k
    let logits = if let Some(k) = params.top_k{
      self.top_k_filter(&logits, k)
    }else{
      logits
    };

    //apply top-p
    let logits = if let Some(p) = params.top_p{
      self.top_p_filter(&logits, p)
    }else{
      logits
    };

    self.sample_from_logits(&logits)
  }

  fn top_k_filter(&self, logits: &[f32], k: usize) -> Vec<f32>{
    let mut indexed: Vec<(usize, f32)> = logits.iter()
      .copied()
      .enumerate()
      .collect();

    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let threshod = indexed[k.min(indexed.len() - 1)].1;
    logits.iter()
      .map(|&x| if x > threshod {x} else {f32::NEG_INFINITY})
      .collect()
  }

  fn top_p_filter(&self, logits: &[f32], p: f64) -> Vec<f32>{
    let mut indexed: Vec<(usize, f32)> = logits.iter()
      .copied()
      .enumerate()
      .collect();

    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let max_logit = indexed[0].1;
    let exp_sum: f32 = indexed.iter()
      .map(|(_, l)| (l - max_logit).exp())
      .sum();

    let mut cumulative_sum = 0.0;
    let mut threshold_idx = indexed.len();

    for(i, (_, logit)) in indexed.iter().enumerate(){
      let prob = (logit-max_logit).exp() / exp_sum;
      cumulative_sum += prob as f64;
      if cumulative_sum >= p{
        threshold_idx = i + 1;
        break;
      }
    }

    let kept_indices: std::collections::HashSet<usize> = indexed.iter()
      .take(threshold_idx)
      .map(|(i, _)| *i)
      .collect();

    logits.iter()
      .enumerate()
      .map(|(i, &x)| if kept_indices.contains(&i) {x} else {f32::NEG_INFINITY})
      .collect()
  }

  fn sample_from_logits(&self, logits: &[f32]) -> Result<u32>{
    let max_logit = logits.iter()
      .copied()
      .fold(f32::NEG_INFINITY, f32::max);

    let exp_logits: Vec<f32> = logits.iter()
      .map(|&x| (x - max_logit).exp())
      .collect();

    let sum: f32 = exp_logits.iter().sum();
    let probs: Vec<f32> = exp_logits.iter()
      .map(|&x| x / sum)
      .collect();

    use std::time::{SystemTime, UNIX_EPOCH};
    let seed = SystemTime::now()
      .duration_since(UNIX_EPOCH)
      .unwrap()
      .as_nanos() as u64;

    let rng = (seed.wrapping_mul(1103515245).wrapping_add(12345)) & (1 << 31);
    let r = (rng as f64) / ((1u64 << 31) as f64);

    let mut cumulative_sum = 0.0;
    for(i, &prob) in probs.iter().enumerate(){
      cumulative_sum += prob as f64;
      if r < cumulative_sum{
        return Ok(i as u32);
      }
    }

    Ok((probs.len() - 1) as u32)
  }

  fn is_eos_token(&self, token: u32) -> bool{
    token == 50256 || token == 0
  }
}
