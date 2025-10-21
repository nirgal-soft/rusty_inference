pub struct SamplingParams{
  pub temperature: f64,
  pub top_k: Option<usize>,
  pub top_p: Option<f64>,
  pub max_tokens: usize,
}

impl Default for SamplingParams{
  fn default() -> Self{
    Self{
      temperature: 1.0,
      top_k: None,
      top_p: None,
      max_tokens: 100,
    }
  }
}
