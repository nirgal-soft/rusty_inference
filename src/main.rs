use candle_core::Device;
pub mod activations;
pub mod block;
pub mod inference;
pub mod layers;
pub mod model;
pub mod sampling;
pub mod utils;

fn main() -> anyhow::Result<()>{
  let model_path = "models/gpt2-small/model.safetensors";

  println!("=== inspecting model file ===");
  utils::list_tensors(model_path)?;
  println!();

  // let device = Device::new_metal(0).unwrap_or(Device::Cpu);
  let device = Device::Cpu;
  println!("device: {:?}", device);

  println!("loading model...");
  let model = model::load_model(model_path, &device)?;

  println!("loading tokenizer...");
  let tokenizer = tokenizers::Tokenizer::from_file("models/gpt2-small/tokenizer.json")
    .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {}", e))?;

  let engine = inference::InferenceEngine::new(model, tokenizer, device);

  let prompt = "once upon a time";
  let params = sampling::SamplingParams{
    temperature: 0.8,
    top_k: Some(50),
    top_p: Some(0.95),
    max_tokens: 100,
  };

  println!("prompt: {}", prompt);
  println!("generated text:");
  let output = engine.generate(prompt, &params)?;
  println!("{}", output);

  Ok(())
}
