use std::fs::File;
use std::io::Read;
use safetensors::SafeTensors;
use candle_core::Result;

pub fn list_tensors(path: &str) -> Result<()>{
  let mut file = File::open(path)
    .map_err(|e| candle_core::Error::Msg(format!("failed to open file: {}", e)))?;
  let mut buffer = Vec::new();
  file.read_to_end(&mut buffer)
    .map_err(|e| candle_core::Error::Msg(format!("failed to read file: {}", e)))?;

  let tensors = SafeTensors::deserialize(&buffer)
    .map_err(|e| candle_core::Error::Msg(format!("failed to deserialize file: {}", e)))?;

  println!("tensors in fiel:");
  let mut names: Vec<_> = tensors.names();
  names.sort();
  for name in names{
    if let Ok(tensor) = tensors.tensor(name){
      println!("  {} : {:?}", name, tensor.shape());
    }
  }

  Ok(())
}
