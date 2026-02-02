use safetensors::SafeTensors;
use memmap2::Mmap;
use std::fs::File;
use std::env;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    let path = args.get(1).map(|s| s.as_str()).unwrap_or("model.safetensors");
    
    println!("Inspecting: {}", path);
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    
    let mut names: Vec<_> = st.names().iter().map(|s| s.to_string()).collect();
    names.sort();
    
    for name in names {
        println!("{}", name);
    }
    Ok(())
}
