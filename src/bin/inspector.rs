use safetensors::SafeTensors;
use memmap2::Mmap;
use std::fs::File;

fn main() -> anyhow::Result<()> {
    let file = File::open("model.safetensors")?;
    let mmap = unsafe { Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let keys = st.names();
    for key in keys {
        println!("{}", key);
    }
    Ok(())
}
