use std::env;
use std::fs;

fn main() {
    // Get the target OS
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    
    if target_os == "macos" {
        let path = "/opt/homebrew/opt/openblas/lib";
        if !fs::exists(path).unwrap() {
            println!("cargo:error=Building on macOS. Note that you may need to install OpenBLAS via Homebrew for this crate to work properly.");
        }
        // Add link search path for OpenBLAS
        println!("cargo:rustc-link-search=/opt/homebrew/opt/openblas/lib");
    }
}