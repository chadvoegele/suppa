// Copyright (c) 2025 Chad Voegele
//
// Licensed under the GPL-2.0 License. See LICENSE file for full license information.

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use std::convert::From;
use std::env;
use std::path::PathBuf;
use tokenizers::tokenizer::Tokenizer;

use suplistmlrs::bert::{Config, MultiBert};
use suplistmlrs::model::infer_text;

fn resolve_lfs_path(path: &str) -> String {
    let text = std::fs::read_to_string(path).expect("Failed to read file");
    let lines: Vec<&str> = text.lines().collect();
    let hash_line = lines.get(1).expect("File format is incorrect");
    let hash = hash_line.split_whitespace().last().expect("Hash not found");
    let lfs_root = env::var("SUPPA_LFS_ROOT").expect("LFS_ROOT environment variable not set");
    let lfs_path = PathBuf::from(lfs_root).join(&hash[7..]);
    lfs_path.to_str().expect("Invalid path").to_string()
}

fn main() {
    let tokenizer_file = "../suplistml/suplistml/models/run+1733494653/tokenizer.json";
    let tokenizer_lfs_file = resolve_lfs_path(tokenizer_file);
    let tokenizer = Tokenizer::from_file(tokenizer_lfs_file).expect("failed to load tokenizer");

    let confile_file = "../suplistml/suplistml/models/run+1733494653/config.json";
    let config_lfs_file = resolve_lfs_path(confile_file);
    let file = std::fs::File::open(config_lfs_file).unwrap();
    let config: Config = serde_json::from_reader(file).unwrap();

    let safetensors_file = "../suplistml/suplistml/models/run+1733494653/model.safetensors";
    let safetensors_lfs_file = resolve_lfs_path(safetensors_file);
    let filenames = safetensors_lfs_file
        .split(',')
        .map(std::path::PathBuf::from)
        .collect::<Vec<_>>();
    let device = &Device::Cpu;
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&filenames, DType::F32, &device).unwrap() };
    let model = MultiBert::load(vb, &config).unwrap();

    let class_tokenizer_file = "../suplistml/suplistml/models/run+1733494653/class_tokenizer.json";
    let class_tokenizer_lfs_file = resolve_lfs_path(class_tokenizer_file);
    let class_tokenizer =
        Tokenizer::from_file(class_tokenizer_lfs_file).expect("failed to load tokenizer");

    let tag_tokenizer_file = "../suplistml/suplistml/models/run+1733494653/tag_tokenizer.json";
    let tag_tokenizer_lfs_file = resolve_lfs_path(tag_tokenizer_file);
    let tag_tokenizer =
        Tokenizer::from_file(tag_tokenizer_lfs_file).expect("failed to load tokenizer");

    // let text = "[CLS] 2 green and small apples, diced";
    let text = "[CLS] salt and coarsely-ground pepper";
    let inferred = infer_text(
        &tokenizer,
        &tag_tokenizer,
        &class_tokenizer,
        &model,
        text.to_string(),
    )
    .unwrap();
    println!("{:?}", inferred.text);
    println!("{:?}", inferred.category);
    println!("{:?}", inferred.name);
}
