// Copyright (c) 2025 Chad Voegele
//
// Licensed under the GPL-2.0 License. See LICENSE file for full license information.

use crate::bert;

use bert::MultiBert;
use candle_core::IndexOp;
use candle_core::{Device, Tensor};
use std::error::Error;
use std::result::Result;
use tokenizers::tokenizer::Tokenizer;

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone, PartialEq, Eq, Ord, PartialOrd)]
pub struct MetaRow {
    pub text: String,
    pub category: String,
    pub name: String,
}

pub fn infer_text(
    tokenizer: &Tokenizer,
    tag_tokenizer: &Tokenizer,
    class_tokenizer: &Tokenizer,
    model: &MultiBert,
    text: String,
) -> Result<MetaRow, Box<dyn Error>> {
    let device = &Device::Cpu;

    let encoded = tokenizer.encode(text.clone(), false);
    let encoded_u = encoded.map_err(|e| e.to_string())?;
    let encoded_ids = encoded_u.get_ids();

    let input_ids = Tensor::new(encoded_ids, &device)?.unsqueeze(0)?;
    let token_type_ids = input_ids.zeros_like()?;
    let attention_mask = input_ids.ones_like()?;
    let output = model.forward(&input_ids, Some(&attention_mask), &token_type_ids)?;

    let class_preds = output.class_logits.argmax(1)?;
    let class = class_tokenizer
        .decode(&class_preds.to_vec1::<u32>()?, false)
        .map_err(|e| e.to_string())?;

    let name_token = "NAME";
    let name_id = tag_tokenizer
        .token_to_id(name_token)
        .expect("Failed to get name token id");
    let tag_preds = output.tag_logits.i((0, ..))?.argmax(1)?;
    let mask = tag_preds.eq(name_id)?;

    let indices: Vec<u8> = mask
        .to_vec1::<u8>()?
        .iter()
        .enumerate()
        .filter_map(|(i, &flag)| if flag == 1 { Some(i as u8) } else { None })
        .collect();
    let nindices = indices.len();

    let non_cls_input_ids = input_ids.i((0, 1..))?;
    let indices_tensor = Tensor::from_vec(indices, (nindices,), device)?;
    let name_input_ids = non_cls_input_ids.index_select(&indices_tensor, 0)?;
    let name = tokenizer
        .decode(&name_input_ids.to_vec1::<u32>()?, false)
        .map_err(|e| e.to_string())?;

    Ok(MetaRow {
        text: text,
        category: class,
        name: name,
    })
}
