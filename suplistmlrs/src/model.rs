// Copyright (c) 2025 Chad Voegele
//
// Licensed under the GPL-2.0 License. See LICENSE file for full license information.

use crate::bert;

use bert::MultiBert;
use candle_core::IndexOp;
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::error::Error;
use std::result::Result;
use tokenizers::tokenizer::Tokenizer;

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone, PartialEq, Eq, Ord, PartialOrd)]
pub struct MetaRow {
    pub text: String,
    pub category: String,
    pub name: String,
    pub qty: String,
    pub unit: String,
}

pub fn dequantize_weights(
    tensors: &HashMap<String, Tensor>,
    device: &Device,
) -> candle_core::Result<HashMap<String, Tensor>> {
    let mut dequantized = HashMap::new();

    let quantized_names = tensors
        .keys()
        .filter(|name| {
            name.ends_with(".weight")
                && tensors
                    .keys()
                    .any(|n| n == &name.replace(".weight", ".scale"))
        })
        .cloned()
        .collect::<Vec<_>>();

    for quantized_name in quantized_names.clone() {
        let scale_name = quantized_name.replace(".weight", ".scale");
        if let (Some(weight_tensor), Some(scale_tensor)) =
            (tensors.get(&quantized_name), tensors.get(&scale_name))
        {
            let weight_i8 = weight_tensor.to_device(device)?;
            let scale_f32 = scale_tensor.to_device(device)?.to_dtype(DType::F32)?;

            let weight_f32 = weight_i8.to_dtype(DType::F32)?;
            let dequantized_weight = weight_f32.broadcast_mul(&scale_f32)?;

            dequantized.insert(quantized_name, dequantized_weight);
        } else {
            return Err(candle_core::Error::Msg(format!(
                "Found scale tensor '{}' without corresponding weight tensor",
                scale_name
            )));
        }
    }

    for (name, tensor) in tensors {
        if !quantized_names.contains(name) && !name.ends_with(".scale") {
            dequantized.insert(
                name.clone(),
                tensor.to_device(device)?.to_dtype(DType::F32)?,
            );
        }
    }

    Ok(dequantized)
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

    let class_pred = output.class_logits.argmax(1)?;
    let class_probs = candle_nn::ops::softmax_last_dim(&output.class_logits)?;
    let class_prob = class_probs.index_select(&class_pred, 1)?.i((0, 0))?;
    let unknown_class = "unknown".to_string();
    let unknown_threshold = 0.85;
    let predicted_class = class_tokenizer
        .decode(&class_pred.to_vec1::<u32>()?, false)
        .map_err(|e| e.to_string())?;
    let class = if class_prob.to_scalar::<f32>()? < unknown_threshold {
        unknown_class
    } else {
        predicted_class
    };

    let tag_preds = output.tag_logits.i((0, ..))?.argmax(1)?;
    let non_cls_input_ids = input_ids.i((0, 1..))?;

    let extract_tag = |tag_token: &str| -> Result<String, Box<dyn Error>> {
        let tag_id = tag_tokenizer
            .token_to_id(tag_token)
            .ok_or_else(|| format!("Failed to get token id for tag '{}'", tag_token))?;
        let mask = tag_preds.eq(tag_id)?;
        let indices: Vec<u8> = mask
            .to_vec1::<u8>()?
            .iter()
            .enumerate()
            .filter_map(|(i, &flag)| if flag == 1 { Some(i as u8) } else { None })
            .collect();
        let nindices = indices.len();
        let indices_tensor = Tensor::from_vec(indices, (nindices,), device)?;
        let tag_input_ids = non_cls_input_ids.index_select(&indices_tensor, 0)?;
        let text = tokenizer
            .decode(&tag_input_ids.to_vec1::<u32>()?, false)
            .map_err(|e| e.to_string())?;
        Ok(text)
    };

    let name = extract_tag("NAME")?;
    let qty = extract_tag("QTY")?;
    let unit = extract_tag("UNIT")?;

    Ok(MetaRow {
        text,
        category: class,
        name,
        qty,
        unit,
    })
}
