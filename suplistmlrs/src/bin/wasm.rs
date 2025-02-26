// Copyright (c) 2025 Chad Voegele
//
// Licensed under the GPL-2.0 License. See LICENSE file for full license information.

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use suplistmlrs::bert::{Config, MultiBert};
use suplistmlrs::model::{infer_text, ModelOutput};
use tokenizers::tokenizer::Tokenizer;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Model {
    tokenizer: Tokenizer,
    tag_tokenizer: Tokenizer,
    class_tokenizer: Tokenizer,
    model: MultiBert,
}

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub fn load(
        tokenizer: Vec<u8>,
        tag_tokenizer: Vec<u8>,
        class_tokenizer: Vec<u8>,
        weights: Vec<u8>,
        config: Vec<u8>,
    ) -> Result<Model, JsError> {
        console_error_panic_hook::set_once();
        let tokenizer =
            Tokenizer::from_bytes(tokenizer).map_err(|x| JsError::new(&x.to_string()))?;
        let tag_tokenizer =
            Tokenizer::from_bytes(tag_tokenizer).map_err(|x| JsError::new(&x.to_string()))?;
        let class_tokenizer =
            Tokenizer::from_bytes(class_tokenizer).map_err(|x| JsError::new(&x.to_string()))?;
        let device = &Device::Cpu;
        let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, device)?;
        let config: Config = serde_json::from_slice(&config)?;
        let model = MultiBert::load(vb, &config)?;

        Ok(Self {
            tokenizer,
            tag_tokenizer,
            class_tokenizer,
            model,
        })
    }

    pub fn run(&mut self, text: String) -> Result<JsValue, JsError> {
        let cls_text = "[CLS] ".to_string() + &text;
        let inferred = infer_text(
            &self.tokenizer,
            &self.tag_tokenizer,
            &self.class_tokenizer,
            &self.model,
            cls_text.to_string(),
        )
        .map_err(|x| JsError::new(&x.to_string()))?;

        let output = ModelOutput {
            text: text,
            category: inferred.category,
            name: inferred.name,
        };

        Ok(serde_wasm_bindgen::to_value(&output)?)
    }
}

fn main() {
    console_error_panic_hook::set_once();
}
