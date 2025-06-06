// Copyright (c) 2025 Chad Voegele
//
// Licensed under the GPL-2.0 License. See LICENSE file for full license information.

use candle_core::IndexOp;
use candle_core::{Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};
use candle_transformers::models::bert::Config as BertConfig;
use candle_transformers::models::bert::{BertModel, HiddenAct, PositionEmbeddingType};
use serde::Deserialize;

#[derive(Debug, Clone)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.fc1)?.gelu()?.apply(&self.fc2)
    }
}

impl Mlp {
    fn load(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let fc1 = linear(input_size, hidden_size, vb.pp("fc1"))?;
        let fc2 = linear(hidden_size, output_size, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }
}

pub struct MultiBert {
    bert: BertModel,
    classifier_head: Mlp,
    tag_head: Mlp,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    n_class_labels: usize,
    n_tag_labels: usize,
}

impl MultiBert {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let bert_config: BertConfig = BertConfig {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            intermediate_size: config.intermediate_size,
            hidden_act: HiddenAct::Gelu,
            hidden_dropout_prob: 0.0,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.0,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: PositionEmbeddingType::Absolute,
            use_cache: true,
            classifier_dropout: None,
            model_type: Some("bert".to_string()),
        };
        let bert = BertModel::load(vb.pp("bert"), &bert_config)?;
        let classifier_head = Mlp::load(
            config.hidden_size,
            config.hidden_size / 2,
            config.n_class_labels,
            vb.pp("classifier_head"),
        )?;
        let tag_head = Mlp::load(
            config.hidden_size,
            config.hidden_size / 2,
            config.n_tag_labels,
            vb.pp("tag_head"),
        )?;
        Ok(Self {
            bert,
            classifier_head,
            tag_head,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Output {
    pub class_logits: Tensor,
    pub tag_logits: Tensor,
}

impl MultiBert {
    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        token_type_ids: &Tensor,
    ) -> Result<Output> {
        let sequence_output = self
            .bert
            .forward(input_ids, token_type_ids, attention_mask)?;
        let class_logits = self.classifier_head.forward(&sequence_output.i((.., 0))?)?;
        let tag_logits = self.tag_head.forward(&sequence_output.i((.., 1..))?)?;
        let output = Output {
            class_logits,
            tag_logits,
        };
        Ok(output)
    }
}
