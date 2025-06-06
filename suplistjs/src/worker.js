/*!
 * Copyright (c) 2025 Chad Voegele
 *
 * Licensed under the GPL-2.0 License. See LICENSE file for full license information.
 */

/* eslint-env worker */
import init, { Model } from '@suplistmlrs'

let model

async function initWorker () {
  await init()
  model = await getModel()
}

async function downloadUrl (url) {
  const res = await fetch(url, { cache: 'force-cache' })
  return new Uint8Array(await res.arrayBuffer())
}

function concatUint8Arrays (arrays) {
  const totalLength = arrays.reduce((acc, arr) => acc + arr.length, 0)
  const result = new Uint8Array(totalLength)
  let offset = 0
  arrays.forEach((arr) => {
    result.set(arr, offset)
    offset += arr.length
  })
  return result
}

async function getModel () {
  const modelId = '1748084792'
  const tokenizer = await downloadUrl(`./model/tokenizer.${modelId}.json`)
  const tagTokenizer = await downloadUrl(`./model/tag_tokenizer.${modelId}.json`)
  const classTokenizer = await downloadUrl(`./model/class_tokenizer.${modelId}.json`)
  const config = await downloadUrl(`./model/config.${modelId}.json`)
  const nWeightParts = 4
  const weightParts = Array.from(Array(nWeightParts).keys()).map((i) => downloadUrl(`./model/model.${modelId}.safetensors.part_0${i}`))
  const weights = concatUint8Arrays(await Promise.all(weightParts))

  const model = new Model(
    tokenizer,
    tagTokenizer,
    classTokenizer,
    weights,
    config
  )

  return model
}

function predict (lines) {
  const rows = model.run(lines)
  return rows
}

async function handler (event) {
  if (event.data.method === 'init') {
    await initWorker()
    self.postMessage('inited')
  } else if (event.data.method === 'predict') {
    const rows = predict(event.data.lines)
    self.postMessage(rows)
  }
}

self.addEventListener('message', handler)
