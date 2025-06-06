#!/usr/bin/env node

import fs from 'fs'
import path from 'path'
import { execSync } from 'child_process'

const args = process.argv.slice(2)
const tag = args[0] || '1748084792'
const outputDir = args[1] || 'static/model'
const inputDir = args[2] || '../suplistml/src/suplistml/models/run+' + tag

const LFS_ROOT = process.env.SUPPA_LFS_ROOT

if (!LFS_ROOT) {
  console.error('LFS_ROOT environment variable not set')
  process.exit(1)
}

function resolveLfsFile (srcPath) {
  const binaryContent = fs.readFileSync(srcPath)
  const content = binaryContent.toString('utf8')

  const sha256Match = content.match(/oid sha256:(\w+)/)
  if (!sha256Match) {
    return srcPath
  }

  const sha256 = sha256Match[1]
  const lfsFilePath = path.join(LFS_ROOT, sha256)

  if (!fs.existsSync(lfsFilePath)) {
    console.warn(`LFS file not found: ${lfsFilePath}. Using original file.`)
    throw new Error(`LFS file not found: ${lfsFilePath}`)
  }

  return lfsFilePath
}

if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true })
}

function copyAndRename (filename) {
  const srcPath = path.join(inputDir, filename)
  const lfsSrcPath = resolveLfsFile(srcPath)
  const destPath = path.join(outputDir, `${path.parse(filename).name}.${tag}.json`)
  fs.copyFileSync(lfsSrcPath, destPath)
  console.log(`Copied ${srcPath} via ${lfsSrcPath} to ${destPath}`)
}

['class_tokenizer.json', 'tag_tokenizer.json', 'tokenizer.json', 'config.json'].forEach(copyAndRename)

const modelSrcPath = path.join(inputDir, 'model.safetensors')
const modelLfsSrcPath = resolveLfsFile(modelSrcPath)
const modelDestPrefix = path.join(outputDir, `model.${tag}.safetensors.part_`)

try {
  execSync(`split -d -b 20M "${modelLfsSrcPath}" "${modelDestPrefix}" --numeric-suffixes=00`)
  console.log(`Split ${modelSrcPath} via ${modelLfsSrcPath} into parts in ${outputDir}`)
} catch (error) {
  console.error('Error splitting the model file:', error.message)
}
