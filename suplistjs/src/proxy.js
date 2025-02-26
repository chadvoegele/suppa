/*!
 * Copyright (c) 2025 Chad Voegele
 *
 * Licensed under the GPL-2.0 License. See LICENSE file for full license information.
 */

/* eslint-env worker */

export async function Proxy () {
  let worker
  let presolve

  await initProxy()
  await initWorker()

  return {
    predict
  }

  function handler (event) {
    if (presolve) {
      presolve(event.data)
      presolve = undefined
    }
  }

  async function initProxy () {
    worker = new Worker(new URL('./worker.js', import.meta.url))
    worker.addEventListener('message', handler)
  }

  async function initWorker () {
    return new Promise((resolve, reject) => {
      presolve = resolve
      worker.postMessage({ method: 'init' })
    })
  }

  async function predict (lines) {
    return new Promise((resolve, reject) => {
      presolve = resolve
      worker.postMessage({ method: 'predict', lines })
    })
  }
}
