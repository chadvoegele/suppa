/*!
 * Copyright (c) 2025 Chad Voegele
 *
 * Licensed under the GPL-2.0 License. See LICENSE file for full license information.
 */

import { Renderer } from './dom.js'
import { Model } from './model.js'
import { Proxy } from './proxy.js'

async function main () {
  const proxy = await Proxy()
  const suplist = Model(proxy).suplist
  const renderer = Renderer(document, suplist)
  renderer.render()
  renderer.setExampleInput()
}

main()
