/*!
 * Copyright (c) 2025 Chad Voegele
 *
 * Licensed under the GPL-2.0 License. See LICENSE file for full license information.
 */

/* eslint-env mocha */

import assert from 'assert'
import { JSDOM } from 'jsdom'
import { Renderer } from '../src/dom.js'

describe('render', function () {
  it('should render', function (done) {
    const fakeList = [
      { text: '1 pear', name: 'pear', category: 'fruit' },
      { text: '2 apples', name: 'apples', category: 'fruit' }
    ]
    const suplistMock = () => new Promise((resolve, reject) => resolve(fakeList))

    const { window } = new JSDOM('<!doctype html><html><body><div id="suppalist"></div></body></html>')
    const document = window.document
    Renderer(document, suplistMock).render()

    const output = document.querySelector('#output')
    const observer = new window.MutationObserver((list, observer) => {
      const listEls = output.querySelectorAll('li')
      assert.equal(listEls.length, 2)

      assert.equal(listEls[0].innerHTML, '<span class="token" title="Aisle: fruit, Name: pear">1 pear</span>')
      assert.equal(listEls[1].innerHTML, '<span class="token" title="Aisle: fruit, Name: apples">2 apples</span>')
      observer.disconnect()
      done()
    })
    observer.observe(output, { attributes: true, childList: true, subtree: true })

    const input = document.querySelector('#input')
    input.value = '1 pear\n2 apples'
    input.dispatchEvent(new window.Event('change'))
  })
})
