/*!
 * Copyright (c) 2025 Chad Voegele
 *
 * Licensed under the GPL-2.0 License. See LICENSE file for full license information.
 */

/* eslint-env mocha */

import assert from 'assert'

import { Model } from '../src/model.js'

describe('suplist', function () {
  it('should suplist', async function () {
    const text = '# Fruit Salad\n\n1 pear\n2 apples'
    const mockProxy = {
      predict: async function (textLines) {
        assert.deepStrictEqual(textLines, ['1 pear', '2 apples'])
        return [
          { text: '1 pear', name: 'pear', category: 'fruit' },
          { text: '2 apples', name: 'apple', category: 'fruit' }
        ]
      }
    }
    const model = Model(mockProxy)
    const list = await model.suplist(text)
    const expectedList = [{
      name: 'apple',
      category: 'fruit',
      text: '2 apples'
    }, {
      name: 'pear',
      category: 'fruit',
      text: '1 pear'
    }]
    assert.deepStrictEqual(list, expectedList)
  })
})
