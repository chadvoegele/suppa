/*!
 * Copyright (c) 2025 Chad Voegele
 *
 * Licensed under the GPL-2.0 License. See LICENSE file for full license information.
 */

import assert from 'assert'

import { Combiner } from '../src/combiner.js'

describe('combine same names', function () {
  it('should combine same names', function () {
    const combiner = Combiner()

    const rows = [
      { text: '1 pear', category: 'fruit', name: 'pear' },
      { text: 'salt', category: 'condiments', name: 'salt' },
      { text: 'apple', category: 'fruit', name: 'apple' },
      { text: '1 tsp salt', category: 'condiments', name: 'salt' }
    ]

    const combinedRows = combiner.combine(rows)

    const expectedRows = [
      { text: 'apple', category: 'fruit', name: 'apple' },
      { text: '1 pear', category: 'fruit', name: 'pear' },
      { text: 'salt + 1 tsp salt', category: 'condiments', name: 'salt' }
    ]

    assert.deepStrictEqual(expectedRows, combinedRows)
  })

  it('should not combine when name is missing', function () {
    const combiner = Combiner()

    const rows = [
      { text: 'salt', category: 'condiments', name: 'salt' },
      { text: 'saltines', category: 'condiments', name: undefined }
    ]

    const combinedRows = combiner.combine(rows)

    const expectedRows = [
      { text: 'salt', category: 'condiments', name: 'salt' },
      { text: 'saltines', category: 'condiments', name: undefined }
    ]

    assert.deepStrictEqual(expectedRows, combinedRows)
  })

  it('should handle plural forms', function () {
    const combiner = Combiner()

    const rows = [
      { text: 'peach', category: 'produce', name: 'peach' },
      { text: '2 peaches', category: 'produce', name: 'peaches' }
    ]

    const combinedRows = combiner.combine(rows)

    const expectedRows = [
      { text: 'peach + 2 peaches', category: 'produce', name: 'peach' }
    ]

    assert.deepStrictEqual(expectedRows, combinedRows)
  })

  it('should not combine unnamed', function () {
    const combiner = Combiner()

    const rows = [
      { text: 'bar', category: 'produce', name: undefined },
      { text: '2 hersheys kiss', category: 'produce', name: undefined }
    ]

    const combinedRows = combiner.combine(rows)

    const expectedRows = [
      { text: 'bar', category: 'produce', name: undefined },
      { text: '2 hersheys kiss', category: 'produce', name: undefined }
    ]

    assert.deepStrictEqual(expectedRows, combinedRows)
  })
})
