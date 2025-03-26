/*!
 * Copyright (c) 2025 Chad Voegele
 *
 * Licensed under the GPL-2.0 License. See LICENSE file for full license information.
 */

export function Model (proxy, combiner) {
  return {
    suplist
  }

  async function suplist (text) {
    const textLines = text.split(/\n/).filter(l => l.length > 0).filter(l => !isComment(l))
    const rows = await proxy.predict(textLines)
    const sortedRows = sortRows(rows)
    return sortedRows
  }

  function isComment (line) {
    return line.trim().startsWith('#')
  }

  function sortRows (rows) {
    const ORDER = ['produce', 'vegetables', 'meat', 'seafood', 'deli', 'bakery', 'dairy', 'baking', 'frozen foods', 'beverages', 'cond', 'condiment', 'condiments', 'grains', 'nuts', 'pasta', 'spices']
    const orderMap = Object.fromEntries(ORDER.map((c, i) => [c, i]))

    rows.sort((a, b) => {
      const aCategory = orderMap[a.category]
      const aSortString = a.name || ''
      const bCategory = orderMap[b.category]
      const bSortString = b.name || ''
      if (aCategory < bCategory) {
        return -1
      }
      if (aCategory > bCategory) {
        return 1
      }
      if (aSortString < bSortString) {
        return -1
      }
      if (aSortString > bSortString) {
        return 1
      }
      return 0
    })
    return rows
  }
}
