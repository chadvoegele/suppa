/*!
 * Copyright (c) 2025 Chad Voegele
 *
 * Licensed under the GPL-2.0 License. See LICENSE file for full license information.
 */

export function Combiner () {
  return {
    combine
  }

  function combine (rows) {
    if (rows.length === 0) {
      return []
    }

    const namedSortedRows = rows.toSorted((a, b) => {
      if (getName(a) === undefined) {
        return 1
      }

      return getName(a).localeCompare(getName(b))
    })

    const combinedRows = []

    let previousRows = [namedSortedRows[0]]
    namedSortedRows.slice(1).forEach((row) => {
      const previousName = getName(previousRows[previousRows.length - 1])
      const name = getName(row)
      if (previousName !== undefined && previousName === name) {
        previousRows.push(row)
        return
      }

      const combinedPreviousRows = combineRows(previousRows)
      combinedRows.push(combinedPreviousRows)
      previousRows = [row]
    })

    const combinedPreviousRows = combineRows(previousRows)
    combinedRows.push(combinedPreviousRows)

    return combinedRows
  }

  function combineRows (rows) {
    const combinedText = [rows[0].text]
    const separator = '+'
    rows.slice(1).forEach((row) => {
      combinedText.push(separator)
      combinedText.push(row.text)
    })
    const combinedRows = {
      text: combinedText.join(' '),
      category: rows[0].category,
      name: getName(rows[0])
    }
    return combinedRows
  }

  function getName (row) {
    if (row.name === undefined) {
      return undefined
    }
    return singularize(row.name)
  }

  function singularize (name) {
    // Portions from https://github.com/ixmatus/inflector
    if (name.endsWith('ches') || name.endsWith('shes') || name.endsWith('sses') || name.endsWith('xes')) {
      const singularName = name.slice(0, -2)
      return singularName
    }

    if (name.endsWith('s')) {
      const singularName = name.slice(0, -1)
      return singularName
    }

    return name
  }
}
