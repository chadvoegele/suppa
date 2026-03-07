/*!
 * Copyright (c) 2025 Chad Voegele
 *
 * Licensed under the GPL-2.0 License. See LICENSE file for full license information.
 */

import './style.css'

export function Renderer (document, suplist) {
  return {
    render,
    setExampleInput
  }

  function render () {
    const page = renderPage()
    const appEl = document.getElementById('suppalist')
    appEl.replaceChildren(page)
  }

  function renderPage () {
    const container = document.createElement('div')
    container.className = 'root'

    const output = renderOutput()
    container.appendChild(renderInput(output))
    container.appendChild(output)
    return container
  }

  function renderOutput () {
    const listOutput = document.createElement('div')
    listOutput.id = 'output'
    return listOutput
  }

  async function setOutput (output, text) {
    Array.from(output.childNodes).forEach(n => output.removeChild(n))
    const list = await suplist(text)
    const formattedList = renderList(list)
    output.appendChild(formattedList)

    const copyButton = renderCopyButton()
    output.appendChild(copyButton)
  }

  function debounce (delay, func) {
    let debounceTimer
    return function (...args) {
      return new Promise((resolve) => {
        clearTimeout(debounceTimer)
        debounceTimer = setTimeout(() => {
          resolve(func.apply(this, args))
        }, delay)
      })
    }
  }

  function renderInput (output) {
    const listInput = document.createElement('textarea')
    listInput.id = 'input'

    listInput.addEventListener('input', debounce(1000, async (event) => {
      const text = event.target.value
      await setOutput(output, text)
    }))
    return listInput
  }

  function renderList (list) {
    const formattedList = list.map(renderRow)
    const listEl = document.createElement('ul')
    formattedList.forEach(t => listEl.appendChild(t))
    return listEl
  }

  function renderRow (row) {
    const textSp = document.createElement('span')
    const nParts = Math.max(row.qty.length, row.unit.length) || 1
    const qtyUnitPart = [...Array(nParts).keys()].map(i => `${row.qty[i] || ''} ${row.unit[i] || ''}`).join(' + ').trim() || ''
    const displayText = qtyUnitPart.length > 0 ? `${qtyUnitPart} ${row.name}` : row.text.join(' + ')
    textSp.appendChild(document.createTextNode(displayText))
    textSp.className = 'token'
    const hoverText = `Aisle: ${row.category}, Texts: ${row.text.join(' + ')}`
    textSp.title = hoverText

    const rowEl = document.createElement('li')
    rowEl.append(textSp)

    return rowEl
  }

  function renderCopyButton () {
    const copyButton = document.createElement('button')
    copyButton.innerHTML = '📋'
    copyButton.title = 'Copy list to clipboard'
    copyButton.className = 'copy-button'
    copyButton.addEventListener('click', () => copyListToClipboard())
    return copyButton
  }

  function getExampleInput () {
    const example = `# Chicken Noodle Soup
1 medium onion, chopped
3 medium carrots, diced
2 medium celery stalks, chopped
4 bone-in chicken thighs
2 cups rotini

# Red Beans and Rice
1 medium onion, chopped fine
1/2 cup chopped green bell pepper
1/2 cup chopped celery
3 small garlic cloves, minced
1 pound small red beans, rinsed and picked over
10 ounces andouille sausage
`
    return example
  }

  function setExampleInput () {
    const input = document.getElementById('input')
    input.value = getExampleInput()

    const output = document.getElementById('output')
    setOutput(output, getExampleInput())
  }

  function copyListToClipboard () {
    const output = document.getElementById('output')
    const ul = output.querySelector('ul')
    if (!ul) return

    const listItems = ul.querySelectorAll('li')
    const textContent = Array.from(listItems)
      .map(li => li.textContent.trim())
      .join('\n')

    navigator.clipboard.writeText(textContent).then(() => {
      const copyButton = output.querySelector('.copy-button')
      const originalText = copyButton.innerHTML
      const originalColor = copyButton.style.color
      copyButton.innerHTML = '✓'
      copyButton.style.color = 'green'
      setTimeout(() => {
        copyButton.innerHTML = originalText
        copyButton.style.color = originalColor
      }, 1000)
    }).catch(err => {
      console.error('Failed to copy text: ', err)
    })
  }
}
