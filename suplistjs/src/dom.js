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
  }

  function renderInput (output) {
    const listInput = document.createElement('textarea')
    listInput.id = 'input'

    listInput.addEventListener('change', async (event) => {
      const text = event.target.value
      await setOutput(output, text)
    })
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
    const textNode = document.createTextNode(row.text)
    textSp.appendChild(textNode)
    textSp.className = 'token'
    const hoverText = `Aisle: ${row.category}, Name: ${row.name}`
    textSp.title = hoverText

    const rowEl = document.createElement('li')
    rowEl.append(textSp)

    return rowEl
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
}
