/** Copy `text`, reporting honestly whether it worked.

    `navigator.clipboard` is absent on non-secure origins — which includes the LAN
    demo this tool actually gets shown on (`http://<ip>:8600`) — and can reject even
    where present. The hidden-textarea + `execCommand` path is the fallback that
    still works there. Returns false rather than throwing, because a Copy button that
    silently does nothing is the failure mode worth engineering against. */
export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    const ta = document.createElement('textarea')
    ta.value = text
    ta.style.position = 'fixed'
    ta.style.opacity = '0'
    document.body.appendChild(ta)
    ta.select()
    let ok = false
    try {
      ok = document.execCommand('copy')
    } catch {
      ok = false
    }
    ta.remove()
    return ok
  }
}
