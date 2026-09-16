"""People deliberately kept off the map, and why.

A Commons search occasionally returns a gallery that is not the person at
all. The consensus step cannot catch it -- the images agree with each other
perfectly well, they are just of somebody else -- and the duplicate-identity
check only helps when the real subject also happens to be in the dataset.

So this is the manual escape hatch. Keep it small: if a case here turns out
to represent a whole class of failure, fix the class instead. Every entry
carries the reason it was added, so a later reader can tell a deliberate
removal from a forgotten one, and re-check it when the collector changes.
"""

EXCLUDED: dict[str, str] = {
    "Dominic Holland": (
        "Commons returns only Plautilla Nelli's 'Madonna and Child with Sts "
        "Dominic and Catherine of Siena' -- the search matched the saint, not "
        "the comedian, and no image in the gallery is of him."
    ),
}
