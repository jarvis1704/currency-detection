package com.biprangshu.currency.recognition

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Test

class NoteHighlighterTest {

    private val highlighter = NoteHighlighter(threshold = 0.7f, holdFrames = 2)
    private val box = NormalizedBox(0.2f, 0.2f, 0.8f, 0.8f)
    private val otherBox = NormalizedBox(0.3f, 0.3f, 0.7f, 0.7f)

    @Test
    fun `outlines a confident note where the locator found it`() {
        assertEquals(NoteHighlight("500", box), highlighter.update("500", 0.9f) { box })
    }

    @Test
    fun `does not look for a note in background frames`() {
        var located = false
        assertNull(highlighter.update("background", 0.99f) { located = true; box })
        assertFalse(located)
    }

    @Test
    fun `does not look for a note in low confidence frames`() {
        var located = false
        assertNull(highlighter.update("500", 0.5f) { located = true; box })
        assertFalse(located)
    }

    @Test
    fun `nothing is outlined when the locator finds nothing`() {
        assertNull(highlighter.update("500", 0.9f) { null })
    }

    @Test
    fun `follows the note as it moves`() {
        highlighter.update("500", 0.9f) { box }
        assertEquals(NoteHighlight("500", otherBox), highlighter.update("500", 0.9f) { otherBox })
    }

    @Test
    fun `keeps the outline through a short gap then drops it`() {
        highlighter.update("500", 0.9f) { box }
        assertEquals(NoteHighlight("500", box), highlighter.update("background", 0.9f) { null })
        assertEquals(NoteHighlight("500", box), highlighter.update("500", 0.9f) { null })
        assertNull(highlighter.update("background", 0.9f) { null })
    }

    @Test
    fun `a fresh box restarts the gap allowance`() {
        highlighter.update("500", 0.9f) { box }
        highlighter.update("background", 0.9f) { null }
        highlighter.update("500", 0.9f) { box }
        highlighter.update("background", 0.9f) { null }
        assertEquals(NoteHighlight("500", box), highlighter.update("background", 0.9f) { null })
    }

    @Test
    fun `reset drops the outline`() {
        highlighter.update("500", 0.9f) { box }
        highlighter.reset()
        assertNull(highlighter.update("background", 0.9f) { null })
    }
}
