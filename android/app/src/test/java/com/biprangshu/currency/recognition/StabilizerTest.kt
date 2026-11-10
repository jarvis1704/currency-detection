package com.biprangshu.currency.recognition

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class StabilizerTest {

    private val stabilizer = Stabilizer(threshold = 0.7f, framesRequired = 5)

    private fun feed(label: String, frames: Int, confidence: Float = 0.9f): List<String?> =
        List(frames) { stabilizer.update(label, confidence) }

    @Test
    fun `reports a note on the fifth consecutive frame`() {
        assertEquals(listOf(null, null, null, null, "100"), feed("100", 5))
    }

    @Test
    fun `confidence exactly at the threshold counts`() {
        assertEquals("100", feed("100", 5, confidence = 0.7f).last())
    }

    @Test
    fun `one low confidence frame does not restart the count`() {
        feed("100", 4)
        assertNull(stabilizer.update("100", 0.5f))
        assertEquals("100", stabilizer.update("100", 0.9f))
    }

    @Test
    fun `low confidence frames never count towards a report`() {
        assertEquals(List(20) { null }, feed("100", 20, confidence = 0.69f))
    }

    @Test
    fun `frames that left the window no longer count`() {
        feed("100", 4)
        feed("100", 6, confidence = 0.5f)
        assertEquals(listOf(null, null, null, null, "100"), feed("100", 5))
    }

    @Test
    fun `default settings report on the fourth of six frames`() {
        val fast = Stabilizer()
        val confidences = listOf(0.9f, 0.3f, 0.9f, 0.3f, 0.9f, 0.9f)
        assertEquals(
            listOf(null, null, null, null, null, "100"),
            confidences.map { fast.update("100", it) },
        )
    }

    @Test
    fun `a different label restarts the count`() {
        feed("100", 4)
        assertNull(stabilizer.update("500", 0.9f))
        assertEquals(listOf(null, null, null, "500"), feed("500", 4))
    }

    @Test
    fun `background is never reported`() {
        assertEquals(List(20) { null }, feed("background", 20, confidence = 0.99f))
    }

    @Test
    fun `does not repeat a note that stays in view`() {
        feed("100", 5)
        assertEquals(List(20) { null }, feed("100", 20))
    }

    @Test
    fun `a brief background flicker does not cause a repeat`() {
        feed("100", 5)
        feed("background", 4)
        assertEquals(List(10) { null }, feed("100", 10))
    }

    @Test
    fun `repeats the same note after background is seen`() {
        feed("100", 5)
        feed("background", 5)
        assertEquals("100", feed("100", 5).last())
    }

    @Test
    fun `reports a different note without needing background`() {
        feed("100", 5)
        assertEquals("500", feed("500", 5).last())
    }

    @Test
    fun `reset allows the last note to be reported again`() {
        feed("100", 5)
        stabilizer.reset()
        assertEquals("100", feed("100", 5).last())
    }
}
