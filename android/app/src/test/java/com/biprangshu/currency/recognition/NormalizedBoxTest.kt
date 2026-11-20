package com.biprangshu.currency.recognition

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class NormalizedBoxTest {

    // A box in the upright image's top-left area: 10–30% across, 20–60% down.
    private val upright = NormalizedBox(left = 0.1f, top = 0.2f, right = 0.3f, bottom = 0.6f)

    private fun assertBox(expected: NormalizedBox, actual: NormalizedBox) {
        assertEquals(expected.left, actual.left, 1e-6f)
        assertEquals(expected.top, actual.top, 1e-6f)
        assertEquals(expected.right, actual.right, 1e-6f)
        assertEquals(expected.bottom, actual.bottom, 1e-6f)
    }

    @Test
    fun `no rotation leaves the box alone`() {
        assertBox(upright, upright.uprightToBuffer(0))
    }

    @Test
    fun `quarter turn maps the upright top-left to the buffer bottom-left`() {
        // Turning the buffer 90° clockwise put its bottom-left corner at the upright top-left.
        assertBox(NormalizedBox(0.2f, 0.7f, 0.6f, 0.9f), upright.uprightToBuffer(90))
    }

    @Test
    fun `half turn mirrors both axes`() {
        assertBox(NormalizedBox(0.7f, 0.4f, 0.9f, 0.8f), upright.uprightToBuffer(180))
    }

    @Test
    fun `three quarter turn maps the upright top-left to the buffer top-right`() {
        assertBox(NormalizedBox(0.4f, 0.1f, 0.8f, 0.3f), upright.uprightToBuffer(270))
    }

    @Test
    fun `a rotation and its inverse cancel out`() {
        for (degrees in listOf(0, 90, 180, 270)) {
            val back = upright.uprightToBuffer(degrees).uprightToBuffer((360 - degrees) % 360)
            assertBox(upright, back)
        }
    }

    @Test(expected = IllegalArgumentException::class)
    fun `rejects rotations that are not quarter turns`() {
        upright.uprightToBuffer(45)
    }

    @Test
    fun `pixels are normalised and clamped to the frame`() {
        assertBox(
            NormalizedBox(0f, 0.25f, 0.5f, 1f),
            NormalizedBox.fromPixels(-10, 120, 320, 500, width = 640, height = 480),
        )
    }

    @Test
    fun `centre square of a landscape frame excludes the side margins`() {
        // 640×480: the classifier's square spans x = 80..560, i.e. 0.125..0.875.
        fun centredAt(x: Float) = NormalizedBox(x - 0.05f, 0.4f, x + 0.05f, 0.6f)
        assertTrue(centredAt(0.5f).isCenteredInSquareCrop(640, 480))
        assertTrue(centredAt(0.13f).isCenteredInSquareCrop(640, 480))
        assertFalse(centredAt(0.1f).isCenteredInSquareCrop(640, 480))
        assertFalse(centredAt(0.9f).isCenteredInSquareCrop(640, 480))
    }

    @Test
    fun `centre square of a portrait frame excludes the top and bottom margins`() {
        fun centredAt(y: Float) = NormalizedBox(0.4f, y - 0.05f, 0.6f, y + 0.05f)
        assertTrue(centredAt(0.5f).isCenteredInSquareCrop(480, 640))
        assertFalse(centredAt(0.05f).isCenteredInSquareCrop(480, 640))
    }
}
